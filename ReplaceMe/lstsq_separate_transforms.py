"""Least Squares Transformation with SEPARATE transforms for Attention and MLP

This version computes separate linear transformations for:
1. Attention output → Attention output (across multiple layers)
2. MLP output → MLP output (across multiple layers)

This gives more fine-grained control and potentially better approximations.
"""

import argparse
import gc
import logging
import os
from typing import Optional, Dict, Tuple
import torch
import yaml
from colorama import Fore, init
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

from .utils import (get_calib_dataloader, select_non_overlapping_blocks, seed_all)

# Initialize colorama
init(autoreset=True)

# Configure logging
logging.basicConfig(
    format=(
        f"{Fore.CYAN}%(asctime)s "
        f"{Fore.YELLOW}[%(levelname)s] "
        f"{Fore.RESET}%(message)s"
    ),
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

seed_all()


def lstsq_separate_transforms(
    model_path: str,
    dataset: str,
    dataset_column: str,
    batch_size: int,
    max_length: int,
    layers_to_skip: int,
    dataset_size: Optional[int] = None,
    dataset_subset: Optional[str] = "eval",
    use_4bit: bool = False,
    save_path: Optional[str] = None,
    token: Optional[str] = None,
    alpha: float = 0,
    distances_path: str = "./distances.pth",
    num_A: int = 1,
    merge_consecutive: bool = True,
) -> Dict:
    """Compute SEPARATE linear transformations for Attention and MLP components.

    This function computes two sets of transformations:
    1. Attention Transform: Maps attention_output[layer_i] → attention_output[layer_j]
    2. MLP Transform: Maps mlp_output[layer_i] → mlp_output[layer_j]

    Args:
        model_path: Path to pretrained model
        dataset: Name of dataset to use for calibration
        dataset_column: Column in dataset containing text
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        layers_to_skip: Number of layers between compared blocks
        dataset_size: Optional size limit for dataset
        dataset_subset: Subset of dataset to use (train/eval)
        use_4bit: Whether to use 4-bit quantization
        save_path: Path to save transformation matrices
        token: Authentication token for private models
        alpha: Regularization strength (Ridge regression)
        distances_path: Path to precomputed distance metrics
        num_A: Number of transformation blocks
        merge_consecutive: Whether to merge consecutive blocks

    Returns:
        Dictionary containing attention and MLP transforms separately
    """
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    quantization_config = None

    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    logging.info(f"Loading model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        quantization_config=quantization_config,
        output_hidden_states=True,
        token=token,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    hidden_size = model.config.hidden_size
    num_layers_total = model.config.num_hidden_layers

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    logging.info(f"Loading calibration dataset: {dataset}")
    dataloader = get_calib_dataloader(
        dataset,
        dataset_subset,
        dataset_column,
        dataset_size,
        batch_size,
        tokenizer,
    )

    # Setup activation hooks for BOTH attention and MLP
    def save_attention_activation(name: str):
        """Hook to capture attention output."""
        def hook(module, input, output):
            # output[0] is the attention output (before residual)
            attention_activations[name] = output[0].detach() if isinstance(output, tuple) else output.detach()
        return hook

    def save_mlp_activation(name: str):
        """Hook to capture MLP output."""
        def hook(module, input, output):
            mlp_activations[name] = output.detach()
        return hook

    hooks = []
    attention_activations = {}
    mlp_activations = {}
    model_type = 'falcon' if 'falcon' in model_path.lower() else 'default'

    if model_type == 'falcon':
        for i, layer in enumerate(model.transformer.h):
            hooks.append(layer.self_attention.register_forward_hook(
                save_attention_activation(f'layer_{i}_attn')))
            hooks.append(layer.mlp.register_forward_hook(
                save_mlp_activation(f'layer_{i}_mlp')))
    else:
        for i, layer in enumerate(model.model.layers):
            hooks.append(layer.self_attn.register_forward_hook(
                save_attention_activation(f'layer_{i}_attn')))
            hooks.append(layer.mlp.register_forward_hook(
                save_mlp_activation(f'layer_{i}_mlp')))

    # Load precomputed distances and select blocks
    logging.info(f"Loading distances from: {distances_path}")
    average_distances = torch.load(distances_path)
    selected_blocks = select_non_overlapping_blocks(
        average_distances,
        layers_to_skip,
        num_blocks=num_A,
        merge_consecutive=merge_consecutive,
    )

    logging.info(f"Selected blocks for transformation: {selected_blocks}")

    # Initialize accumulation matrices for BOTH attention and MLP
    start_ids = sorted([x[0] for x in selected_blocks])
    end_ids = sorted([x[1] for x in selected_blocks])

    # Attention transforms
    attn_a1t_a1 = [
        torch.zeros(hidden_size, hidden_size, device='cuda').to(torch.float64)
        for _ in range(len(selected_blocks))
    ]
    attn_a1t_a2 = [
        torch.zeros(hidden_size, hidden_size, device='cuda').to(torch.float64)
        for _ in range(len(selected_blocks))
    ]

    # MLP transforms
    mlp_a1t_a1 = [
        torch.zeros(hidden_size, hidden_size, device='cuda').to(torch.float64)
        for _ in range(len(selected_blocks))
    ]
    mlp_a1t_a2 = [
        torch.zeros(hidden_size, hidden_size, device='cuda').to(torch.float64)
        for _ in range(len(selected_blocks))
    ]

    # Process batches
    logging.info("Processing batches to compute transformation matrices...")
    for batch in tqdm(
        dataloader,
        desc=f"{Fore.GREEN}Computing Separate Transforms{Fore.RESET}",
        dynamic_ncols=True,
        colour="green",
    ):
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding="longest",
            max_length=max_length,
            truncation=True,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # Process each selected block
        for idx, (start_id, end_id) in enumerate(selected_blocks):
            # Get attention activations
            attn_start = attention_activations[f'layer_{start_id-1}_attn'].view(-1, hidden_size).to(torch.float64)
            attn_end = attention_activations[f'layer_{end_id-1}_attn'].view(-1, hidden_size).to(torch.float64)

            # Get MLP activations
            mlp_start = mlp_activations[f'layer_{start_id-1}_mlp'].view(-1, hidden_size).to(torch.float64)
            mlp_end = mlp_activations[f'layer_{end_id-1}_mlp'].view(-1, hidden_size).to(torch.float64)

            # Accumulate for attention transform
            attn_a1t_a1[idx] += attn_start.t() @ attn_start
            attn_a1t_a2[idx] += attn_start.t() @ attn_end

            # Accumulate for MLP transform
            mlp_a1t_a1[idx] += mlp_start.t() @ mlp_start
            mlp_a1t_a2[idx] += mlp_start.t() @ mlp_end

    # Compute transformations for both attention and MLP
    logging.info("Computing transformation matrices using least squares...")
    reg_term = alpha * torch.eye(hidden_size, device='cuda').to(torch.float64)

    attention_transforms = []
    mlp_transforms = []

    for idx in range(len(selected_blocks)):
        # Compute attention transform
        attn_transform = torch.linalg.inv(attn_a1t_a1[idx] + reg_term) @ attn_a1t_a2[idx]
        attention_transforms.append(attn_transform)

        # Compute MLP transform
        mlp_transform = torch.linalg.inv(mlp_a1t_a1[idx] + reg_term) @ mlp_a1t_a2[idx]
        mlp_transforms.append(mlp_transform)

        logging.info(f"Block {idx+1}/{len(selected_blocks)} (layers {selected_blocks[idx][0]}-{selected_blocks[idx][1]}):")
        logging.info(f"  Attention transform norm: {attn_transform.norm().item():.4f}")
        logging.info(f"  MLP transform norm: {mlp_transform.norm().item():.4f}")

    # Clean up hooks and model
    for hook in hooks:
        hook.remove()
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Prepare save path
    if save_path is None:
        os.makedirs('transforms_separate', exist_ok=True)
        layer_indices_for_name = '__'.join([f"{start_ids[i]}_{end_ids[i]}" for i in range(len(selected_blocks))])
        save_path = os.path.join(
            "transforms_separate",
            f"{model_path}_{layers_to_skip}layers_{layer_indices_for_name}_{dataset}_{dataset_size}".replace("/", "_")
        )

    # Save transforms with metadata
    save_dict = {
        'attention_transforms': [t.cpu() for t in attention_transforms],
        'mlp_transforms': [t.cpu() for t in mlp_transforms],
        'selected_blocks': selected_blocks,
        'start_ids': start_ids,
        'end_ids': end_ids,
        'metadata': {
            'model_path': model_path,
            'layers_to_skip': layers_to_skip,
            'dataset': dataset,
            'dataset_size': dataset_size,
            'hidden_size': hidden_size,
            'num_layers_total': num_layers_total,
            'alpha': alpha,
            'method': 'lstsq_separate',
            'note': 'Separate transforms for attention and MLP components'
        }
    }

    final_save_path = f"{save_path}_separate_transforms.pt"
    torch.save(save_dict, final_save_path)
    logging.info(f"{Fore.GREEN}✓ Separate transformation matrices saved to: {final_save_path}{Fore.RESET}")

    # Save human-readable summary
    summary_path = f"{save_path}_separate_transforms_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("SEPARATE LINEAR TRANSFORMATIONS FOR ATTENTION AND MLP\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Total layers in model: {num_layers_total}\n")
        f.write(f"Hidden size: {hidden_size}\n")
        f.write(f"Layers to skip: {layers_to_skip}\n")
        f.write(f"Number of blocks: {len(selected_blocks)}\n")
        f.write(f"Regularization (alpha): {alpha}\n\n")

        f.write("=" * 80 + "\n")
        f.write("TRANSFORM DETAILS\n")
        f.write("=" * 80 + "\n\n")

        for i, (start, end) in enumerate(selected_blocks):
            f.write(f"Block {i+1}: Layers {start} to {end} ({end-start} layers removed)\n")
            f.write(f"\n  Attention Transform:\n")
            f.write(f"    Shape: {attention_transforms[i].shape}\n")
            f.write(f"    Norm: {attention_transforms[i].norm().item():.6f}\n")
            f.write(f"    Mean: {attention_transforms[i].mean().item():.6f}\n")
            f.write(f"    Std: {attention_transforms[i].std().item():.6f}\n")

            attn_identity_diff = (attention_transforms[i] - torch.eye(hidden_size, dtype=attention_transforms[i].dtype)).abs().mean().item()
            f.write(f"    Distance from identity: {attn_identity_diff:.6f}\n")

            f.write(f"\n  MLP Transform:\n")
            f.write(f"    Shape: {mlp_transforms[i].shape}\n")
            f.write(f"    Norm: {mlp_transforms[i].norm().item():.6f}\n")
            f.write(f"    Mean: {mlp_transforms[i].mean().item():.6f}\n")
            f.write(f"    Std: {mlp_transforms[i].std().item():.6f}\n")

            mlp_identity_diff = (mlp_transforms[i] - torch.eye(hidden_size, dtype=mlp_transforms[i].dtype)).abs().mean().item()
            f.write(f"    Distance from identity: {mlp_identity_diff:.6f}\n\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("HOW TO USE THESE TRANSFORMS:\n")
        f.write("=" * 80 + "\n")
        f.write("1. Load: data = torch.load('separate_transforms.pt')\n")
        f.write("2. Access attention transforms: data['attention_transforms']\n")
        f.write("3. Access MLP transforms: data['mlp_transforms']\n")
        f.write("4. Apply to model:\n")
        f.write("   - Remove layers from selected_blocks[i][0] to selected_blocks[i][1]\n")
        f.write("   - Apply attention_transform to self_attn.o_proj of layer before removed block\n")
        f.write("   - Apply mlp_transform to mlp.down_proj of layer before removed block\n")
        f.write("   - Formula: new_weight = transform.T @ old_weight\n")

    logging.info(f"✓ Summary saved to: {summary_path}")

    return {
        'attention_transforms': attention_transforms,
        'mlp_transforms': mlp_transforms,
        'selected_blocks': selected_blocks,
        'save_path': final_save_path,
        'metadata': save_dict['metadata']
    }


def read_config(config_path: str) -> dict:
    """Read and parse YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_from_config() -> None:
    """Run separate transform computation from configuration file."""
    parser = argparse.ArgumentParser(
        description="Compute separate LSTSQ transforms for Attention and MLP."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file.",
    )
    args = parser.parse_args()
    config = read_config(args.config)
    result = lstsq_separate_transforms(**config)

    print("\n" + "="*80)
    print(f"{Fore.GREEN}SUCCESS!{Fore.RESET}")
    print("="*80)
    print(f"Computed {len(result['attention_transforms'])} attention transforms")
    print(f"Computed {len(result['mlp_transforms'])} MLP transforms")
    print(f"Saved to: {result['save_path']}")
    print("\nTo load:")
    print(f"  data = torch.load('{result['save_path']}')")
    print("  attention_transforms = data['attention_transforms']")
    print("  mlp_transforms = data['mlp_transforms']")
