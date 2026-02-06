"""Least Squares Transformation module - TRANSFORM ONLY VERSION

This modified version computes and saves ONLY the linear transformation matrices
without pruning the model or merging weights.
"""

import argparse
import gc
import logging
import os
from typing import Optional, List, Dict
import torch
import yaml
from colorama import Fore, init
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

from .utils import (get_calib_dataloader, select_non_overlapping_blocks, seed_all)

# Initialize colorama for Windows compatibility
init(autoreset=True)

# Configure logging to display colored messages and timestamps
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


def lstsq_transform_only(
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
    diag: bool = False,
    alpha: float = 0,
    distances_path: str = "./distances.pth",
    num_A: int = 1,
    merge_consecutive: bool = True,
) -> Dict:
    """Compute least squares transformations between model layers WITHOUT pruning.

    This function ONLY computes and saves the linear transformation matrices.
    It does NOT prune the model or merge weights.

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
        diag: Whether to use diagonal matrix approximation
        alpha: Regularization strength (Ridge regression)
        distances_path: Path to precomputed distance metrics
        num_A: Number of transformation matrices to compute
        merge_consecutive: Whether to merge consecutive blocks

    Returns:
        Dictionary containing:
            - 'transforms': List of transformation matrices (as tensors)
            - 'selected_blocks': List of (start_layer, end_layer) tuples
            - 'save_path': Path where transforms were saved
            - 'metadata': Additional information about the transforms
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
    # Load model and tokenizer
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

    # Setup activation hooks
    def save_mlp_activation(name: str):
        """Returns a hook function that saves module outputs."""
        def hook(module, input, output):
            mlp_activations[name] = output.detach()
        return hook

    hooks = []
    mlp_activations = {}
    model_type = 'falcon' if 'falcon' in model_path.lower() else 'default'

    if model_type == 'falcon':
        for i, layer in enumerate(model.transformer.h):
            hooks.append(layer.mlp.register_forward_hook(save_mlp_activation(f'layer_{i}_mlp')))
    else:
        for i, layer in enumerate(model.model.layers):
            hooks.append(layer.mlp.register_forward_hook(save_mlp_activation(f'layer_{i}_mlp')))

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

    # Initialize accumulation matrices
    start_ids = sorted([x[0] for x in selected_blocks])
    end_ids = sorted([x[1] for x in selected_blocks])
    num_layers = [end_ids[i] - start_ids[i] for i in range(len(start_ids))]
    num_layers_cumsum = [sum(num_layers[:i]) for i in range(len(start_ids)+1)]

    a1t_a1 = [
        torch.zeros(hidden_size, hidden_size, device='cuda').to(torch.float64)
        for _ in range(len(selected_blocks))
    ]
    a1t_a2 = [
        torch.zeros(hidden_size, hidden_size, device='cuda').to(torch.float64)
        for _ in range(len(selected_blocks))
    ]

    # Process batches
    logging.info("Processing batches to compute transformation matrices...")
    for batch in tqdm(
        dataloader,
        desc=f"{Fore.GREEN}Computing LSTSQ Transformations{Fore.RESET}",
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

        hidden_states = outputs.hidden_states[1:]

        # Get relevant hidden states
        hidden_states_mlp = [
            mlp_activations[f'layer_{i}_mlp'].view(-1, hidden_size).to(torch.float64)
            for i in range(num_layers_total) if i+1 in start_ids
        ]
        hidden_states_i = [
            hidden_states[i].view(-1, hidden_size).to(torch.float64)
            for i in range(num_layers_total) if i+1 in start_ids
        ]
        hidden_states_n = [
            hidden_states[i].view(-1, hidden_size).to(torch.float64)
            for i in range(num_layers_total) if i+1 in end_ids
        ]

        # Accumulate matrices
        for idx in range(len(selected_blocks)):
            dev = hidden_states_mlp[idx].device
            if idx == 0 or start_ids[idx] != end_ids[idx-1]:
                a1_batch = hidden_states_mlp[idx].to(dev)
                a2_batch = hidden_states_n[idx].to(dev) + hidden_states_mlp[idx].to(dev) - hidden_states_i[idx].to(dev)
            else:
                a1_batch = hidden_states_i[idx].to(dev)
                a2_batch = hidden_states_n[idx].to(dev)

            a1t_a1[idx] += a1_batch.t().to(a1t_a1[idx].device) @ a1_batch.to(a1t_a1[idx].device)
            a1t_a2[idx] += a1_batch.t().to(a1t_a2[idx].device) @ a2_batch.to(a1t_a2[idx].device)

    # Compute transformations
    logging.info("Computing transformation matrices using least squares...")
    transforms = []
    for idx in range(len(selected_blocks)):
        if diag:
            transform = torch.diag(
                torch.linalg.inv(
                    a1t_a1[idx] * torch.eye(hidden_size, device='cuda').to(torch.float64)
                ) @ torch.diag(a1t_a2[idx])
            )
        else:
            reg_term = alpha * torch.eye(hidden_size, device='cuda').to(torch.float64)
            transform = torch.linalg.inv(a1t_a1[idx] + reg_term) @ a1t_a2[idx]
        transforms.append(transform)
        logging.info(f"Transform {idx+1}/{len(selected_blocks)}: shape={transform.shape}, "
                    f"dtype={transform.dtype}, norm={transform.norm().item():.4f}")

    # Clean up hooks and model
    for hook in hooks:
        hook.remove()
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Prepare save path
    if save_path is None:
        os.makedirs('transforms_only', exist_ok=True)
        layer_indices_for_name = '__'.join([f"{start_ids[i]}_{end_ids[i]}" for i in range(len(selected_blocks))])
        save_path = os.path.join(
            "transforms_only",
            f"{model_path}_{layers_to_skip}layers_{layer_indices_for_name}_{dataset}_{dataset_size}".replace("/", "_")
        )

    # Save transforms with metadata
    save_dict = {
        'transforms': [t.cpu() for t in transforms],
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
            'diag': diag,
            'method': 'lstsq'
        }
    }

    final_save_path = f"{save_path}_transforms_only.pt"
    torch.save(save_dict, final_save_path)
    logging.info(f"{Fore.GREEN}✓ Transformation matrices saved to: {final_save_path}{Fore.RESET}")

    # Also save a human-readable summary
    summary_path = f"{save_path}_transforms_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("LINEAR TRANSFORMATION MATRICES SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Total layers in model: {num_layers_total}\n")
        f.write(f"Hidden size: {hidden_size}\n")
        f.write(f"Layers to skip: {layers_to_skip}\n")
        f.write(f"Number of transforms: {len(transforms)}\n")
        f.write(f"Regularization (alpha): {alpha}\n")
        f.write(f"Diagonal only: {diag}\n\n")

        f.write("Selected blocks for transformation:\n")
        for i, (start, end) in enumerate(selected_blocks):
            f.write(f"  Block {i+1}: Layers {start} to {end} ({end-start} layers removed)\n")
            f.write(f"    Transform shape: {transforms[i].shape}\n")
            f.write(f"    Transform norm: {transforms[i].norm().item():.6f}\n")
            f.write(f"    Transform dtype: {transforms[i].dtype}\n\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("HOW TO USE THESE TRANSFORMS:\n")
        f.write("=" * 80 + "\n")
        f.write("1. Load the transforms: data = torch.load('transforms_only.pt')\n")
        f.write("2. Access transforms: transforms = data['transforms']\n")
        f.write("3. Access block info: selected_blocks = data['selected_blocks']\n")
        f.write("4. To apply to a model:\n")
        f.write("   - Remove layers from selected_blocks[i][0] to selected_blocks[i][1]\n")
        f.write("   - Multiply transform[i] into the MLP down_proj of layer before removed block\n")
        f.write("   - Formula: new_weight = transform.T @ old_weight\n")

    logging.info(f"✓ Summary saved to: {summary_path}")

    return {
        'transforms': transforms,
        'selected_blocks': selected_blocks,
        'save_path': final_save_path,
        'metadata': save_dict['metadata']
    }


def read_config(config_path: str) -> dict:
    """Read and parse YAML configuration file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Parsed configuration dictionary
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_from_config() -> None:
    """Run transform-only computation from configuration file."""
    parser = argparse.ArgumentParser(
        description="Compute LSTSQ linear transforms WITHOUT pruning the model."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file.",
    )
    args = parser.parse_args()
    config = read_config(args.config)
    result = lstsq_transform_only(**config)

    print("\n" + "="*80)
    print(f"{Fore.GREEN}SUCCESS!{Fore.RESET}")
    print("="*80)
    print(f"Computed {len(result['transforms'])} transformation matrices")
    print(f"Saved to: {result['save_path']}")
    print("\nTo load these transforms:")
    print(f"  data = torch.load('{result['save_path']}')")
    print("  transforms = data['transforms']")
    print("  selected_blocks = data['selected_blocks']")
