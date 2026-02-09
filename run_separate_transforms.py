#!/usr/bin/env python3
"""
Simple script to run SEPARATE transform computation (Attention + MLP).

Usage:
    python run_separate_transforms.py
"""

import os
import sys
import torch
from pathlib import Path

# Add ReplaceMe to path
sys.path.insert(0, str(Path(__file__).parent / "ReplaceMe"))

from ReplaceMe.lstsq_separate_transforms import lstsq_separate_transforms


def main():
    print("="*80)
    print("SEPARATE TRANSFORMS COMPUTATION (Attention + MLP)")
    print("="*80)

    # Configuration
    config = {
        'model_path': 'meta-llama/Llama-2-7b-hf',
        'dataset': 'Open-Orca/SlimOrca',
        'dataset_column': 'text',
        'batch_size': 4,
        'max_length': 512,
        'layers_to_skip': 8,
        'dataset_size': 100,
        'dataset_subset': 'train',
        'use_4bit': True,
        'alpha': 0.0,
        'distances_path': './distances.pth',
        'num_A': 1,
        'merge_consecutive': True,
        'save_path': './my_separate_transforms'
    }

    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Check if distances.pth exists
    if not os.path.exists(config['distances_path']):
        print("\n" + "!"*80)
        print("WARNING: distances.pth not found!")
        print("You need to run distance computation first.")
        print("!"*80)
        return

    print("\n" + "="*80)
    print("Starting separate transform computation...")
    print("="*80 + "\n")

    try:
        result = lstsq_separate_transforms(**config)
        print("\n" + "="*80)
        print("✓ SUCCESS!")
        print("="*80)

        inspect_results(result)

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


def inspect_results(result):
    """Inspect and display the computed transforms."""

    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    attention_transforms = result['attention_transforms']
    mlp_transforms = result['mlp_transforms']
    selected_blocks = result['selected_blocks']
    save_path = result['save_path']
    metadata = result['metadata']

    print(f"\n📁 Saved to: {save_path}")
    print(f"\n📊 Number of blocks: {len(selected_blocks)}")
    print(f"   - Attention transforms: {len(attention_transforms)}")
    print(f"   - MLP transforms: {len(mlp_transforms)}")
    print(f"\n🔧 Model: {metadata['model_path']}")
    print(f"📏 Hidden size: {metadata['hidden_size']}")
    print(f"🔢 Total layers: {metadata['num_layers_total']}")
    print(f"⏭️  Layers to skip: {metadata['layers_to_skip']}")

    print("\n" + "-"*80)
    print("TRANSFORM DETAILS")
    print("-"*80)

    for i, block in enumerate(selected_blocks):
        start, end = block
        attn_transform = attention_transforms[i]
        mlp_transform = mlp_transforms[i]

        print(f"\n📦 Block {i+1}: Layers {start} to {end} ({end-start} layers removed)")

        print(f"\n   🔵 ATTENTION Transform:")
        print(f"      Shape: {attn_transform.shape}")
        print(f"      Norm: {attn_transform.norm().item():.6f}")
        print(f"      Mean: {attn_transform.mean().item():.6f}")
        print(f"      Std: {attn_transform.std().item():.6f}")

        attn_identity = torch.eye(attn_transform.shape[0], dtype=attn_transform.dtype)
        attn_diff = (attn_transform.cpu() - attn_identity).abs().mean().item()
        print(f"      Distance from identity: {attn_diff:.6f}")

        print(f"\n   🟢 MLP Transform:")
        print(f"      Shape: {mlp_transform.shape}")
        print(f"      Norm: {mlp_transform.norm().item():.6f}")
        print(f"      Mean: {mlp_transform.mean().item():.6f}")
        print(f"      Std: {mlp_transform.std().item():.6f}")

        mlp_identity = torch.eye(mlp_transform.shape[0], dtype=mlp_transform.dtype)
        mlp_diff = (mlp_transform.cpu() - mlp_identity).abs().mean().item()
        print(f"      Distance from identity: {mlp_diff:.6f}")

        # Compare attention vs MLP
        print(f"\n   📊 Comparison:")
        if attn_diff < mlp_diff:
            print(f"      → Attention more redundant (smaller change: {attn_diff:.6f} vs {mlp_diff:.6f})")
        else:
            print(f"      → MLP more redundant (smaller change: {mlp_diff:.6f} vs {attn_diff:.6f})")

    print("\n" + "="*80)
    print("HOW TO LOAD THESE TRANSFORMS")
    print("="*80)
    print(f"""
import torch

# Load the transforms
data = torch.load('{save_path}')

# Access components
attention_transforms = data['attention_transforms']
mlp_transforms = data['mlp_transforms']
selected_blocks = data['selected_blocks']

# Example: Use first block
attn_transform = attention_transforms[0]
mlp_transform = mlp_transforms[0]
start_layer, end_layer = selected_blocks[0]

print(f"Block removes layers {{start_layer}} to {{end_layer}}")
print(f"Attention transform: {{attn_transform.shape}}")
print(f"MLP transform: {{mlp_transform.shape}}")
    """)

    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    print("""
✓ You now have SEPARATE transforms for attention and MLP
✓ Can analyze which component (attention vs MLP) is more compressible
✓ Can selectively compress different components
✓ More interpretable than single mixed transform
    """)


if __name__ == "__main__":
    main()
