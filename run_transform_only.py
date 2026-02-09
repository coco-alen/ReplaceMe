#!/usr/bin/env python3
"""
Simple script to run transform-only computation and inspect results.

Usage:
    python run_transform_only.py
"""

import os
import sys
import torch
from pathlib import Path

# Add ReplaceMe to path
sys.path.insert(0, str(Path(__file__).parent / "ReplaceMe"))

from ReplaceMe.lstsq_transform_only import lstsq_transform_only


def main():
    print("="*80)
    print("TRANSFORM-ONLY COMPUTATION (No Pruning)")
    print("="*80)

    # Configuration
    config = {
        'model_path': 'meta-llama/Llama-2-7b-hf',  # Change to your model
        'dataset': 'Open-Orca/SlimOrca',
        'dataset_column': 'text',
        'batch_size': 4,
        'max_length': 512,
        'layers_to_skip': 8,
        'dataset_size': 100,  # Small for quick testing
        'dataset_subset': 'train',
        'use_4bit': True,  # Use 4-bit to save memory
        'alpha': 0.0,  # No regularization
        'distances_path': './distances.pth',
        'num_A': 1,  # Number of blocks to transform
        'merge_consecutive': True,
        'save_path': './my_transforms'
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
        print("\nRun this first:")
        print("""
from ReplaceMe.distance import profile_distances
profile_distances(
    model_path='meta-llama/Llama-2-7b-hf',
    dataset='Open-Orca/SlimOrca',
    dataset_column='text',
    batch_size=4,
    max_length=512,
    layers_to_skip=8,
    dataset_size=100,
    use_4bit=True
)
        """)
        return

    print("\n" + "="*80)
    print("Starting transform computation...")
    print("="*80 + "\n")

    # Run transform computation
    try:
        result = lstsq_transform_only(**config)
        print("\n" + "="*80)
        print("✓ SUCCESS!")
        print("="*80)

        # Inspect results
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

    transforms = result['transforms']
    selected_blocks = result['selected_blocks']
    save_path = result['save_path']
    metadata = result['metadata']

    print(f"\n📁 Saved to: {save_path}")
    print(f"\n📊 Number of transforms: {len(transforms)}")
    print(f"🔧 Model: {metadata['model_path']}")
    print(f"📏 Hidden size: {metadata['hidden_size']}")
    print(f"🔢 Total layers: {metadata['num_layers_total']}")
    print(f"⏭️  Layers to skip: {metadata['layers_to_skip']}")

    print("\n" + "-"*80)
    print("TRANSFORM DETAILS")
    print("-"*80)

    for i, (transform, block) in enumerate(zip(transforms, selected_blocks)):
        start, end = block
        print(f"\n📦 Block {i+1}:")
        print(f"   Layers removed: {start} to {end} ({end-start} layers)")
        print(f"   Transform shape: {transform.shape}")
        print(f"   Transform dtype: {transform.dtype}")
        print(f"   Transform norm: {transform.norm().item():.6f}")
        print(f"   Transform mean: {transform.mean().item():.6f}")
        print(f"   Transform std: {transform.std().item():.6f}")

        # Check distance from identity
        identity = torch.eye(transform.shape[0], dtype=transform.dtype)
        identity_diff = (transform.cpu() - identity).abs().mean().item()
        print(f"   Distance from identity: {identity_diff:.6f}")

        if identity_diff < 0.01:
            print(f"   → Very close to identity (minimal transformation needed)")
        elif identity_diff < 0.1:
            print(f"   → Moderate transformation")
        else:
            print(f"   → Significant transformation")

    print("\n" + "="*80)
    print("HOW TO LOAD THESE TRANSFORMS")
    print("="*80)
    print(f"""
import torch

# Load the transforms
data = torch.load('{save_path}')

# Access components
transforms = data['transforms']  # List of transformation matrices
selected_blocks = data['selected_blocks']  # [(start, end), ...]
metadata = data['metadata']  # Model info, hyperparameters

# Example: Use first transform
transform = transforms[0]
start_layer, end_layer = selected_blocks[0]

print(f"Transform shape: {{transform.shape}}")
print(f"Removes layers {{start_layer}} to {{end_layer}}")
    """)

    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("""
1. Analyze the transforms:
   - Check which layers are most compressible
   - Compare transform norms across blocks

2. Apply to a model (optional):
   - Load the model
   - Remove the selected layers
   - Update weights with transforms

3. Evaluate performance:
   - Compare original vs compressed model
   - Measure speedup and accuracy loss
    """)


if __name__ == "__main__":
    main()
