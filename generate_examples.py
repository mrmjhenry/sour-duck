"""
Generate Example Datasets

Run this script to generate pre-made example datasets in the examples/ directory.
Requires: numpy, matplotlib (optional for visualizations)

Usage:
    python generate_examples.py
"""

import sys
try:
    import numpy as np
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from pathlib import Path
from simple_test_data import (
    generate_binary_classification,
    generate_multiclass_classification,
    generate_regression_data,
    generate_xor_data,
    generate_spiral_data,
    save_dataset,
    visualize_2d_data,
    create_toy_network
)


def generate_all_examples():
    """Generate all example datasets."""

    # Create examples directory
    examples_dir = Path('examples')
    examples_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("GENERATING EXAMPLE DATASETS")
    print("=" * 70)

    # 1. Tiny Binary Classification
    print("\n1. Generating tiny binary classification...")
    X, y = generate_binary_classification(
        n_samples=50,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        random_state=42
    )
    save_dataset(
        X, y,
        'examples/tiny_binary.npz',
        metadata={
            'name': 'tiny_binary',
            'type': 'binary_classification',
            'n_samples': 50,
            'n_features': 5,
            'description': 'Tiny binary classification dataset for quick tests'
        }
    )

    # 2. Small Binary Classification
    print("\n2. Generating small binary classification...")
    X, y = generate_binary_classification(
        n_samples=100,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        random_state=42
    )
    save_dataset(
        X, y,
        'examples/small_binary.npz',
        metadata={
            'name': 'small_binary',
            'type': 'binary_classification',
            'n_samples': 100,
            'n_features': 10,
            'description': 'Small binary classification dataset'
        }
    )

    # 3. Medium Multi-class Classification
    print("\n3. Generating medium multi-class classification...")
    X, y = generate_multiclass_classification(
        n_samples=300,
        n_features=20,
        n_classes=3,
        random_state=42
    )
    save_dataset(
        X, y,
        'examples/medium_multiclass.npz',
        metadata={
            'name': 'medium_multiclass',
            'type': 'multiclass_classification',
            'n_samples': 300,
            'n_features': 20,
            'n_classes': 3,
            'description': 'Medium-sized 3-class classification dataset'
        }
    )

    # 4. XOR Dataset
    print("\n4. Generating XOR dataset...")
    X, y = generate_xor_data(n_samples=200, noise=0.1, random_state=42)
    save_dataset(
        X, y,
        'examples/xor.npz',
        metadata={
            'name': 'xor',
            'type': 'binary_classification',
            'n_samples': 200,
            'n_features': 2,
            'description': 'Classic XOR problem (non-linearly separable)'
        }
    )

    if HAS_MATPLOTLIB:
        visualize_2d_data(X, y, "XOR Dataset", "examples/xor_visualization.png")

    # 5. Spiral Dataset
    print("\n5. Generating spiral dataset...")
    X, y = generate_spiral_data(n_samples=300, n_classes=3, noise=0.2, random_state=42)
    save_dataset(
        X, y,
        'examples/spiral.npz',
        metadata={
            'name': 'spiral',
            'type': 'multiclass_classification',
            'n_samples': 300,
            'n_features': 2,
            'n_classes': 3,
            'description': 'Three-spiral dataset (challenging non-linear problem)'
        }
    )

    if HAS_MATPLOTLIB:
        visualize_2d_data(X, y, "Spiral Dataset", "examples/spiral_visualization.png")

    # 6. Regression Dataset
    print("\n6. Generating regression dataset...")
    X, y = generate_regression_data(
        n_samples=100,
        n_features=10,
        n_informative=5,
        noise=0.1,
        random_state=42
    )
    save_dataset(
        X, y,
        'examples/regression.npz',
        metadata={
            'name': 'regression',
            'type': 'regression',
            'n_samples': 100,
            'n_features': 10,
            'description': 'Simple regression dataset'
        }
    )

    # 7. Large Multi-class
    print("\n7. Generating large multi-class dataset...")
    X, y = generate_multiclass_classification(
        n_samples=1000,
        n_features=50,
        n_classes=5,
        random_state=42
    )
    save_dataset(
        X, y,
        'examples/large_multiclass.npz',
        metadata={
            'name': 'large_multiclass',
            'type': 'multiclass_classification',
            'n_samples': 1000,
            'n_features': 50,
            'n_classes': 5,
            'description': 'Large 5-class classification dataset for realistic testing'
        }
    )

    # 8. Save toy network weights
    print("\n8. Generating toy network weights...")
    weights = create_toy_network([10, 20, 10, 2], random_state=42)
    np.savez_compressed(
        'examples/toy_network_weights.npz',
        **{f'layer_{i}': w for i, w in enumerate(weights)},
        metadata=np.array([{
            'architecture': [10, 20, 10, 2],
            'n_layers': len(weights),
            'total_params': sum(w.size for w in weights),
            'description': 'Toy network weights for testing'
        }])
    )
    print("✓ Saved to examples/toy_network_weights.npz")

    print("\n" + "=" * 70)
    print("ALL EXAMPLE DATASETS GENERATED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\nFiles created in: {examples_dir.absolute()}")
    print("\nGenerated files:")
    for file in sorted(examples_dir.glob('*.npz')):
        size_kb = file.stat().st_size / 1024
        print(f"  - {file.name:30s} ({size_kb:.1f} KB)")

    if HAS_MATPLOTLIB:
        print("\nVisualizations:")
        for file in sorted(examples_dir.glob('*.png')):
            print(f"  - {file.name}")

    print("\nUsage:")
    print("  from simple_test_data import load_dataset")
    print("  X, y, metadata = load_dataset('examples/xor.npz')")


if __name__ == '__main__':
    try:
        generate_all_examples()
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        print("\nMake sure you have installed required dependencies:", file=sys.stderr)
        print("  pip install numpy matplotlib", file=sys.stderr)
        sys.exit(1)
