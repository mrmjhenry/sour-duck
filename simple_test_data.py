"""
Simple Test Data Generator for Moduli Space Pruning

This module provides easy-to-use test datasets for experimenting with the
moduli space pruning method. No PyTorch required for data generation.

Usage:
    from simple_test_data import *

    # Generate simple classification data
    X, y = generate_binary_classification(n_samples=100, n_features=10)

    # Load pre-made examples
    data = load_example_dataset('spiral')

    # Create toy network weights
    network = create_toy_network([10, 20, 10, 2])
"""

import numpy as np
import json
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import matplotlib.pyplot as plt


# =============================================================================
# Simple Dataset Generators
# =============================================================================

def generate_binary_classification(
    n_samples: int = 100,
    n_features: int = 10,
    n_informative: int = 5,
    n_redundant: int = 2,
    random_state: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a simple binary classification dataset.

    Args:
        n_samples: Number of samples
        n_features: Total number of features
        n_informative: Number of informative features
        n_redundant: Number of redundant features
        random_state: Random seed for reproducibility

    Returns:
        X: Feature matrix of shape (n_samples, n_features)
        y: Labels of shape (n_samples,) with values in {0, 1}

    Example:
        >>> X, y = generate_binary_classification(n_samples=50, n_features=5)
        >>> print(f"Shape: {X.shape}, Classes: {np.unique(y)}")
        Shape: (50, 5), Classes: [0 1]
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Generate informative features
    X_informative = np.random.randn(n_samples, n_informative)

    # Create labels based on informative features
    decision_boundary = X_informative[:, 0] + 0.5 * X_informative[:, 1] > 0
    y = decision_boundary.astype(int)

    # Add redundant features (linear combinations)
    X_redundant = np.zeros((n_samples, n_redundant))
    for i in range(n_redundant):
        weights = np.random.rand(n_informative)
        X_redundant[:, i] = X_informative @ weights

    # Add noise features
    n_noise = n_features - n_informative - n_redundant
    X_noise = np.random.randn(n_samples, n_noise) * 0.1

    # Combine all features
    X = np.hstack([X_informative, X_redundant, X_noise])

    return X, y


def generate_multiclass_classification(
    n_samples: int = 150,
    n_features: int = 10,
    n_classes: int = 3,
    random_state: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a simple multi-class classification dataset.

    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_classes: Number of classes
        random_state: Random seed

    Returns:
        X: Feature matrix of shape (n_samples, n_features)
        y: Labels of shape (n_samples,) with values in {0, 1, ..., n_classes-1}

    Example:
        >>> X, y = generate_multiclass_classification(n_samples=90, n_classes=3)
        >>> print(f"Shape: {X.shape}, Classes: {np.unique(y)}")
        Shape: (90, 10), Classes: [0 1 2]
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Generate samples for each class as Gaussian clusters
    samples_per_class = n_samples // n_classes
    X = []
    y = []

    for class_idx in range(n_classes):
        # Each class has a different center
        center = np.random.randn(n_features) * 3
        class_samples = np.random.randn(samples_per_class, n_features) + center
        X.append(class_samples)
        y.append(np.full(samples_per_class, class_idx))

    X = np.vstack(X)
    y = np.hstack(y)

    # Shuffle
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


def generate_regression_data(
    n_samples: int = 100,
    n_features: int = 10,
    n_informative: int = 5,
    noise: float = 0.1,
    random_state: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a simple regression dataset.

    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_informative: Number of features that contribute to the target
        noise: Standard deviation of Gaussian noise
        random_state: Random seed

    Returns:
        X: Feature matrix of shape (n_samples, n_features)
        y: Target values of shape (n_samples,)

    Example:
        >>> X, y = generate_regression_data(n_samples=50, n_features=5)
        >>> print(f"X shape: {X.shape}, y shape: {y.shape}")
        X shape: (50, 5), y shape: (50,)
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Generate features
    X = np.random.randn(n_samples, n_features)

    # Generate true weights for informative features
    true_weights = np.zeros(n_features)
    true_weights[:n_informative] = np.random.randn(n_informative)

    # Compute target
    y = X @ true_weights + np.random.randn(n_samples) * noise

    return X, y


def generate_xor_data(
    n_samples: int = 200,
    noise: float = 0.1,
    random_state: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate XOR dataset (classic non-linearly separable problem).

    Args:
        n_samples: Number of samples
        noise: Amount of noise to add
        random_state: Random seed

    Returns:
        X: Feature matrix of shape (n_samples, 2)
        y: Binary labels of shape (n_samples,)

    Example:
        >>> X, y = generate_xor_data(n_samples=100)
        >>> print(f"XOR data: {X.shape}, unique labels: {np.unique(y)}")
        XOR data: (100, 2), unique labels: [0 1]
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Generate points in four quadrants
    n_per_quadrant = n_samples // 4
    X = []
    y = []

    # Quadrant 1: (++), label 1
    X.append(np.random.rand(n_per_quadrant, 2) + np.array([0.5, 0.5]))
    y.append(np.ones(n_per_quadrant))

    # Quadrant 2: (-+), label 0
    X.append(np.random.rand(n_per_quadrant, 2) + np.array([-1.5, 0.5]))
    y.append(np.zeros(n_per_quadrant))

    # Quadrant 3: (--), label 1
    X.append(np.random.rand(n_per_quadrant, 2) + np.array([-1.5, -1.5]))
    y.append(np.ones(n_per_quadrant))

    # Quadrant 4: (+-), label 0
    X.append(np.random.rand(n_per_quadrant, 2) + np.array([0.5, -1.5]))
    y.append(np.zeros(n_per_quadrant))

    X = np.vstack(X)
    y = np.hstack(y)

    # Add noise
    X += np.random.randn(*X.shape) * noise

    # Shuffle
    idx = np.random.permutation(len(X))
    return X[idx], y[idx].astype(int)


def generate_spiral_data(
    n_samples: int = 300,
    n_classes: int = 3,
    noise: float = 0.2,
    random_state: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate spiral dataset (challenging non-linear classification).

    Args:
        n_samples: Number of samples
        n_classes: Number of spiral arms
        noise: Amount of noise
        random_state: Random seed

    Returns:
        X: Feature matrix of shape (n_samples, 2)
        y: Labels of shape (n_samples,)

    Example:
        >>> X, y = generate_spiral_data(n_samples=150, n_classes=3)
        >>> print(f"Spiral data: {X.shape}")
        Spiral data: (150, 2)
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_per_class = n_samples // n_classes
    X = []
    y = []

    for class_idx in range(n_classes):
        # Generate spiral
        t = np.linspace(0, 1, n_per_class)
        r = t * 5  # Radius grows with t
        theta = t * 2 * np.pi + (class_idx * 2 * np.pi / n_classes)

        # Convert to Cartesian coordinates
        x = r * np.cos(theta)
        y_coord = r * np.sin(theta)

        # Add noise
        x += np.random.randn(n_per_class) * noise
        y_coord += np.random.randn(n_per_class) * noise

        X.append(np.column_stack([x, y_coord]))
        y.append(np.full(n_per_class, class_idx))

    X = np.vstack(X)
    y = np.hstack(y)

    # Shuffle
    idx = np.random.permutation(len(X))
    return X[idx], y[idx].astype(int)


# =============================================================================
# Toy Network Generators
# =============================================================================

def create_toy_network(
    layer_sizes: List[int],
    initialization: str = 'normal',
    random_state: Optional[int] = 42
) -> List[np.ndarray]:
    """
    Create toy network weights for testing.

    Args:
        layer_sizes: List of layer sizes, e.g., [10, 20, 10, 2]
        initialization: 'normal', 'uniform', or 'xavier'
        random_state: Random seed

    Returns:
        List of weight matrices

    Example:
        >>> weights = create_toy_network([5, 10, 2])
        >>> print(f"Network: {len(weights)} layers")
        >>> print(f"Layer 1: {weights[0].shape}")
        Network: 2 layers
        Layer 1: (10, 5)
    """
    if random_state is not None:
        np.random.seed(random_state)

    weights = []
    for i in range(len(layer_sizes) - 1):
        in_size = layer_sizes[i]
        out_size = layer_sizes[i + 1]

        if initialization == 'normal':
            W = np.random.randn(out_size, in_size) * 0.1
        elif initialization == 'uniform':
            W = np.random.uniform(-0.1, 0.1, (out_size, in_size))
        elif initialization == 'xavier':
            scale = np.sqrt(2.0 / (in_size + out_size))
            W = np.random.randn(out_size, in_size) * scale
        else:
            raise ValueError(f"Unknown initialization: {initialization}")

        weights.append(W)

    return weights


def create_sparse_network(
    layer_sizes: List[int],
    sparsity: float = 0.5,
    random_state: Optional[int] = 42
) -> List[np.ndarray]:
    """
    Create toy network with pre-defined sparsity.

    Args:
        layer_sizes: List of layer sizes
        sparsity: Fraction of weights to set to zero (0 to 1)
        random_state: Random seed

    Returns:
        List of weight matrices with sparsity applied

    Example:
        >>> weights = create_sparse_network([5, 10, 2], sparsity=0.7)
        >>> sparsity = np.mean(weights[0] == 0)
        >>> print(f"Actual sparsity: {sparsity:.2f}")
        Actual sparsity: 0.70
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Create normal network first
    weights = create_toy_network(layer_sizes, random_state=random_state)

    # Apply sparsity
    for W in weights:
        mask = np.random.rand(*W.shape) > sparsity
        W *= mask

    return weights


# =============================================================================
# Data Loading and Saving
# =============================================================================

def save_dataset(
    X: np.ndarray,
    y: np.ndarray,
    filename: str,
    metadata: Optional[Dict] = None
):
    """
    Save dataset to file (supports .npz and .json).

    Args:
        X: Feature matrix
        y: Labels
        filename: Output filename
        metadata: Optional dictionary with dataset metadata

    Example:
        >>> X, y = generate_xor_data(100)
        >>> save_dataset(X, y, 'xor_data.npz',
        ...              metadata={'type': 'xor', 'n_samples': 100})
    """
    filepath = Path(filename)

    if filepath.suffix == '.npz':
        # Save as NumPy compressed format
        if metadata:
            np.savez_compressed(filepath, X=X, y=y, metadata=np.array([metadata]))
        else:
            np.savez_compressed(filepath, X=X, y=y)
        print(f"✓ Saved to {filepath}")

    elif filepath.suffix == '.json':
        # Save as JSON (less efficient but human-readable)
        data = {
            'X': X.tolist(),
            'y': y.tolist(),
            'metadata': metadata or {}
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✓ Saved to {filepath}")
    else:
        raise ValueError("Filename must end with .npz or .json")


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load dataset from file.

    Args:
        filename: Input filename (.npz or .json)

    Returns:
        X: Feature matrix
        y: Labels
        metadata: Dictionary with metadata (empty if none)

    Example:
        >>> X, y, meta = load_dataset('xor_data.npz')
        >>> print(f"Loaded {len(X)} samples")
    """
    filepath = Path(filename)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    if filepath.suffix == '.npz':
        data = np.load(filepath, allow_pickle=True)
        X = data['X']
        y = data['y']
        metadata = data['metadata'].item() if 'metadata' in data else {}

    elif filepath.suffix == '.json':
        with open(filepath, 'r') as f:
            data = json.load(f)
        X = np.array(data['X'])
        y = np.array(data['y'])
        metadata = data.get('metadata', {})
    else:
        raise ValueError("Filename must end with .npz or .json")

    print(f"✓ Loaded from {filepath}: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y, metadata


# =============================================================================
# Pre-defined Example Datasets
# =============================================================================

EXAMPLE_DATASETS = {
    'tiny': {
        'func': lambda: generate_binary_classification(50, 5, 3, 1),
        'description': 'Tiny binary classification (50 samples, 5 features)'
    },
    'small': {
        'func': lambda: generate_binary_classification(100, 10, 5, 2),
        'description': 'Small binary classification (100 samples, 10 features)'
    },
    'medium': {
        'func': lambda: generate_multiclass_classification(300, 20, 3),
        'description': 'Medium multi-class (300 samples, 20 features, 3 classes)'
    },
    'xor': {
        'func': lambda: generate_xor_data(200),
        'description': 'XOR problem (200 samples, 2 features, non-linear)'
    },
    'spiral': {
        'func': lambda: generate_spiral_data(300, 3),
        'description': 'Spiral problem (300 samples, 2 features, 3 spirals)'
    },
    'regression': {
        'func': lambda: generate_regression_data(100, 10, 5),
        'description': 'Regression problem (100 samples, 10 features)'
    }
}


def load_example_dataset(name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a pre-defined example dataset by name.

    Args:
        name: One of 'tiny', 'small', 'medium', 'xor', 'spiral', 'regression'

    Returns:
        X: Feature matrix
        y: Labels/targets

    Example:
        >>> X, y = load_example_dataset('xor')
        >>> print(f"Loaded XOR data: {X.shape}")
        Loaded XOR data: (200, 2)
    """
    if name not in EXAMPLE_DATASETS:
        available = ', '.join(EXAMPLE_DATASETS.keys())
        raise ValueError(f"Unknown dataset '{name}'. Available: {available}")

    dataset_info = EXAMPLE_DATASETS[name]
    print(f"Loading: {dataset_info['description']}")
    X, y = dataset_info['func']()
    return X, y


def list_example_datasets():
    """
    Print all available example datasets.

    Example:
        >>> list_example_datasets()
    """
    print("Available Example Datasets:")
    print("=" * 70)
    for name, info in EXAMPLE_DATASETS.items():
        print(f"  {name:12s} - {info['description']}")
    print("=" * 70)


# =============================================================================
# Visualization Utilities
# =============================================================================

def visualize_2d_data(
    X: np.ndarray,
    y: np.ndarray,
    title: str = "2D Dataset",
    save_path: Optional[str] = None
):
    """
    Visualize 2D dataset (only works if X has 2 features).

    Args:
        X: Feature matrix with shape (n_samples, 2)
        y: Labels
        title: Plot title
        save_path: Optional path to save figure

    Example:
        >>> X, y = generate_xor_data(200)
        >>> visualize_2d_data(X, y, "XOR Dataset", "xor.png")
    """
    if X.shape[1] != 2:
        print(f"Warning: Data has {X.shape[1]} features, expected 2. Skipping visualization.")
        return

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis',
                         alpha=0.6, edgecolors='k', linewidth=0.5)
    plt.colorbar(scatter, label='Class')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization to {save_path}")
    else:
        plt.show()


def print_dataset_stats(X: np.ndarray, y: np.ndarray):
    """
    Print statistics about a dataset.

    Args:
        X: Feature matrix
        y: Labels

    Example:
        >>> X, y = generate_binary_classification(100, 10)
        >>> print_dataset_stats(X, y)
    """
    print("\nDataset Statistics:")
    print("-" * 50)
    print(f"Number of samples:  {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Feature mean:       {X.mean():.4f}")
    print(f"Feature std:        {X.std():.4f}")
    print(f"Feature range:      [{X.min():.4f}, {X.max():.4f}]")
    print(f"\nTarget statistics:")
    unique, counts = np.unique(y, return_counts=True)
    for val, count in zip(unique, counts):
        print(f"  Class {val}: {count} samples ({count/len(y)*100:.1f}%)")
    print("-" * 50)


# =============================================================================
# Main Demo
# =============================================================================

def demo():
    """
    Run a demonstration of all test data generators.
    """
    print("\n" + "=" * 70)
    print("SIMPLE TEST DATA GENERATOR - DEMONSTRATION")
    print("=" * 70)

    # Show available datasets
    list_example_datasets()
    print()

    # Generate and show stats for each type
    datasets_to_test = [
        ('Binary Classification',
         lambda: generate_binary_classification(100, 10)),
        ('Multi-class Classification',
         lambda: generate_multiclass_classification(150, 10, 3)),
        ('Regression',
         lambda: generate_regression_data(100, 10)),
        ('XOR (Non-linear)',
         lambda: generate_xor_data(200)),
        ('Spiral (Complex)',
         lambda: generate_spiral_data(300, 3))
    ]

    for name, generator in datasets_to_test:
        print(f"\n{name}:")
        print("-" * 70)
        X, y = generator()
        print_dataset_stats(X, y)

    # Create toy network
    print("\n\nToy Network Generation:")
    print("-" * 70)
    weights = create_toy_network([10, 20, 10, 2])
    print(f"Created network with {len(weights)} layers:")
    for i, W in enumerate(weights):
        print(f"  Layer {i+1}: {W.shape[1]} → {W.shape[0]} ({W.size} parameters)")

    total_params = sum(W.size for W in weights)
    print(f"Total parameters: {total_params}")

    # Create sparse network
    print("\nSparse Network (50% sparsity):")
    sparse_weights = create_sparse_network([10, 20, 10, 2], sparsity=0.5)
    for i, W in enumerate(sparse_weights):
        sparsity = np.mean(W == 0)
        print(f"  Layer {i+1}: {sparsity*100:.1f}% zeros")

    print("\n" + "=" * 70)
    print("Demo complete! Use these functions in your experiments.")
    print("=" * 70)


if __name__ == '__main__':
    demo()
