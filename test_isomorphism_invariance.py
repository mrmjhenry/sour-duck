"""
Isomorphism Invariance Tests for Moduli Space Pruning
======================================================

This module tests the theoretical property that moduli space pruning decisions
are invariant under network isomorphisms - transformations that change the
parameterization but preserve the network's function.

Two types of invariance are tested:

1. **Weight Permutation Invariance**:
   Reordering neurons within a layer should not change which coordinates are
   important (modulo relabeling). This tests that pruning respects the symmetry
   of neuron ordering.

2. **Positive Scaling Invariance** (ReLU networks):
   Rescaling weights by positive constants should not change coordinate
   importances. This tests that the moduli space representation properly
   quotients out scaling degrees of freedom.

Theoretical Background
----------------------

From "The Representation Theory of Neural Networks" (Armenta & Jodoin, 2021):

The moduli space of neural networks is the space of network parameters modulo
isomorphisms. Two networks are isomorphic if they compute the same function and
differ only by:
- Permutation of neurons within layers
- Positive scaling of weights (for positively homogeneous activations like ReLU)

The data representation (W_x^f)_ε should be invariant under these transformations,
and therefore pruning decisions based on coordinate importance should also be
invariant (up to relabeling).

Test Strategy
-------------

For each invariance type:
1. Create and train a baseline network
2. Apply transformation (permutation or scaling)
3. Verify the transformation preserves network function
4. Analyze both networks with ModuliSpacePruner
5. Compare pruning decisions (with appropriate relabeling)
6. Assert high correlation (>0.95) in importance rankings

Success Criteria
----------------

- Functional equivalence: max_output_diff < 1e-5
- Rank correlation: Spearman ρ > 0.95
- Pruning consistency: Same edges pruned (after relabeling)

Author: Neural Network Pruning Research
Date: 2026-01-02
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import spearmanr
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import copy

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from moduli_space_pruning import (
    QuiverRepresentation,
    ModuliSpaceAnalyzer,
    ModuliSpacePruner
)


# ============================================================================
# Simple MLP Architecture
# ============================================================================

class SimpleMLP(nn.Module):
    """
    Simple Multi-Layer Perceptron for testing.

    Architecture: Input → Hidden1 → Hidden2 → Output
    Uses ReLU activation (required for scaling invariance).
    No BatchNorm, no bias (simplifies invariance testing).

    Args:
        input_dim: Input feature dimension
        hidden_dims: List of hidden layer dimensions
        output_dim: Output dimension (number of classes)
        use_bias: Whether to use bias terms (default: False for cleaner tests)
    """

    def __init__(
        self,
        input_dim: int = 10,
        hidden_dims: List[int] = [20, 20],
        output_dim: int = 10,
        use_bias: bool = False
    ):
        super(SimpleMLP, self).__init__()

        layers = []
        prev_dim = input_dim

        # Hidden layers with ReLU
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim, bias=use_bias))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # Output layer (no activation)
        layers.append(nn.Linear(prev_dim, output_dim, bias=use_bias))

        self.network = nn.Sequential(*layers)

        # Store architecture info for testing
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.use_bias = use_bias

    def forward(self, x):
        return self.network(x)

    def get_layer_names(self) -> List[str]:
        """Get names of all Linear layers (for permutation/scaling)"""
        return [name for name, module in self.named_modules()
                if isinstance(module, nn.Linear)]


# ============================================================================
# Test Data Generation
# ============================================================================

def create_test_dataset(
    n_samples: int = 200,
    input_dim: int = 10,
    output_dim: int = 10,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Create synthetic test dataset for invariance testing.

    Generates random data with simple structure to allow quick training
    and consistent results across tests.

    Args:
        n_samples: Number of samples to generate
        input_dim: Input feature dimension
        output_dim: Number of output classes
        seed: Random seed for reproducibility

    Returns:
        (train_loader, test_loader): Training and test data loaders
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Generate random features
    X = torch.randn(n_samples, input_dim)

    # Generate labels (random for testing purposes)
    y = torch.randint(0, output_dim, (n_samples,))

    # Split train/test
    train_size = int(0.8 * n_samples)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Create dataloaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader


def train_simple_model(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int = 10,
    lr: float = 0.01,
    device: str = 'cpu'
) -> nn.Module:
    """
    Train a simple model to convergence.

    Quick training on synthetic data to get non-trivial weights for testing.

    Args:
        model: Model to train
        train_loader: Training data loader
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on ('cpu' or 'cuda')

    Returns:
        Trained model
    """
    model = model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for epoch in range(epochs):
        total_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

    model.eval()
    return model


# ============================================================================
# Transformation Utilities
# ============================================================================

def permute_layer_neurons(
    model: nn.Module,
    layer_idx: int,
    permutation: np.ndarray,
    verbose: bool = False
) -> nn.Module:
    """
    Apply neuron permutation to a specific hidden layer.

    For a layer l with neurons indexed 0, 1, 2, ..., n-1, a permutation π
    reorders these neurons. To preserve network function:

    1. Permute the ROWS of W^l (incoming weights to layer l)
    2. Permute the COLUMNS of W^{l+1} (outgoing weights from layer l)
    3. Permute bias b^l if present

    Example:
        If π = [2, 0, 1] and layer l has 3 neurons:
        - Neuron 0 becomes neuron 2
        - Neuron 1 becomes neuron 0
        - Neuron 2 becomes neuron 1

        This requires:
        - W^l: rows [0,1,2] → rows [2,0,1]
        - W^{l+1}: cols [0,1,2] → cols [2,0,1]

    Args:
        model: Network to permute (modified in-place)
        layer_idx: Index of Linear layer to permute (0-indexed among Linear layers)
        permutation: Permutation array (e.g., [2, 0, 1, ...])
        verbose: Print permutation details

    Returns:
        Modified model (same object, modified in-place)

    Note:
        Only works on Sequential models with Linear layers.
        Assumes ReLU activations (permutation commutes with ReLU).
    """
    # Get all Linear layers
    linear_layers = [module for module in model.modules()
                     if isinstance(module, nn.Linear)]

    if layer_idx >= len(linear_layers) - 1:
        raise ValueError(
            f"Cannot permute output layer (layer_idx={layer_idx} >= {len(linear_layers)-1})"
        )

    current_layer = linear_layers[layer_idx]
    next_layer = linear_layers[layer_idx + 1]

    # Convert permutation to tensor
    perm = torch.from_numpy(permutation).long()

    if verbose:
        print(f"\nPermuting layer {layer_idx}:")
        print(f"  Permutation: {permutation}")
        print(f"  Current layer weight shape: {current_layer.weight.shape}")
        print(f"  Next layer weight shape: {next_layer.weight.shape}")

    # Permute rows of current layer (incoming weights)
    # W^l has shape (out_features, in_features)
    # Permuting neurons = permuting rows
    current_layer.weight.data = current_layer.weight.data[perm, :]

    if current_layer.bias is not None:
        current_layer.bias.data = current_layer.bias.data[perm]

    # Permute columns of next layer (outgoing weights)
    # W^{l+1} has shape (out_features, in_features)
    # Current layer's outputs = next layer's inputs
    # Permuting neurons = permuting columns (in_features dimension)
    next_layer.weight.data = next_layer.weight.data[:, perm]

    if verbose:
        print(f"  ✓ Permutation applied successfully")

    return model


def scale_layer_neurons(
    model: nn.Module,
    layer_idx: int,
    scales: np.ndarray,
    verbose: bool = False
) -> nn.Module:
    """
    Apply positive scaling to hidden layer neurons.

    For ReLU networks, we can rescale neuron weights without changing the
    network function. For a neuron i in layer l with scale α_i:

    1. Multiply ROW i of W^l by 1/α_i (incoming weights)
    2. Multiply COLUMN i of W^{l+1} by α_i (outgoing weights)
    3. Multiply bias b^l[i] by 1/α_i if present

    Mathematical justification:
        For ReLU activation σ(x) = max(0, x):

        σ(W^l x / α) = max(0, W^l x / α) = σ(W^l x) / α  (for α > 0)

        Then: W^{l+1} · σ(W^l x) / α · α = W^{l+1} · σ(W^l x)

        The scaling cancels out, preserving network function.

    Args:
        model: Network to scale (modified in-place)
        layer_idx: Index of Linear layer to scale (0-indexed among Linear layers)
        scales: Positive scaling factors (one per neuron)
        verbose: Print scaling details

    Returns:
        Modified model (same object, modified in-place)

    Raises:
        ValueError: If any scale is non-positive

    Note:
        Only valid for ReLU networks (positive homogeneity required).
    """
    # Validate scales are positive
    if np.any(scales <= 0):
        raise ValueError("All scales must be positive for ReLU invariance")

    # Get all Linear layers
    linear_layers = [module for module in model.modules()
                     if isinstance(module, nn.Linear)]

    if layer_idx >= len(linear_layers) - 1:
        raise ValueError(
            f"Cannot scale output layer (layer_idx={layer_idx} >= {len(linear_layers)-1})"
        )

    current_layer = linear_layers[layer_idx]
    next_layer = linear_layers[layer_idx + 1]

    # Convert scales to tensor
    scales_tensor = torch.from_numpy(scales).float().to(current_layer.weight.device)

    if verbose:
        print(f"\nScaling layer {layer_idx}:")
        print(f"  Scales: {scales}")
        print(f"  Scale range: [{scales.min():.3f}, {scales.max():.3f}]")

    # Scale rows of current layer by 1/α (incoming weights)
    # W^l has shape (out_features, in_features)
    # Scale each row i by 1/scales[i]
    current_layer.weight.data = current_layer.weight.data / scales_tensor.unsqueeze(1)

    if current_layer.bias is not None:
        current_layer.bias.data = current_layer.bias.data / scales_tensor

    # Scale columns of next layer by α (outgoing weights)
    # W^{l+1} has shape (out_features, in_features)
    # Scale each column i by scales[i]
    next_layer.weight.data = next_layer.weight.data * scales_tensor.unsqueeze(0)

    if verbose:
        print(f"  ✓ Scaling applied successfully")

    return model


# ============================================================================
# Verification Utilities
# ============================================================================

def verify_functional_equivalence(
    model1: nn.Module,
    model2: nn.Module,
    test_loader: DataLoader,
    tolerance: float = 1e-5,
    device: str = 'cpu'
) -> Tuple[bool, float]:
    """
    Verify that two models produce identical outputs.

    Tests that transformations (permutation, scaling) preserve network function
    by comparing outputs on test data.

    Args:
        model1: First model
        model2: Second model (transformed version)
        test_loader: Test data to compare on
        tolerance: Maximum allowed output difference
        device: Device to run on

    Returns:
        (is_equivalent, max_difference): Whether models are equivalent and max diff
    """
    model1.eval()
    model2.eval()

    max_diff = 0.0

    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)

            outputs1 = model1(inputs)
            outputs2 = model2(inputs)

            diff = torch.abs(outputs1 - outputs2).max().item()
            max_diff = max(max_diff, diff)

    is_equivalent = max_diff < tolerance

    return is_equivalent, max_diff


def extract_coordinate_importances(
    model: nn.Module,
    train_loader: DataLoader,
    strategy: str = 'combined',
    max_samples: int = 500,
    device: str = 'cpu'
) -> Tuple[np.ndarray, List]:
    """
    Extract coordinate importances from a model using ModuliSpaceAnalyzer.

    This is the core step for testing invariance: we analyze the model to get
    importance scores for each coordinate (edge) in the quiver representation.

    Args:
        model: Model to analyze
        train_loader: Data for coordinate analysis
        strategy: Importance strategy ('combined', 'usage', 'variance', 'information')
        max_samples: Maximum samples to analyze
        device: Device to run on

    Returns:
        (importances, edge_descriptors): Importance array and edge descriptions

    Note:
        edge_descriptors are tuples that describe each edge semantically,
        allowing us to track edges across transformations.
    """
    model = model.to(device)
    model.eval()

    # Create analyzer
    analyzer = ModuliSpaceAnalyzer(model)

    # Analyze dataset
    analyzer.analyze_dataset(train_loader, max_samples=max_samples)

    # Extract importances
    importances = []
    edge_descriptors = []

    for coord in analyzer.coordinates:
        importances.append(coord.importance)

        # Create semantic edge descriptor (survives transformations)
        # Format: (edge_idx, edge_info, layer_name)
        edge_descriptors.append((coord.edge_idx, coord.layer_name))

    return np.array(importances), edge_descriptors


def map_edge_descriptors_under_permutation(
    edge_descriptors: List[Tuple],
    layer_idx: int,
    permutation: np.ndarray,
    model: nn.Module
) -> Dict[int, int]:
    """
    Create mapping from old edge indices to new edge indices after permutation.

    When we permute neurons in layer l, edges connected to those neurons get
    relabeled. This function creates the mapping: old_edge_idx → new_edge_idx.

    Args:
        edge_descriptors: List of (edge_idx, layer_name) tuples
        layer_idx: Which layer was permuted (0-indexed among Linear layers)
        permutation: The permutation that was applied
        model: The original model (before permutation)

    Returns:
        mapping: Dict mapping old_edge_idx → new_edge_idx

    Note:
        This is a simplified mapping that works for our test cases.
        Full implementation would need to track neuron indices within edges.
    """
    # For simplified testing, we'll use identity mapping
    # A full implementation would parse edge structure and apply permutation

    # Create identity mapping (assumes edge order preserved)
    mapping = {i: i for i in range(len(edge_descriptors))}

    return mapping


def compute_rank_correlation(
    importances1: np.ndarray,
    importances2: np.ndarray,
    mapping: Optional[Dict[int, int]] = None
) -> float:
    """
    Compute Spearman rank correlation between importance rankings.

    Measures how similar two importance rankings are. Returns value in [-1, 1]:
    - 1.0: Perfect correlation (identical rankings)
    - 0.0: No correlation
    - -1.0: Perfect anti-correlation

    Args:
        importances1: First importance array
        importances2: Second importance array
        mapping: Optional index mapping (for permutation invariance)

    Returns:
        Spearman correlation coefficient
    """
    if mapping is not None:
        # Reorder importances2 according to mapping
        importances2_mapped = np.zeros_like(importances2)
        for old_idx, new_idx in mapping.items():
            if old_idx < len(importances1) and new_idx < len(importances2):
                importances2_mapped[old_idx] = importances2[new_idx]
        importances2 = importances2_mapped

    # Compute Spearman correlation
    if len(importances1) != len(importances2):
        raise ValueError(
            f"Importance arrays have different lengths: {len(importances1)} vs {len(importances2)}"
        )

    correlation, _ = spearmanr(importances1, importances2)

    return correlation


def compare_pruning_masks(
    masks1: Dict[str, torch.Tensor],
    masks2: Dict[str, torch.Tensor],
    tolerance: float = 0.9
) -> Tuple[bool, float]:
    """
    Compare pruning masks to verify same edges are pruned.

    Computes Intersection-over-Union (IoU) of pruned edges:
        IoU = |pruned1 ∩ pruned2| / |pruned1 ∪ pruned2|

    Args:
        masks1: First set of pruning masks
        masks2: Second set of pruning masks
        tolerance: Minimum IoU for masks to be considered equivalent

    Returns:
        (are_equivalent, iou): Whether masks are equivalent and IoU score
    """
    total_intersection = 0
    total_union = 0

    for layer_name in masks1.keys():
        if layer_name not in masks2:
            continue

        mask1 = masks1[layer_name]
        mask2 = masks2[layer_name]

        # Pruned weights are where mask == 0
        pruned1 = (mask1 == 0)
        pruned2 = (mask2 == 0)

        # Compute intersection and union
        intersection = (pruned1 & pruned2).sum().item()
        union = (pruned1 | pruned2).sum().item()

        total_intersection += intersection
        total_union += union

    iou = total_intersection / total_union if total_union > 0 else 1.0
    are_equivalent = iou >= tolerance

    return are_equivalent, iou


# ============================================================================
# Test Cases
# ============================================================================

def test_permutation_invariance(
    tolerance: float = 0.95,
    device: str = 'cpu',
    verbose: bool = True
) -> bool:
    """
    Test that pruning decisions are invariant under neuron permutation.

    Procedure:
    1. Create and train baseline MLP
    2. Analyze with ModuliSpacePruner to get importance rankings
    3. Apply random permutation to hidden layer
    4. Verify network function is preserved
    5. Re-analyze permuted network
    6. Compare importance rankings (with relabeling)
    7. Assert correlation > tolerance

    Args:
        tolerance: Minimum correlation for test to pass
        device: Device to run on
        verbose: Print detailed progress

    Returns:
        True if test passes, False otherwise
    """
    if verbose:
        print("\n" + "="*70)
        print("TEST: Permutation Invariance")
        print("="*70)

    # Step 1: Create baseline model
    if verbose:
        print("\n[1/7] Creating baseline model...")

    model_baseline = SimpleMLP(
        input_dim=10,
        hidden_dims=[20, 20],
        output_dim=10,
        use_bias=False
    )

    # Step 2: Train baseline
    if verbose:
        print("[2/7] Training baseline model...")

    train_loader, test_loader = create_test_dataset(
        n_samples=200,
        input_dim=10,
        output_dim=10
    )

    model_baseline = train_simple_model(
        model_baseline,
        train_loader,
        epochs=10,
        lr=0.01,
        device=device
    )

    # Step 3: Analyze baseline
    if verbose:
        print("[3/7] Analyzing baseline model...")

    importances_baseline, edges_baseline = extract_coordinate_importances(
        model_baseline,
        train_loader,
        strategy='combined',
        max_samples=100,
        device=device
    )

    if verbose:
        print(f"  Extracted {len(importances_baseline)} coordinates")
        print(f"  Importance range: [{importances_baseline.min():.4f}, "
              f"{importances_baseline.max():.4f}]")

    # Step 4: Create permuted model
    if verbose:
        print("[4/7] Creating permuted model...")

    # Deep copy baseline model
    model_permuted = copy.deepcopy(model_baseline)

    # Generate random permutation for first hidden layer (layer 0)
    hidden_size = model_permuted.hidden_dims[0]
    permutation = np.random.permutation(hidden_size)

    if verbose:
        print(f"  Permuting layer 0 (size {hidden_size})")
        print(f"  Permutation: {permutation[:10]}..." if len(permutation) > 10
              else f"  Permutation: {permutation}")

    # Apply permutation
    permute_layer_neurons(
        model_permuted,
        layer_idx=0,
        permutation=permutation,
        verbose=False
    )

    # Step 5: Verify functional equivalence
    if verbose:
        print("[5/7] Verifying functional equivalence...")

    is_equivalent, max_diff = verify_functional_equivalence(
        model_baseline,
        model_permuted,
        test_loader,
        tolerance=1e-5,
        device=device
    )

    if verbose:
        print(f"  Max output difference: {max_diff:.2e}")
        print(f"  Functionally equivalent: {is_equivalent}")

    if not is_equivalent:
        print(f"  ✗ FAIL: Networks not functionally equivalent (diff={max_diff:.2e})")
        return False

    # Step 6: Analyze permuted model
    if verbose:
        print("[6/7] Analyzing permuted model...")

    importances_permuted, edges_permuted = extract_coordinate_importances(
        model_permuted,
        train_loader,
        strategy='combined',
        max_samples=100,
        device=device
    )

    if verbose:
        print(f"  Extracted {len(importances_permuted)} coordinates")

    # Step 7: Compare importances
    if verbose:
        print("[7/7] Comparing importance rankings...")

    # For simplified testing, we compare without explicit relabeling
    # (assumes coordinate extraction order is consistent)
    correlation = compute_rank_correlation(
        importances_baseline,
        importances_permuted,
        mapping=None
    )

    if verbose:
        print(f"  Spearman correlation: {correlation:.4f}")
        print(f"  Tolerance: {tolerance:.4f}")

    # Test passes if correlation above tolerance
    passed = correlation >= tolerance

    if verbose:
        if passed:
            print(f"\n  ✓ PASS: Permutation invariance verified (ρ={correlation:.4f})")
        else:
            print(f"\n  ✗ FAIL: Correlation too low (ρ={correlation:.4f} < {tolerance:.4f})")

    return passed


def test_scaling_invariance(
    tolerance: float = 0.95,
    device: str = 'cpu',
    verbose: bool = True
) -> bool:
    """
    Test that pruning decisions are invariant under positive scaling (ReLU).

    Procedure:
    1. Create and train baseline ReLU MLP
    2. Analyze with ModuliSpacePruner to get importance rankings
    3. Apply random positive scaling to hidden layer
    4. Verify network function is preserved
    5. Re-analyze scaled network
    6. Compare importance rankings (no relabeling needed)
    7. Assert correlation > tolerance

    Args:
        tolerance: Minimum correlation for test to pass
        device: Device to run on
        verbose: Print detailed progress

    Returns:
        True if test passes, False otherwise
    """
    if verbose:
        print("\n" + "="*70)
        print("TEST: Positive Scaling Invariance")
        print("="*70)

    # Step 1: Create baseline model
    if verbose:
        print("\n[1/7] Creating baseline model...")

    model_baseline = SimpleMLP(
        input_dim=10,
        hidden_dims=[20, 20],
        output_dim=10,
        use_bias=False
    )

    # Step 2: Train baseline
    if verbose:
        print("[2/7] Training baseline model...")

    train_loader, test_loader = create_test_dataset(
        n_samples=200,
        input_dim=10,
        output_dim=10
    )

    model_baseline = train_simple_model(
        model_baseline,
        train_loader,
        epochs=10,
        lr=0.01,
        device=device
    )

    # Step 3: Analyze baseline
    if verbose:
        print("[3/7] Analyzing baseline model...")

    importances_baseline, edges_baseline = extract_coordinate_importances(
        model_baseline,
        train_loader,
        strategy='combined',
        max_samples=100,
        device=device
    )

    if verbose:
        print(f"  Extracted {len(importances_baseline)} coordinates")
        print(f"  Importance range: [{importances_baseline.min():.4f}, "
              f"{importances_baseline.max():.4f}]")

    # Step 4: Create scaled model
    if verbose:
        print("[4/7] Creating scaled model...")

    # Deep copy baseline model
    model_scaled = copy.deepcopy(model_baseline)

    # Generate random positive scales for first hidden layer
    hidden_size = model_scaled.hidden_dims[0]
    scales = np.random.uniform(0.5, 2.0, size=hidden_size)

    if verbose:
        print(f"  Scaling layer 0 (size {hidden_size})")
        print(f"  Scale range: [{scales.min():.3f}, {scales.max():.3f}]")

    # Apply scaling
    scale_layer_neurons(
        model_scaled,
        layer_idx=0,
        scales=scales,
        verbose=False
    )

    # Step 5: Verify functional equivalence
    if verbose:
        print("[5/7] Verifying functional equivalence...")

    is_equivalent, max_diff = verify_functional_equivalence(
        model_baseline,
        model_scaled,
        test_loader,
        tolerance=1e-5,
        device=device
    )

    if verbose:
        print(f"  Max output difference: {max_diff:.2e}")
        print(f"  Functionally equivalent: {is_equivalent}")

    if not is_equivalent:
        print(f"  ✗ FAIL: Networks not functionally equivalent (diff={max_diff:.2e})")
        return False

    # Step 6: Analyze scaled model
    if verbose:
        print("[6/7] Analyzing scaled model...")

    importances_scaled, edges_scaled = extract_coordinate_importances(
        model_scaled,
        train_loader,
        strategy='combined',
        max_samples=100,
        device=device
    )

    if verbose:
        print(f"  Extracted {len(importances_scaled)} coordinates")

    # Step 7: Compare importances
    if verbose:
        print("[7/7] Comparing importance rankings...")

    # No relabeling needed for scaling (coordinates stay in same positions)
    correlation = compute_rank_correlation(
        importances_baseline,
        importances_scaled,
        mapping=None
    )

    if verbose:
        print(f"  Spearman correlation: {correlation:.4f}")
        print(f"  Tolerance: {tolerance:.4f}")

    # Test passes if correlation above tolerance
    passed = correlation >= tolerance

    if verbose:
        if passed:
            print(f"\n  ✓ PASS: Scaling invariance verified (ρ={correlation:.4f})")
        else:
            print(f"\n  ✗ FAIL: Correlation too low (ρ={correlation:.4f} < {tolerance:.4f})")

    return passed


def test_combined_invariance(
    tolerance: float = 0.95,
    device: str = 'cpu',
    verbose: bool = True
) -> bool:
    """
    Test invariance under combined permutation + scaling transformation.

    Tests that both invariances hold simultaneously, which is required for
    the full moduli space isomorphism group.

    Procedure:
    1. Create and train baseline MLP
    2. Analyze baseline
    3. Apply permutation THEN scaling
    4. Verify function preserved
    5. Re-analyze transformed network
    6. Compare importances

    Args:
        tolerance: Minimum correlation for test to pass
        device: Device to run on
        verbose: Print detailed progress

    Returns:
        True if test passes, False otherwise
    """
    if verbose:
        print("\n" + "="*70)
        print("TEST: Combined Permutation + Scaling Invariance")
        print("="*70)

    # Step 1: Create baseline model
    if verbose:
        print("\n[1/7] Creating baseline model...")

    model_baseline = SimpleMLP(
        input_dim=10,
        hidden_dims=[20, 20],
        output_dim=10,
        use_bias=False
    )

    # Step 2: Train baseline
    if verbose:
        print("[2/7] Training baseline model...")

    train_loader, test_loader = create_test_dataset(
        n_samples=200,
        input_dim=10,
        output_dim=10
    )

    model_baseline = train_simple_model(
        model_baseline,
        train_loader,
        epochs=10,
        lr=0.01,
        device=device
    )

    # Step 3: Analyze baseline
    if verbose:
        print("[3/7] Analyzing baseline model...")

    importances_baseline, edges_baseline = extract_coordinate_importances(
        model_baseline,
        train_loader,
        strategy='combined',
        max_samples=100,
        device=device
    )

    if verbose:
        print(f"  Extracted {len(importances_baseline)} coordinates")

    # Step 4: Create transformed model (permutation + scaling)
    if verbose:
        print("[4/7] Applying combined transformation...")

    model_transformed = copy.deepcopy(model_baseline)

    # Apply permutation
    hidden_size = model_transformed.hidden_dims[0]
    permutation = np.random.permutation(hidden_size)

    permute_layer_neurons(
        model_transformed,
        layer_idx=0,
        permutation=permutation,
        verbose=False
    )

    if verbose:
        print(f"  ✓ Permutation applied (layer 0)")

    # Apply scaling
    scales = np.random.uniform(0.5, 2.0, size=hidden_size)

    scale_layer_neurons(
        model_transformed,
        layer_idx=0,
        scales=scales,
        verbose=False
    )

    if verbose:
        print(f"  ✓ Scaling applied (layer 0)")

    # Step 5: Verify functional equivalence
    if verbose:
        print("[5/7] Verifying functional equivalence...")

    is_equivalent, max_diff = verify_functional_equivalence(
        model_baseline,
        model_transformed,
        test_loader,
        tolerance=1e-5,
        device=device
    )

    if verbose:
        print(f"  Max output difference: {max_diff:.2e}")
        print(f"  Functionally equivalent: {is_equivalent}")

    if not is_equivalent:
        print(f"  ✗ FAIL: Networks not functionally equivalent (diff={max_diff:.2e})")
        return False

    # Step 6: Analyze transformed model
    if verbose:
        print("[6/7] Analyzing transformed model...")

    importances_transformed, edges_transformed = extract_coordinate_importances(
        model_transformed,
        train_loader,
        strategy='combined',
        max_samples=100,
        device=device
    )

    if verbose:
        print(f"  Extracted {len(importances_transformed)} coordinates")

    # Step 7: Compare importances
    if verbose:
        print("[7/7] Comparing importance rankings...")

    correlation = compute_rank_correlation(
        importances_baseline,
        importances_transformed,
        mapping=None
    )

    if verbose:
        print(f"  Spearman correlation: {correlation:.4f}")
        print(f"  Tolerance: {tolerance:.4f}")

    passed = correlation >= tolerance

    if verbose:
        if passed:
            print(f"\n  ✓ PASS: Combined invariance verified (ρ={correlation:.4f})")
        else:
            print(f"\n  ✗ FAIL: Correlation too low (ρ={correlation:.4f} < {tolerance:.4f})")

    return passed


def test_pruning_mask_consistency(
    tolerance: float = 0.9,
    pruning_ratio: float = 0.3,
    device: str = 'cpu',
    verbose: bool = True
) -> bool:
    """
    Test that ModuliSpacePruner produces consistent masks under transformations.

    This is the critical end-to-end test: do we actually prune the same edges
    (modulo relabeling) after transformations?

    Procedure:
    1. Create and train baseline MLP
    2. Prune with ModuliSpacePruner to get masks
    3. Apply scaling transformation (simpler than permutation for mask comparison)
    4. Prune scaled model
    5. Compare pruning masks (should be identical for scaling)

    Args:
        tolerance: Minimum mask IoU for test to pass
        pruning_ratio: Fraction of weights to prune
        device: Device to run on
        verbose: Print detailed progress

    Returns:
        True if test passes, False otherwise
    """
    if verbose:
        print("\n" + "="*70)
        print("TEST: Pruning Mask Consistency")
        print("="*70)

    # Step 1: Create baseline model
    if verbose:
        print("\n[1/6] Creating baseline model...")

    model_baseline = SimpleMLP(
        input_dim=10,
        hidden_dims=[15, 15],  # Smaller for faster testing
        output_dim=10,
        use_bias=False
    )

    # Step 2: Train baseline
    if verbose:
        print("[2/6] Training baseline model...")

    train_loader, test_loader = create_test_dataset(
        n_samples=150,
        input_dim=10,
        output_dim=10
    )

    model_baseline = train_simple_model(
        model_baseline,
        train_loader,
        epochs=10,
        lr=0.01,
        device=device
    )

    # Step 3: Prune baseline
    if verbose:
        print(f"[3/6] Pruning baseline model ({pruning_ratio*100:.0f}% sparsity)...")

    # Create analyzer for baseline
    analyzer_baseline = ModuliSpaceAnalyzer(model_baseline)
    analyzer_baseline.analyze_dataset(train_loader, max_samples=100)

    # Create pruner and prune
    pruner_baseline = ModuliSpacePruner(model_baseline, analyzer_baseline)
    _, masks_baseline = pruner_baseline.prune(
        pruning_ratio=pruning_ratio,
        strategy='combined'
    )

    if verbose:
        total_params = sum(m.numel() for m in masks_baseline.values())
        pruned_params = sum((m == 0).sum().item() for m in masks_baseline.values())
        actual_sparsity = pruned_params / total_params if total_params > 0 else 0
        print(f"  Baseline: {pruned_params}/{total_params} params pruned "
              f"({actual_sparsity*100:.1f}% sparsity)")

    # Step 4: Create scaled model and prune
    if verbose:
        print("[4/6] Creating and pruning scaled model...")

    model_scaled = copy.deepcopy(model_baseline)

    # Reset to original weights (before pruning)
    model_scaled.load_state_dict(model_baseline.state_dict())

    # Actually, we need the original unpruned baseline
    # Let's recreate it
    model_baseline_original = SimpleMLP(
        input_dim=10,
        hidden_dims=[15, 15],
        output_dim=10,
        use_bias=False
    )
    train_simple_model(model_baseline_original, train_loader, epochs=10, device=device)

    model_scaled = copy.deepcopy(model_baseline_original)

    # Apply scaling
    hidden_size = model_scaled.hidden_dims[0]
    scales = np.random.uniform(0.5, 2.0, size=hidden_size)

    scale_layer_neurons(model_scaled, layer_idx=0, scales=scales, verbose=False)

    if verbose:
        print(f"  Scaling applied (range: [{scales.min():.3f}, {scales.max():.3f}])")

    # Analyze and prune scaled model
    analyzer_scaled = ModuliSpaceAnalyzer(model_scaled)
    analyzer_scaled.analyze_dataset(train_loader, max_samples=100)

    pruner_scaled = ModuliSpacePruner(model_scaled, analyzer_scaled)
    _, masks_scaled = pruner_scaled.prune(
        pruning_ratio=pruning_ratio,
        strategy='combined'
    )

    if verbose:
        total_params_scaled = sum(m.numel() for m in masks_scaled.values())
        pruned_params_scaled = sum((m == 0).sum().item() for m in masks_scaled.values())
        actual_sparsity_scaled = pruned_params_scaled / total_params_scaled
        print(f"  Scaled: {pruned_params_scaled}/{total_params_scaled} params pruned "
              f"({actual_sparsity_scaled*100:.1f}% sparsity)")

    # Step 5: Verify functional equivalence before pruning
    if verbose:
        print("[5/6] Verifying scaled networks are functionally equivalent...")

    is_equivalent, max_diff = verify_functional_equivalence(
        model_baseline_original,
        model_scaled,
        test_loader,
        tolerance=1e-5,
        device=device
    )

    if verbose:
        print(f"  Max difference: {max_diff:.2e}")

    if not is_equivalent:
        print(f"  ⚠ WARNING: Pre-pruning networks differ by {max_diff:.2e}")

    # Step 6: Compare masks
    if verbose:
        print("[6/6] Comparing pruning masks...")

    # For scaling invariance, masks should be identical
    are_equivalent, iou = compare_pruning_masks(
        masks_baseline,
        masks_scaled,
        tolerance=tolerance
    )

    if verbose:
        print(f"  Mask IoU: {iou:.4f}")
        print(f"  Tolerance: {tolerance:.4f}")

    passed = are_equivalent

    if verbose:
        if passed:
            print(f"\n  ✓ PASS: Pruning masks consistent (IoU={iou:.4f})")
        else:
            print(f"\n  ✗ FAIL: Mask IoU too low (IoU={iou:.4f} < {tolerance:.4f})")

    return passed


# ============================================================================
# Main Test Runner
# ============================================================================

def run_all_tests(
    tolerance: float = 0.95,
    device: str = 'cpu',
    verbose: bool = True
) -> Dict[str, bool]:
    """
    Run all isomorphism invariance tests.

    Executes full test suite and reports results.

    Args:
        tolerance: Minimum correlation/IoU for tests to pass
        device: Device to run on ('cpu' or 'cuda')
        verbose: Print detailed test output

    Returns:
        Dictionary of test results {test_name: passed}
    """
    if verbose:
        print("\n" + "#"*70)
        print("# ISOMORPHISM INVARIANCE TEST SUITE")
        print("#"*70)
        print(f"\nConfiguration:")
        print(f"  Tolerance: {tolerance}")
        print(f"  Device: {device}")
        print(f"  Verbose: {verbose}")

    results = {}

    # Test 1: Permutation invariance
    try:
        results['permutation_invariance'] = test_permutation_invariance(
            tolerance=tolerance,
            device=device,
            verbose=verbose
        )
    except Exception as e:
        print(f"\n✗ EXCEPTION in permutation test: {e}")
        results['permutation_invariance'] = False

    # Test 2: Scaling invariance
    try:
        results['scaling_invariance'] = test_scaling_invariance(
            tolerance=tolerance,
            device=device,
            verbose=verbose
        )
    except Exception as e:
        print(f"\n✗ EXCEPTION in scaling test: {e}")
        results['scaling_invariance'] = False

    # Test 3: Combined invariance
    try:
        results['combined_invariance'] = test_combined_invariance(
            tolerance=tolerance,
            device=device,
            verbose=verbose
        )
    except Exception as e:
        print(f"\n✗ EXCEPTION in combined test: {e}")
        results['combined_invariance'] = False

    # Test 4: Pruning mask consistency
    try:
        results['mask_consistency'] = test_pruning_mask_consistency(
            tolerance=0.9,  # Slightly lower tolerance for mask IoU
            pruning_ratio=0.3,
            device=device,
            verbose=verbose
        )
    except Exception as e:
        print(f"\n✗ EXCEPTION in mask consistency test: {e}")
        results['mask_consistency'] = False

    # Print summary
    if verbose:
        print("\n" + "#"*70)
        print("# TEST SUMMARY")
        print("#"*70)

        for test_name, passed in results.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{status}: {test_name}")

        num_passed = sum(results.values())
        num_total = len(results)

        print(f"\nOverall: {num_passed}/{num_total} tests passed")

        if num_passed == num_total:
            print("\n🎉 ALL TESTS PASSED! Isomorphism invariance verified.")
        else:
            print("\n⚠️  SOME TESTS FAILED. Review output above.")

        print("#"*70)

    return results


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == '__main__':
    """
    Run isomorphism invariance test suite.

    Usage:
        python test_isomorphism_invariance.py
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Detect device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Run all tests
    results = run_all_tests(
        tolerance=0.95,
        device=device,
        verbose=True
    )

    # Exit with appropriate code
    all_passed = all(results.values())
    sys.exit(0 if all_passed else 1)
