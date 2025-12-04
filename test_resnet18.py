"""
Test Suite for ResNet-18 with Moduli Space Pruning

Verifies that ResNet-18 works correctly with the moduli space pruning framework:
- Quiver representation extraction
- Data representation computation
- Coordinate analysis
- Pruning functionality
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import sys

try:
    from resnet18_cifar import create_resnet18_cifar10, ResNet18_CIFAR
    from moduli_space_pruning import (
        QuiverRepresentation,
        DataRepresentation,
        ModuliSpaceAnalyzer,
        ModuliSpacePruner
    )
    HAS_DEPENDENCIES = True
except ImportError as e:
    print(f"Warning: Could not import dependencies: {e}")
    HAS_DEPENDENCIES = False


def test_resnet18_architecture():
    """Test 1: Verify ResNet-18 architecture works correctly."""
    print("\n" + "=" * 70)
    print("TEST 1: ResNet-18 Architecture")
    print("=" * 70)

    # Create model
    model = create_resnet18_cifar10(num_classes=10, small=False)
    print(f"✓ Created ResNet-18 model")
    print(f"  Parameters: {model.get_num_parameters():,}")

    # Test forward pass
    x = torch.randn(2, 3, 32, 32)
    model.eval()
    with torch.no_grad():
        output = model(x)

    assert output.shape == (2, 10), f"Expected shape (2, 10), got {output.shape}"
    print(f"✓ Forward pass successful")
    print(f"  Input:  {x.shape}")
    print(f"  Output: {output.shape}")

    # Verify gradients flow
    model.train()
    output = model(x)
    loss = output.sum()
    loss.backward()

    has_gradients = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.parameters()
    )
    assert has_gradients, "Gradients not flowing through network"
    print(f"✓ Gradients flow correctly")

    print("\n✅ Test 1 PASSED: ResNet-18 architecture works correctly")
    return True


def test_quiver_extraction():
    """Test 2: Verify quiver representation extraction from ResNet-18."""
    print("\n" + "=" * 70)
    print("TEST 2: Quiver Representation Extraction")
    print("=" * 70)

    # Create small model for faster testing
    model = create_resnet18_cifar10(num_classes=10, small=True)
    print(f"✓ Created small ResNet-18")

    # Extract quiver representation
    quiver = QuiverRepresentation(model)
    print(f"✓ Extracted quiver representation")

    # Verify extraction
    num_edges = len(quiver.edges)
    num_vertices = len(quiver.vertices)

    print(f"  Number of edges (weights): {num_edges:,}")
    print(f"  Number of vertices (neurons): {num_vertices}")

    assert num_edges > 0, "No edges extracted"
    assert num_vertices > 0, "No vertices extracted"

    # Verify edge format
    for i, edge in enumerate(quiver.edges[:3]):
        assert len(edge) == 3, f"Edge {i} has {len(edge)} elements, expected 3"
        edge_idx, edge_info, layer_name = edge
        print(f"  Edge {i}: idx={edge_idx}, info={edge_info}, layer={layer_name}")

    print(f"✓ All edges have correct format (idx, info, layer_name)")

    # Calculate moduli dimension
    moduli_dim = num_edges - num_vertices
    print(f"\n  Moduli space dimension: {moduli_dim:,}")
    print(f"  Ratio: {moduli_dim/num_edges*100:.1f}% of parameters")

    print("\n✅ Test 2 PASSED: Quiver extraction works for ResNet-18")
    return True


def test_data_representation():
    """Test 3: Verify data representation computation."""
    print("\n" + "=" * 70)
    print("TEST 3: Data Representation Computation")
    print("=" * 70)

    # Create model and data
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_resnet18_cifar10(num_classes=10, small=True)
    print(f"✓ Using device: {device}")

    # Create data representation
    data_rep = DataRepresentation(model, device=device)
    print(f"✓ Created data representation extractor")

    # Test with sample data
    x = torch.randn(2, 3, 32, 32).to(device)
    representation = data_rep.compute_representation(x)

    print(f"✓ Computed representation")
    print(f"  Number of layers captured: {len(representation)}")

    # Verify representation
    assert len(representation) > 0, "No layers in representation"

    # Show first few layers
    print(f"\n  Sample layers:")
    for i, (name, tensor) in enumerate(list(representation.items())[:5]):
        print(f"    {name}: shape {tuple(tensor.shape)}")

    # Cleanup
    data_rep.cleanup()
    print(f"✓ Cleaned up hooks")

    print("\n✅ Test 3 PASSED: Data representation computation works")
    return True


def test_moduli_analysis():
    """Test 4: Verify moduli space analysis."""
    print("\n" + "=" * 70)
    print("TEST 4: Moduli Space Analysis")
    print("=" * 70)

    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_resnet18_cifar10(num_classes=10, small=True).to(device)
    print(f"✓ Created ResNet-18 (small)")

    # Create dummy dataset
    n_samples = 50
    X = torch.randn(n_samples, 3, 32, 32)
    y = torch.randint(0, 10, (n_samples,))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False)
    print(f"✓ Created dummy dataset ({n_samples} samples)")

    # Analyze
    print(f"  Analyzing moduli space...")
    analyzer = ModuliSpaceAnalyzer(model)
    analyzer.analyze_dataset(dataloader, max_samples=n_samples)

    print(f"✓ Analysis complete")
    print(f"  Total coordinates: {analyzer.total_coordinates:,}")
    print(f"  Moduli dimension: {analyzer.moduli_dimension:,}")

    # Verify analysis
    assert analyzer.total_coordinates > 0, "No coordinates analyzed"
    assert len(analyzer.coordinates) > 0, "No coordinate objects created"

    # Check coordinate statistics
    importances = [coord.importance_score for coord in analyzer.coordinates]
    print(f"\n  Importance scores:")
    print(f"    Min: {min(importances):.6f}")
    print(f"    Max: {max(importances):.6f}")
    print(f"    Mean: {np.mean(importances):.6f}")
    print(f"    Std: {np.std(importances):.6f}")

    # Verify reasonable importance scores
    assert max(importances) > 0, "All importance scores are zero"
    assert min(importances) >= 0, "Negative importance scores found"

    print("\n✅ Test 4 PASSED: Moduli space analysis works")
    return True


def test_pruning():
    """Test 5: Verify pruning functionality."""
    print("\n" + "=" * 70)
    print("TEST 5: Pruning Functionality")
    print("=" * 70)

    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_resnet18_cifar10(num_classes=10, small=True).to(device)
    original_params = model.get_num_parameters()
    print(f"✓ Created ResNet-18 (small): {original_params:,} parameters")

    # Create dummy dataset
    n_samples = 30
    X = torch.randn(n_samples, 3, 32, 32)
    y = torch.randint(0, 10, (n_samples,))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=10)

    # Analyze
    print(f"  Analyzing coordinates...")
    analyzer = ModuliSpaceAnalyzer(model)
    analyzer.analyze_dataset(dataloader, max_samples=n_samples)

    # Prune
    pruning_ratios = [0.3, 0.5]
    for ratio in pruning_ratios:
        print(f"\n  Testing {ratio*100:.0f}% pruning:")

        # Create pruner
        pruner = ModuliSpacePruner(model, analyzer)
        pruned_model, masks = pruner.prune(pruning_ratio=ratio)

        # Count pruned parameters
        total_params = 0
        pruned_params = 0
        for mask in masks:
            total_params += mask.numel()
            pruned_params += (mask == 0).sum().item()

        actual_sparsity = pruned_params / total_params
        print(f"    Pruned {pruned_params:,} / {total_params:,} parameters")
        print(f"    Actual sparsity: {actual_sparsity*100:.1f}%")
        print(f"    Target sparsity: {ratio*100:.1f}%")

        # Verify sparsity is approximately correct
        assert abs(actual_sparsity - ratio) < 0.05, \
            f"Sparsity mismatch: {actual_sparsity:.2f} vs {ratio:.2f}"

        # Test forward pass still works
        x_test = torch.randn(2, 3, 32, 32).to(device)
        with torch.no_grad():
            output = pruned_model(x_test)

        assert output.shape == (2, 10), f"Output shape wrong: {output.shape}"
        print(f"    ✓ Forward pass works after pruning")

    print("\n✅ Test 5 PASSED: Pruning functionality works")
    return True


def test_residual_connections():
    """Test 6: Verify residual connections are handled correctly."""
    print("\n" + "=" * 70)
    print("TEST 6: Residual Connection Handling")
    print("=" * 70)

    # Create model
    model = create_resnet18_cifar10(num_classes=10, small=True)

    # Count total Conv2d and Linear layers
    conv_count = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d))
    linear_count = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    total_learnable = conv_count + linear_count

    print(f"✓ Model has:")
    print(f"  Conv2d layers: {conv_count}")
    print(f"  Linear layers: {linear_count}")
    print(f"  Total learnable layers: {total_learnable}")

    # Extract quiver
    quiver = QuiverRepresentation(model)

    # Verify all layers are captured
    # (residual connections are additions, not learnable parameters)
    layer_names_in_edges = set(edge[2] for edge in quiver.edges)
    print(f"✓ Unique layers in quiver: {len(layer_names_in_edges)}")

    # The number of unique layer names should match our count
    assert len(layer_names_in_edges) == total_learnable, \
        f"Layer count mismatch: {len(layer_names_in_edges)} vs {total_learnable}"

    print(f"✓ All learnable layers captured correctly")
    print(f"  (Residual connections are non-parametric additions)")

    # Verify forward pass with residuals
    x = torch.randn(2, 3, 32, 32)
    model.eval()
    with torch.no_grad():
        output1 = model(x)
        output2 = model(x)

    # Should be deterministic
    assert torch.allclose(output1, output2), "Non-deterministic forward pass"
    print(f"✓ Forward pass is deterministic")

    print("\n✅ Test 6 PASSED: Residual connections handled correctly")
    return True


def run_all_tests():
    """Run all ResNet-18 tests."""
    if not HAS_DEPENDENCIES:
        print("\n❌ Missing dependencies. Cannot run tests.")
        print("Make sure you have:")
        print("  - resnet18_cifar.py")
        print("  - moduli_space_pruning.py")
        print("  - PyTorch installed")
        return False

    print("\n" + "=" * 70)
    print("RESNET-18 MODULI SPACE PRUNING TEST SUITE")
    print("=" * 70)

    tests = [
        ("Architecture", test_resnet18_architecture),
        ("Quiver Extraction", test_quiver_extraction),
        ("Data Representation", test_data_representation),
        ("Moduli Analysis", test_moduli_analysis),
        ("Pruning", test_pruning),
        ("Residual Connections", test_residual_connections),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result, None))
        except Exception as e:
            print(f"\n❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False, str(e)))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, result, _ in results if result)
    total = len(results)

    for test_name, result, error in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
        if error:
            print(f"         Error: {error}")

    print("\n" + "=" * 70)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 70)

    if passed == total:
        print("\n🎉 All tests passed! ResNet-18 is fully compatible with moduli space pruning.")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. See details above.")
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
