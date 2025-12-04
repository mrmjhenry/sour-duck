"""
Test Suite for Moduli Space Pruning Implementation
Verifies core functionality and demonstrates basic usage
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from moduli_space_pruning import (
    QuiverRepresentation,
    DataRepresentation,
    ModuliSpaceAnalyzer,
    ModuliSpacePruner
)

def test_quiver_representation():
    """Test quiver representation extraction"""
    print("\nTest 1: Quiver Representation")
    print("-" * 40)
    
    # Create simple network
    network = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 2)
    )
    
    # Extract quiver representation
    quiver = QuiverRepresentation(network)
    
    print(f"Number of edges: {len(quiver.edges)}")
    print(f"Number of vertices: {len(quiver.vertices)}")
    
    assert len(quiver.edges) > 0, "Failed to extract edges"
    assert len(quiver.vertices) > 0, "Failed to extract vertices"
    
    print("✓ Quiver representation test passed")
    return True


def test_data_representation():
    """Test data representation computation"""
    print("\nTest 2: Data Representation")
    print("-" * 40)
    
    # Create simple network
    network = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 2)
    )
    
    # Create data representation extractor
    data_rep = DataRepresentation(network)
    
    # Test with random input
    x = torch.randn(1, 10)
    representation = data_rep.compute_representation(x)
    
    print(f"Number of layers in representation: {len(representation)}")
    for name, tensor in representation.items():
        print(f"  {name}: shape {tensor.shape}")
    
    assert len(representation) > 0, "Failed to compute representation"
    
    # Cleanup hooks
    data_rep.cleanup()
    
    print("✓ Data representation test passed")
    return True


def test_moduli_space_analysis():
    """Test moduli space analysis"""
    print("\nTest 3: Moduli Space Analysis")
    print("-" * 40)
    
    # Create simple network
    network = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 2)
    )
    
    # Create analyzer
    analyzer = ModuliSpaceAnalyzer(network)
    
    # Create dummy dataset
    from torch.utils.data import TensorDataset, DataLoader
    X = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=10)
    
    # Analyze dataset
    analyzer.analyze_dataset(dataloader, max_samples=50)
    
    print(f"Number of coordinates analyzed: {len(analyzer.coordinate_stats)}")
    
    if analyzer.coordinate_stats:
        # Get sample coordinate
        sample_coord = list(analyzer.coordinate_stats.values())[0]
        print(f"Sample coordinate stats:")
        print(f"  Usage rate: {sample_coord.usage_rate:.3f}")
        print(f"  Variance: {sample_coord.variance:.6f}")
        print(f"  Information content: {sample_coord.information_content:.3f}")
        print(f"  Importance score: {sample_coord.importance_score:.3f}")
    
    assert len(analyzer.coordinate_stats) > 0, "Failed to analyze coordinates"
    
    print("✓ Moduli space analysis test passed")
    return analyzer


def test_pruning():
    """Test pruning functionality"""
    print("\nTest 4: Pruning")
    print("-" * 40)
    
    # Create and analyze network
    network = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 2)
    )
    
    analyzer = ModuliSpaceAnalyzer(network)
    
    # Create dummy dataset
    from torch.utils.data import TensorDataset, DataLoader
    X = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=10)
    
    # Analyze
    analyzer.analyze_dataset(dataloader, max_samples=50)
    
    # Get prunable coordinates
    prunable = analyzer.get_prunable_coordinates(pruning_ratio=0.3)
    print(f"Number of prunable coordinates: {len(prunable)}")
    
    # Apply pruning
    pruner = ModuliSpacePruner(network, analyzer)
    pruned_network, masks = pruner.prune(pruning_ratio=0.3)
    
    # Check sparsity
    total_weights = 0
    zero_weights = 0
    for name, module in pruned_network.named_modules():
        if isinstance(module, nn.Linear):
            total_weights += module.weight.numel()
            zero_weights += (module.weight == 0).sum().item()
    
    sparsity = zero_weights / total_weights if total_weights > 0 else 0
    print(f"Network sparsity after pruning: {sparsity:.2%}")
    
    assert sparsity > 0, "No weights were pruned"
    
    print("✓ Pruning test passed")
    return True


def test_network_functionality():
    """Test that pruned network still functions"""
    print("\nTest 5: Network Functionality After Pruning")
    print("-" * 40)
    
    # Create network
    network = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 2)
    )
    
    # Get initial output
    x = torch.randn(5, 10)
    initial_output = network(x).detach().clone()
    
    # Analyze and prune
    analyzer = ModuliSpaceAnalyzer(network)
    
    from torch.utils.data import TensorDataset, DataLoader
    X = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=10)
    
    analyzer.analyze_dataset(dataloader, max_samples=50)
    
    pruner = ModuliSpacePruner(network, analyzer)
    pruned_network, _ = pruner.prune(pruning_ratio=0.3)
    
    # Test pruned network
    pruned_output = pruned_network(x)
    
    print(f"Initial output shape: {initial_output.shape}")
    print(f"Pruned output shape: {pruned_output.shape}")
    print(f"Output difference (L2): {(initial_output - pruned_output).norm():.4f}")
    
    assert pruned_output.shape == initial_output.shape, "Output shape changed after pruning"
    
    print("✓ Network functionality test passed")
    return True


def test_visualization():
    """Test visualization capabilities"""
    print("\nTest 6: Visualization")
    print("-" * 40)
    
    try:
        from analysis import ModuliSpaceVisualizer
        
        # Create simple network
        network = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )
        
        # Analyze
        analyzer = ModuliSpaceAnalyzer(network)
        
        from torch.utils.data import TensorDataset, DataLoader
        X = torch.randn(100, 10)
        y = torch.randint(0, 2, (100,))
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=10)
        
        analyzer.analyze_dataset(dataloader, max_samples=50)
        
        # Create visualizer
        visualizer = ModuliSpaceVisualizer(analyzer)
        
        # Extract embeddings
        embeddings, labels = visualizer.extract_coordinate_embeddings(dataloader, max_samples=20)
        
        print(f"Embedding shape: {embeddings.shape}")
        print(f"Labels shape: {labels.shape}")
        
        assert embeddings.shape[0] == labels.shape[0], "Embedding/label mismatch"
        
        print("✓ Visualization test passed")
        return True
        
    except ImportError:
        print("⚠ Visualization test skipped (missing dependencies)")
        return True


def run_integration_test():
    """Run a complete integration test"""
    print("\nIntegration Test: Complete Pipeline")
    print("=" * 40)
    
    # Create a more complex network
    class TestNet(nn.Module):
        def __init__(self):
            super(TestNet, self).__init__()
            self.conv1 = nn.Conv2d(1, 10, 3, padding=1)
            self.pool = nn.MaxPool2d(2)
            self.fc1 = nn.Linear(10 * 14 * 14, 50)
            self.fc2 = nn.Linear(50, 10)
            
        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = x.view(x.size(0), -1)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    network = TestNet()
    
    # Create dataset
    from torch.utils.data import TensorDataset, DataLoader
    X = torch.randn(200, 1, 28, 28)
    y = torch.randint(0, 10, (200,))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 1. Train network briefly
    print("1. Training network...")
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    network.train()
    for epoch in range(2):
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            output = network(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            break  # Just one batch per epoch for testing
    
    print(f"   Training loss: {loss.item():.4f}")
    
    # 2. Analyze moduli space
    print("2. Analyzing moduli space...")
    analyzer = ModuliSpaceAnalyzer(network)
    analyzer.analyze_dataset(dataloader, max_samples=100)
    
    coords = list(analyzer.coordinate_stats.values())
    avg_usage = np.mean([c.usage_rate for c in coords])
    print(f"   Average coordinate usage: {avg_usage:.3f}")
    
    # 3. Prune network
    print("3. Pruning network...")
    pruner = ModuliSpacePruner(network, analyzer)
    pruned_network, masks = pruner.prune(pruning_ratio=0.4)
    
    # Calculate sparsity
    total_params = sum(p.numel() for p in network.parameters())
    zero_params = sum((p == 0).sum().item() for p in network.parameters())
    sparsity = zero_params / total_params
    print(f"   Network sparsity: {sparsity:.2%}")
    
    # 4. Test pruned network
    print("4. Testing pruned network...")
    network.eval()
    with torch.no_grad():
        test_input = torch.randn(10, 1, 28, 28)
        output = pruned_network(test_input)
        print(f"   Output shape: {output.shape}")
        print(f"   Output range: [{output.min():.2f}, {output.max():.2f}]")
    
    print("\n✓ Integration test completed successfully!")
    return True


def main():
    """Run all tests"""
    print("="*60)
    print("Moduli Space Pruning - Test Suite")
    print("="*60)
    
    # Create results directory
    Path('results').mkdir(exist_ok=True)
    
    tests = [
        ("Quiver Representation", test_quiver_representation),
        ("Data Representation", test_data_representation),
        ("Moduli Space Analysis", test_moduli_space_analysis),
        ("Pruning", test_pruning),
        ("Network Functionality", test_network_functionality),
        ("Visualization", test_visualization),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ {name} test failed with error: {e}")
            failed += 1
    
    # Run integration test
    print("\n" + "="*60)
    try:
        run_integration_test()
        passed += 1
    except Exception as e:
        print(f"✗ Integration test failed with error: {e}")
        failed += 1
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print(f"Tests passed: {passed}")
    print(f"Tests failed: {failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed successfully!")
    else:
        print(f"\n⚠️  {failed} test(s) failed")
    
    return failed == 0


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
