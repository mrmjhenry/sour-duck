"""
Test script to verify the dimension mismatch fix in moduli space pruning
"""

import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_conv_layer_fix():
    """Test that the compute_representation fix works for Conv2d layers"""
    print("Testing Conv2d layer dimension fix...")
    
    # Create a simple network with Conv2d layers
    class TestNet(nn.Module):
        def __init__(self):
            super(TestNet, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.relu = nn.ReLU()
            self.pool = nn.MaxPool2d(2)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.fc = nn.Linear(64 * 16 * 16, 10)
        
        def forward(self, x):
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    try:
        from moduli_space_pruning import DataRepresentation
        
        # Create network and data representation
        network = TestNet()
        data_rep = DataRepresentation(network)
        
        # Test with batch of different sizes
        test_sizes = [(1, 3, 32, 32), (4, 3, 32, 32), (32, 3, 32, 32)]
        
        for size in test_sizes:
            print(f"  Testing with input shape: {size}")
            x = torch.randn(size)
            
            try:
                representation = data_rep.compute_representation(x)
                print(f"    ✓ Successfully computed representation")
                print(f"    Number of layers in representation: {len(representation)}")
            except RuntimeError as e:
                if "size of tensor" in str(e):
                    print(f"    ✗ Dimension mismatch error: {e}")
                    return False
                else:
                    raise
        
        # Cleanup hooks
        data_rep.cleanup()
        print("✓ All Conv2d tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        return False


def test_linear_layer():
    """Test that Linear layers still work correctly"""
    print("\nTesting Linear layer compatibility...")
    
    class SimpleNet(nn.Module):
        def __init__(self):
            super(SimpleNet, self).__init__()
            self.fc1 = nn.Linear(10, 20)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(20, 5)
        
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    try:
        from moduli_space_pruning import DataRepresentation
        
        network = SimpleNet()
        data_rep = DataRepresentation(network)
        
        # Test with different batch sizes
        for batch_size in [1, 4, 16]:
            x = torch.randn(batch_size, 10)
            representation = data_rep.compute_representation(x)
            print(f"  ✓ Batch size {batch_size}: {len(representation)} layers processed")
        
        data_rep.cleanup()
        print("✓ All Linear layer tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def test_mixed_architecture():
    """Test with mixed Conv2d and Linear layers"""
    print("\nTesting mixed architecture...")
    
    class MixedNet(nn.Module):
        def __init__(self):
            super(MixedNet, self).__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(16)
            self.relu = nn.ReLU()
            self.pool = nn.MaxPool2d(2)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(32)
            self.fc1 = nn.Linear(32 * 8 * 8, 64)
            self.fc2 = nn.Linear(64, 10)
        
        def forward(self, x):
            x = self.pool(self.relu(self.bn1(self.conv1(x))))
            x = self.pool(self.relu(self.bn2(self.conv2(x))))
            x = x.view(x.size(0), -1)
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    try:
        from moduli_space_pruning import DataRepresentation, ModuliSpaceAnalyzer
        
        network = MixedNet()
        
        # Test data representation
        data_rep = DataRepresentation(network)
        x = torch.randn(8, 3, 32, 32)
        representation = data_rep.compute_representation(x)
        print(f"  ✓ Data representation computed: {len(representation)} layers")
        data_rep.cleanup()
        
        # Test full analyzer
        print("  Testing full analyzer...")
        analyzer = ModuliSpaceAnalyzer(network)
        
        # Create small dataset for testing
        from torch.utils.data import TensorDataset, DataLoader
        X = torch.randn(50, 3, 32, 32)
        y = torch.randint(0, 10, (50,))
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=8)
        
        # Analyze dataset
        analyzer.analyze_dataset(dataloader, max_samples=20)
        
        print(f"  ✓ Analyzer processed {len(analyzer.coordinate_stats)} coordinates")
        
        # Get prunable coordinates
        prunable = analyzer.get_prunable_coordinates(pruning_ratio=0.3)
        print(f"  ✓ Identified {len(prunable)} prunable coordinates")
        
        print("✓ Mixed architecture test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("Testing Dimension Mismatch Fix")
    print("="*60)
    
    tests = [
        ("Conv2d Layer Fix", test_conv_layer_fix),
        ("Linear Layer Compatibility", test_linear_layer),
        ("Mixed Architecture", test_mixed_architecture),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ {name} test failed with unexpected error: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print(f"Tests passed: {passed}")
    print(f"Tests failed: {failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed! The dimension mismatch issue is fixed.")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the implementation.")
    
    return failed == 0


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
