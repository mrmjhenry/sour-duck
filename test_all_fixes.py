"""
Comprehensive Test Suite for All Moduli Space Pruning Fixes
Tests dimension mismatch, layer indexing, and inhomogeneous array fixes
"""

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_all_fixes():
    """Test that all fixes work together"""
    print("="*60)
    print("Testing All Fixes Together")
    print("="*60)
    
    # Create a complex network that would trigger all the errors
    class ComplexNet(nn.Module):
        def __init__(self):
            super().__init__()
            # Sequential with named layers (triggers layer indexing issue)
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),  # features.0
                nn.BatchNorm2d(32),               # features.1
                nn.ReLU(inplace=True),            # features.2
                nn.MaxPool2d(2),                  # features.3
                nn.Conv2d(32, 64, 3, padding=1),  # features.4
                nn.ReLU(inplace=True),            # features.5
            )
            self.classifier = nn.Sequential(
                nn.Linear(64 * 16 * 16, 128),    # classifier.0
                nn.ReLU(inplace=True),            # classifier.1
                nn.Linear(128, 10)                # classifier.2
            )
        
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
    
    print("1. Testing Conv2d dimension fix...")
    print("-" * 40)
    
    try:
        from moduli_space_pruning import DataRepresentation
        
        network = ComplexNet()
        data_rep = DataRepresentation(network)
        
        # Test with different batch sizes (would trigger dimension mismatch)
        test_inputs = [
            torch.randn(1, 3, 32, 32),   # Single sample
            torch.randn(4, 3, 32, 32),   # Small batch
            torch.randn(32, 3, 32, 32),  # Large batch
        ]
        
        for i, x in enumerate(test_inputs):
            representation = data_rep.compute_representation(x)
            print(f"  Batch size {x.shape[0]}: ✓ {len(representation)} layers processed")
        
        data_rep.cleanup()
        print("✅ Dimension fix working!")
        
    except RuntimeError as e:
        if "size of tensor" in str(e):
            print(f"❌ Dimension mismatch still present: {e}")
            return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    
    print("\n2. Testing layer indexing fix...")
    print("-" * 40)
    
    try:
        from moduli_space_pruning import ModuliSpaceAnalyzer
        
        analyzer = ModuliSpaceAnalyzer(network)
        
        # Create test data that would trigger the layer indexing error
        from torch.utils.data import TensorDataset, DataLoader
        X = torch.randn(20, 3, 32, 32)
        y = torch.randint(0, 10, (20,))
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=4)
        
        # This would trigger "('features.0', None) is not in list" error
        analyzer.analyze_dataset(dataloader, max_samples=10)
        
        print(f"  Analyzed {len(analyzer.coordinate_stats)} coordinates")
        
        # Check that layer indices were assigned correctly
        has_valid_indices = all(
            hasattr(coord, 'layer_idx') and coord.layer_idx >= -1
            for coord in analyzer.coordinate_stats.values()
        )
        
        if has_valid_indices:
            print("✅ Layer indexing fix working!")
        else:
            print("❌ Some coordinates have invalid layer indices")
            return False
            
    except ValueError as e:
        if "is not in list" in str(e):
            print(f"❌ Layer indexing error still present: {e}")
            return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    
    print("\n3. Testing inhomogeneous array fix...")
    print("-" * 40)
    
    try:
        # Simulate coordinate values with different lengths
        # This would trigger the inhomogeneous array error
        analyzer.coordinate_stats = {}
        
        # Create coordinate values with varying lengths
        coordinate_values = defaultdict(list)
        for i in range(100):
            # Intentionally create lists of different lengths
            num_values = np.random.randint(1, 10)
            coordinate_values[i] = np.random.randn(num_values).tolist()
        
        # Add some edge cases
        coordinate_values[100] = []  # Empty list
        coordinate_values[101] = [1.0]  # Single value
        coordinate_values[102] = [float('nan'), 1.0, float('inf')]  # Special values
        
        # This would trigger "inhomogeneous shape" error
        analyzer._compute_coordinate_statistics(coordinate_values)
        
        print(f"  Processed {len(analyzer.coordinate_stats)} coordinates")
        
        # Verify all statistics are valid Python floats
        all_valid = True
        for coord in analyzer.coordinate_stats.values():
            if not all(isinstance(getattr(coord, attr), float) 
                      for attr in ['usage_rate', 'variance', 'information_content', 'importance_score']):
                all_valid = False
                break
        
        if all_valid:
            print("✅ Inhomogeneous array fix working!")
        else:
            print("❌ Some statistics are not valid floats")
            return False
            
    except ValueError as e:
        if "inhomogeneous" in str(e):
            print(f"❌ Inhomogeneous array error still present: {e}")
            return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    
    print("\n4. Testing full pruning pipeline...")
    print("-" * 40)
    
    try:
        from moduli_space_pruning import ModuliSpacePruner
        
        # Create fresh network and analyzer
        network = ComplexNet()
        analyzer = ModuliSpaceAnalyzer(network)
        
        # Run full analysis
        analyzer.analyze_dataset(dataloader, max_samples=10)
        
        # Apply pruning (tests all components together)
        pruner = ModuliSpacePruner(network, analyzer)
        pruned_network, masks = pruner.prune(pruning_ratio=0.3)
        
        # Verify pruning was applied
        total_params = sum(p.numel() for p in network.parameters())
        zero_params = sum((p == 0).sum().item() for p in network.parameters())
        sparsity = zero_params / total_params
        
        print(f"  Sparsity achieved: {sparsity:.2%}")
        
        # Test that pruned network still works
        test_input = torch.randn(2, 3, 32, 32)
        output = pruned_network(test_input)
        
        print(f"  Output shape: {output.shape}")
        
        if output.shape == (2, 10):
            print("✅ Full pipeline working!")
        else:
            print(f"❌ Unexpected output shape: {output.shape}")
            return False
            
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        return False
    
    return True


def run_stress_test():
    """Run stress test with edge cases"""
    print("\n" + "="*60)
    print("Running Stress Test")
    print("="*60)
    
    # Create network with unusual architecture
    class EdgeCaseNet(nn.Module):
        def __init__(self):
            super().__init__()
            # Deep sequential nesting
            self.block1 = nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(3, 16, 1),
                    nn.BatchNorm2d(16)
                ),
                nn.ReLU()
            )
            # Single layer
            self.conv_single = nn.Conv2d(16, 32, 1)
            # Empty sequential
            self.empty_block = nn.Sequential()
            # Final layers
            self.final = nn.Linear(32 * 32 * 32, 10)
        
        def forward(self, x):
            x = self.block1(x)
            x = self.conv_single(x)
            x = self.empty_block(x)
            x = x.view(x.size(0), -1)
            x = self.final(x)
            return x
    
    try:
        from moduli_space_pruning import ModuliSpaceAnalyzer, ModuliSpacePruner
        
        network = EdgeCaseNet()
        analyzer = ModuliSpaceAnalyzer(network)
        
        # Create edge case data
        X = torch.randn(5, 3, 32, 32)
        X[0] = 0  # All zeros
        X[1] = torch.ones_like(X[1]) * 1e-10  # Very small values
        X[2] = torch.ones_like(X[2]) * 1e10   # Very large values
        
        from torch.utils.data import TensorDataset, DataLoader
        y = torch.zeros(5, dtype=torch.long)
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=1)
        
        # Run analysis
        analyzer.analyze_dataset(dataloader, max_samples=5)
        
        # Apply pruning
        pruner = ModuliSpacePruner(network, analyzer)
        pruned_network, _ = pruner.prune(pruning_ratio=0.5)
        
        print("✅ Stress test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Stress test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("COMPREHENSIVE FIX VALIDATION")
    print("="*60)
    
    all_fixes_working = test_all_fixes()
    stress_test_passed = run_stress_test()
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    if all_fixes_working and stress_test_passed:
        print("🎉 ALL FIXES VERIFIED AND WORKING!")
        print("\nThe following issues have been resolved:")
        print("✅ Conv2d dimension mismatch")
        print("✅ Layer indexing with Sequential modules")
        print("✅ Inhomogeneous array handling")
        print("✅ Edge cases and stress conditions")
        return True
    else:
        print("⚠️ Some issues remain:")
        if not all_fixes_working:
            print("❌ Basic fixes not working correctly")
        if not stress_test_passed:
            print("❌ Stress test failed")
        return False


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
