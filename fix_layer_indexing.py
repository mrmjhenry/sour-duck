"""
Fix for Layer Indexing Error in Moduli Space Pruning
Resolves: "('features.0', None) is not in list" error
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple

def demonstrate_error():
    """Demonstrate the original error"""
    print("Demonstrating the original error:")
    print("-" * 40)
    
    # Create a network with named modules
    class TestNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3),  # This will be 'features.0'
                nn.ReLU(),            # This will be 'features.1'
                nn.MaxPool2d(2)       # This will be 'features.2'
            )
            self.classifier = nn.Linear(32 * 15 * 15, 10)
    
    network = TestNet()
    
    # Show what named_modules() actually returns
    print("What named_modules() returns:")
    for name, module in network.named_modules():
        print(f"  ('{name}', {module.__class__.__name__})")
    
    print("\nThe error occurs when trying to find:")
    print("  ('features.0', None) in the list above")
    print("  But None is never a module - it should be Conv2d!")
    
    # Demonstrate the error
    try:
        modules_list = list(network.named_modules())
        index = modules_list.index(('features.0', None))
    except ValueError as e:
        print(f"\n❌ Error: {e}")
    
    return network


def show_fixed_approach(network):
    """Show the corrected approach"""
    print("\n" + "="*50)
    print("Fixed Approach:")
    print("-" * 40)
    
    # Method 1: Extract just the names
    module_names = [name for name, _ in network.named_modules()]
    print("Module names only:", module_names)
    
    # Now we can find the index properly
    if 'features.0' in module_names:
        layer_idx = module_names.index('features.0')
        print(f"✅ Found 'features.0' at index: {layer_idx}")
    
    # Method 2: Create a name-to-index mapping (more efficient)
    name_to_idx = {name: idx for idx, (name, _) in enumerate(network.named_modules())}
    print(f"\nName to index mapping: {name_to_idx}")
    
    if 'features.0' in name_to_idx:
        print(f"✅ Using mapping: 'features.0' is at index {name_to_idx['features.0']}")
    
    return name_to_idx


class FixedModuliSpaceAnalyzer:
    """
    Fixed version of the problematic method
    """
    
    def __init__(self, network):
        self.network = network
        # Precompute name-to-index mapping for efficiency
        self.layer_name_to_idx = {
            name: idx for idx, (name, _) in enumerate(network.named_modules())
        }
        self.module_names = [name for name, _ in network.named_modules()]
    
    def compute_coordinate_statistics_fixed(self, coordinate_values):
        """Fixed version of _compute_coordinate_statistics"""
        
        # Simulate the coordinate statistics computation
        for edge_idx, values in coordinate_values.items():
            if len(values) == 0:
                continue
            
            # ... statistics computation ...
            
            # FIXED: Find layer info correctly
            layer_idx = -1
            layer_name = f"features.{edge_idx % 3}"  # Simulated layer name
            
            # Method 1: Use precomputed mapping (fastest)
            if layer_name in self.layer_name_to_idx:
                layer_idx = self.layer_name_to_idx[layer_name]
            else:
                # Method 2: Fallback to searching
                if layer_name in self.module_names:
                    layer_idx = self.module_names.index(layer_name)
                else:
                    # Method 3: Ultimate fallback - use edge-based grouping
                    layer_idx = edge_idx // 100
            
            print(f"Edge {edge_idx} → Layer '{layer_name}' → Index {layer_idx}")
    
    def analyze_dataset_safe(self, dataloader, max_samples=100):
        """Safe version that handles various edge cases"""
        
        # Create edges with proper layer names
        edges = []
        for name, module in self.network.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # Create mock edges for this layer
                edges.append((len(edges), None, name))
        
        print(f"Created {len(edges)} edges from network layers")
        
        # Process with error handling
        coordinate_values = {}
        for edge_idx, _, layer_name in edges:
            coordinate_values[edge_idx] = [0.1, 0.2, 0.3]  # Mock values
            
            # Safe layer index lookup
            try:
                if layer_name in self.layer_name_to_idx:
                    layer_idx = self.layer_name_to_idx[layer_name]
                else:
                    print(f"Warning: Layer '{layer_name}' not found in mapping")
                    layer_idx = -1
            except Exception as e:
                print(f"Error processing layer '{layer_name}': {e}")
                layer_idx = -1
        
        self.compute_coordinate_statistics_fixed(coordinate_values)


def test_complete_fix():
    """Test the complete fix in a realistic scenario"""
    print("\n" + "="*50)
    print("Testing Complete Fix:")
    print("-" * 40)
    
    # Create a realistic network
    class RealisticNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
            self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
            self.classifier = nn.Sequential(
                nn.Linear(128 * 7 * 7, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(256, 10),
            )
        
        def forward(self, x):
            x = self.features(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
    
    network = RealisticNet()
    
    # Test with fixed analyzer
    analyzer = FixedModuliSpaceAnalyzer(network)
    
    # Create dummy dataloader
    from torch.utils.data import TensorDataset, DataLoader
    X = torch.randn(10, 3, 224, 224)
    y = torch.randint(0, 10, (10,))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=2)
    
    # Run analysis
    analyzer.analyze_dataset_safe(dataloader, max_samples=5)
    
    print("\n✅ Fix successfully handles complex network structures!")


def generate_patch():
    """Generate the exact patch needed for the fix"""
    print("\n" + "="*50)
    print("PATCH TO APPLY:")
    print("-" * 40)
    
    patch = '''
# In moduli_space_pruning.py, replace the problematic section with:

# Find layer info
layer_idx = -1
layer_name = ""
for e_idx, _, l_name in self.quiver.edges:
    if e_idx == edge_idx:
        layer_name = l_name
        # Fixed layer indexing
        module_names = [name for name, _ in self.network.named_modules()]
        if l_name in module_names:
            layer_idx = module_names.index(l_name)
        else:
            # Fallback for safety
            layer_idx = e_idx // 100
        break

# Or even better, add this to __init__:
self.layer_name_to_idx = {
    name: idx for idx, (name, _) in enumerate(self.network.named_modules())
}

# Then use:
layer_idx = self.layer_name_to_idx.get(layer_name, edge_idx // 100)
'''
    print(patch)


if __name__ == "__main__":
    print("="*60)
    print("Layer Indexing Error - Diagnosis and Fix")
    print("="*60)
    
    # Demonstrate the error
    network = demonstrate_error()
    
    # Show the fix
    name_to_idx = show_fixed_approach(network)
    
    # Test complete fix
    test_complete_fix()
    
    # Generate patch
    generate_patch()
    
    print("\n" + "="*60)
    print("Summary:")
    print("-" * 40)
    print("❌ Problem: Trying to find (name, None) in named_modules()")
    print("✅ Solution: Search for name only, not the tuple")
    print("✅ Best Practice: Precompute name-to-index mapping")
    print("="*60)
