"""
Diagnostic and Fix for "not enough values to unpack" Error
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Any

def diagnose_unpacking_error():
    """Diagnose the unpacking error"""
    print("="*60)
    print("Diagnosing Unpacking Error")
    print("="*60)
    
    # Simulate what might be happening
    print("\n1. The Problem:")
    print("-" * 40)
    
    # Some edges might be created with 2 elements
    edges_mixed = [
        (0, (0, 1), 'layer1'),  # 3 elements - correct
        (1, 'layer2'),           # 2 elements - missing edge_info!
        (2, (1, 0), 'layer3'),   # 3 elements - correct
    ]
    
    print("Edges with mixed formats:")
    for edge in edges_mixed:
        print(f"  {edge} - {len(edge)} elements")
    
    print("\nTrying to unpack with 3 variables...")
    try:
        for edge_idx, edge_info, layer_name in edges_mixed:
            print(f"  Processing edge {edge_idx}")
    except ValueError as e:
        print(f"❌ Error: {e}")
        print("   This happens on edge:", edges_mixed[1])


def show_robust_unpacking():
    """Show robust unpacking methods"""
    print("\n2. Robust Solutions:")
    print("-" * 40)
    
    edges_mixed = [
        (0, (0, 1), 'layer1'),
        (1, 'layer2'),
        (2, (1, 0), 'layer3'),
        (3,),  # Even worse - only 1 element!
    ]
    
    print("Method 1: Check length before unpacking")
    for edge in edges_mixed:
        if len(edge) == 3:
            edge_idx, edge_info, layer_name = edge
            print(f"  Edge {edge_idx}: info={edge_info}, layer={layer_name}")
        elif len(edge) == 2:
            edge_idx, layer_name = edge
            edge_info = None
            print(f"  Edge {edge_idx}: info=None, layer={layer_name}")
        else:
            print(f"  Skipping malformed edge: {edge}")
    
    print("\nMethod 2: Use * operator for flexible unpacking")
    for edge in edges_mixed:
        if len(edge) >= 2:
            edge_idx, *rest = edge
            layer_name = rest[-1] if rest else 'unknown'
            edge_info = rest[0] if len(rest) > 1 else None
            print(f"  Edge {edge_idx}: info={edge_info}, layer={layer_name}")
    
    print("\nMethod 3: Pad edges to ensure consistent format")
    fixed_edges = []
    for edge in edges_mixed:
        if len(edge) == 3:
            fixed_edges.append(edge)
        elif len(edge) == 2:
            # Add None for missing edge_info
            fixed_edges.append((edge[0], None, edge[1]))
        elif len(edge) == 1:
            # Add defaults for missing values
            fixed_edges.append((edge[0], None, 'unknown'))
        else:
            # Skip or handle empty edges
            continue
    
    print("Fixed edges:")
    for edge_idx, edge_info, layer_name in fixed_edges:
        print(f"  Edge {edge_idx}: info={edge_info}, layer={layer_name}")


class RobustQuiverRepresentation:
    """Fixed version that ensures all edges have 3 elements"""
    
    def __init__(self, network: nn.Module):
        self.network = network
        self.edges = self._extract_edges()
        self.vertices = self._extract_vertices()
    
    def _extract_edges(self) -> List[Tuple[int, Any, str]]:
        """Extract edges ensuring all have exactly 3 elements"""
        edges = []
        edge_idx = 0
        
        for name, module in self.network.named_modules():
            if isinstance(module, nn.Linear):
                # For Linear layers, include connection info
                for i in range(min(module.in_features, 10)):  # Limit for demo
                    for j in range(min(module.out_features, 10)):
                        # Always include 3 elements: (idx, info, name)
                        edges.append((edge_idx, (i, j), name))
                        edge_idx += 1
                        
            elif isinstance(module, nn.Conv2d):
                # For Conv2d, include channel info
                for i in range(min(module.out_channels, 10)):  # Limit for demo
                    # Always include 3 elements: (idx, info, name)
                    edges.append((edge_idx, i, name))
                    edge_idx += 1
        
        # Validate all edges have 3 elements
        for edge in edges:
            assert len(edge) == 3, f"Edge {edge} doesn't have 3 elements!"
        
        return edges
    
    def _extract_vertices(self) -> dict:
        """Extract vertices from network"""
        vertices = {}
        for name, module in self.network.named_modules():
            if isinstance(module, nn.Linear):
                vertices[f"{name}_out"] = module.out_features
            elif isinstance(module, nn.Conv2d):
                vertices[f"{name}_out"] = module.out_channels
        return vertices


def test_fixed_implementation():
    """Test the fixed implementation"""
    print("\n3. Testing Fixed Implementation:")
    print("-" * 40)
    
    # Create test network
    class TestNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3)
            self.fc1 = nn.Linear(16 * 30 * 30, 10)
        
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = x.view(x.size(0), -1)
            return self.fc1(x)
    
    network = TestNet()
    quiver = RobustQuiverRepresentation(network)
    
    print(f"Created {len(quiver.edges)} edges")
    print("First 5 edges:")
    for edge in quiver.edges[:5]:
        print(f"  {edge}")
    
    # Test unpacking works
    print("\nTesting unpacking...")
    try:
        for edge_idx, edge_info, layer_name in quiver.edges[:5]:
            print(f"  ✓ Edge {edge_idx} unpacked successfully")
        print("✅ All edges unpack correctly!")
    except ValueError as e:
        print(f"❌ Unpacking failed: {e}")
        return False
    
    return True


def create_safe_unpacking_wrapper():
    """Create a wrapper function for safe unpacking"""
    print("\n4. Safe Unpacking Wrapper:")
    print("-" * 40)
    
    def safe_unpack_edge(edge_tuple):
        """
        Safely unpack edge tuple, handling various formats
        
        Returns: (edge_idx, edge_info, layer_name)
        """
        if not edge_tuple:
            return None, None, None
        
        if len(edge_tuple) == 3:
            return edge_tuple
        elif len(edge_tuple) == 2:
            # Assume format is (edge_idx, layer_name)
            return edge_tuple[0], None, edge_tuple[1]
        elif len(edge_tuple) == 1:
            # Only edge_idx provided
            return edge_tuple[0], None, 'unknown'
        else:
            # Too many elements - take first 3
            return edge_tuple[0], edge_tuple[1], edge_tuple[2]
    
    # Test the wrapper
    test_cases = [
        (0, (0, 1), 'layer1'),  # Normal
        (1, 'layer2'),           # Missing info
        (2,),                    # Only index
        (),                      # Empty
        (3, 4, 5, 6),           # Too many
    ]
    
    print("Testing safe unpacking:")
    for edge in test_cases:
        idx, info, name = safe_unpack_edge(edge)
        print(f"  {edge} → idx={idx}, info={info}, name={name}")
    
    return safe_unpack_edge


def generate_complete_fix():
    """Generate the complete fix"""
    print("\n" + "="*60)
    print("COMPLETE FIX:")
    print("-" * 40)
    
    fix_code = '''
# Option 1: Fix at the source (in _extract_edges):
def _extract_edges(self):
    edges = []
    edge_idx = 0
    for name, module in self.network.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # ALWAYS create 3-element tuples
            if isinstance(module, nn.Linear):
                for i in range(module.in_features):
                    for j in range(module.out_features):
                        edges.append((edge_idx, (i, j), name))  # 3 elements
                        edge_idx += 1
            elif isinstance(module, nn.Conv2d):
                for i in range(module.out_channels):
                    edges.append((edge_idx, i, name))  # 3 elements
                    edge_idx += 1
    return edges

# Option 2: Robust unpacking everywhere:
for edge_tuple in self.quiver.edges:
    # Safe unpacking
    if len(edge_tuple) == 3:
        edge_idx, edge_info, layer_name = edge_tuple
    elif len(edge_tuple) == 2:
        edge_idx, layer_name = edge_tuple
        edge_info = None
    else:
        continue  # Skip malformed edges
    
    # Process edge...

# Option 3: Add validation in __init__:
def __init__(self, network):
    self.network = network
    self.edges = self._extract_edges()
    # Validate edges
    for i, edge in enumerate(self.edges):
        if len(edge) != 3:
            print(f"Warning: Edge {i} has {len(edge)} elements: {edge}")
            # Fix it
            if len(edge) == 2:
                self.edges[i] = (edge[0], None, edge[1])
'''
    print(fix_code)


if __name__ == "__main__":
    # Run diagnostics
    diagnose_unpacking_error()
    show_robust_unpacking()
    
    # Test fixes
    success = test_fixed_implementation()
    safe_unpack = create_safe_unpacking_wrapper()
    
    # Generate fix
    generate_complete_fix()
    
    print("\n" + "="*60)
    print("Summary:")
    print("-" * 40)
    print("❌ Problem: Some edges have 2 elements instead of 3")
    print("✅ Solution 1: Ensure edges are created with 3 elements")
    print("✅ Solution 2: Use robust unpacking that handles both formats")
    print("✅ Solution 3: Add validation and auto-fix malformed edges")
    print("="*60)
