"""
Fix for Inhomogeneous Array Error in Moduli Space Pruning
Resolves: "setting an array element with a sequence" error
"""

import numpy as np
from collections import defaultdict
from dataclasses import dataclass

@dataclass
class ModuliCoordinate:
    """Represents a coordinate in the moduli space"""
    edge_idx: int
    layer_idx: int
    usage_rate: float
    variance: float
    information_content: float
    importance_score: float


def demonstrate_error():
    """Demonstrate the original inhomogeneous array error"""
    print("Demonstrating the inhomogeneous array error:")
    print("-" * 40)
    
    # Create coordinate_values with lists of different lengths
    # This happens when different edges are processed different numbers of times
    coordinate_values = {
        0: [0.1, 0.2, 0.3],        # 3 values
        1: [0.4, 0.5],             # 2 values (different length!)
        2: [0.6, 0.7, 0.8, 0.9],   # 4 values
        3: []                      # Empty list
    }
    
    print("Coordinate values (note different lengths):")
    for edge_idx, values in coordinate_values.items():
        print(f"  Edge {edge_idx}: {values} (length: {len(values)})")
    
    print("\nTrying to convert to numpy array directly...")
    
    # This causes the error
    try:
        for edge_idx, values in coordinate_values.items():
            if len(values) == 0:
                continue
            # This line causes the error with inhomogeneous data
            values_array = np.array(values)  # Works fine for single list
            print(f"✓ Edge {edge_idx}: converted to array shape {values_array.shape}")
    except Exception as e:
        print(f"✓ Single lists work fine")
    
    # But this fails when trying to create array from all values
    try:
        all_values = list(coordinate_values.values())
        combined = np.array(all_values)  # This fails!
    except ValueError as e:
        print(f"\n❌ Error when combining: {e}")
    
    return coordinate_values


def show_fixed_approach(coordinate_values):
    """Show the corrected approach"""
    print("\n" + "="*50)
    print("Fixed Approach:")
    print("-" * 40)
    
    coordinate_stats = {}
    
    for edge_idx, values in coordinate_values.items():
        if len(values) == 0:
            print(f"Edge {edge_idx}: Skipped (empty)")
            continue
        
        # FIX 1: Explicitly specify dtype to avoid inference issues
        values = np.array(values, dtype=np.float32)
        
        # FIX 2: Ensure values is 1D (flatten if needed)
        if values.ndim > 1:
            values = values.flatten()
        
        # Now compute statistics safely
        usage_rate = np.mean(np.abs(values) > 1e-6)
        variance = np.var(values)
        
        # FIX 3: Convert numpy values to Python floats for storage
        # This prevents serialization issues later
        usage_rate = float(usage_rate)
        variance = float(variance)
        
        # Information content
        if 0 < usage_rate < 1:
            info_content = -usage_rate * np.log(usage_rate + 1e-8) - \
                          (1 - usage_rate) * np.log(1 - usage_rate + 1e-8)
        else:
            info_content = 0.0
        
        importance = usage_rate * np.sqrt(variance) * (1 + info_content)
        
        # Store results (all as Python floats)
        coordinate_stats[edge_idx] = ModuliCoordinate(
            edge_idx=edge_idx,
            layer_idx=edge_idx // 100,  # Simplified
            usage_rate=float(usage_rate),
            variance=float(variance),
            information_content=float(info_content),
            importance_score=float(importance)
        )
        
        print(f"Edge {edge_idx}: ✓ Processed successfully")
        print(f"  Usage: {usage_rate:.3f}, Variance: {variance:.3f}, Importance: {importance:.3f}")
    
    return coordinate_stats


def test_edge_cases():
    """Test various edge cases that could cause the error"""
    print("\n" + "="*50)
    print("Testing Edge Cases:")
    print("-" * 40)
    
    edge_cases = {
        "empty_list": [],
        "single_value": [0.5],
        "all_zeros": [0.0, 0.0, 0.0],
        "mixed_types": [1, 2.0, np.float32(3.0)],  # Different numeric types
        "nested_list": [[0.1], [0.2]],  # Accidentally nested
        "very_small": [1e-10, 1e-11, 1e-12],
        "very_large": [1e10, 1e11, 1e12],
    }
    
    for case_name, values in edge_cases.items():
        print(f"\nTesting {case_name}: {values}")
        try:
            if len(values) == 0:
                print("  → Skipped (empty)")
                continue
            
            # Apply all fixes
            arr = np.array(values, dtype=np.float32)
            if arr.ndim > 1:
                arr = arr.flatten()
            
            usage_rate = float(np.mean(np.abs(arr) > 1e-6))
            variance = float(np.var(arr))
            
            print(f"  ✓ Success: usage={usage_rate:.3f}, var={variance:.3e}")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")


def create_robust_implementation():
    """Create a fully robust implementation"""
    print("\n" + "="*50)
    print("Robust Implementation:")
    print("-" * 40)
    
    class RobustCoordinateProcessor:
        """Robust processor that handles all edge cases"""
        
        @staticmethod
        def process_coordinate_values(coordinate_values):
            """Process coordinate values with comprehensive error handling"""
            stats = {}
            
            for edge_idx, values in coordinate_values.items():
                try:
                    # Skip empty
                    if not values or len(values) == 0:
                        continue
                    
                    # Convert to numpy array with error handling
                    if isinstance(values, (list, tuple)):
                        # Handle nested lists/arrays
                        values_flat = []
                        for v in values:
                            if isinstance(v, (list, tuple, np.ndarray)):
                                values_flat.extend(np.array(v).flatten())
                            else:
                                values_flat.append(v)
                        values = np.array(values_flat, dtype=np.float32)
                    else:
                        values = np.array(values, dtype=np.float32)
                    
                    # Ensure 1D
                    values = values.flatten()
                    
                    # Remove NaN/Inf values
                    values = values[np.isfinite(values)]
                    
                    if len(values) == 0:
                        continue
                    
                    # Compute statistics
                    usage_rate = float(np.mean(np.abs(values) > 1e-6))
                    variance = float(np.var(values))
                    
                    # Safe information content
                    if 0 < usage_rate < 1:
                        info_content = float(
                            -usage_rate * np.log(usage_rate + 1e-8) - 
                            (1 - usage_rate) * np.log(1 - usage_rate + 1e-8)
                        )
                    else:
                        info_content = 0.0
                    
                    importance = float(usage_rate * np.sqrt(variance + 1e-8) * (1 + info_content))
                    
                    stats[edge_idx] = {
                        'usage_rate': usage_rate,
                        'variance': variance,
                        'info_content': info_content,
                        'importance': importance,
                        'num_samples': len(values)
                    }
                    
                except Exception as e:
                    print(f"Warning: Failed to process edge {edge_idx}: {e}")
                    # Provide default values
                    stats[edge_idx] = {
                        'usage_rate': 0.0,
                        'variance': 0.0,
                        'info_content': 0.0,
                        'importance': 0.0,
                        'num_samples': 0
                    }
            
            return stats
    
    # Test the robust implementation
    test_values = {
        0: [0.1, 0.2, 0.3],
        1: [[0.4], [0.5]],  # Nested
        2: [1, 2.0, np.float64(3)],  # Mixed types
        3: [],  # Empty
        4: [np.nan, 1.0, np.inf, 2.0],  # Contains NaN/Inf
    }
    
    processor = RobustCoordinateProcessor()
    results = processor.process_coordinate_values(test_values)
    
    print("\nRobust processing results:")
    for edge_idx, stats in results.items():
        print(f"Edge {edge_idx}: {stats}")
    
    return processor


def generate_patch():
    """Generate the complete patch for the fix"""
    print("\n" + "="*50)
    print("COMPLETE PATCH:")
    print("-" * 40)
    
    patch = '''
# In moduli_space_pruning.py, update _compute_coordinate_statistics:

def _compute_coordinate_statistics(self, coordinate_values):
    """Compute usage statistics for each coordinate"""
    
    for edge_idx, values in coordinate_values.items():
        if len(values) == 0:
            continue
        
        # FIX: Safely convert to numpy array
        values = np.array(values, dtype=np.float32)
        
        # FIX: Ensure 1D array
        if values.ndim > 1:
            values = values.flatten()
        
        # Compute statistics...
        usage_rate = np.mean(np.abs(values) > 1e-6)
        variance = np.var(values)
        
        # ... rest of computation ...
        
        # FIX: Convert to Python floats for storage
        self.coordinate_stats[edge_idx] = ModuliCoordinate(
            edge_idx=edge_idx,
            layer_idx=layer_idx,
            usage_rate=float(usage_rate),  # Convert to Python float
            variance=float(variance),
            information_content=float(info_content),
            importance_score=float(importance)
        )
'''
    print(patch)


if __name__ == "__main__":
    print("="*60)
    print("Inhomogeneous Array Error - Diagnosis and Fix")
    print("="*60)
    
    # Demonstrate the error
    coordinate_values = demonstrate_error()
    
    # Show the fix
    stats = show_fixed_approach(coordinate_values)
    
    # Test edge cases
    test_edge_cases()
    
    # Create robust implementation
    processor = create_robust_implementation()
    
    # Generate patch
    generate_patch()
    
    print("\n" + "="*60)
    print("Summary of Fixes:")
    print("-" * 40)
    print("✅ 1. Specify dtype=np.float32 explicitly")
    print("✅ 2. Flatten arrays if multidimensional")
    print("✅ 3. Convert numpy scalars to Python floats")
    print("✅ 4. Handle empty lists gracefully")
    print("✅ 5. Add error handling for edge cases")
    print("="*60)
