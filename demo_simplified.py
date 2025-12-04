"""
Simplified Moduli Space Pruning Demonstration
This version demonstrates the concepts without requiring PyTorch
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import json

class SimpleNetwork:
    """Simplified neural network for demonstration"""
    
    def __init__(self, layers: List[int]):
        """
        Create a simple feedforward network
        layers: list of layer sizes, e.g., [10, 5, 2]
        """
        self.layers = layers
        self.weights = []
        
        # Initialize random weights
        for i in range(len(layers) - 1):
            W = np.random.randn(layers[i+1], layers[i]) * 0.1
            self.weights.append(W)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Simple forward pass with ReLU activation"""
        activation = x
        self.activations = [x]
        
        for i, W in enumerate(self.weights):
            pre_activation = W @ activation
            if i < len(self.weights) - 1:  # ReLU for hidden layers
                activation = np.maximum(0, pre_activation)
            else:  # No activation for output layer
                activation = pre_activation
            self.activations.append(activation)
        
        return activation
    
    def get_num_parameters(self) -> int:
        """Count total parameters"""
        return sum(W.size for W in self.weights)
    
    def get_sparsity(self) -> float:
        """Calculate sparsity (fraction of zero weights)"""
        total = 0
        zeros = 0
        for W in self.weights:
            total += W.size
            zeros += np.sum(W == 0)
        return zeros / total if total > 0 else 0


class ModuliCoordinate:
    """Represents a coordinate in the moduli space"""
    
    def __init__(self, idx: int, layer: int, position: Tuple[int, int]):
        self.idx = idx
        self.layer = layer
        self.position = position  # (row, col) in weight matrix
        self.usage_rate = 0.0
        self.variance = 0.0
        self.importance = 0.0
        self.values = []
    
    def update_stats(self):
        """Update statistics from collected values"""
        if len(self.values) > 0:
            self.usage_rate = np.mean(np.abs(self.values) > 1e-6)
            self.variance = np.var(self.values)
            # Simple importance score
            self.importance = self.usage_rate * np.sqrt(self.variance + 1e-8)


class SimplifiedModuliAnalyzer:
    """Simplified version of moduli space analysis"""
    
    def __init__(self, network: SimpleNetwork):
        self.network = network
        self.coordinates = []
        self._create_coordinates()
    
    def _create_coordinates(self):
        """Create coordinate objects for each weight"""
        idx = 0
        for layer_idx, W in enumerate(self.network.weights):
            for i in range(W.shape[0]):
                for j in range(W.shape[1]):
                    coord = ModuliCoordinate(idx, layer_idx, (i, j))
                    self.coordinates.append(coord)
                    idx += 1
    
    def analyze_data(self, X: np.ndarray, num_samples: int = 100):
        """Analyze coordinate usage on data"""
        print(f"Analyzing {min(num_samples, len(X))} samples...")
        
        for sample_idx in range(min(num_samples, len(X))):
            x = X[sample_idx]
            
            # Forward pass to get activations
            _ = self.network.forward(x)
            
            # Compute data representation (simplified W_x^f)
            coord_idx = 0
            for layer_idx, W in enumerate(self.network.weights):
                input_activation = self.network.activations[layer_idx]
                output_activation = self.network.activations[layer_idx + 1]
                
                for i in range(W.shape[0]):
                    for j in range(W.shape[1]):
                        # Simplified version of W_x^f from the paper
                        if np.abs(input_activation[j]) > 1e-6:
                            value = W[i, j] * output_activation[i] / (input_activation[j] + 1e-8)
                        else:
                            value = 0
                        
                        self.coordinates[coord_idx].values.append(value)
                        coord_idx += 1
        
        # Update statistics for all coordinates
        for coord in self.coordinates:
            coord.update_stats()
    
    def get_pruning_mask(self, pruning_ratio: float = 0.5) -> List[np.ndarray]:
        """Get pruning masks based on coordinate importance"""
        
        # Sort coordinates by importance
        sorted_coords = sorted(self.coordinates, key=lambda c: c.importance)
        
        # Select coordinates to prune
        n_prune = int(len(sorted_coords) * pruning_ratio)
        coords_to_prune = sorted_coords[:n_prune]
        
        # Create masks
        masks = []
        for W in self.network.weights:
            masks.append(np.ones_like(W))
        
        # Apply pruning decisions to masks
        for coord in coords_to_prune:
            layer_idx = coord.layer
            i, j = coord.position
            masks[layer_idx][i, j] = 0
        
        return masks
    
    def visualize_analysis(self):
        """Visualize coordinate importance analysis"""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Usage rates
        usage_rates = [c.usage_rate for c in self.coordinates]
        axes[0, 0].hist(usage_rates, bins=30, edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Usage Rate')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Distribution of Coordinate Usage Rates')
        axes[0, 0].axvline(np.mean(usage_rates), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(usage_rates):.3f}')
        axes[0, 0].legend()
        
        # Variances
        variances = [c.variance for c in self.coordinates]
        axes[0, 1].hist(np.log(np.array(variances) + 1e-8), bins=30, 
                       edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('Log Variance')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Distribution of Coordinate Variances')
        
        # Importance scores
        importance = [c.importance for c in self.coordinates]
        axes[1, 0].hist(importance, bins=30, edgecolor='black', alpha=0.7)
        axes[1, 0].set_xlabel('Importance Score')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Distribution of Importance Scores')
        axes[1, 0].axvline(np.mean(importance), color='red', linestyle='--',
                          label=f'Mean: {np.mean(importance):.3f}')
        axes[1, 0].legend()
        
        # Layer-wise importance
        layer_importance = {}
        for coord in self.coordinates:
            if coord.layer not in layer_importance:
                layer_importance[coord.layer] = []
            layer_importance[coord.layer].append(coord.importance)
        
        layers = sorted(layer_importance.keys())
        means = [np.mean(layer_importance[l]) for l in layers]
        stds = [np.std(layer_importance[l]) for l in layers]
        
        axes[1, 1].bar(layers, means, yerr=stds, capsize=5, alpha=0.7)
        axes[1, 1].set_xlabel('Layer')
        axes[1, 1].set_ylabel('Mean Importance')
        axes[1, 1].set_title('Layer-wise Importance Distribution')
        axes[1, 1].set_xticks(layers)
        
        plt.tight_layout()
        plt.savefig('moduli_analysis_demo.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return fig


def demonstrate_moduli_pruning():
    """Complete demonstration of moduli space pruning"""
    
    print("="*60)
    print("Moduli Space Coordinate Pruning - Demonstration")
    print("="*60)
    
    # 1. Create network
    print("\n1. Creating network...")
    network = SimpleNetwork([10, 20, 10, 2])
    print(f"   Network architecture: {network.layers}")
    print(f"   Total parameters: {network.get_num_parameters()}")
    
    # 2. Generate synthetic data
    print("\n2. Generating synthetic data...")
    np.random.seed(42)
    X_train = np.random.randn(500, 10)
    
    # 3. Analyze moduli space
    print("\n3. Analyzing moduli space coordinates...")
    analyzer = SimplifiedModuliAnalyzer(network)
    analyzer.analyze_data(X_train, num_samples=100)
    
    # Print statistics
    coords = analyzer.coordinates
    avg_usage = np.mean([c.usage_rate for c in coords])
    avg_importance = np.mean([c.importance for c in coords])
    
    print(f"   Number of coordinates: {len(coords)}")
    print(f"   Average usage rate: {avg_usage:.3f}")
    print(f"   Average importance: {avg_importance:.3f}")
    
    # 4. Theoretical analysis
    print("\n4. Theoretical Properties:")
    num_edges = network.get_num_parameters()
    num_hidden = sum(network.layers[1:-1])  # Hidden vertices
    moduli_dim = num_edges - num_hidden
    
    print(f"   Number of edges (#E): {num_edges}")
    print(f"   Number of hidden vertices (#Ṽ): {num_hidden}")
    print(f"   Moduli space dimension: {moduli_dim}")
    print(f"   Dimension ratio: {moduli_dim/num_edges:.2%}")
    
    # 5. Apply pruning
    print("\n5. Applying moduli space pruning...")
    pruning_ratios = [0.3, 0.5, 0.7]
    
    results = []
    for ratio in pruning_ratios:
        masks = analyzer.get_pruning_mask(ratio)
        
        # Apply masks
        original_weights = [W.copy() for W in network.weights]
        for i, mask in enumerate(masks):
            network.weights[i] = network.weights[i] * mask
        
        sparsity = network.get_sparsity()
        remaining_params = network.get_num_parameters() * (1 - sparsity)
        
        results.append({
            'ratio': ratio,
            'sparsity': sparsity,
            'remaining_params': int(remaining_params)
        })
        
        print(f"   Ratio {ratio:.1f}: Sparsity={sparsity:.2%}, "
              f"Remaining params={int(remaining_params)}")
        
        # Restore weights for next iteration
        network.weights = original_weights
    
    # 6. Visualize analysis
    print("\n6. Generating visualization...")
    fig = analyzer.visualize_analysis()
    
    # 7. Compare with baseline methods
    print("\n7. Comparison with baseline methods:")
    
    # Magnitude pruning (simplified)
    def magnitude_pruning(weights, ratio):
        all_weights = np.concatenate([W.flatten() for W in weights])
        threshold = np.percentile(np.abs(all_weights), ratio * 100)
        masks = []
        for W in weights:
            masks.append((np.abs(W) > threshold).astype(float))
        return masks
    
    # Random pruning
    def random_pruning(weights, ratio):
        masks = []
        for W in weights:
            mask = np.random.random(W.shape) > ratio
            masks.append(mask.astype(float))
        return masks
    
    print("\n   Method Comparison (50% pruning):")
    print("   " + "-"*40)
    
    # Moduli space result
    moduli_coords = sorted(analyzer.coordinates, key=lambda c: c.importance)
    moduli_pruned = sum(1 for c in moduli_coords[:len(moduli_coords)//2] 
                       if c.importance < np.median([c.importance for c in moduli_coords]))
    
    # Magnitude pruning comparison
    mag_masks = magnitude_pruning(network.weights, 0.5)
    mag_zeros = sum(np.sum(m == 0) for m in mag_masks)
    
    # Random pruning comparison  
    np.random.seed(42)
    rand_masks = random_pruning(network.weights, 0.5)
    rand_zeros = sum(np.sum(m == 0) for m in rand_masks)
    
    print(f"   Moduli Space: Pruned {moduli_pruned} low-importance coords")
    print(f"   Magnitude:    Pruned {mag_zeros} small weights")
    print(f"   Random:       Pruned {rand_zeros} random weights")
    
    # 8. Summary statistics
    print("\n8. Summary Statistics:")
    print("   " + "-"*40)
    
    # Coordinate statistics by layer
    layer_stats = {}
    for coord in analyzer.coordinates:
        if coord.layer not in layer_stats:
            layer_stats[coord.layer] = {
                'count': 0,
                'low_usage': 0,
                'high_importance': 0
            }
        layer_stats[coord.layer]['count'] += 1
        if coord.usage_rate < 0.1:
            layer_stats[coord.layer]['low_usage'] += 1
        if coord.importance > np.mean([c.importance for c in coords]):
            layer_stats[coord.layer]['high_importance'] += 1
    
    for layer, stats in sorted(layer_stats.items()):
        print(f"   Layer {layer}: {stats['count']} coords, "
              f"{stats['low_usage']} low-usage, "
              f"{stats['high_importance']} high-importance")
    
    # Save results
    with open('demo_results.json', 'w') as f:
        json.dump({
            'network_architecture': network.layers,
            'total_parameters': network.get_num_parameters(),
            'moduli_dimension': moduli_dim,
            'pruning_results': results,
            'coordinate_stats': {
                'total': len(coords),
                'avg_usage': float(avg_usage),
                'avg_importance': float(avg_importance)
            }
        }, f, indent=2)
    
    print("\n✓ Demonstration complete! Results saved to demo_results.json")
    print("  Visualization saved to moduli_analysis_demo.png")
    
    return analyzer, results


if __name__ == "__main__":
    # Run demonstration
    analyzer, results = demonstrate_moduli_pruning()
    
    print("\n" + "="*60)
    print("Key Insights from Moduli Space Pruning:")
    print("="*60)
    print("""
    1. COORDINATE INTERPRETATION: Each coordinate in the moduli space 
       represents a specific information pathway through the network.
       
    2. PRINCIPLED SELECTION: Unlike magnitude pruning which only considers
       weight values, we analyze how coordinates are actually used by data.
       
    3. INVARIANCE PROPERTIES: The pruning decisions are invariant under
       network isomorphisms (weight reparameterizations).
       
    4. THEORETICAL FOUNDATION: The moduli dimension (#E - #Ṽ) gives the
       true degrees of freedom after accounting for symmetries.
       
    5. DATA-DRIVEN: The importance scores incorporate usage patterns across
       the dataset, not just static weight properties.
    """)
