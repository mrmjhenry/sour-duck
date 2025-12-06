"""
Moduli Space Coordinate Pruning Implementation
Based on "The Representation Theory of Neural Networks" by Armenta & Jodoin

This module implements pruning via moduli space coordinates, where neural networks
are represented as quiver representations and pruning decisions are made based on
coordinate usage in the geometric moduli space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

@dataclass
class ModuliCoordinate:
    """Represents a coordinate in the moduli space"""
    edge_idx: int
    layer_idx: int
    usage_rate: float
    variance: float
    information_content: float
    importance_score: float


class QuiverRepresentation:
    """
    Represents a neural network as a quiver representation.
    This is the core mathematical object from the paper.
    """
    
    def __init__(self, network: nn.Module):
        self.network = network
        self.edges = self._extract_edges()
        self.vertices = self._extract_vertices()
        
    def _extract_edges(self) -> List[Tuple[int, int, str]]:
        """Extract edges (connections) from the network"""
        edges = []
        edge_idx = 0
        
        for name, module in self.network.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # Each weight represents an edge in the quiver
                if isinstance(module, nn.Linear):
                    for i in range(module.in_features):
                        for j in range(module.out_features):
                            # Ensure all edges have exactly 3 elements: (idx, info, name)
                            edges.append((edge_idx, (i, j), name))
                            edge_idx += 1
                elif isinstance(module, nn.Conv2d):
                    # Simplified: treat each filter as an edge
                    for i in range(module.out_channels):
                        # Ensure all edges have exactly 3 elements: (idx, info, name)
                        edges.append((edge_idx, i, name))
                        edge_idx += 1
                        
        return edges
    
    def _extract_vertices(self) -> Dict[str, int]:
        """Extract vertices (neurons) from the network"""
        vertices = {}
        vertex_idx = 0
        
        for name, module in self.network.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if isinstance(module, nn.Linear):
                    vertices[f"{name}_out"] = module.out_features
                elif isinstance(module, nn.Conv2d):
                    vertices[f"{name}_out"] = module.out_channels
                    
        return vertices


class DataRepresentation:
    """
    Computes W_x^f from the paper - the thin quiver representation
    induced by a forward pass of data x through network (W,f).
    """
    
    def __init__(self, network: nn.Module, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.network = network.to(device)
        self.device = device
        self.activation_cache = {}
        self.pre_activation_cache = {}
        self.hooks = []
        self._register_hooks()
        
    def _register_hooks(self):
        """Register forward hooks to capture activations"""
        def get_activation(name):
            def hook(model, input, output):
                self.activation_cache[name] = output.detach()
                if len(input) > 0 and isinstance(input[0], torch.Tensor):
                    self.pre_activation_cache[name] = input[0].detach()
            return hook
        
        for name, module in self.network.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.ReLU)):
                self.hooks.append(module.register_forward_hook(get_activation(name)))
    
    def compute_representation(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute W_x^f following Equation (5) from the paper:
        (W_x^f)_ε = W_ε * a(W,f)_s(ε)(x) / Σ_δ∈ζ_s(ε) W_δ * a(W,f)_s(δ)(x)
        """
        # Clear caches
        self.activation_cache.clear()
        self.pre_activation_cache.clear()
        
        # Forward pass to populate caches
        with torch.no_grad():
            _ = self.network(x)
        
        representation = {}
        
        for name, module in self.network.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                weight = module.weight.data.clone()
                
                if name in self.activation_cache:
                    activation = self.activation_cache[name]
                    
                    # Handle different layer types appropriately
                    if isinstance(module, nn.Linear):
                        # For linear layers, use simplified computation
                        if name in self.pre_activation_cache:
                            pre_activation = self.pre_activation_cache[name]
                            # Avoid division by zero
                            pre_act_sum = pre_activation.abs().mean() + 1e-8
                            w_xf = weight * (activation.abs().mean() / pre_act_sum)
                        else:
                            # First layer
                            w_xf = weight * x.abs().mean()
                    
                    elif isinstance(module, nn.Conv2d):
                        # For conv layers, use mean activation values
                        # This avoids dimension mismatch issues
                        act_mean = activation.abs().mean().item()
                        
                        if name in self.pre_activation_cache:
                            pre_activation = self.pre_activation_cache[name]
                            pre_act_mean = pre_activation.abs().mean().item() + 1e-8
                            scale_factor = act_mean / pre_act_mean
                        else:
                            # First conv layer
                            scale_factor = act_mean
                        
                        # Apply scaling to weights
                        w_xf = weight * scale_factor
                    
                    representation[name] = w_xf
                    
        return representation
    
    def cleanup(self):
        """Remove hooks"""
        for hook in self.hooks:
            hook.remove()


class ModuliSpaceAnalyzer:
    """
    Analyzes the moduli space usage patterns for pruning decisions.
    Implements the principled coordinate selection strategy.
    """
    
    def __init__(self, network: nn.Module, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.network = network.to(device)
        self.device = device
        self.quiver = QuiverRepresentation(network)
        self.data_rep = DataRepresentation(network, device)
        self.coordinate_stats = {}

    @property
    def total_coordinates(self):
        """Total number of coordinates (edges in quiver)"""
        return len(self.quiver.edges)

    @property
    def moduli_dimension(self):
        """Moduli space dimension: #E - #Ṽ"""
        return len(self.quiver.edges) - len(self.quiver.vertices)

    @property
    def coordinates(self):
        """List of analyzed ModuliCoordinate objects"""
        return list(self.coordinate_stats.values())

        
    def analyze_dataset(self, dataloader, max_samples=1000):
        """Analyze coordinate usage across dataset"""
        print("Analyzing moduli space coordinates...")
        
        coordinate_values = {edge[0]: [] for edge in self.quiver.edges}
        samples_processed = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(dataloader)):
            if samples_processed >= max_samples:
                break
                
            data = data.to(self.device)
            batch_size = data.size(0)
            
            for i in range(batch_size):
                if samples_processed >= max_samples:
                    break
                    
                x = data[i].unsqueeze(0)
                representation = self.data_rep.compute_representation(x)
                
                # Extract coordinate values (OPTIMIZED)
                # Precompute all layer values once to avoid redundant tensor operations
                layer_values = {
                    name: tensor.abs().mean().item() 
                    for name, tensor in representation.items()
                }
                
                # Single pass through edges with O(1) lookup
                for edge_tuple in self.quiver.edges:
                    # Robust unpacking - handle both 2 and 3 element tuples
                    if len(edge_tuple) == 3:
                        edge_idx, edge_info, layer_name = edge_tuple
                    elif len(edge_tuple) == 2:
                        edge_idx, layer_name = edge_tuple
                        edge_info = None
                    else:
                        print(f"Warning: Unexpected edge format: {edge_tuple}")
                        continue
                    
                    if layer_name in layer_values:  # O(1) dictionary lookup
                        coordinate_values[edge_idx].append(layer_values[layer_name])
                
                samples_processed += 1
        
        # Compute statistics for each coordinate
        self._compute_coordinate_statistics(coordinate_values)
        
    def _compute_coordinate_statistics(self, coordinate_values):
        """Compute usage statistics for each coordinate"""
        
        for edge_idx, values in coordinate_values.items():
            if len(values) == 0:
                continue
            
            # Convert to numpy array safely (handles lists of varying lengths)
            values = np.array(values, dtype=np.float32)
            
            # Ensure values is 1D array
            if values.ndim > 1:
                values = values.flatten()
            
            # Usage rate: fraction of non-zero values
            usage_rate = np.mean(np.abs(values) > 1e-6)
            
            # Variance
            variance = np.var(values)
            
            # Information content (entropy-based)
            if usage_rate > 0 and usage_rate < 1:
                info_content = -usage_rate * np.log(usage_rate + 1e-8) - \
                               (1 - usage_rate) * np.log(1 - usage_rate + 1e-8)
            else:
                info_content = 0
            
            # Combined importance score
            importance = usage_rate * np.sqrt(variance) * (1 + info_content)
            
            # Find layer info
            layer_idx = -1
            layer_name = ""
            for edge_tuple in self.quiver.edges:
                # Robust unpacking
                if len(edge_tuple) == 3:
                    e_idx, _, l_name = edge_tuple
                elif len(edge_tuple) == 2:
                    e_idx, l_name = edge_tuple
                else:
                    continue
                    
                if e_idx == edge_idx:
                    layer_name = l_name
                    # Fixed layer indexing - find by name only
                    module_names = [name for name, _ in self.network.named_modules()]
                    if l_name in module_names:
                        layer_idx = module_names.index(l_name)
                    else:
                        # Fallback: use a simple counter based on edge position
                        layer_idx = e_idx // 100  # Group every 100 edges as a "layer"
                    break
            
            self.coordinate_stats[edge_idx] = ModuliCoordinate(
                edge_idx=edge_idx,
                layer_idx=layer_idx,
                usage_rate=float(usage_rate),  # Ensure it's a Python float
                variance=float(variance),      # Ensure it's a Python float
                information_content=float(info_content),  # Ensure it's a Python float
                importance_score=float(importance)  # Ensure it's a Python float
            )
    
    def get_prunable_coordinates(self, pruning_ratio=0.5, strategy='combined'):
        """
        Identify coordinates to prune based on importance scores.
        
        Args:
            pruning_ratio: Fraction of coordinates to prune
            strategy: 'usage', 'variance', 'information', or 'combined'
        """
        if not self.coordinate_stats:
            raise ValueError("Must call analyze_dataset first")
        
        # Sort coordinates by importance
        coords = list(self.coordinate_stats.values())
        
        if strategy == 'usage':
            coords.sort(key=lambda c: c.usage_rate)
        elif strategy == 'variance':
            coords.sort(key=lambda c: c.variance)
        elif strategy == 'information':
            coords.sort(key=lambda c: c.information_content)
        else:  # combined
            coords.sort(key=lambda c: c.importance_score)
        
        # Select bottom coordinates for pruning
        n_prune = int(len(coords) * pruning_ratio)
        prunable = coords[:n_prune]
        
        return prunable
    
    def visualize_coordinate_usage(self, save_path=None):
        """Visualize coordinate usage patterns"""
        if not self.coordinate_stats:
            raise ValueError("Must call analyze_dataset first")
        
        coords = list(self.coordinate_stats.values())
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Usage rates
        axes[0, 0].hist([c.usage_rate for c in coords], bins=30, edgecolor='black')
        axes[0, 0].set_xlabel('Usage Rate')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Distribution of Coordinate Usage Rates')
        
        # Variances
        axes[0, 1].hist([np.log(c.variance + 1e-8) for c in coords], bins=30, edgecolor='black')
        axes[0, 1].set_xlabel('Log Variance')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Distribution of Coordinate Variances')
        
        # Information content
        axes[1, 0].hist([c.information_content for c in coords], bins=30, edgecolor='black')
        axes[1, 0].set_xlabel('Information Content')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Distribution of Information Content')
        
        # Importance scores
        axes[1, 1].hist([c.importance_score for c in coords], bins=30, edgecolor='black')
        axes[1, 1].set_xlabel('Importance Score')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Distribution of Combined Importance Scores')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        return fig


class ModuliSpacePruner:
    """
    Implements the actual pruning based on moduli space analysis.
    Maps coordinates back to network weights for pruning.
    """
    
    def __init__(self, network: nn.Module, analyzer: ModuliSpaceAnalyzer):
        self.network = network
        self.analyzer = analyzer
        self.original_state = network.state_dict()
        
    def prune(self, pruning_ratio=0.5, strategy='combined'):
        """
        Prune network based on moduli space coordinate analysis.

        Returns:
            pruned_network: The pruned network
            pruning_mask: Binary mask indicating pruned weights
        """
        # Create pruning masks for each layer
        pruning_masks = {}

        for name, module in self.network.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                mask = torch.ones_like(module.weight.data)
                pruning_masks[name] = mask

        # Calculate total number of prunable weights
        total_weights = sum(mask.numel() for mask in pruning_masks.values())
        target_pruned = int(total_weights * pruning_ratio)

        # Get coordinates sorted by importance (lowest first)
        prunable_coords = self.analyzer.get_prunable_coordinates(pruning_ratio, strategy)

        # Calculate how many weights to prune per coordinate to achieve target
        if len(prunable_coords) > 0:
            weights_per_coord = max(1, target_pruned // len(prunable_coords))
        else:
            weights_per_coord = 1

        # Map coordinates to weights and create masks
        total_pruned = 0
        for coord in prunable_coords:
            if total_pruned >= target_pruned:
                break

            # Find corresponding layer
            for edge_tuple in self.analyzer.quiver.edges:
                # Robust unpacking
                if len(edge_tuple) == 3:
                    edge_idx, edge_info, layer_name = edge_tuple
                elif len(edge_tuple) == 2:
                    edge_idx, layer_name = edge_tuple
                    edge_info = None
                else:
                    continue

                if edge_idx == coord.edge_idx:
                    if layer_name in pruning_masks:
                        mask = pruning_masks[layer_name]
                        if mask.numel() > 0:
                            # Prune multiple weights per coordinate to achieve target ratio
                            flat_mask = mask.view(-1)
                            remaining = min(weights_per_coord, target_pruned - total_pruned)

                            # Find indices to prune (starting from edge_idx position)
                            for i in range(remaining):
                                flat_idx = (edge_idx + i) % mask.numel()
                                if flat_mask[flat_idx] == 1:  # Only prune if not already pruned
                                    flat_mask[flat_idx] = 0
                                    total_pruned += 1
                    break

        # Apply masks to network
        self._apply_masks(pruning_masks)

        return self.network, pruning_masks
    
    def _apply_masks(self, masks):
        """Apply pruning masks to network weights"""
        for name, module in self.network.named_modules():
            if name in masks:
                module.weight.data *= masks[name]
    
    def restore_original(self):
        """Restore network to original state"""
        self.network.load_state_dict(self.original_state)
