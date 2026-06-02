"""
Magnitude Pruning
=================

Prune weights with smallest magnitude (L1 or L2 norm).
Based on Han et al., "Learning both Weights and Connections for Efficient Neural Networks", NIPS 2015.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple
from .base import BasePruner


class MagnitudePruner(BasePruner):
    """
    Prune weights based on magnitude.

    Removes weights with smallest L1 or L2 norm globally across the network.
    """

    def __init__(self, model: nn.Module, norm: str = 'l1'):
        """
        Initialize magnitude pruner.

        Args:
            model: Neural network to prune
            norm: Norm to use ('l1' or 'l2')
        """
        super().__init__(model)
        self.norm = norm.lower()

        if self.norm not in ['l1', 'l2']:
            raise ValueError(f"Norm must be 'l1' or 'l2', got {self.norm}")

    def prune(
        self,
        pruning_ratio: float,
        **kwargs
    ) -> Tuple[nn.Module, Dict[str, torch.Tensor]]:
        """
        Prune weights based on magnitude.

        Args:
            pruning_ratio: Fraction of weights to prune

        Returns:
            (pruned_model, masks)
        """
        # Collect all weights and their magnitudes
        all_weights = []
        weight_locations = []  # (layer_name, index)

        prunable_layers = self.get_prunable_layers(self.model)

        for name, module in prunable_layers.items():
            weights_flat = module.weight.data.view(-1)

            # Compute magnitude
            if self.norm == 'l1':
                magnitudes = weights_flat.abs()
            else:  # l2
                magnitudes = weights_flat ** 2

            for idx, mag in enumerate(magnitudes):
                all_weights.append((mag.item(), name, idx))

        # Sort by magnitude (smallest first)
        all_weights.sort(key=lambda x: x[0])

        # Determine how many weights to prune
        total_weights = len(all_weights)
        num_prune = int(total_weights * pruning_ratio)

        # Create masks
        masks = {name: torch.ones_like(module.weight.data)
                for name, module in prunable_layers.items()}

        # Prune smallest weights
        for i in range(num_prune):
            _, layer_name, weight_idx = all_weights[i]
            flat_mask = masks[layer_name].view(-1)
            flat_mask[weight_idx] = 0

        # Apply masks
        self.apply_masks(masks)

        return self.model, masks


class LayerWiseMagnitudePruner(BasePruner):
    """
    Prune weights based on magnitude, but independently per layer.

    This ensures each layer is pruned by the same ratio.
    """

    def __init__(self, model: nn.Module, norm: str = 'l1'):
        """
        Initialize layer-wise magnitude pruner.

        Args:
            model: Neural network to prune
            norm: Norm to use ('l1' or 'l2')
        """
        super().__init__(model)
        self.norm = norm.lower()

    def prune(
        self,
        pruning_ratio: float,
        **kwargs
    ) -> Tuple[nn.Module, Dict[str, torch.Tensor]]:
        """
        Prune weights layer-wise based on magnitude.

        Args:
            pruning_ratio: Fraction of weights to prune per layer

        Returns:
            (pruned_model, masks)
        """
        prunable_layers = self.get_prunable_layers(self.model)
        masks = {}

        for name, module in prunable_layers.items():
            # Get weight magnitudes
            weights_flat = module.weight.data.abs().view(-1)

            if self.norm == 'l2':
                weights_flat = weights_flat ** 2

            # Find threshold for this layer
            num_weights = weights_flat.numel()
            num_prune = int(num_weights * pruning_ratio)

            if num_prune > 0:
                # Get indices of smallest weights
                _, indices = torch.topk(weights_flat, num_prune, largest=False)

                # Create mask
                mask = torch.ones_like(module.weight.data)
                flat_mask = mask.view(-1)
                flat_mask[indices] = 0

                masks[name] = mask
            else:
                masks[name] = torch.ones_like(module.weight.data)

        # Apply masks
        self.apply_masks(masks)

        return self.model, masks
