"""
Random Pruning
==============

Baseline pruning method that randomly selects weights to prune.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple
from .base import BasePruner


class RandomPruner(BasePruner):
    """
    Randomly prune weights from the network.

    This serves as a baseline to compare against more sophisticated methods.
    """

    def __init__(self, model: nn.Module, seed: int = 42):
        """
        Initialize random pruner.

        Args:
            model: Neural network to prune
            seed: Random seed for reproducibility
        """
        super().__init__(model)
        self.seed = seed

    def prune(
        self,
        pruning_ratio: float,
        **kwargs
    ) -> Tuple[nn.Module, Dict[str, torch.Tensor]]:
        """
        Randomly prune weights.

        Args:
            pruning_ratio: Fraction of weights to prune

        Returns:
            (pruned_model, masks)
        """
        # Set random seed for reproducibility
        torch.manual_seed(self.seed)

        # Get all prunable layers
        prunable_layers = self.get_prunable_layers(self.model)

        # Create masks for each layer
        masks = {}

        for name, module in prunable_layers.items():
            # Start with all weights kept
            mask = torch.ones_like(module.weight.data)

            # Randomly select weights to prune
            num_weights = mask.numel()
            num_prune = int(num_weights * pruning_ratio)

            # Generate random indices
            flat_mask = mask.view(-1)
            prune_indices = torch.randperm(num_weights)[:num_prune]

            # Set selected weights to 0
            flat_mask[prune_indices] = 0

            masks[name] = mask

        # Apply masks
        self.apply_masks(masks)

        return self.model, masks
