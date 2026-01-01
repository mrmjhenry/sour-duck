"""
Base Pruner Class
=================

Provides common interface and utilities for all pruning methods.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional
import copy


class BasePruner(ABC):
    """
    Abstract base class for pruning methods.

    All pruning methods should inherit from this class and implement
    the `prune()` method.
    """

    def __init__(self, model: nn.Module):
        """
        Initialize pruner.

        Args:
            model: Neural network to prune
        """
        self.model = model
        self.original_state = copy.deepcopy(model.state_dict())

    @abstractmethod
    def prune(
        self,
        pruning_ratio: float,
        **kwargs
    ) -> Tuple[nn.Module, Dict[str, torch.Tensor]]:
        """
        Prune the network.

        Args:
            pruning_ratio: Fraction of weights to prune (0.0 to 1.0)
            **kwargs: Method-specific arguments

        Returns:
            (pruned_model, masks):
                - pruned_model: The pruned network
                - masks: Dictionary mapping layer names to binary masks
        """
        pass

    def get_masks(self) -> Dict[str, torch.Tensor]:
        """
        Get current pruning masks for all layers.

        Returns:
            Dictionary mapping layer names to masks (1 = keep, 0 = prune)
        """
        masks = {}
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                masks[name] = torch.ones_like(module.weight.data)
        return masks

    def apply_masks(self, masks: Dict[str, torch.Tensor]):
        """
        Apply pruning masks to network weights.

        Args:
            masks: Dictionary mapping layer names to binary masks
        """
        for name, module in self.model.named_modules():
            if name in masks:
                module.weight.data *= masks[name]

    def get_sparsity(self, masks: Optional[Dict[str, torch.Tensor]] = None) -> float:
        """
        Calculate current network sparsity.

        Args:
            masks: Optional dictionary of masks. If None, uses current weights.

        Returns:
            Sparsity ratio (fraction of zero weights)
        """
        if masks is None:
            # Count zeros in actual weights
            total = 0
            zeros = 0
            for module in self.model.modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    total += module.weight.numel()
                    zeros += (module.weight == 0).sum().item()
        else:
            # Count zeros in masks
            total = 0
            zeros = 0
            for mask in masks.values():
                total += mask.numel()
                zeros += (mask == 0).sum().item()

        return zeros / total if total > 0 else 0.0

    def restore_original(self):
        """Restore network to original unpruned state"""
        self.model.load_state_dict(self.original_state)

    def count_parameters(self) -> Tuple[int, int]:
        """
        Count total and non-zero parameters.

        Returns:
            (total_params, active_params)
        """
        total = 0
        active = 0

        for module in self.model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                total += module.weight.numel()
                active += (module.weight != 0).sum().item()

        return total, active

    def get_model_size_mb(self) -> float:
        """
        Get model size in MB.

        Returns:
            Model size in megabytes (float32 assumption)
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        return total_params * 4 / (1024 ** 2)  # 4 bytes per float32

    @staticmethod
    def get_prunable_layers(model: nn.Module) -> Dict[str, nn.Module]:
        """
        Get all prunable layers (Linear and Conv2d).

        Args:
            model: Neural network

        Returns:
            Dictionary mapping layer names to modules
        """
        prunable = {}
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                prunable[name] = module
        return prunable
