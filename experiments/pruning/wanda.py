"""
Wanda Pruning
=============

Pruning by Weights and activations (Wanda).
Based on Sun et al., "A Simple and Effective Pruning Approach for Large Language Models", 2024.

Note: This is a simplified implementation. The full Wanda method includes
additional optimizations for large language models.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from torch.utils.data import DataLoader
from .base import BasePruner


class WandaPruner(BasePruner):
    """
    Wanda: Pruning by Weights AND activations.

    Combines weight magnitude with activation importance, similar to
    moduli space pruning but using a simpler activation-based metric.
    """

    def __init__(self, model: nn.Module, use_activations: bool = True):
        """
        Initialize Wanda pruner.

        Args:
            model: Neural network to prune
            use_activations: Whether to use activation-based importance
        """
        super().__init__(model)
        self.use_activations = use_activations
        self.activation_cache = {}

    def _register_hooks(self):
        """Register forward hooks to capture activations"""
        hooks = []

        def get_activation(name):
            def hook(model, input, output):
                # Store mean absolute activation
                self.activation_cache[name] = output.abs().mean(dim=0).detach()
            return hook

        # Register hooks on all Conv2d and Linear layers
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                hooks.append(module.register_forward_hook(get_activation(name)))

        return hooks

    def _compute_activation_importance(
        self,
        dataloader: DataLoader,
        max_samples: int = 100
    ):
        """
        Compute activation importance by running forward passes.

        Args:
            dataloader: DataLoader with calibration data
            max_samples: Maximum number of samples to use
        """
        self.model.eval()
        hooks = self._register_hooks()

        samples_seen = 0

        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(dataloader):
                if samples_seen >= max_samples:
                    break

                data = data.to(next(self.model.parameters()).device)
                _ = self.model(data)

                samples_seen += data.size(0)

        # Remove hooks
        for hook in hooks:
            hook.remove()

    def prune(
        self,
        pruning_ratio: float,
        dataloader: Optional[DataLoader] = None,
        **kwargs
    ) -> Tuple[nn.Module, Dict[str, torch.Tensor]]:
        """
        Prune using Wanda method.

        Args:
            pruning_ratio: Fraction of weights to prune
            dataloader: DataLoader for activation computation (required if use_activations=True)

        Returns:
            (pruned_model, masks)
        """
        if self.use_activations:
            if dataloader is None:
                raise ValueError("Must provide dataloader when use_activations=True")

            # Compute activation importance
            self._compute_activation_importance(dataloader)

        # Collect all weights with their importance scores
        all_weights = []
        prunable_layers = self.get_prunable_layers(self.model)

        for name, module in prunable_layers.items():
            weights_flat = module.weight.data.abs().view(-1)

            if self.use_activations and name in self.activation_cache:
                # Wanda score: weight magnitude × activation importance
                activations = self.activation_cache[name]

                # Broadcast activations to match weight shape
                if isinstance(module, nn.Linear):
                    # For Linear: activations shape [in_features]
                    # weights shape [out_features, in_features]
                    act_importance = activations.unsqueeze(0).expand_as(module.weight.data)
                elif isinstance(module, nn.Conv2d):
                    # For Conv2d: activations shape [out_channels, H, W]
                    # weights shape [out_channels, in_channels, K, K]
                    # Use mean activation per output channel
                    act_importance = activations.mean(dim=(1, 2) if activations.dim() > 1 else 0)
                    act_importance = act_importance.view(-1, 1, 1, 1).expand_as(module.weight.data)

                # Combine weight and activation importance
                importance = weights_flat * act_importance.view(-1)
            else:
                # Fallback to magnitude-only pruning
                importance = weights_flat

            for idx, imp in enumerate(importance):
                all_weights.append((imp.item(), name, idx))

        # Sort by importance (lowest first)
        all_weights.sort(key=lambda x: x[0])

        # Determine how many to prune
        num_prune = int(len(all_weights) * pruning_ratio)

        # Create masks
        masks = {name: torch.ones_like(module.weight.data)
                for name, module in prunable_layers.items()}

        # Prune lowest importance weights
        for i in range(num_prune):
            _, layer_name, weight_idx = all_weights[i]
            flat_mask = masks[layer_name].view(-1)
            flat_mask[weight_idx] = 0

        # Apply masks
        self.apply_masks(masks)

        return self.model, masks
