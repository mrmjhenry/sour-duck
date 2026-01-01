"""
Lottery Ticket Pruning
======================

Implements the Lottery Ticket Hypothesis.
Based on Frankle & Carbin, "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks", ICLR 2019.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
import copy
from .base import BasePruner


class LotteryTicketPruner(BasePruner):
    """
    Iterative magnitude pruning with weight rewinding.

    The Lottery Ticket Hypothesis states that dense networks contain sparse
    subnetworks ("winning tickets") that can train to comparable accuracy.
    """

    def __init__(
        self,
        model: nn.Module,
        initial_weights: Optional[Dict] = None,
        reset_to_init: bool = True
    ):
        """
        Initialize lottery ticket pruner.

        Args:
            model: Neural network to prune
            initial_weights: Initial weights to rewind to (if None, uses current)
            reset_to_init: Whether to reset to initial weights after pruning
        """
        super().__init__(model)
        self.reset_to_init = reset_to_init

        # Store initial weights for rewinding
        if initial_weights is None:
            self.initial_weights = copy.deepcopy(model.state_dict())
        else:
            self.initial_weights = initial_weights

    def prune(
        self,
        pruning_ratio: float,
        iterations: int = 1,
        **kwargs
    ) -> Tuple[nn.Module, Dict[str, torch.Tensor]]:
        """
        Perform iterative magnitude pruning.

        Args:
            pruning_ratio: Fraction of weights to prune
            iterations: Number of pruning iterations
                       (each iteration prunes remaining weights by ratio)

        Returns:
            (pruned_model, masks)
        """
        # Calculate per-iteration pruning ratio
        # If we want to prune P% total over N iterations:
        # per_iter_ratio = 1 - (1 - P)^(1/N)
        if iterations > 1:
            per_iter_ratio = 1 - ((1 - pruning_ratio) ** (1 / iterations))
        else:
            per_iter_ratio = pruning_ratio

        # Initialize masks (all weights kept initially)
        prunable_layers = self.get_prunable_layers(self.model)
        masks = {name: torch.ones_like(module.weight.data)
                for name, module in prunable_layers.items()}

        # Iterative pruning
        for iter_idx in range(iterations):
            print(f"  Lottery Ticket iteration {iter_idx + 1}/{iterations}")

            # Collect weights that haven't been pruned yet
            unpruned_weights = []

            for name, module in prunable_layers.items():
                # Only consider weights that haven't been pruned
                current_mask = masks[name].view(-1)
                weights_flat = module.weight.data.abs().view(-1)

                for idx in range(len(weights_flat)):
                    if current_mask[idx] == 1:  # Not yet pruned
                        unpruned_weights.append((
                            weights_flat[idx].item(),
                            name,
                            idx
                        ))

            # Sort by magnitude
            unpruned_weights.sort(key=lambda x: x[0])

            # Determine how many to prune this iteration
            num_to_prune = int(len(unpruned_weights) * per_iter_ratio)

            # Prune smallest weights
            for i in range(num_to_prune):
                _, layer_name, weight_idx = unpruned_weights[i]
                flat_mask = masks[layer_name].view(-1)
                flat_mask[weight_idx] = 0

        # Reset weights to initial values (lottery ticket hypothesis)
        if self.reset_to_init:
            self.model.load_state_dict(self.initial_weights)

        # Apply final masks
        self.apply_masks(masks)

        return self.model, masks

    def get_winning_ticket_init(self) -> Dict:
        """
        Get the winning ticket initialization.

        Returns:
            State dict of the winning ticket at initialization
        """
        return copy.deepcopy(self.initial_weights)
