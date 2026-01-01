"""
Pruning Methods Package
=======================

Provides implementations of various neural network pruning methods.
"""

from .base import BasePruner
from .random_pruning import RandomPruner
from .magnitude_pruning import MagnitudePruner, LayerWiseMagnitudePruner
from .lottery_ticket import LotteryTicketPruner
from .moduli_space_pruning import ModuliSpacePruner, ModuliSpacePrunerAblation
from .wanda import WandaPruner


def get_pruner(method: str, model, **kwargs):
    """
    Factory function to get pruner by name.

    Args:
        method: Pruning method name
               ('random', 'magnitude', 'lottery_ticket', 'moduli_space', 'wanda')
        model: Neural network to prune
        **kwargs: Method-specific arguments

    Returns:
        Pruner instance
    """
    method = method.lower()

    if method == 'random':
        return RandomPruner(model, **kwargs)

    elif method == 'magnitude':
        return MagnitudePruner(model, **kwargs)

    elif method == 'magnitude_layerwise':
        return LayerWiseMagnitudePruner(model, **kwargs)

    elif method == 'lottery_ticket':
        return LotteryTicketPruner(model, **kwargs)

    elif method == 'moduli_space':
        return ModuliSpacePruner(model, **kwargs)

    elif method == 'wanda':
        return WandaPruner(model, **kwargs)

    else:
        raise ValueError(f"Unknown pruning method: {method}")


__all__ = [
    'BasePruner',
    'RandomPruner',
    'MagnitudePruner',
    'LayerWiseMagnitudePruner',
    'LotteryTicketPruner',
    'ModuliSpacePruner',
    'ModuliSpacePrunerAblation',
    'WandaPruner',
    'get_pruner',
]
