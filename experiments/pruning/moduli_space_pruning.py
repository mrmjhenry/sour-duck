"""
Moduli Space Pruning
====================

Pruning based on moduli space coordinate analysis.
Integrates with the existing moduli_space_pruning.py implementation.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from torch.utils.data import DataLoader
import sys
from pathlib import Path

# Add parent directory to path to import main moduli space code
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from moduli_space_pruning import ModuliSpaceAnalyzer, ModuliSpacePruner as MainPruner
from .base import BasePruner


class ModuliSpacePruner(BasePruner):
    """
    Prune based on moduli space coordinate importance.

    Uses representation theory to identify and remove low-importance
    information pathways in the network.
    """

    def __init__(
        self,
        model: nn.Module,
        strategy: str = 'combined',
        max_samples: int = 1000
    ):
        """
        Initialize moduli space pruner.

        Args:
            model: Neural network to prune
            strategy: Importance strategy ('combined', 'usage', 'variance', 'information')
            max_samples: Maximum number of samples for coordinate analysis
        """
        super().__init__(model)
        self.strategy = strategy
        self.max_samples = max_samples
        self.analyzer = None

    def analyze_coordinates(
        self,
        dataloader: DataLoader,
        max_samples: Optional[int] = None
    ):
        """
        Analyze coordinate importance across dataset.

        Args:
            dataloader: DataLoader with training data
            max_samples: Maximum samples to analyze (default: self.max_samples)
        """
        if max_samples is None:
            max_samples = self.max_samples

        print(f"  Analyzing moduli space coordinates (max {max_samples} samples)...")

        # Create analyzer
        self.analyzer = ModuliSpaceAnalyzer(self.model)

        # Analyze dataset
        self.analyzer.analyze_dataset(dataloader, max_samples=max_samples)

        print(f"  ✓ Analysis complete")

    def prune(
        self,
        pruning_ratio: float,
        dataloader: Optional[DataLoader] = None,
        **kwargs
    ) -> Tuple[nn.Module, Dict[str, torch.Tensor]]:
        """
        Prune based on moduli space coordinate importance.

        Args:
            pruning_ratio: Fraction of weights to prune
            dataloader: DataLoader for coordinate analysis (required if not already analyzed)
            **kwargs: Additional arguments

        Returns:
            (pruned_model, masks)
        """
        # Analyze coordinates if not already done
        if self.analyzer is None:
            if dataloader is None:
                raise ValueError(
                    "Must provide dataloader for coordinate analysis or "
                    "call analyze_coordinates() first"
                )
            self.analyze_coordinates(dataloader)

        # Use main pruner for actual pruning
        main_pruner = MainPruner(self.model, self.analyzer)
        pruned_model, masks = main_pruner.prune(
            pruning_ratio=pruning_ratio,
            strategy=self.strategy
        )

        return pruned_model, masks

    def get_coordinate_statistics(self) -> Dict:
        """
        Get statistics about analyzed coordinates.

        Returns:
            Dictionary with coordinate analysis results
        """
        if self.analyzer is None:
            return {}

        # Get coordinates
        coords = list(self.analyzer.coordinate_stats.values())

        if len(coords) == 0:
            return {}

        return {
            'num_coordinates': len(coords),
            'mean_usage': sum(c.usage_rate for c in coords) / len(coords),
            'mean_variance': sum(c.variance for c in coords) / len(coords),
            'mean_information': sum(c.information_content for c in coords) / len(coords),
            'mean_importance': sum(c.importance_score for c in coords) / len(coords),
        }


class ModuliSpacePrunerAblation(ModuliSpacePruner):
    """
    Moduli space pruner with component ablation support.

    Allows testing individual components of the importance score:
    - usage_only: Only usage rate
    - variance_only: Only variance
    - information_only: Only information content
    - combined: All three (default)
    """

    def __init__(
        self,
        model: nn.Module,
        component: str = 'combined',
        max_samples: int = 1000
    ):
        """
        Initialize ablation pruner.

        Args:
            model: Neural network to prune
            component: Which component to use for importance
                      ('usage_only', 'variance_only', 'information_only', 'combined')
            max_samples: Maximum number of samples for analysis
        """
        # Map component to strategy
        strategy_map = {
            'usage_only': 'usage',
            'variance_only': 'variance',
            'information_only': 'information',
            'combined': 'combined'
        }

        strategy = strategy_map.get(component, 'combined')
        super().__init__(model, strategy=strategy, max_samples=max_samples)

        self.component = component
