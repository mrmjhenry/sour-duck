"""
Metrics Tracking for Pruning Experiments
=========================================

Tracks accuracy, memory, FLOPs, and other metrics across experiments.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
import time
import numpy as np
from dataclasses import dataclass, asdict
import json
from pathlib import Path


@dataclass
class PruningMetrics:
    """Container for pruning experiment metrics"""

    # Basic info
    method: str
    sparsity: float

    # Accuracy metrics
    test_accuracy: float
    top5_accuracy: Optional[float] = None
    train_accuracy: Optional[float] = None

    # Model characteristics
    total_params: int = 0
    active_params: int = 0
    model_size_mb: float = 0.0

    # Performance metrics
    inference_time_ms: Optional[float] = None
    flops: Optional[int] = None
    memory_mb: Optional[float] = None

    # Training metrics
    pretrain_accuracy: Optional[float] = None
    finetune_epochs: Optional[int] = None

    # Moduli space specific (if applicable)
    moduli_dimension: Optional[int] = None
    num_coordinates: Optional[int] = None
    mean_importance: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)

    def save(self, path: str):
        """Save metrics to JSON file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str):
        """Load metrics from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


class MetricsTracker:
    """
    Track and compute metrics for pruning experiments.
    """

    def __init__(self, device: str = 'cuda'):
        """
        Initialize metrics tracker.

        Args:
            device: Device for computation
        """
        self.device = device
        self.metrics_history: List[PruningMetrics] = []

    def compute_accuracy(
        self,
        model: nn.Module,
        dataloader,
        top5: bool = False,
        device: Optional[str] = None
    ) -> tuple:
        """
        Compute accuracy on a dataset.

        Args:
            model: Neural network
            dataloader: Data loader
            top5: Whether to compute top-5 accuracy
            device: Device to use (default: self.device)

        Returns:
            (top1_accuracy, top5_accuracy) if top5=True
            (top1_accuracy, None) otherwise
        """
        if device is None:
            device = self.device

        model.eval()
        model = model.to(device)

        correct_top1 = 0
        correct_top5 = 0
        total = 0

        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(device), target.to(device)
                output = model(data)

                # Top-1 accuracy
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct_top1 += (predicted == target).sum().item()

                # Top-5 accuracy
                if top5:
                    _, top5_pred = output.topk(5, dim=1)
                    correct_top5 += (top5_pred == target.unsqueeze(1)).any(dim=1).sum().item()

        top1_acc = 100.0 * correct_top1 / total
        top5_acc = 100.0 * correct_top5 / total if top5 else None

        return top1_acc, top5_acc

    def measure_inference_time(
        self,
        model: nn.Module,
        input_shape: tuple,
        num_iterations: int = 100,
        device: Optional[str] = None
    ) -> float:
        """
        Measure average inference time.

        Args:
            model: Neural network
            input_shape: Shape of input tensor (batch_size, C, H, W)
            num_iterations: Number of iterations to average
            device: Device to use

        Returns:
            Average inference time in milliseconds
        """
        if device is None:
            device = self.device

        model.eval()
        model = model.to(device)

        # Warm-up
        x = torch.randn(input_shape).to(device)
        with torch.no_grad():
            for _ in range(10):
                _ = model(x)

        # Measure
        torch.cuda.synchronize() if device == 'cuda' else None
        start_time = time.time()

        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(x)

        torch.cuda.synchronize() if device == 'cuda' else None
        end_time = time.time()

        avg_time_ms = (end_time - start_time) / num_iterations * 1000
        return avg_time_ms

    def estimate_flops(
        self,
        model: nn.Module,
        input_shape: tuple
    ) -> int:
        """
        Estimate FLOPs (simplified calculation).

        Args:
            model: Neural network
            input_shape: Input shape (batch_size, C, H, W)

        Returns:
            Estimated FLOPs
        """
        total_flops = 0

        # Simple FLOP estimation for Conv2d and Linear
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                # FLOPs = 2 * in_channels * out_channels * kernel_size^2 * output_h * output_w
                # This is a rough estimate
                in_c = module.in_channels
                out_c = module.out_channels
                k = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size

                # Estimate output size (simplified)
                flops = 2 * in_c * out_c * k * k * 32 * 32  # Rough estimate
                total_flops += flops

            elif isinstance(module, nn.Linear):
                # FLOPs = 2 * in_features * out_features
                flops = 2 * module.in_features * module.out_features
                total_flops += flops

        return total_flops

    def get_memory_footprint(
        self,
        model: nn.Module,
        input_shape: tuple,
        device: Optional[str] = None
    ) -> float:
        """
        Measure memory footprint during inference.

        Args:
            model: Neural network
            input_shape: Input shape
            device: Device to use

        Returns:
            Memory usage in MB
        """
        if device is None:
            device = self.device

        if device != 'cuda':
            # For CPU, estimate from model size
            return sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)

        model = model.to(device)
        x = torch.randn(input_shape).to(device)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        model.eval()
        with torch.no_grad():
            _ = model(x)

        memory_bytes = torch.cuda.max_memory_allocated()
        memory_mb = memory_bytes / (1024 ** 2)

        return memory_mb

    def create_metrics(
        self,
        method: str,
        model: nn.Module,
        test_loader,
        sparsity: float,
        **kwargs
    ) -> PruningMetrics:
        """
        Create comprehensive metrics for a pruned model.

        Args:
            method: Pruning method name
            model: Pruned model
            test_loader: Test data loader
            sparsity: Pruning sparsity ratio
            **kwargs: Additional metrics to include

        Returns:
            PruningMetrics object
        """
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        active_params = sum((p != 0).sum().item() for p in model.parameters())
        model_size_mb = total_params * 4 / (1024 ** 2)

        # Compute accuracy
        top1_acc, top5_acc = self.compute_accuracy(
            model, test_loader, top5=(kwargs.get('num_classes', 10) > 10)
        )

        # Create metrics
        metrics = PruningMetrics(
            method=method,
            sparsity=sparsity,
            test_accuracy=top1_acc,
            top5_accuracy=top5_acc,
            total_params=total_params,
            active_params=active_params,
            model_size_mb=model_size_mb,
            **kwargs
        )

        self.metrics_history.append(metrics)

        return metrics

    def compare_methods(
        self,
        metrics_list: List[PruningMetrics],
        save_path: Optional[str] = None
    ) -> Dict:
        """
        Compare metrics across different pruning methods.

        Args:
            metrics_list: List of PruningMetrics
            save_path: Optional path to save comparison

        Returns:
            Dictionary with comparison results
        """
        comparison = {
            'methods': [m.method for m in metrics_list],
            'sparsities': [m.sparsity for m in metrics_list],
            'accuracies': [m.test_accuracy for m in metrics_list],
            'model_sizes_mb': [m.model_size_mb for m in metrics_list],
            'active_params': [m.active_params for m in metrics_list],
        }

        if save_path:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as f:
                json.dump(comparison, f, indent=2)

        return comparison

    def save_all(self, output_dir: str):
        """Save all tracked metrics"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, metrics in enumerate(self.metrics_history):
            filename = f"{metrics.method}_sparsity{metrics.sparsity:.2f}.json"
            metrics.save(output_dir / filename)


if __name__ == '__main__':
    # Example usage
    print("Metrics Tracker Example")
    print("=" * 50)

    # Create dummy metrics
    metrics = PruningMetrics(
        method="moduli_space",
        sparsity=0.5,
        test_accuracy=92.5,
        total_params=1000000,
        active_params=500000,
        model_size_mb=3.8
    )

    print(metrics)
    print("\nMetrics dict:")
    print(json.dumps(metrics.to_dict(), indent=2))
