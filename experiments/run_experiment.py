"""
Experiment Orchestrator
=======================

Main runner for pruning experiments. Handles:
- Loading configuration from YAML
- Training baseline models
- Pruning with multiple methods at multiple sparsity levels
- Fine-tuning pruned models
- Tracking metrics
- Generating visualizations
- Saving results

Usage:
    python experiments/run_experiment.py --config experiments/config/cifar10_resnet18.yaml
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.config import load_config, Config
from experiments.datasets.data_loaders import DatasetFactory
from experiments.models import get_model
from experiments.pruning import get_pruner
from experiments.utils.metrics import MetricsTracker, PruningMetrics
from experiments.utils.visualization import ExperimentVisualizer


class ExperimentOrchestrator:
    """Orchestrates the entire pruning experiment pipeline"""

    def __init__(self, config: Config):
        """
        Initialize experiment orchestrator.

        Args:
            config: Experiment configuration
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create output directory
        self.output_dir = Path(config.experiment.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.metrics_tracker = MetricsTracker(device=self.device)
        self.visualizer = ExperimentVisualizer(
            output_dir=str(self.output_dir / 'plots')
        )

        # Storage for results
        self.all_results: Dict[str, List[PruningMetrics]] = {}
        self.baseline_accuracy = None

        print(f"Experiment: {config.experiment.name}")
        print(f"Device: {self.device}")
        print(f"Output: {self.output_dir}")
        print()

    def setup_data(self) -> Tuple:
        """
        Load and prepare datasets.

        Returns:
            (train_loader, test_loader, num_classes)
        """
        print("Setting up dataset...")

        train_loader, test_loader, num_classes = DatasetFactory.get_dataset(
            name=self.config.dataset.name,
            data_dir=self.config.dataset.data_dir,
            batch_size=self.config.dataset.batch_size,
            num_workers=self.config.dataset.num_workers,
            augmentation=self.config.dataset.augmentation,
            download=self.config.dataset.download
        )

        print(f"  Dataset: {self.config.dataset.name}")
        print(f"  Classes: {num_classes}")
        print(f"  Training batches: {len(train_loader)}")
        print(f"  Test batches: {len(test_loader)}")
        print()

        return train_loader, test_loader, num_classes

    def create_model(self, num_classes: int) -> nn.Module:
        """
        Create and initialize model.

        Args:
            num_classes: Number of output classes

        Returns:
            Initialized model
        """
        model = get_model(
            model_name=self.config.model.name,
            num_classes=num_classes,
            pretrained=self.config.model.pretrained,
            dataset_name=self.config.dataset.name
        )

        model = model.to(self.device)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model: {self.config.model.name}")
        print(f"  Total parameters: {total_params:,}")
        print()

        return model

    def train_baseline(
        self,
        model: nn.Module,
        train_loader,
        test_loader
    ) -> nn.Module:
        """
        Train baseline (unpruned) model.

        Args:
            model: Model to train
            train_loader: Training data loader
            test_loader: Test data loader

        Returns:
            Trained model
        """
        print("Training baseline model...")

        # Setup optimizer and scheduler
        optimizer = optim.SGD(
            model.parameters(),
            lr=self.config.training.learning_rate,
            momentum=self.config.training.momentum,
            weight_decay=self.config.training.weight_decay
        )

        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=self.config.training.lr_milestones,
            gamma=self.config.training.lr_gamma
        )

        criterion = nn.CrossEntropyLoss()

        best_accuracy = 0.0

        for epoch in range(self.config.training.baseline_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.training.baseline_epochs}")
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()

                pbar.set_postfix({
                    'loss': train_loss / (pbar.n + 1),
                    'acc': 100. * train_correct / train_total
                })

            scheduler.step()

            # Evaluation phase
            test_accuracy = self.metrics_tracker.compute_accuracy(model, test_loader)

            print(f"  Epoch {epoch+1}: Train Acc = {100.*train_correct/train_total:.2f}%, "
                  f"Test Acc = {test_accuracy:.2f}%")

            # Save best model
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                torch.save(
                    model.state_dict(),
                    self.output_dir / 'baseline_best.pth'
                )

        # Load best model
        model.load_state_dict(torch.load(self.output_dir / 'baseline_best.pth'))
        self.baseline_accuracy = best_accuracy

        print(f"  Best baseline accuracy: {best_accuracy:.2f}%")
        print()

        return model

    def prune_and_finetune(
        self,
        baseline_model: nn.Module,
        pruning_method: str,
        sparsity: float,
        train_loader,
        test_loader
    ) -> Tuple[nn.Module, Dict, PruningMetrics]:
        """
        Prune model and fine-tune.

        Args:
            baseline_model: Baseline model to prune
            pruning_method: Pruning method name
            sparsity: Target sparsity level
            train_loader: Training data loader
            test_loader: Test data loader

        Returns:
            (pruned_model, masks, metrics)
        """
        # Create fresh copy of baseline model
        model = get_model(
            model_name=self.config.model.name,
            num_classes=test_loader.dataset.targets.max().item() + 1
            if hasattr(test_loader.dataset, 'targets') else 10,
            pretrained=False,
            dataset_name=self.config.dataset.name
        ).to(self.device)

        model.load_state_dict(baseline_model.state_dict())

        # Get pruner
        pruner_kwargs = {}

        if pruning_method == 'moduli_space':
            pruner_kwargs['strategy'] = self.config.pruning.moduli_space.strategy
            pruner_kwargs['max_samples'] = self.config.pruning.moduli_space.max_samples
        elif pruning_method == 'lottery_ticket':
            pruner_kwargs['initial_weights'] = baseline_model.state_dict()

        pruner = get_pruner(pruning_method, model, **pruner_kwargs)

        # Perform pruning
        print(f"  Pruning with {pruning_method} at {sparsity*100:.0f}% sparsity...")

        if pruning_method in ['moduli_space', 'wanda']:
            # These methods need data for analysis
            pruned_model, masks = pruner.prune(
                pruning_ratio=sparsity,
                dataloader=train_loader
            )
        else:
            pruned_model, masks = pruner.prune(pruning_ratio=sparsity)

        # Fine-tune pruned model
        if self.config.training.finetune_epochs > 0:
            print(f"  Fine-tuning for {self.config.training.finetune_epochs} epochs...")

            optimizer = optim.SGD(
                pruned_model.parameters(),
                lr=self.config.training.finetune_lr,
                momentum=self.config.training.momentum,
                weight_decay=self.config.training.weight_decay
            )

            criterion = nn.CrossEntropyLoss()

            for epoch in range(self.config.training.finetune_epochs):
                pruned_model.train()

                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    optimizer.zero_grad()
                    outputs = pruned_model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()

                    # Reapply masks after gradient step
                    optimizer.step()
                    pruner.apply_masks(masks)

        # Compute metrics
        metrics = self.metrics_tracker.create_metrics(
            method=pruning_method,
            model=pruned_model,
            test_loader=test_loader,
            sparsity=sparsity,
            masks=masks
        )

        # Add moduli-specific metrics if applicable
        if pruning_method == 'moduli_space' and hasattr(pruner, 'analyzer'):
            if pruner.analyzer is not None:
                metrics.moduli_dimension = len(pruner.analyzer.quiver.edges)

        print(f"  Result: Accuracy = {metrics.test_accuracy:.2f}%, "
              f"Actual Sparsity = {metrics.sparsity*100:.2f}%, "
              f"Size = {metrics.model_size_mb:.2f} MB")

        return pruned_model, masks, metrics

    def run_pruning_experiments(
        self,
        baseline_model: nn.Module,
        train_loader,
        test_loader
    ):
        """
        Run pruning experiments for all methods and sparsity levels.

        Args:
            baseline_model: Trained baseline model
            train_loader: Training data loader
            test_loader: Test data loader
        """
        print("Running pruning experiments...")
        print()

        for method in self.config.pruning.methods:
            print(f"Method: {method}")
            print("-" * 50)

            method_results = []

            for sparsity in self.config.pruning.sparsity_levels:
                _, _, metrics = self.prune_and_finetune(
                    baseline_model=baseline_model,
                    pruning_method=method,
                    sparsity=sparsity,
                    train_loader=train_loader,
                    test_loader=test_loader
                )

                method_results.append(metrics)

            self.all_results[method] = method_results
            print()

    def run_ablation_study(
        self,
        baseline_model: nn.Module,
        train_loader,
        test_loader
    ):
        """
        Run ablation study for moduli space components.

        Args:
            baseline_model: Trained baseline model
            train_loader: Training data loader
            test_loader: Test data loader
        """
        if not self.config.pruning.moduli_space.ablation.enabled:
            return

        if 'moduli_space' not in self.config.pruning.methods:
            return

        print("Running moduli space ablation study...")
        print("-" * 50)

        ablation_results = {}

        for component in self.config.pruning.moduli_space.ablation.components:
            print(f"Component: {component}")
            component_results = []

            for sparsity in self.config.pruning.sparsity_levels:
                # Create pruner with specific component
                model = get_model(
                    model_name=self.config.model.name,
                    num_classes=test_loader.dataset.targets.max().item() + 1
                    if hasattr(test_loader.dataset, 'targets') else 10,
                    pretrained=False,
                    dataset_name=self.config.dataset.name
                ).to(self.device)

                model.load_state_dict(baseline_model.state_dict())

                # Map component to strategy
                strategy_map = {
                    'usage_only': 'usage',
                    'variance_only': 'variance',
                    'information_only': 'information',
                    'combined': 'combined'
                }
                strategy = strategy_map.get(component, 'combined')

                pruner = get_pruner(
                    'moduli_space',
                    model,
                    strategy=strategy,
                    max_samples=self.config.pruning.moduli_space.max_samples
                )

                pruned_model, masks = pruner.prune(
                    pruning_ratio=sparsity,
                    dataloader=train_loader
                )

                # Quick fine-tune
                if self.config.training.finetune_epochs > 0:
                    optimizer = optim.SGD(
                        pruned_model.parameters(),
                        lr=self.config.training.finetune_lr,
                        momentum=self.config.training.momentum
                    )
                    criterion = nn.CrossEntropyLoss()

                    for epoch in range(min(5, self.config.training.finetune_epochs)):
                        pruned_model.train()
                        for inputs, targets in train_loader:
                            inputs, targets = inputs.to(self.device), targets.to(self.device)
                            optimizer.zero_grad()
                            loss = criterion(pruned_model(inputs), targets)
                            loss.backward()
                            optimizer.step()
                            pruner.apply_masks(masks)

                accuracy = self.metrics_tracker.compute_accuracy(pruned_model, test_loader)
                component_results.append(accuracy)

                print(f"  {sparsity*100:.0f}% sparsity: {accuracy:.2f}%")

            ablation_results[component] = component_results

        # Save ablation plot
        self.visualizer.plot_ablation_study(
            components=self.config.pruning.moduli_space.ablation.components,
            accuracies=ablation_results,
            sparsities=[s * 100 for s in self.config.pruning.sparsity_levels],
            save_name='moduli_ablation_study.png'
        )

        print()

    def generate_visualizations(self):
        """Generate all visualizations from results"""
        print("Generating visualizations...")

        # 1. Accuracy vs Sparsity
        accuracy_results = {
            method: [(m.sparsity * 100, m.test_accuracy) for m in metrics]
            for method, metrics in self.all_results.items()
        }

        self.visualizer.plot_accuracy_vs_sparsity(
            results=accuracy_results,
            save_name='accuracy_vs_sparsity.png',
            title=f'Accuracy vs Sparsity - {self.config.model.name} on {self.config.dataset.name}'
        )

        # 2. Memory Reduction
        memory_results = {
            method: [(m.sparsity, m.model_size_mb) for m in metrics]
            for method, metrics in self.all_results.items()
        }

        self.visualizer.plot_memory_reduction(
            results=memory_results,
            save_name='memory_reduction.png'
        )

        # 3. Moduli-specific visualizations
        if 'moduli_space' in self.all_results:
            moduli_metrics = self.all_results['moduli_space']

            # Extract moduli dimensions if available
            moduli_dims = [m.moduli_dimension for m in moduli_metrics
                          if m.moduli_dimension is not None]

            if moduli_dims:
                # Dimension vs prunable parameters correlation
                prunable_params = [m.active_params for m in moduli_metrics
                                  if m.moduli_dimension is not None]

                self.visualizer.plot_dimension_vs_prunable(
                    moduli_dims=moduli_dims,
                    prunable_params=prunable_params,
                    labels=[f"{m.sparsity*100:.0f}%" for m in moduli_metrics
                           if m.moduli_dimension is not None],
                    save_name='dimension_vs_prunable.png'
                )

        print("  ✓ All visualizations generated")
        print()

    def save_results(self):
        """Save all results to JSON"""
        print("Saving results...")

        # Convert results to JSON-serializable format
        results_dict = {
            'experiment': {
                'name': self.config.experiment.name,
                'dataset': self.config.dataset.name,
                'model': self.config.model.name,
                'baseline_accuracy': self.baseline_accuracy
            },
            'methods': {}
        }

        for method, metrics_list in self.all_results.items():
            results_dict['methods'][method] = [
                {
                    'sparsity': m.sparsity,
                    'test_accuracy': m.test_accuracy,
                    'total_params': m.total_params,
                    'active_params': m.active_params,
                    'model_size_mb': m.model_size_mb,
                    'inference_time_ms': m.inference_time_ms,
                    'flops': m.flops,
                    'moduli_dimension': m.moduli_dimension
                }
                for m in metrics_list
            ]

        # Save to file
        results_path = self.output_dir / 'results.json'
        with open(results_path, 'w') as f:
            json.dump(results_dict, f, indent=2)

        print(f"  ✓ Results saved to {results_path}")
        print()

    def run(self):
        """Run complete experiment pipeline"""
        print("=" * 70)
        print(f"EXPERIMENT: {self.config.experiment.name}")
        print("=" * 70)
        print()

        # 1. Setup data
        train_loader, test_loader, num_classes = self.setup_data()

        # 2. Create model
        baseline_model = self.create_model(num_classes)

        # 3. Train baseline (or load if exists)
        baseline_path = self.output_dir / 'baseline_best.pth'

        if baseline_path.exists() and not self.config.training.retrain_baseline:
            print("Loading existing baseline model...")
            baseline_model.load_state_dict(torch.load(baseline_path))
            self.baseline_accuracy = self.metrics_tracker.compute_accuracy(
                baseline_model, test_loader
            )
            print(f"  Baseline accuracy: {self.baseline_accuracy:.2f}%")
            print()
        else:
            baseline_model = self.train_baseline(
                baseline_model, train_loader, test_loader
            )

        # 4. Run pruning experiments
        self.run_pruning_experiments(baseline_model, train_loader, test_loader)

        # 5. Run ablation study
        if self.config.pruning.moduli_space.ablation.enabled:
            self.run_ablation_study(baseline_model, train_loader, test_loader)

        # 6. Generate visualizations
        self.generate_visualizations()

        # 7. Save results
        self.save_results()

        # 8. Print summary
        self.print_summary()

        print("=" * 70)
        print("EXPERIMENT COMPLETE")
        print("=" * 70)

    def print_summary(self):
        """Print experiment summary"""
        print()
        print("=" * 70)
        print("EXPERIMENT SUMMARY")
        print("=" * 70)
        print()
        print(f"Baseline Accuracy: {self.baseline_accuracy:.2f}%")
        print()

        for method, metrics_list in self.all_results.items():
            print(f"{method}:")
            for m in metrics_list:
                acc_drop = self.baseline_accuracy - m.test_accuracy
                print(f"  {m.sparsity*100:>3.0f}% sparsity: "
                      f"{m.test_accuracy:5.2f}% accuracy "
                      f"({acc_drop:+5.2f}% drop), "
                      f"{m.model_size_mb:5.2f} MB")
            print()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Run neural network pruning experiments'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML configuration file'
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Create and run orchestrator
    orchestrator = ExperimentOrchestrator(config)
    orchestrator.run()


if __name__ == '__main__':
    main()
