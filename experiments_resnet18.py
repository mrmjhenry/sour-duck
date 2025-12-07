"""
ResNet-18 Experiments with Moduli Space Pruning

Dedicated experiment script for evaluating ResNet-18 with moduli space pruning.
Compares against magnitude, random, and structured pruning baselines.

Usage:
    python experiments_resnet18.py [--quick] [--small]

Options:
    --quick: Run quick test with reduced epochs
    --small: Use ResNet-18 Small (half channels) for faster testing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from pathlib import Path
import json
import time
import argparse
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import datetime

from resnet18_cifar import create_resnet18_cifar10
from moduli_space_pruning import ModuliSpaceAnalyzer, ModuliSpacePruner
from experiments import ExperimentRunner, BasePruningMethods

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100


class ResNet18ExperimentRunner(ExperimentRunner):
    """Extended experiment runner specifically for ResNet-18."""

    def __init__(self, use_small_resnet: bool = False):
        super().__init__()
        self.use_small_resnet = use_small_resnet

    def create_resnet18(self, num_classes: int = 10):
        """Create ResNet-18 model."""
        model = create_resnet18_cifar10(
            num_classes=num_classes,
            small=self.use_small_resnet
        )
        print(f"Created ResNet-18 ({'small' if self.use_small_resnet else 'standard'})")
        print(f"Total parameters: {model.get_num_parameters():,}")
        return model

    def prepare_data(self, dataset='CIFAR10', batch_size=128):
        """Prepare datasets"""
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        if dataset == 'CIFAR10':
            trainset = torchvision.datasets.CIFAR10(
                root='./data', train=True, download=True, transform=transform_train)
            testset = torchvision.datasets.CIFAR10(
                root='./data', train=False, download=True, transform=transform_test)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        return trainloader, testloader

    def run_resnet18_experiment(
        self,
        pruning_method: str = 'moduli_space',
        pruning_ratio: float = 0.5,
        pretrain_epochs: int = 1,
        finetune_epochs: int = 1,
        batch_size: int = 128
    ) -> Dict:
        """
        Run single ResNet-18 pruning experiment.

        Args:
            pruning_method: 'moduli_space', 'magnitude', 'random', or 'structured'
            pruning_ratio: Fraction of weights to prune (0 to 1)
            pretrain_epochs: Epochs to train before pruning
            finetune_epochs: Epochs to fine-tune after pruning
            batch_size: Training batch size

        Returns:
            Dictionary with experiment results
        """
        print(f"\n{'='*70}")
        print(f"ResNet-18 Experiment: {pruning_method}, ratio={pruning_ratio}")
        print(f"{'='*70}")

        device = self.device
        results = {
            'network': 'ResNet18_Small' if self.use_small_resnet else 'ResNet18',
            'method': pruning_method,
            'pruning_ratio': pruning_ratio
        }

        # Create model
        network = self.create_resnet18().to(device)

        # Load data
        train_loader, test_loader = self.prepare_data()

        # Pre-train
        print(f"\n1. Pre-training for {pretrain_epochs} epochs...")
        network, train_losses = self.train_network(
            network, train_loader, pretrain_epochs
        )
        pretrain_acc, pretrain_time = self.evaluate_network(network, test_loader)
        print(f"   Pre-train accuracy: {pretrain_acc:.2f}%")
        results['pretrain_acc'] = pretrain_acc

        # Measure baseline inference time
        print(f"   Baseline inference time: {pretrain_time:.4f}s")
        results['baseline_inference_time'] = pretrain_time

        # Prune
        print(f"\n2. Pruning with {pruning_method} (ratio={pruning_ratio})...")
        prune_start = time.time()

        if pruning_method == 'moduli_space':
            # Moduli space pruning
            print("   Analyzing moduli space...")
            analyzer = ModuliSpaceAnalyzer(network)
            analyzer.analyze_dataset(train_loader, max_samples=500)

            print(f"   Moduli dimension: {analyzer.moduli_dimension:,}")
            print(f"   Total coordinates: {analyzer.total_coordinates:,}")

            pruner = ModuliSpacePruner(network, analyzer)
            pruned_network, masks = pruner.prune(pruning_ratio=pruning_ratio)

        elif pruning_method == 'magnitude':
            pruned_network, masks = BasePruningMethods.magnitude_pruning(
                network, pruning_ratio
            )

        elif pruning_method == 'random':
            pruned_network, masks = BasePruningMethods.random_pruning(
                network, pruning_ratio
            )

        elif pruning_method == 'structured':
            pruned_network, masks = BasePruningMethods.structured_pruning(
                network, pruning_ratio
            )

        else:
            raise ValueError(f"Unknown pruning method: {pruning_method}")

        prune_time = time.time() - prune_start
        print(f"   Pruning completed in {prune_time:.2f}s")
        results['prune_time'] = prune_time

        # Calculate actual sparsity
        total_params = 0
        zero_params = 0
        for layer_name, mask in masks.items():
            total_params += mask.numel()
            zero_params += (mask == 0).sum().item()

        actual_sparsity = zero_params / total_params
        print(f"   Actual sparsity: {actual_sparsity:.2%}")
        results['sparsity'] = actual_sparsity

        # Evaluate immediately after pruning
        pruned_network.eval()
        immediate_acc, immediate_time = self.evaluate_network(pruned_network, test_loader)
        print(f"   Immediate accuracy (before fine-tuning): {immediate_acc:.2f}%")
        results['immediate_acc'] = immediate_acc

        # Fine-tune
        print(f"\n3. Fine-tuning for {finetune_epochs} epochs...")
        final_network, final_network_losses = self.train_network(
            #pruned_network, train_loader, test_loader, epochs=finetune_epochs
            pruned_network, train_loader, epochs=finetune_epochs
        )

        final_acc, pruned_time = self.evaluate_network(final_network, test_loader)
        print(f"   Final accuracy: {final_acc:.2f}%")
        results['final_acc'] = final_acc

        # Measure pruned inference time
        speedup = pretrain_time / pruned_time
        print(f"   Pruned inference time: {pruned_time:.4f}s")
        print(f"   Speedup: {speedup:.2f}x")
        results['pruned_inference_time'] = pruned_time
        results['speedup'] = speedup

        # Calculate metrics
        results['accuracy_drop'] = pretrain_acc - final_acc
        results['immediate_drop'] = pretrain_acc - immediate_acc

        print(f"\n{'='*70}")
        print(f"Results Summary:")
        print(f"  Pre-train: {pretrain_acc:.2f}%")
        print(f"  Immediate: {immediate_acc:.2f}% (drop: {results['immediate_drop']:.2f}%)")
        print(f"  Final:     {final_acc:.2f}% (drop: {results['accuracy_drop']:.2f}%)")
        print(f"  Sparsity:  {actual_sparsity:.1%}")
        print(f"  Speedup:   {speedup:.2f}x")
        print(f"{'='*70}")

        return results

    def run_comparison_suite(
        self,
        methods: List[str] = None,
        pruning_ratios: List[float] = None,
        pretrain_epochs: int = 30,
        finetune_epochs: int = 10
    ) -> List[Dict]:
        """
        Run comprehensive comparison across methods and ratios.

        Args:
            methods: List of pruning methods to compare
            pruning_ratios: List of pruning ratios to test
            pretrain_epochs: Epochs for pre-training
            finetune_epochs: Epochs for fine-tuning

        Returns:
            List of result dictionaries
        """
        if methods is None:
            methods = ['moduli_space', 'magnitude', 'random']

        if pruning_ratios is None:
            pruning_ratios = [0.3, 0.5, 0.7]

        print(f"\n{'='*70}")
        print("RESNET-18 PRUNING COMPARISON SUITE")
        print(f"{'='*70}")
        print(f"Methods: {methods}")
        print(f"Pruning ratios: {pruning_ratios}")
        print(f"Pre-train epochs: {pretrain_epochs}")
        print(f"Fine-tune epochs: {finetune_epochs}")
        print(f"{'='*70}\n")

        # Create results directory
        Path('results').mkdir(exist_ok=True)

        all_results = []
        total_experiments = len(methods) * len(pruning_ratios)
        experiment_num = 0

        for method in methods:
            for ratio in pruning_ratios:
                experiment_num += 1
                print(f"\n[Experiment {experiment_num}/{total_experiments}]")

                try:
                    result = self.run_resnet18_experiment(
                        pruning_method=method,
                        pruning_ratio=ratio,
                        pretrain_epochs=pretrain_epochs,
                        finetune_epochs=finetune_epochs
                    )
                    all_results.append(result)

                    # Save intermediate results
                    df = pd.DataFrame(all_results)
                    df.to_csv('results/resnet18_results.csv', index=False)

                except Exception as e:
                    print(f"\n❌ Experiment failed: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

        return all_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='ResNet-18 Moduli Space Pruning Experiments'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick test with reduced epochs'
    )
    parser.add_argument(
        '--small',
        action='store_true',
        help='Use small ResNet-18 (half channels)'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='moduli_space',
        choices=['moduli_space', 'magnitude', 'random', 'structured'],
        help='Pruning method to use'
    )
    parser.add_argument(
        '--ratio',
        type=float,
        default=0.5,
        help='Pruning ratio (0 to 1)'
    )
    parser.add_argument(
        '--full-suite',
        action='store_true',
        help='Run full comparison suite'
    )

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Create runner
    runner = ResNet18ExperimentRunner(use_small_resnet=args.small)

    if args.full_suite:
        # Run full comparison
        print("\n🚀 Running full comparison suite...")

        if args.quick:
            pretrain_epochs = 10
            finetune_epochs = 5
        else:
            pretrain_epochs = 30
            finetune_epochs = 10

        results = runner.run_comparison_suite(
            methods=['moduli_space', 'magnitude', 'random'],
            pruning_ratios=[0.3, 0.5, 0.7],
            pretrain_epochs=pretrain_epochs,
            finetune_epochs=finetune_epochs
        )

        # Save and plot
        df = pd.DataFrame(results)
        now = datetime.now()
        date_str = now.isoformat(sep='', timespec='milliseconds')
        fname = f"results/resnet18_full_results-{date_str}.csv"
        df.to_csv(fname, index=False)
        print(f"\n✓ Results saved to results/resnet18_full_results.csv")

        # Generate plots
        runner.plot_results(df)
        print(f"✓ Plots saved to results/")

    else:
        # Run single experiment
        print(f"\n🚀 Running single experiment...")

        if args.quick:
            pretrain_epochs = 5
            finetune_epochs = 3
        else:
            pretrain_epochs = 1
            finetune_epochs = 1

        result = runner.run_resnet18_experiment(
            pruning_method=args.method,
            pruning_ratio=args.ratio,
            pretrain_epochs=pretrain_epochs,
            finetune_epochs=finetune_epochs
        )

        # Save result
        results_file = 'results/resnet18_single_result.json'
        with open(results_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n✓ Result saved to {results_file}")


if __name__ == '__main__':
    main()
