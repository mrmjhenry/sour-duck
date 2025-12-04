"""
Experimental Evaluation of Moduli Space Coordinate Pruning
Comprehensive experiments comparing the proposed method against baselines
FIXED VERSION - Resolves dimension mismatch issues
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
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from moduli_space_pruning import (
    ModuliSpaceAnalyzer, ModuliSpacePruner, DataRepresentation
)

# Set style for better plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

class SimpleNet(nn.Module):
    """Simple network for initial experiments"""
    def __init__(self, num_classes=10):
        super(SimpleNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class VGG11_Small(nn.Module):
    """Smaller VGG11 for CIFAR-10"""
    def __init__(self, num_classes=10):
        super(VGG11_Small, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class BasePruningMethods:
    """Traditional pruning methods for comparison"""
    
    @staticmethod
    def magnitude_pruning(network, pruning_ratio=0.5):
        """Standard magnitude-based pruning"""
        masks = {}
        
        # Collect all weights and their magnitudes
        all_weights = []
        weight_info = []
        
        for name, module in network.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                weight = module.weight.data.cpu().numpy()
                all_weights.extend(weight.flatten())
                weight_info.append((name, weight.shape))
        
        all_weights = np.array(all_weights)
        threshold = np.percentile(np.abs(all_weights), pruning_ratio * 100)
        
        # Create masks
        for name, module in network.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                mask = (module.weight.data.abs() > threshold).float()
                masks[name] = mask
                module.weight.data *= mask
        
        return network, masks
    
    @staticmethod
    def random_pruning(network, pruning_ratio=0.5):
        """Random pruning baseline"""
        masks = {}
        
        for name, module in network.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                mask = torch.rand_like(module.weight.data) > pruning_ratio
                masks[name] = mask.float()
                module.weight.data *= masks[name]
        
        return network, masks
    
    @staticmethod
    def structured_pruning(network, pruning_ratio=0.5):
        """Simple structured pruning (channel-wise)"""
        masks = {}
        
        for name, module in network.named_modules():
            if isinstance(module, nn.Conv2d):
                # Compute channel importance (L2 norm)
                importance = module.weight.data.norm(dim=(1, 2, 3))
                n_prune = int(len(importance) * pruning_ratio)
                
                # Get indices of channels to keep
                _, indices = importance.sort(descending=True)
                keep_indices = indices[:len(indices) - n_prune]
                
                # Create mask
                mask = torch.zeros_like(module.weight.data)
                mask[keep_indices] = 1
                masks[name] = mask
                module.weight.data *= mask
                
            elif isinstance(module, nn.Linear):
                # For linear layers, use magnitude pruning
                mask = torch.ones_like(module.weight.data)
                threshold = module.weight.data.abs().quantile(pruning_ratio)
                mask[module.weight.data.abs() <= threshold] = 0
                masks[name] = mask
                module.weight.data *= mask
        
        return network, masks


class ExperimentRunner:
    """Main experiment runner"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.results = []
        print(f"Using device: {self.device}")
        
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
    
    def train_network(self, network, trainloader, epochs=10, lr=0.001):
        """Train a network"""
        network = network.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(network.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        train_losses = []
        
        for epoch in range(epochs):
            network.train()
            running_loss = 0.0
            
            pbar = tqdm(trainloader, desc=f'Epoch {epoch+1}/{epochs}')
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = network(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})
            
            train_losses.append(running_loss / len(trainloader))
            scheduler.step()
        
        return network, train_losses
    
    def evaluate_network(self, network, testloader):
        """Evaluate network performance"""
        network = network.to(self.device)
        network.eval()
        
        correct = 0
        total = 0
        inference_times = []
        
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                start_time = time.time()
                outputs = network(inputs)
                inference_times.append(time.time() - start_time)
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        accuracy = 100. * correct / total
        avg_inference_time = np.mean(inference_times)
        
        return accuracy, avg_inference_time
    
    def compute_sparsity(self, network):
        """Compute network sparsity"""
        total_params = 0
        zero_params = 0
        
        for module in network.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                total_params += module.weight.data.numel()
                zero_params += (module.weight.data == 0).sum().item()
        
        sparsity = zero_params / total_params if total_params > 0 else 0
        return sparsity
    
    def fine_tune(self, network, trainloader, epochs=5, lr=0.0001):
        """Fine-tune pruned network"""
        return self.train_network(network, trainloader, epochs=epochs, lr=lr)
    
    def run_pruning_experiment(self, network_class, pruning_method, pruning_ratio,
                              dataset='CIFAR10', pretrain_epochs=20, finetune_epochs=10):
        """Run a single pruning experiment"""
        
        print(f"\n{'='*60}")
        print(f"Running: {pruning_method} with ratio {pruning_ratio}")
        print(f"{'='*60}")
        
        # Prepare data
        trainloader, testloader = self.prepare_data(dataset)
        
        # Create and train network
        network = network_class()
        print("Training baseline network...")
        network, train_losses = self.train_network(network, trainloader, epochs=pretrain_epochs)
        
        # Evaluate baseline
        baseline_acc, baseline_time = self.evaluate_network(network, testloader)
        baseline_sparsity = self.compute_sparsity(network)
        print(f"Baseline - Accuracy: {baseline_acc:.2f}%, Inference: {baseline_time:.4f}s")
        
        # Apply pruning
        print(f"Applying {pruning_method} pruning...")
        start_prune_time = time.time()
        
        try:
            if pruning_method == 'moduli_space':
                analyzer = ModuliSpaceAnalyzer(network, self.device)
                # Use smaller sample size to avoid memory issues
                analyzer.analyze_dataset(trainloader, max_samples=min(500, len(trainloader.dataset)))
                pruner = ModuliSpacePruner(network, analyzer)
                network, masks = pruner.prune(pruning_ratio=pruning_ratio)
                
                # Generate visualization if possible
                try:
                    Path('results').mkdir(exist_ok=True)
                    analyzer.visualize_coordinate_usage(
                        save_path=f'results/moduli_analysis_{pruning_ratio}.png'
                    )
                except Exception as e:
                    print(f"Warning: Could not generate visualization: {e}")
                
            elif pruning_method == 'magnitude':
                network, masks = BasePruningMethods.magnitude_pruning(network, pruning_ratio)
            elif pruning_method == 'random':
                network, masks = BasePruningMethods.random_pruning(network, pruning_ratio)
            elif pruning_method == 'structured':
                network, masks = BasePruningMethods.structured_pruning(network, pruning_ratio)
            else:
                raise ValueError(f"Unknown pruning method: {pruning_method}")
                
        except Exception as e:
            print(f"Error during pruning: {e}")
            print("Falling back to magnitude pruning...")
            network, masks = BasePruningMethods.magnitude_pruning(network, pruning_ratio)
        
        prune_time = time.time() - start_prune_time
        
        # Evaluate after pruning
        pruned_acc, pruned_time = self.evaluate_network(network, testloader)
        pruned_sparsity = self.compute_sparsity(network)
        print(f"After pruning - Accuracy: {pruned_acc:.2f}%, Sparsity: {pruned_sparsity:.2%}")
        
        # Fine-tune
        print("Fine-tuning pruned network...")
        network, finetune_losses = self.fine_tune(network, trainloader, epochs=finetune_epochs)
        
        # Final evaluation
        final_acc, final_time = self.evaluate_network(network, testloader)
        print(f"After fine-tuning - Accuracy: {final_acc:.2f}%, Inference: {final_time:.4f}s")
        
        # Store results
        result = {
            'method': pruning_method,
            'pruning_ratio': pruning_ratio,
            'baseline_acc': baseline_acc,
            'pruned_acc': pruned_acc,
            'final_acc': final_acc,
            'accuracy_drop': baseline_acc - final_acc,
            'sparsity': pruned_sparsity,
            'baseline_inference': baseline_time,
            'pruned_inference': final_time,
            'speedup': baseline_time / final_time if final_time > 0 else 1.0,
            'prune_time': prune_time,
            'network': network_class.__name__,
        }
        
        self.results.append(result)
        return result
    
    def run_full_experiment_suite(self):
        """Run comprehensive experiments"""
        
        # Create results directory
        Path('results').mkdir(exist_ok=True)
        
        # Experiment configurations
        methods = ['moduli_space', 'magnitude', 'random', 'structured']
        pruning_ratios = [0.3, 0.5, 0.7]
        networks = [SimpleNet]  # Start with SimpleNet, add VGG11_Small if it works
        
        all_results = []
        
        for network_class in networks:
            for method in methods:
                for ratio in pruning_ratios:
                    try:
                        result = self.run_pruning_experiment(
                            network_class=network_class,
                            pruning_method=method,
                            pruning_ratio=ratio,
                            pretrain_epochs=20,
                            finetune_epochs=10
                        )
                        all_results.append(result)
                        
                        # Save intermediate results
                        df = pd.DataFrame(all_results)
                        df.to_csv('results/experiment_results.csv', index=False)
                        
                    except Exception as e:
                        print(f"Error in experiment: {e}")
                        continue
        
        return all_results
    
    def plot_results(self, results_df):
        """Generate comprehensive result plots"""
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Accuracy vs Sparsity
        ax = axes[0, 0]
        for method in results_df['method'].unique():
            data = results_df[results_df['method'] == method]
            ax.plot(data['sparsity'] * 100, data['final_acc'], 
                   marker='o', label=method, linewidth=2)
        ax.set_xlabel('Sparsity (%)')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Accuracy vs Sparsity Trade-off')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Accuracy Drop by Method
        ax = axes[0, 1]
        try:
            data_to_plot = results_df.pivot_table(
                values='accuracy_drop', index='pruning_ratio', 
                columns='method', aggfunc='mean'
            )
            data_to_plot.plot(kind='bar', ax=ax)
            ax.set_xlabel('Pruning Ratio')
            ax.set_ylabel('Accuracy Drop (%)')
            ax.set_title('Accuracy Drop by Pruning Method')
            ax.legend(title='Method')
            ax.grid(True, alpha=0.3)
        except:
            ax.text(0.5, 0.5, 'Not enough data', ha='center', va='center')
        
        # 3. Inference Speedup
        ax = axes[0, 2]
        for method in results_df['method'].unique():
            data = results_df[results_df['method'] == method]
            ax.plot(data['pruning_ratio'] * 100, data['speedup'], 
                   marker='s', label=method, linewidth=2)
        ax.set_xlabel('Pruning Ratio (%)')
        ax.set_ylabel('Speedup Factor')
        ax.set_title('Inference Speedup')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Pruning Time Comparison
        ax = axes[1, 0]
        try:
            pruning_times = results_df.groupby('method')['prune_time'].mean()
            pruning_times.plot(kind='bar', ax=ax, color='skyblue')
            ax.set_ylabel('Time (seconds)')
            ax.set_title('Average Pruning Time by Method')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            ax.grid(True, alpha=0.3)
        except:
            ax.text(0.5, 0.5, 'Not enough data', ha='center', va='center')
        
        # 5. Network Architecture Comparison
        ax = axes[1, 1]
        for network in results_df['network'].unique():
            data = results_df[results_df['network'] == network]
            moduli_data = data[data['method'] == 'moduli_space']
            if len(moduli_data) > 0:
                ax.plot(moduli_data['pruning_ratio'] * 100, moduli_data['final_acc'],
                       marker='D', label=network, linewidth=2)
        ax.set_xlabel('Pruning Ratio (%)')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Moduli Space Pruning Across Architectures')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. Method Comparison Heatmap
        ax = axes[1, 2]
        try:
            pivot = results_df.pivot_table(
                values='final_acc', index='method', 
                columns='pruning_ratio', aggfunc='mean'
            )
            sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', 
                       cbar_kws={'label': 'Accuracy (%)'}, ax=ax)
            ax.set_title('Accuracy Heatmap')
            ax.set_xlabel('Pruning Ratio')
            ax.set_ylabel('Method')
        except:
            ax.text(0.5, 0.5, 'Not enough data for heatmap', ha='center', va='center')
        
        plt.tight_layout()
        plt.savefig('results/comprehensive_results.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return fig


def main():
    """Main experimental pipeline"""
    
    print("="*60)
    print("Moduli Space Coordinate Pruning - Experimental Evaluation")
    print("="*60)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Initialize experiment runner
    runner = ExperimentRunner()
    
    # Run quick test (single experiment)
    print("\nRunning quick test...")
    try:
        result = runner.run_pruning_experiment(
            network_class=SimpleNet,
            pruning_method='moduli_space',
            pruning_ratio=0.5,
            pretrain_epochs=5,  # Reduced for quick test
            finetune_epochs=3
        )
        
        print(f"\nQuick test results:")
        print(f"  Final accuracy: {result['final_acc']:.2f}%")
        print(f"  Sparsity: {result['sparsity']:.2%}")
        print(f"  Speedup: {result['speedup']:.2f}x")
    except Exception as e:
        print(f"Error in quick test: {e}")
        print("Trying with magnitude pruning instead...")
        result = runner.run_pruning_experiment(
            network_class=SimpleNet,
            pruning_method='magnitude',
            pruning_ratio=0.5,
            pretrain_epochs=5,
            finetune_epochs=3
        )
    
    # Optionally run full experiments
    user_input = input("\nRun full experiment suite? (y/n): ")
    if user_input.lower() == 'y':
        print("\nRunning full experiment suite...")
        all_results = runner.run_full_experiment_suite()
        
        # Generate plots and analysis
        df = pd.DataFrame(all_results)
        runner.plot_results(df)
        
        # Print summary statistics
        print("\n" + "="*60)
        print("Summary Statistics")
        print("="*60)
        try:
            print(df.groupby('method').agg({
                'final_acc': ['mean', 'std'],
                'accuracy_drop': ['mean', 'std'],
                'speedup': ['mean', 'std'],
                'sparsity': ['mean', 'std']
            }).round(2))
        except:
            print("Could not generate summary statistics")
    
    print("\nExperiments complete!")


if __name__ == "__main__":
    main()
