"""
Visualization for Pruning Experiments
======================================

Creates plots for accuracy vs sparsity, moduli space analysis, etc.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100


class ExperimentVisualizer:
    """Create visualizations for pruning experiments"""

    def __init__(self, output_dir: str = "experiments/results/plots"):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_accuracy_vs_sparsity(
        self,
        results: Dict[str, List[tuple]],
        save_name: str = "accuracy_vs_sparsity.png",
        title: str = "Accuracy vs Sparsity"
    ):
        """
        Plot accuracy vs sparsity for multiple methods.

        Args:
            results: Dict mapping method name to list of (sparsity, accuracy) tuples
            save_name: Filename to save plot
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        colors = plt.cm.Set2(np.linspace(0, 1, len(results)))

        for (method, data), color in zip(results.items(), colors):
            if not data:
                continue

            sparsities = [d[0] for d in data]
            accuracies = [d[1] for d in data]

            # Sort by sparsity
            sorted_data = sorted(zip(sparsities, accuracies))
            sparsities = [d[0] for d in sorted_data]
            accuracies = [d[1] for d in sorted_data]

            ax.plot(sparsities, accuracies, marker='o', label=method,
                   linewidth=2, markersize=8, color=color)

        ax.set_xlabel('Sparsity (%)', fontsize=12)
        ax.set_ylabel('Test Accuracy (%)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ Saved: {self.output_dir / save_name}")

    def plot_memory_reduction(
        self,
        results: Dict[str, List[tuple]],
        save_name: str = "memory_reduction.png"
    ):
        """
        Plot memory footprint reduction.

        Args:
            results: Dict mapping method to list of (sparsity, memory_mb) tuples
            save_name: Filename to save plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        colors = plt.cm.Set2(np.linspace(0, 1, len(results)))

        for (method, data), color in zip(results.items(), colors):
            if not data:
                continue

            sparsities = [d[0] * 100 for d in data]  # Convert to percentage
            memory = [d[1] for d in data]

            ax.plot(sparsities, memory, marker='s', label=method,
                   linewidth=2, markersize=8, color=color)

        ax.set_xlabel('Sparsity (%)', fontsize=12)
        ax.set_ylabel('Model Size (MB)', fontsize=12)
        ax.set_title('Memory Footprint vs Sparsity', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ Saved: {self.output_dir / save_name}")

    def plot_moduli_dimension_vs_architecture(
        self,
        architectures: List[str],
        theoretical_dims: List[int],
        actual_dims: List[int],
        save_name: str = "moduli_dimension_vs_architecture.png"
    ):
        """
        Plot theoretical vs actual moduli space dimensions.

        Args:
            architectures: List of architecture names
            theoretical_dims: Theoretical dimensions (#E - #V)
            actual_dims: Actual analyzed dimensions
            save_name: Filename to save
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(architectures))
        width = 0.35

        bars1 = ax.bar(x - width/2, theoretical_dims, width,
                      label='Theoretical (#E - #Ṽ)', color='steelblue')
        bars2 = ax.bar(x + width/2, actual_dims, width,
                      label='Actual (Analyzed)', color='coral')

        ax.set_xlabel('Architecture', fontsize=12)
        ax.set_ylabel('Moduli Space Dimension', fontsize=12)
        ax.set_title('Theoretical vs Actual Moduli Dimension', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(architectures, rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height):,}',
                       ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ Saved: {self.output_dir / save_name}")

    def plot_dimension_vs_prunable(
        self,
        moduli_dims: List[int],
        prunable_params: List[int],
        labels: Optional[List[str]] = None,
        save_name: str = "dimension_vs_prunable.png"
    ):
        """
        Plot correlation between moduli dimension and prunable parameters.

        Args:
            moduli_dims: List of moduli dimensions
            prunable_params: List of prunable parameter counts
            labels: Optional labels for each point
            save_name: Filename to save
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Scatter plot
        ax.scatter(moduli_dims, prunable_params, s=100, alpha=0.6, color='steelblue')

        # Add labels if provided
        if labels:
            for i, label in enumerate(labels):
                ax.annotate(label, (moduli_dims[i], prunable_params[i]),
                          xytext=(5, 5), textcoords='offset points',
                          fontsize=9, alpha=0.7)

        # Fit and plot trend line
        if len(moduli_dims) > 1:
            z = np.polyfit(moduli_dims, prunable_params, 1)
            p = np.poly1d(z)
            ax.plot(moduli_dims, p(moduli_dims), "r--", alpha=0.5,
                   label=f'Trend: y = {z[0]:.2f}x + {z[1]:.2f}')

        # Calculate correlation
        if len(moduli_dims) > 1:
            corr = np.corrcoef(moduli_dims, prunable_params)[0, 1]
            ax.text(0.05, 0.95, f'Correlation: {corr:.3f}',
                   transform=ax.transAxes, fontsize=11,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_xlabel('Moduli Space Dimension', fontsize=12)
        ax.set_ylabel('Prunable Parameters', fontsize=12)
        ax.set_title('Moduli Dimension vs Prunable Parameters', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ Saved: {self.output_dir / save_name}")

    def plot_ablation_study(
        self,
        components: List[str],
        accuracies: Dict[str, List[float]],
        sparsities: List[float],
        save_name: str = "ablation_study.png"
    ):
        """
        Plot ablation study results for moduli space components.

        Args:
            components: List of component names (e.g., ['usage_only', 'variance_only', 'combined'])
            accuracies: Dict mapping component to list of accuracies at different sparsities
            sparsities: List of sparsity levels
            save_name: Filename to save
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        colors = plt.cm.Set2(np.linspace(0, 1, len(components)))
        markers = ['o', 's', '^', 'D', 'v']

        for i, (component, color) in enumerate(zip(components, colors)):
            if component in accuracies:
                marker = markers[i % len(markers)]
                ax.plot(sparsities, accuracies[component],
                       marker=marker, label=component.replace('_', ' ').title(),
                       linewidth=2, markersize=8, color=color)

        ax.set_xlabel('Sparsity (%)', fontsize=12)
        ax.set_ylabel('Test Accuracy (%)', fontsize=12)
        ax.set_title('Moduli Space Component Ablation Study', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ Saved: {self.output_dir / save_name}")

    def create_summary_table(
        self,
        results: Dict[str, Dict],
        save_name: str = "summary_table.png"
    ):
        """
        Create a summary table comparing all methods.

        Args:
            results: Nested dict with method -> sparsity -> metrics
            save_name: Filename to save
        """
        # Extract data for table
        methods = list(results.keys())
        sparsities = sorted(set(
            sp for method_results in results.values()
            for sp in method_results.keys()
        ))

        # Create table data
        table_data = []
        for method in methods:
            row = [method]
            for sp in sparsities:
                if sp in results[method]:
                    acc = results[method][sp].get('accuracy', 0)
                    row.append(f"{acc:.2f}%")
                else:
                    row.append("-")
            table_data.append(row)

        # Create figure
        fig, ax = plt.subplots(figsize=(12, len(methods) * 0.5 + 1))
        ax.axis('tight')
        ax.axis('off')

        # Create table
        col_labels = ['Method'] + [f'{sp*100:.0f}% Sparsity' for sp in sparsities]
        table = ax.table(cellText=table_data, colLabels=col_labels,
                        cellLoc='center', loc='center',
                        bbox=[0, 0, 1, 1])

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Style header
        for i in range(len(col_labels)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Style cells
        for i in range(1, len(table_data) + 1):
            for j in range(len(col_labels)):
                if j == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
                    table[(i, j)].set_text_props(weight='bold')

        plt.title('Pruning Methods Comparison', fontsize=14,
                 fontweight='bold', pad=20)
        plt.savefig(self.output_dir / save_name, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ Saved: {self.output_dir / save_name}")


if __name__ == '__main__':
    # Example usage
    viz = ExperimentVisualizer()

    # Example data
    results = {
        'Moduli Space': [(0.1, 92.5), (0.3, 91.2), (0.5, 88.7), (0.7, 84.3)],
        'Magnitude': [(0.1, 91.8), (0.3, 89.5), (0.5, 85.2), (0.7, 78.1)],
        'Random': [(0.1, 90.2), (0.3, 85.7), (0.5, 78.4), (0.7, 68.9)],
    }

    viz.plot_accuracy_vs_sparsity(results)
    print("✓ Example plot created")
