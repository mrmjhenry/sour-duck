"""
Moduli Space Analysis and Visualization
Interactive analysis of the moduli space properties and pruning behavior
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from pathlib import Path

from moduli_space_pruning import (
    ModuliSpaceAnalyzer, DataRepresentation, QuiverRepresentation
)

class ModuliSpaceVisualizer:
    """Advanced visualization of moduli space properties"""
    
    def __init__(self, analyzer: ModuliSpaceAnalyzer):
        self.analyzer = analyzer
        self.coordinate_embeddings = None
        
    def extract_coordinate_embeddings(self, dataloader, max_samples=100):
        """Extract coordinate embeddings for visualization"""
        
        print("Extracting coordinate embeddings...")
        embeddings = []
        labels = []
        
        samples_processed = 0
        for batch_idx, (data, target) in enumerate(dataloader):
            if samples_processed >= max_samples:
                break
                
            data = data.to(self.analyzer.device)
            batch_size = data.size(0)
            
            for i in range(min(batch_size, max_samples - samples_processed)):
                x = data[i].unsqueeze(0)
                representation = self.analyzer.data_rep.compute_representation(x)
                
                # Flatten representation to create embedding vector
                embedding = []
                for layer_name, rep_tensor in representation.items():
                    embedding.extend(rep_tensor.flatten().cpu().numpy())
                
                embeddings.append(embedding)
                labels.append(target[i].item())
                samples_processed += 1
        
        # Pad embeddings to same length if needed
        max_len = max(len(e) for e in embeddings)
        embeddings = [e + [0] * (max_len - len(e)) for e in embeddings]
        
        self.coordinate_embeddings = np.array(embeddings)
        self.embedding_labels = np.array(labels)
        
        return self.coordinate_embeddings, self.embedding_labels
    
    def visualize_moduli_space_2d(self, method='pca'):
        """Visualize moduli space in 2D using dimensionality reduction"""
        
        if self.coordinate_embeddings is None:
            raise ValueError("Must call extract_coordinate_embeddings first")
        
        # Apply dimensionality reduction
        if method == 'pca':
            reducer = PCA(n_components=2)
            embeddings_2d = reducer.fit_transform(self.coordinate_embeddings)
            title = "Moduli Space Projection (PCA)"
            explained_var = reducer.explained_variance_ratio_
            subtitle = f"Explained variance: {explained_var[0]:.2%}, {explained_var[1]:.2%}"
        elif method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
            embeddings_2d = reducer.fit_transform(self.coordinate_embeddings)
            title = "Moduli Space Projection (t-SNE)"
            subtitle = ""
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        scatter = ax.scatter(
            embeddings_2d[:, 0], embeddings_2d[:, 1],
            c=self.embedding_labels, cmap='tab10',
            s=50, alpha=0.7, edgecolors='black', linewidth=0.5
        )
        
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_title(title)
        if subtitle:
            ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, 
                   ha='center', fontsize=10)
        
        plt.colorbar(scatter, ax=ax, label='Class')
        plt.tight_layout()
        
        return fig
    
    def visualize_coordinate_importance_3d(self):
        """3D visualization of coordinate importance"""
        
        if not self.analyzer.coordinate_stats:
            raise ValueError("Must run analysis first")
        
        coords = list(self.analyzer.coordinate_stats.values())
        
        # Create 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=[c.usage_rate for c in coords],
            y=[np.log(c.variance + 1e-8) for c in coords],
            z=[c.information_content for c in coords],
            mode='markers',
            marker=dict(
                size=5,
                color=[c.importance_score for c in coords],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Importance Score"),
                opacity=0.8
            ),
            text=[f"Edge {c.edge_idx}" for c in coords],
            hovertemplate='<b>Edge %{text}</b><br>' +
                         'Usage: %{x:.3f}<br>' +
                         'Log Variance: %{y:.3f}<br>' +
                         'Information: %{z:.3f}<br>' +
                         '<extra></extra>'
        )])
        
        fig.update_layout(
            title="3D Coordinate Importance Space",
            scene=dict(
                xaxis_title="Usage Rate",
                yaxis_title="Log Variance",
                zaxis_title="Information Content"
            ),
            width=900,
            height=700
        )
        
        return fig
    
    def analyze_layer_wise_pruning(self):
        """Analyze pruning distribution across layers"""
        
        if not self.analyzer.coordinate_stats:
            raise ValueError("Must run analysis first")
        
        # Group coordinates by layer
        layer_stats = {}
        for coord in self.analyzer.coordinate_stats.values():
            layer_idx = coord.layer_idx
            if layer_idx not in layer_stats:
                layer_stats[layer_idx] = []
            layer_stats[layer_idx].append(coord)
        
        # Compute statistics per layer
        layer_data = []
        for layer_idx, coords in layer_stats.items():
            layer_data.append({
                'layer': layer_idx,
                'num_coords': len(coords),
                'avg_usage': np.mean([c.usage_rate for c in coords]),
                'avg_variance': np.mean([c.variance for c in coords]),
                'avg_importance': np.mean([c.importance_score for c in coords]),
                'prunable_ratio': np.mean([c.importance_score < 0.5 for c in coords])
            })
        
        df = pd.DataFrame(layer_data)
        
        # Create visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Average Usage Rate', 'Average Variance',
                          'Average Importance', 'Prunable Ratio')
        )
        
        # Usage rate
        fig.add_trace(
            go.Bar(x=df['layer'], y=df['avg_usage'], name='Usage'),
            row=1, col=1
        )
        
        # Variance
        fig.add_trace(
            go.Bar(x=df['layer'], y=df['avg_variance'], name='Variance'),
            row=1, col=2
        )
        
        # Importance
        fig.add_trace(
            go.Bar(x=df['layer'], y=df['avg_importance'], name='Importance'),
            row=2, col=1
        )
        
        # Prunable ratio
        fig.add_trace(
            go.Bar(x=df['layer'], y=df['prunable_ratio'], name='Prunable'),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="Layer-wise Pruning Analysis",
            showlegend=False,
            height=600
        )
        
        return fig, df
    
    def compare_pruning_strategies(self, pruning_ratios=[0.3, 0.5, 0.7]):
        """Compare different pruning strategies"""
        
        if not self.analyzer.coordinate_stats:
            raise ValueError("Must run analysis first")
        
        strategies = ['usage', 'variance', 'information', 'combined']
        results = []
        
        for ratio in pruning_ratios:
            for strategy in strategies:
                prunable = self.analyzer.get_prunable_coordinates(ratio, strategy)
                
                # Compute statistics for pruned coordinates
                results.append({
                    'pruning_ratio': ratio,
                    'strategy': strategy,
                    'num_pruned': len(prunable),
                    'avg_importance_pruned': np.mean([c.importance_score for c in prunable]),
                    'total_usage_pruned': np.sum([c.usage_rate for c in prunable]),
                })
        
        df = pd.DataFrame(results)
        
        # Create comparison plot
        fig = px.line(df, x='pruning_ratio', y='avg_importance_pruned',
                     color='strategy', markers=True,
                     title='Average Importance of Pruned Coordinates by Strategy',
                     labels={
                         'pruning_ratio': 'Pruning Ratio',
                         'avg_importance_pruned': 'Average Importance Score',
                         'strategy': 'Strategy'
                     })
        
        fig.update_layout(
            xaxis_tickformat='.0%',
            yaxis_tickformat='.3f',
            height=500
        )
        
        return fig, df


class TheoreticalAnalysis:
    """Theoretical analysis of moduli space properties"""
    
    @staticmethod
    def compute_moduli_dimension(network):
        """
        Compute theoretical moduli space dimension: #E - #Ṽ
        """
        num_edges = 0
        num_hidden_vertices = 0
        
        for name, module in network.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # Count edges (weights)
                num_edges += module.weight.data.numel()
                
                # Count hidden vertices (output neurons)
                if isinstance(module, nn.Linear):
                    num_hidden_vertices += module.out_features
                elif isinstance(module, nn.Conv2d):
                    # Simplified: count output channels
                    num_hidden_vertices += module.out_channels
        
        moduli_dimension = num_edges - num_hidden_vertices
        
        return {
            'num_edges': num_edges,
            'num_hidden_vertices': num_hidden_vertices,
            'moduli_dimension': moduli_dimension,
            'dimension_ratio': moduli_dimension / num_edges if num_edges > 0 else 0
        }
    
    @staticmethod
    def analyze_isomorphism_classes(network, dataloader, max_samples=50):
        """
        Analyze the distribution of isomorphism classes in the data
        """
        data_rep = DataRepresentation(network)
        
        # Collect representations
        representations = []
        for batch_idx, (data, _) in enumerate(dataloader):
            if len(representations) >= max_samples:
                break
            
            for i in range(min(data.size(0), max_samples - len(representations))):
                x = data[i].unsqueeze(0)
                rep = data_rep.compute_representation(x)
                
                # Convert to vector for comparison
                rep_vector = []
                for layer_rep in rep.values():
                    rep_vector.extend(layer_rep.flatten().cpu().numpy())
                representations.append(rep_vector)
        
        # Pad to same length
        max_len = max(len(r) for r in representations)
        representations = [r + [0] * (max_len - len(r)) for r in representations]
        representations = np.array(representations)
        
        # Compute pairwise distances (proxy for isomorphism classes)
        from scipy.spatial.distance import pdist, squareform
        distances = squareform(pdist(representations, metric='euclidean'))
        
        # Estimate number of distinct isomorphism classes
        # (using clustering as proxy)
        from sklearn.cluster import DBSCAN
        clustering = DBSCAN(eps=0.5, min_samples=2).fit(representations)
        num_classes = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
        
        data_rep.cleanup()
        
        return {
            'num_samples': len(representations),
            'estimated_isomorphism_classes': num_classes,
            'avg_distance': np.mean(distances),
            'std_distance': np.std(distances),
            'distance_matrix': distances
        }
    
    @staticmethod
    def compute_capacity_measures(network):
        """
        Compute various capacity measures related to moduli space
        """
        total_params = sum(p.numel() for p in network.parameters())
        trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
        
        # Compute layer-wise capacity
        layer_capacity = []
        for name, module in network.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if isinstance(module, nn.Linear):
                    capacity = module.in_features * module.out_features
                else:
                    capacity = (module.in_channels * module.kernel_size[0] * 
                              module.kernel_size[1] * module.out_channels)
                
                layer_capacity.append({
                    'layer': name,
                    'type': module.__class__.__name__,
                    'capacity': capacity,
                    'parameters': module.weight.numel()
                })
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'layer_capacity': layer_capacity,
            'average_layer_capacity': np.mean([l['capacity'] for l in layer_capacity])
        }


def run_analysis_demo():
    """Run demonstration analysis"""
    
    print("="*60)
    print("Moduli Space Analysis and Visualization")
    print("="*60)
    
    # Create simple network for analysis
    from experiments import SimpleNet
    network = SimpleNet()
    
    # Compute theoretical properties
    print("\nTheoretical Analysis:")
    print("-" * 40)
    
    theory = TheoreticalAnalysis()
    
    # Moduli dimension
    dim_info = theory.compute_moduli_dimension(network)
    print(f"Number of edges (weights): {dim_info['num_edges']:,}")
    print(f"Number of hidden vertices: {dim_info['num_hidden_vertices']:,}")
    print(f"Moduli space dimension: {dim_info['moduli_dimension']:,}")
    print(f"Dimension ratio: {dim_info['dimension_ratio']:.2%}")
    
    # Capacity measures
    capacity = theory.compute_capacity_measures(network)
    print(f"\nTotal parameters: {capacity['total_parameters']:,}")
    print(f"Average layer capacity: {capacity['average_layer_capacity']:.0f}")
    
    # Prepare sample data
    from experiments import ExperimentRunner
    runner = ExperimentRunner()
    trainloader, _ = runner.prepare_data('CIFAR10', batch_size=32)
    
    # Create analyzer and run analysis
    print("\nRunning moduli space analysis...")
    analyzer = ModuliSpaceAnalyzer(network)
    analyzer.analyze_dataset(trainloader, max_samples=100)
    
    # Create visualizer
    visualizer = ModuliSpaceVisualizer(analyzer)
    
    # Extract embeddings
    print("Extracting coordinate embeddings...")
    embeddings, labels = visualizer.extract_coordinate_embeddings(trainloader, max_samples=50)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # 2D moduli space
    fig_pca = visualizer.visualize_moduli_space_2d('pca')
    fig_pca.savefig('results/moduli_space_pca.png', dpi=150, bbox_inches='tight')
    
    # 3D coordinate importance
    fig_3d = visualizer.visualize_coordinate_importance_3d()
    fig_3d.write_html('results/coordinate_importance_3d.html')
    
    # Layer-wise analysis
    fig_layers, df_layers = visualizer.analyze_layer_wise_pruning()
    fig_layers.write_html('results/layer_wise_pruning.html')
    
    # Strategy comparison
    fig_strategies, df_strategies = visualizer.compare_pruning_strategies()
    fig_strategies.write_html('results/strategy_comparison.html')
    
    # Isomorphism class analysis
    print("\nAnalyzing isomorphism classes...")
    iso_analysis = theory.analyze_isomorphism_classes(network, trainloader, max_samples=30)
    print(f"Estimated isomorphism classes: {iso_analysis['estimated_isomorphism_classes']}")
    print(f"Average representation distance: {iso_analysis['avg_distance']:.2f}")
    
    # Generate summary report
    generate_report(dim_info, capacity, analyzer.coordinate_stats, iso_analysis)
    
    print("\nAnalysis complete! Check the 'results' directory for outputs.")


def generate_report(dim_info, capacity, coordinate_stats, iso_analysis):
    """Generate analysis report"""
    
    report = f"""
    Moduli Space Analysis Report
    ============================
    
    Network Architecture Analysis
    -----------------------------
    Total Parameters: {capacity['total_parameters']:,}
    Number of Edges: {dim_info['num_edges']:,}
    Number of Hidden Vertices: {dim_info['num_hidden_vertices']:,}
    
    Moduli Space Properties
    -----------------------
    Dimension: {dim_info['moduli_dimension']:,}
    Dimension Ratio: {dim_info['dimension_ratio']:.2%}
    Estimated Isomorphism Classes: {iso_analysis['estimated_isomorphism_classes']}
    
    Coordinate Statistics
    --------------------
    Total Coordinates Analyzed: {len(coordinate_stats)}
    Average Usage Rate: {np.mean([c.usage_rate for c in coordinate_stats.values()]):.3f}
    Average Variance: {np.mean([c.variance for c in coordinate_stats.values()]):.6f}
    Average Information Content: {np.mean([c.information_content for c in coordinate_stats.values()]):.3f}
    
    Pruning Potential
    ----------------
    Coordinates with <10% usage: {sum(c.usage_rate < 0.1 for c in coordinate_stats.values())}
    Coordinates with <50% usage: {sum(c.usage_rate < 0.5 for c in coordinate_stats.values())}
    Estimated prunable ratio: {np.mean([c.importance_score < 0.5 for c in coordinate_stats.values()]):.2%}
    
    Data Representation Analysis
    ---------------------------
    Average inter-representation distance: {iso_analysis['avg_distance']:.2f}
    Distance standard deviation: {iso_analysis['std_distance']:.2f}
    
    Theoretical Insights
    -------------------
    The moduli space dimension of {dim_info['moduli_dimension']:,} represents the 
    degrees of freedom in the network's representation space after accounting for
    the action of the change of basis group G̃.
    
    The presence of {iso_analysis['estimated_isomorphism_classes']} distinct 
    isomorphism classes suggests that the data manifold occupies a relatively
    small portion of the available moduli space, indicating potential for pruning.
    """
    
    with open('results/analysis_report.txt', 'w') as f:
        f.write(report)
    
    print("Report saved to results/analysis_report.txt")


if __name__ == "__main__":
    # Create results directory
    Path('results').mkdir(exist_ok=True)
    
    # Run analysis
    run_analysis_demo()
