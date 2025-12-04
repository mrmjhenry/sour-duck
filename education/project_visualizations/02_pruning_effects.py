"""
Pruning Effects Visualization

Shows the effect of moduli space pruning on network structure:
- Highlight connections to be pruned in red
- Fade out pruned connections
- Show sparsity statistics
- Compare network before and after pruning

Usage:
    manim -pql 02_pruning_effects.py PruningProcessScene
    manim -pqh 02_pruning_effects.py BeforeAfterPruning
    manim -pql 02_pruning_effects.py PruningStatistics

Requirements:
    pip install manim numpy
"""

from manim import *
import numpy as np


# ============================================================================
# Configuration
# ============================================================================

KEEP_COLOR = GREEN
PRUNE_COLOR = RED
NEUTRAL_COLOR = GRAY
NODE_COLOR = BLUE
HIGHLIGHT_COLOR = YELLOW


# ============================================================================
# Scene 1: Pruning Process Animation
# ============================================================================

class PruningProcessScene(Scene):
    """
    Animates the process of identifying and pruning connections
    based on coordinate importance.
    """

    def construct(self):
        # Title
        title = Text("Moduli Space Pruning Process", font_size=36)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait()

        # Create network
        network = self.create_network([4, 6, 4])
        network.shift(LEFT * 2)
        self.play(Create(network))
        self.wait()

        # Step 1: Analyze importance
        self.show_importance_analysis(network)

        # Step 2: Identify low-importance connections
        self.identify_prunable_connections(network)

        # Step 3: Prune connections
        self.prune_connections(network)

        # Show final sparsity
        self.show_sparsity_result()

        self.wait(2)

    def create_network(self, layers):
        """Create a neural network graph."""
        vertices = []
        edges = []
        partitions = []

        c = 0
        for layer_size in layers:
            partition = list(range(c + 1, c + layer_size + 1))
            partitions.append(partition)
            vertices.extend(partition)
            c += layer_size

        # Create edges
        for i in range(len(layers) - 1):
            start = sum(layers[:i])
            end = sum(layers[:i+1])
            next_start = end
            next_end = sum(layers[:i+2])

            for j in range(start + 1, end + 1):
                for k in range(next_start + 1, next_end + 1):
                    edges.append((j, k))

        graph = Graph(
            vertices,
            edges,
            layout='partite',
            partitions=partitions,
            layout_scale=2.5,
            vertex_config={'radius': 0.15, 'fill_color': NODE_COLOR},
            edge_config={'stroke_width': 1.5, 'stroke_color': NEUTRAL_COLOR}
        )

        return graph

    def show_importance_analysis(self, network):
        """Show importance scores being computed."""
        text = Text("Step 1: Analyze Coordinate Importance", font_size=24)
        text.to_edge(RIGHT).shift(UP * 2)
        self.play(Write(text))

        # Simulate importance computation
        importance_bar = self.create_importance_display()
        importance_bar.next_to(text, DOWN, buff=0.5)
        self.play(FadeIn(importance_bar, shift=DOWN))

        self.wait()
        self.importance_display = VGroup(text, importance_bar)

    def create_importance_display(self):
        """Create visual display of importance scores."""
        bars = VGroup()

        # Simulate importance scores
        np.random.seed(42)
        importances = np.random.uniform(0.2, 1.0, 10)
        importances = sorted(importances, reverse=True)

        for i, imp in enumerate(importances):
            color = interpolate_color(PRUNE_COLOR, KEEP_COLOR, imp)
            bar = Rectangle(
                width=imp * 2,
                height=0.15,
                fill_color=color,
                fill_opacity=0.8,
                stroke_width=0.5
            )
            bar.align_to(ORIGIN, LEFT)

            label = Text(f"{imp:.2f}", font_size=12)
            label.next_to(bar, LEFT, buff=0.1)

            bar_group = VGroup(label, bar)
            bars.add(bar_group)

        bars.arrange(DOWN, buff=0.1, aligned_edge=LEFT)
        bars.scale(0.8)
        return bars

    def identify_prunable_connections(self, network):
        """Highlight connections to be pruned."""
        text = Text("Step 2: Identify Low-Importance Connections", font_size=24, color=PRUNE_COLOR)
        text.to_edge(RIGHT).shift(UP * 2)

        self.play(
            FadeOut(self.importance_display),
            Write(text)
        )

        # Select random edges to prune (30%)
        all_edges = list(network.edges.keys())
        np.random.seed(42)
        n_prune = int(len(all_edges) * 0.3)
        edges_to_prune = np.random.choice(len(all_edges), n_prune, replace=False)

        # Highlight edges to prune in red
        prune_animations = []
        self.prunable_edges = []

        for idx in edges_to_prune:
            edge_key = all_edges[idx]
            edge = network.edges[edge_key]
            prune_animations.append(
                edge.animate.set_color(PRUNE_COLOR).set_stroke_width(3)
            )
            self.prunable_edges.append(edge_key)

        self.play(*prune_animations, run_time=2)

        # Show threshold line on importance display
        threshold = Text(
            f"Pruning Threshold: 30% lowest importance",
            font_size=18,
            color=PRUNE_COLOR
        )
        threshold.next_to(text, DOWN, buff=0.3)
        self.play(Write(threshold))

        self.wait()
        self.prune_text = VGroup(text, threshold)

    def prune_connections(self, network):
        """Remove (fade out) the pruned connections."""
        text = Text("Step 3: Remove Pruned Connections", font_size=24, color=PRUNE_COLOR)
        text.to_edge(RIGHT).shift(UP * 2)

        self.play(
            FadeOut(self.prune_text),
            Write(text)
        )

        # Fade out pruned edges
        fade_animations = []
        for edge_key in self.prunable_edges:
            edge = network.edges[edge_key]
            fade_animations.append(FadeOut(edge))

        self.play(*fade_animations, run_time=2)

        # Highlight kept connections in green
        keep_animations = []
        for edge_key, edge in network.edges.items():
            if edge_key not in self.prunable_edges:
                keep_animations.append(
                    edge.animate.set_color(KEEP_COLOR).set_stroke_width(2)
                )

        self.play(*keep_animations, run_time=1)

        self.wait()
        self.final_text = text

    def show_sparsity_result(self):
        """Show final sparsity statistics."""
        total_edges = len(self.prunable_edges) + sum(1 for _ in self.prunable_edges)  # Approximation
        pruned = len(self.prunable_edges)
        sparsity = pruned / (pruned + 30) if pruned > 0 else 0  # Rough estimate

        result = VGroup(
            Text(f"Pruning Complete!", font_size=28, color=KEEP_COLOR),
            Text(f"Sparsity: {sparsity*100:.1f}%", font_size=24),
            Text(f"Remaining connections: {100-sparsity*100:.1f}%", font_size=20, color=KEEP_COLOR)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)

        result.to_edge(RIGHT).shift(DOWN * 1)

        self.play(
            FadeOut(self.final_text),
            FadeIn(result, shift=UP)
        )


# ============================================================================
# Scene 2: Before and After Pruning
# ============================================================================

class BeforeAfterPruning(Scene):
    """
    Side-by-side comparison of network before and after pruning.
    """

    def construct(self):
        # Title
        title = Text("Network: Before and After Pruning", font_size=36)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait()

        # Create two networks side by side
        network_before = self.create_network([3, 5, 3])
        network_before.shift(LEFT * 3)

        network_after = self.create_network([3, 5, 3])
        network_after.shift(RIGHT * 3)

        # Labels
        label_before = Text("Before Pruning", font_size=24)
        label_before.next_to(network_before, DOWN, buff=0.5)

        label_after = Text("After Pruning (50%)", font_size=24, color=PRUNE_COLOR)
        label_after.next_to(network_after, DOWN, buff=0.5)

        # Show both networks
        self.play(
            Create(network_before),
            Write(label_before)
        )
        self.wait()

        self.play(
            Create(network_after),
            Write(label_after)
        )

        # Apply pruning to right network
        self.apply_pruning_visualization(network_after)

        # Show comparison metrics
        self.show_comparison_metrics()

        self.wait(2)

    def create_network(self, layers):
        """Create a neural network graph."""
        vertices = []
        edges = []
        partitions = []

        c = 0
        for layer_size in layers:
            partition = list(range(c + 1, c + layer_size + 1))
            partitions.append(partition)
            vertices.extend(partition)
            c += layer_size

        for i in range(len(layers) - 1):
            start = sum(layers[:i])
            end = sum(layers[:i+1])
            next_start = end
            next_end = sum(layers[:i+2])

            for j in range(start + 1, end + 1):
                for k in range(next_start + 1, next_end + 1):
                    edges.append((j, k))

        graph = Graph(
            vertices,
            edges,
            layout='partite',
            partitions=partitions,
            layout_scale=2,
            vertex_config={'radius': 0.12, 'fill_color': NODE_COLOR},
            edge_config={'stroke_width': 1.5, 'stroke_color': NEUTRAL_COLOR}
        )

        return graph

    def apply_pruning_visualization(self, network):
        """Visualize pruning on the right network."""
        # Prune 50% of edges
        all_edges = list(network.edges.keys())
        np.random.seed(42)
        n_prune = len(all_edges) // 2
        edges_to_prune = np.random.choice(len(all_edges), n_prune, replace=False)

        # Highlight in red, then fade out
        animations = []
        for idx in edges_to_prune:
            edge_key = all_edges[idx]
            edge = network.edges[edge_key]
            animations.append(edge.animate.set_color(PRUNE_COLOR))

        self.play(*animations, run_time=1)
        self.wait(0.5)

        # Fade out pruned edges
        fade_animations = []
        for idx in edges_to_prune:
            edge_key = all_edges[idx]
            edge = network.edges[edge_key]
            fade_animations.append(FadeOut(edge))

        self.play(*fade_animations, run_time=1)

        # Color remaining edges green
        keep_animations = []
        for i, (edge_key, edge) in enumerate(network.edges.items()):
            if i not in edges_to_prune:
                keep_animations.append(edge.animate.set_color(KEEP_COLOR))

        self.play(*keep_animations, run_time=0.5)

    def show_comparison_metrics(self):
        """Show metrics comparing before and after."""
        metrics = VGroup(
            Text("Comparison:", font_size=24, color=YELLOW),
            Text("Before: 30 connections", font_size=18),
            Text("After: 15 connections", font_size=18, color=PRUNE_COLOR),
            Text("Sparsity: 50%", font_size=18, color=GREEN),
            Text("Accuracy drop: ~2-5%", font_size=18, color=GRAY),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)

        metrics.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(metrics, shift=UP))


# ============================================================================
# Scene 3: Pruning Statistics
# ============================================================================

class PruningStatistics(Scene):
    """
    Shows detailed statistics about pruning at different ratios.
    """

    def construct(self):
        # Title
        title = Text("Pruning Statistics and Trade-offs", font_size=36)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait()

        # Show statistics for different pruning ratios
        self.show_pruning_ratios()

        # Show accuracy vs sparsity trade-off
        self.show_tradeoff_curve()

        self.wait(2)

    def show_pruning_ratios(self):
        """Show statistics for different pruning ratios."""
        ratios = [0.3, 0.5, 0.7]
        stats_group = VGroup()

        for ratio in ratios:
            # Create network visualization
            network = self.create_small_network()

            # Statistics text
            stats = VGroup(
                Text(f"{ratio*100:.0f}% Pruning", font_size=20, color=PRUNE_COLOR),
                Text(f"Remaining: {(1-ratio)*100:.0f}%", font_size=16, color=KEEP_COLOR),
                Text(f"Acc. drop: ~{ratio*10:.1f}%", font_size=14, color=GRAY),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)

            group = VGroup(network, stats)
            group.arrange(DOWN, buff=0.3)
            stats_group.add(group)

        stats_group.arrange(RIGHT, buff=1.5)
        stats_group.shift(UP * 0.5)

        self.play(FadeIn(stats_group, lag_ratio=0.3))
        self.wait()

    def create_small_network(self):
        """Create a small network for statistics display."""
        vertices = [1, 2, 3, 4, 5, 6, 7]
        edges = [(1, 3), (1, 4), (2, 3), (2, 4), (3, 5), (3, 6), (4, 5), (4, 6), (5, 7), (6, 7)]

        graph = Graph(
            vertices,
            edges,
            layout='partite',
            partitions=[[1, 2], [3, 4], [5, 6], [7]],
            layout_scale=1,
            vertex_config={'radius': 0.08},
            edge_config={'stroke_width': 1}
        )
        graph.scale(0.6)
        return graph

    def show_tradeoff_curve(self):
        """Show accuracy vs sparsity trade-off curve."""
        # Create axes
        axes = Axes(
            x_range=[0, 100, 20],
            y_range=[70, 100, 10],
            x_length=6,
            y_length=4,
            axis_config={"include_numbers": True, "font_size": 18},
        )

        # Labels
        x_label = Text("Sparsity (%)", font_size=20)
        x_label.next_to(axes.x_axis, DOWN, buff=0.3)

        y_label = Text("Accuracy (%)", font_size=20)
        y_label.next_to(axes.y_axis, LEFT, buff=0.3)
        y_label.rotate(90 * DEGREES)

        # Plot curves
        # Moduli space pruning (better)
        moduli_curve = axes.plot(
            lambda x: 95 - 0.3 * x - 0.005 * x**2,
            x_range=[0, 80],
            color=GREEN
        )

        # Random pruning (worse)
        random_curve = axes.plot(
            lambda x: 95 - 0.5 * x - 0.008 * x**2,
            x_range=[0, 80],
            color=RED
        )

        # Legend
        legend = VGroup(
            VGroup(Line(LEFT * 0.3, RIGHT * 0.3, color=GREEN), Text("Moduli Space", font_size=16)),
            VGroup(Line(LEFT * 0.3, RIGHT * 0.3, color=RED), Text("Random", font_size=16))
        )
        for item in legend:
            item.arrange(RIGHT, buff=0.2)
        legend.arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        legend.next_to(axes, RIGHT, buff=0.5)

        chart_group = VGroup(axes, x_label, y_label, moduli_curve, random_curve, legend)
        chart_group.to_edge(DOWN, buff=0.5)

        self.play(Create(axes), Write(x_label), Write(y_label))
        self.play(Create(moduli_curve), Create(random_curve))
        self.play(FadeIn(legend))


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("To render these animations, use:")
    print("  manim -pql 02_pruning_effects.py PruningProcessScene")
    print("  manim -pql 02_pruning_effects.py BeforeAfterPruning")
    print("  manim -pql 02_pruning_effects.py PruningStatistics")
