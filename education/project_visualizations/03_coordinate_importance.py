"""
Coordinate Importance Visualization
====================================

This module creates Manim animations demonstrating coordinate importance in the
moduli space pruning framework. It visualizes how importance scores are computed
and how they affect pruning decisions.

Scenes:
-------
1. ImportanceComputationScene: Shows the three components of importance
   - Usage rate (ρ): Fraction of samples activating this coordinate
   - Variance (σ²): Variability of coordinate values
   - Information content (I): Entropy-based measure
   - Combined score: Importance = ρ × √σ² × (1 + I)

2. NodeScalingScene: Scales node sizes by importance scores
   - Larger nodes = higher importance
   - Color gradient: Red (low) → Yellow (medium) → Green (high)
   - Shows which neurons are most critical

3. ImportanceEvolutionScene: Shows how importance changes during training
   - Animates importance scores evolving over epochs
   - Demonstrates that importance stabilizes after early training

Usage:
------
Render specific scenes:
    manim -pql 03_coordinate_importance.py ImportanceComputationScene
    manim -pql 03_coordinate_importance.py NodeScalingScene
    manim -pql 03_coordinate_importance.py ImportanceEvolutionScene

Render all scenes:
    manim -pql 03_coordinate_importance.py

High quality:
    manim -pqh 03_coordinate_importance.py
"""

from manim import *
import numpy as np

# Color scheme for importance visualization
HIGH_IMPORTANCE = GREEN
MEDIUM_IMPORTANCE = YELLOW
LOW_IMPORTANCE = RED
USAGE_COLOR = BLUE
VARIANCE_COLOR = PURPLE
INFO_COLOR = ORANGE

# Try to import ManimML for professional neural network visualization
try:
    from manim_ml.neural_network import NeuralNetwork, FeedForwardLayer
    HAS_MANIM_ML = True
except ImportError:
    HAS_MANIM_ML = False
    print("Warning: manim_ml not available. Using fallback visualization.")


class ImportanceComputationScene(Scene):
    """
    Demonstrates how coordinate importance is computed from three metrics.

    Shows the formula:
        Importance = ρ × √σ² × (1 + I)
    where:
        ρ = usage rate (fraction of samples activating coordinate)
        σ² = variance (variability of coordinate values)
        I = information content (entropy measure)
    """

    def construct(self):
        # Title
        title = Text("Coordinate Importance Computation", font_size=40)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)

        # Show the three components
        self.show_components()
        self.wait(1)

        # Show the formula
        self.show_formula()
        self.wait(1)

        # Demonstrate with example
        self.show_example()
        self.wait(2)

    def show_components(self):
        """Display the three components of importance"""

        # Component 1: Usage Rate
        usage_title = Text("1. Usage Rate (ρ)", font_size=28, color=USAGE_COLOR)
        usage_title.shift(UP * 2 + LEFT * 3.5)
        usage_desc = Text(
            "Fraction of data points\nactivating this coordinate",
            font_size=20, color=WHITE
        ).next_to(usage_title, DOWN, buff=0.3)

        # Component 2: Variance
        variance_title = Text("2. Variance (σ²)", font_size=28, color=VARIANCE_COLOR)
        variance_title.shift(UP * 2)
        variance_desc = Text(
            "Variability of\ncoordinate values",
            font_size=20, color=WHITE
        ).next_to(variance_title, DOWN, buff=0.3)

        # Component 3: Information Content
        info_title = Text("3. Information (I)", font_size=28, color=INFO_COLOR)
        info_title.shift(UP * 2 + RIGHT * 3.5)
        info_desc = Text(
            "Entropy-based\nimportance measure",
            font_size=20, color=WHITE
        ).next_to(info_title, DOWN, buff=0.3)

        self.play(
            Write(usage_title), Write(usage_desc),
            Write(variance_title), Write(variance_desc),
            Write(info_title), Write(info_desc),
            run_time=2
        )

        # Store for later reference
        self.components = VGroup(
            usage_title, usage_desc,
            variance_title, variance_desc,
            info_title, info_desc
        )

    def show_formula(self):
        """Display the combined importance formula"""

        # Fade out component descriptions
        self.play(FadeOut(self.components), run_time=0.5)

        # Main formula
        formula = MathTex(
            r"\text{Importance} = \rho \times \sqrt{\sigma^2} \times (1 + I)",
            font_size=36
        )
        formula.shift(UP * 1.5)

        # Color code the formula components
        formula[0][12:13].set_color(USAGE_COLOR)  # ρ
        formula[0][14:19].set_color(VARIANCE_COLOR)  # √σ²
        formula[0][20:21].set_color(INFO_COLOR)  # I

        self.play(Write(formula))

        # Explanation
        explanation = Text(
            "Combines activation frequency, value spread, and information content",
            font_size=22, color=GRAY
        ).next_to(formula, DOWN, buff=0.5)

        self.play(Write(explanation))

        # Store formula
        self.formula = VGroup(formula, explanation)

    def show_example(self):
        """Show a concrete example calculation"""

        # Move formula up
        self.play(self.formula.animate.shift(UP * 1), run_time=0.5)

        # Example values
        example_title = Text("Example Calculation:", font_size=28)
        example_title.shift(DOWN * 0.5)

        # Create a table of values
        values = [
            ["Metric", "Value"],
            ["ρ (usage)", "0.85"],
            ["σ² (variance)", "0.36"],
            ["I (information)", "0.42"],
        ]

        table = self.create_table(values)
        table.scale(0.8)
        table.next_to(example_title, DOWN, buff=0.5)
        table.shift(LEFT * 2)

        self.play(Write(example_title))
        self.play(Create(table), run_time=1.5)

        # Calculate importance
        calculation = MathTex(
            r"\text{Importance} &= 0.85 \times \sqrt{0.36} \times (1 + 0.42)\\",
            r"&= 0.85 \times 0.6 \times 1.42\\",
            r"&= 0.72",
            font_size=28
        )
        calculation.next_to(table, RIGHT, buff=1)

        self.play(Write(calculation[0]))
        self.wait(0.5)
        self.play(Write(calculation[1]))
        self.wait(0.5)
        self.play(Write(calculation[2]))

        # Highlight the result
        result_box = SurroundingRectangle(calculation[2], color=GREEN, buff=0.2)
        self.play(Create(result_box))

        # Interpretation
        interpretation = Text(
            "High importance (0.72) → Keep this coordinate",
            font_size=22, color=GREEN
        )
        interpretation.next_to(calculation, DOWN, buff=0.8)
        self.play(Write(interpretation))

    def create_table(self, data):
        """Create a simple table from 2D list"""
        rows = []
        for row_data in data:
            row = VGroup(*[
                Text(str(cell), font_size=22)
                for cell in row_data
            ])
            row.arrange(RIGHT, buff=1.5)
            rows.append(row)

        table = VGroup(*rows)
        table.arrange(DOWN, buff=0.3, aligned_edge=LEFT)

        # Add borders
        for row in rows:
            for cell in row:
                rect = SurroundingRectangle(cell, color=WHITE, buff=0.15)
                cell.add(rect)

        # Highlight header
        rows[0].set_color(BLUE)

        return table


class NodeScalingScene(Scene):
    """
    Visualizes a neural network with nodes scaled by importance scores.
    Larger nodes = more important, color gradient from red to green.
    """

    def construct(self):
        # Title
        title = Text("Node Importance Visualization", font_size=40)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)

        # Create network structure [3, 5, 4, 2]
        layers = [3, 5, 4, 2]

        # Generate random importance scores for hidden + output layers
        # Input layer doesn't get pruned
        np.random.seed(42)
        importance_scores = {
            'layer1': np.random.beta(2, 2, layers[1]),  # Hidden layer 1
            'layer2': np.random.beta(2, 2, layers[2]),  # Hidden layer 2
            'layer3': np.random.beta(2, 2, layers[3]),  # Output layer
        }

        # Normalize to [0, 1] range
        for key in importance_scores:
            scores = importance_scores[key]
            importance_scores[key] = (scores - scores.min()) / (scores.max() - scores.min())

        # Show uniform network first
        subtitle1 = Text("Before Importance Analysis", font_size=28, color=GRAY)
        subtitle1.next_to(title, DOWN, buff=0.5)
        self.play(Write(subtitle1))

        network_uniform = self.create_network(layers, uniform=True)
        network_uniform.shift(DOWN * 0.5)
        self.play(Create(network_uniform), run_time=2)
        self.wait(1)

        # Transform to importance-scaled network
        self.play(FadeOut(subtitle1))
        subtitle2 = Text("After Importance Analysis", font_size=28, color=GREEN)
        subtitle2.next_to(title, DOWN, buff=0.5)
        self.play(Write(subtitle2))

        network_scaled = self.create_network(layers, uniform=False,
                                            importance_scores=importance_scores)
        network_scaled.shift(DOWN * 0.5)

        self.play(Transform(network_uniform, network_scaled), run_time=2)
        self.wait(1)

        # Add legend
        legend = self.create_importance_legend()
        legend.to_edge(DOWN).shift(UP * 0.5)
        self.play(FadeIn(legend))
        self.wait(2)

    def create_network(self, layers, uniform=True, importance_scores=None):
        """
        Create a neural network graph with nodes scaled by importance.

        Args:
            layers: List of layer sizes [input, hidden1, ..., output]
            uniform: If True, all nodes same size. If False, scale by importance.
            importance_scores: Dict mapping layer names to importance arrays
        """
        vertices = []
        edges = []
        partitions = []
        vertex_config = {}

        # Create vertices for each layer
        node_idx = 0
        for layer_idx, layer_size in enumerate(layers):
            layer_vertices = []
            for i in range(layer_size):
                vertices.append(node_idx)
                layer_vertices.append(node_idx)

                # Determine node radius and color based on importance
                if uniform or layer_idx == 0:  # Input layer or uniform mode
                    radius = 0.15
                    color = BLUE
                else:
                    # Get importance score for this neuron
                    layer_key = f'layer{layer_idx}'
                    score = importance_scores[layer_key][i]

                    # Scale radius: 0.08 to 0.25 based on importance
                    radius = 0.08 + 0.17 * score

                    # Color gradient: red (low) → yellow (medium) → green (high)
                    if score < 0.5:
                        # Red to Yellow
                        t = score * 2
                        color = interpolate_color(LOW_IMPORTANCE, MEDIUM_IMPORTANCE, t)
                    else:
                        # Yellow to Green
                        t = (score - 0.5) * 2
                        color = interpolate_color(MEDIUM_IMPORTANCE, HIGH_IMPORTANCE, t)

                vertex_config[node_idx] = {
                    'radius': radius,
                    'fill_color': color,
                    'fill_opacity': 0.8,
                    'stroke_color': WHITE,
                    'stroke_width': 1
                }

                node_idx += 1

            partitions.append(layer_vertices)

        # Create edges between consecutive layers
        for layer_idx in range(len(layers) - 1):
            layer_start = sum(layers[:layer_idx])
            next_layer_start = sum(layers[:layer_idx + 1])

            for i in range(layers[layer_idx]):
                for j in range(layers[layer_idx + 1]):
                    from_node = layer_start + i
                    to_node = next_layer_start + j
                    edges.append((from_node, to_node))

        # Create graph
        graph = Graph(
            vertices, edges,
            layout='partite',
            partitions=partitions,
            layout_scale=2.5,
            vertex_config=vertex_config,
            edge_config={'stroke_width': 0.5, 'stroke_color': GRAY, 'stroke_opacity': 0.5}
        )

        # Add layer labels
        labels = VGroup()
        layer_names = ['Input', 'Hidden 1', 'Hidden 2', 'Output']
        for idx, (partition, name) in enumerate(zip(partitions, layer_names)):
            # Get position of first node in this layer
            first_node_pos = graph.vertices[partition[0]].get_center()
            label = Text(name, font_size=18, color=GRAY)
            label.next_to(first_node_pos, UP, buff=0.8)
            labels.add(label)

        return VGroup(graph, labels)

    def create_importance_legend(self):
        """Create a color legend for importance scores"""

        legend_title = Text("Importance Score", font_size=22)

        # Create gradient bar
        n_segments = 50
        gradient_bar = VGroup()
        for i in range(n_segments):
            t = i / (n_segments - 1)
            if t < 0.5:
                color = interpolate_color(LOW_IMPORTANCE, MEDIUM_IMPORTANCE, t * 2)
            else:
                color = interpolate_color(MEDIUM_IMPORTANCE, HIGH_IMPORTANCE, (t - 0.5) * 2)

            segment = Rectangle(
                width=0.08, height=0.4,
                fill_color=color, fill_opacity=1,
                stroke_width=0
            )
            segment.move_to(RIGHT * (i - n_segments/2) * 0.08)
            gradient_bar.add(segment)

        gradient_bar.scale(0.8)

        # Add labels
        low_label = Text("Low", font_size=18, color=LOW_IMPORTANCE)
        low_label.next_to(gradient_bar, LEFT, buff=0.3)

        high_label = Text("High", font_size=18, color=HIGH_IMPORTANCE)
        high_label.next_to(gradient_bar, RIGHT, buff=0.3)

        # Add size indicator
        size_text = Text("Size ∝ Importance", font_size=18, color=GRAY)
        size_text.next_to(gradient_bar, DOWN, buff=0.4)

        legend = VGroup(legend_title, gradient_bar, low_label, high_label, size_text)
        legend.arrange(DOWN, buff=0.3)

        return legend


class ImportanceEvolutionScene(Scene):
    """
    Shows how importance scores evolve during training.
    Demonstrates that importance stabilizes after early training.
    """

    def construct(self):
        # Title
        title = Text("Importance Score Evolution During Training", font_size=38)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)

        # Create axes for plotting
        axes = Axes(
            x_range=[0, 100, 20],
            y_range=[0, 1, 0.2],
            x_length=10,
            y_length=5,
            axis_config={"include_tip": True, "numbers_to_include": np.arange(0, 101, 20)},
            y_axis_config={"numbers_to_include": np.arange(0, 1.1, 0.2)},
        )
        axes.shift(DOWN * 0.8)

        # Add labels
        x_label = Text("Training Epoch", font_size=24)
        x_label.next_to(axes.x_axis, DOWN, buff=0.3)

        y_label = Text("Importance Score", font_size=24)
        y_label.rotate(PI/2).next_to(axes.y_axis, LEFT, buff=0.4)

        self.play(Create(axes), Write(x_label), Write(y_label))
        self.wait(0.5)

        # Generate importance evolution curves for multiple neurons
        n_neurons = 5
        np.random.seed(42)

        curves = []
        labels = []
        colors = [RED, ORANGE, YELLOW, GREEN, BLUE]

        for i in range(n_neurons):
            # Generate curve: rapid change early, stabilization later
            epochs = np.linspace(0, 100, 200)

            # Different patterns for different neurons
            if i == 0:
                # High importance, quick stabilization
                importance = 0.8 + 0.15 * np.exp(-epochs / 10) * np.cos(epochs / 5)
            elif i == 1:
                # Medium-high, moderate stabilization
                importance = 0.6 + 0.2 * np.exp(-epochs / 20) * np.cos(epochs / 8)
            elif i == 2:
                # Medium, slow stabilization
                importance = 0.5 + 0.3 * np.exp(-epochs / 30) * np.cos(epochs / 10)
            elif i == 3:
                # Low-medium, becomes less important over time
                importance = 0.5 - 0.2 * (1 - np.exp(-epochs / 25))
            else:
                # Low importance, quick stabilization to low value
                importance = 0.2 + 0.15 * np.exp(-epochs / 15) * np.cos(epochs / 6)

            # Clip to [0, 1]
            importance = np.clip(importance, 0, 1)

            # Create graph
            points = [(e, imp) for e, imp in zip(epochs, importance)]
            curve = axes.plot_line_graph(
                epochs, importance,
                line_color=colors[i],
                stroke_width=3,
                add_vertex_dots=False
            )

            curves.append(curve)

            # Create label
            final_value = importance[-1]
            label_text = f"Neuron {i+1} (final: {final_value:.2f})"
            label = Text(label_text, font_size=18, color=colors[i])
            labels.append(label)

        # Animate curves appearing
        self.play(*[Create(curve) for curve in curves], run_time=3)
        self.wait(0.5)

        # Add legend
        legend = VGroup(*labels)
        legend.arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        legend.scale(0.9)
        legend.to_corner(UR).shift(LEFT * 0.5 + DOWN * 1.5)

        self.play(FadeIn(legend), run_time=1)
        self.wait(1)

        # Highlight stabilization region
        stabilization_box = Rectangle(
            width=2.5, height=5.5,
            stroke_color=GREEN, stroke_width=2, stroke_opacity=0.8,
            fill_opacity=0
        )
        stabilization_box.move_to(axes.c2p(70, 0.5))

        stabilization_text = Text(
            "Stabilization\nRegion", font_size=22, color=GREEN
        )
        stabilization_text.next_to(stabilization_box, UP, buff=0.3)

        self.play(Create(stabilization_box), Write(stabilization_text))
        self.wait(1)

        # Add insight
        insight = Text(
            "Key Insight: Importance scores stabilize after ~30-40 epochs",
            font_size=24, color=YELLOW
        )
        insight.to_edge(DOWN).shift(UP * 0.5)

        self.play(Write(insight))
        self.wait(2)

        # Add pruning threshold line
        threshold = 0.4
        threshold_line = DashedLine(
            axes.c2p(0, threshold),
            axes.c2p(100, threshold),
            color=RED, stroke_width=2
        )

        threshold_label = Text(
            f"Pruning Threshold ({threshold})", font_size=20, color=RED
        )
        threshold_label.next_to(threshold_line, LEFT, buff=0.2)

        self.play(Create(threshold_line), Write(threshold_label))
        self.wait(1)

        # Highlight neurons below threshold
        prune_text = Text(
            "Neurons 4 & 5 would be pruned", font_size=22, color=RED
        )
        prune_text.next_to(insight, UP, buff=0.5)

        self.play(Write(prune_text))
        self.wait(2)


class ComparisonBeforeAfterScene(Scene):
    """
    Side-by-side comparison showing the same network before and after
    importance-based visualization.
    """

    def construct(self):
        # Title
        title = Text("Before vs After Importance Analysis", font_size=40)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)

        # Create two networks side by side
        layers = [4, 6, 5, 3]

        # Generate importance scores
        np.random.seed(42)
        importance_scores = {
            'layer1': np.random.beta(2, 2, layers[1]),
            'layer2': np.random.beta(2, 2, layers[2]),
            'layer3': np.random.beta(2, 2, layers[3]),
        }

        # Normalize
        for key in importance_scores:
            scores = importance_scores[key]
            importance_scores[key] = (scores - scores.min()) / (scores.max() - scores.min())

        # Before (uniform)
        before_label = Text("Before", font_size=28, color=GRAY)
        before_label.shift(UP * 2.5 + LEFT * 3.5)

        network_before = self.create_simple_network(layers, uniform=True)
        network_before.scale(0.7).shift(LEFT * 3.5)

        # After (scaled)
        after_label = Text("After", font_size=28, color=GREEN)
        after_label.shift(UP * 2.5 + RIGHT * 3.5)

        network_after = self.create_simple_network(layers, uniform=False,
                                                   importance_scores=importance_scores)
        network_after.scale(0.7).shift(RIGHT * 3.5)

        # Show both
        self.play(
            Write(before_label), Create(network_before),
            Write(after_label), Create(network_after),
            run_time=2
        )
        self.wait(1)

        # Add annotations
        annotation_before = Text(
            "All neurons\ntreated equally", font_size=20, color=GRAY
        )
        annotation_before.next_to(network_before, DOWN, buff=0.8)

        annotation_after = Text(
            "Size & color show\nimportance", font_size=20, color=GREEN
        )
        annotation_after.next_to(network_after, DOWN, buff=0.8)

        self.play(Write(annotation_before), Write(annotation_after))
        self.wait(2)

    def create_simple_network(self, layers, uniform=True, importance_scores=None):
        """Simplified network creation for comparison"""
        vertices = []
        edges = []
        partitions = []
        vertex_config = {}

        node_idx = 0
        for layer_idx, layer_size in enumerate(layers):
            layer_vertices = []
            for i in range(layer_size):
                vertices.append(node_idx)
                layer_vertices.append(node_idx)

                if uniform or layer_idx == 0:
                    radius = 0.12
                    color = BLUE
                else:
                    layer_key = f'layer{layer_idx}'
                    score = importance_scores[layer_key][i]
                    radius = 0.06 + 0.12 * score

                    if score < 0.5:
                        t = score * 2
                        color = interpolate_color(LOW_IMPORTANCE, MEDIUM_IMPORTANCE, t)
                    else:
                        t = (score - 0.5) * 2
                        color = interpolate_color(MEDIUM_IMPORTANCE, HIGH_IMPORTANCE, t)

                vertex_config[node_idx] = {
                    'radius': radius,
                    'fill_color': color,
                    'fill_opacity': 0.8,
                    'stroke_color': WHITE,
                    'stroke_width': 1
                }

                node_idx += 1
            partitions.append(layer_vertices)

        # Edges
        for layer_idx in range(len(layers) - 1):
            layer_start = sum(layers[:layer_idx])
            next_layer_start = sum(layers[:layer_idx + 1])

            for i in range(layers[layer_idx]):
                for j in range(layers[layer_idx + 1]):
                    edges.append((layer_start + i, next_layer_start + j))

        graph = Graph(
            vertices, edges,
            layout='partite',
            partitions=partitions,
            layout_scale=2,
            vertex_config=vertex_config,
            edge_config={'stroke_width': 0.5, 'stroke_color': GRAY, 'stroke_opacity': 0.3}
        )

        return graph


# Entry point for rendering
if __name__ == "__main__":
    # This allows running: python 03_coordinate_importance.py
    # Manim will render all scenes
    pass
