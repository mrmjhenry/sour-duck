"""
Moduli Space Visualization

Demonstrates the concept of moduli space for neural networks, showing how
different parameter configurations can represent the same function.

Usage:
    manim -pql moduli_space_visualization.py ModuliSpaceIntroScene
    manim -pqh moduli_space_visualization.py DataRepresentationScene

Requirements:
    pip install manim manim-ml numpy
"""

from manim import *
import numpy as np


# ============================================================================
# Configuration
# ============================================================================

PARAM_COLOR = BLUE
EQUIV_COLOR = GREEN
MODULI_COLOR = YELLOW
COORDINATE_COLOR = RED


# ============================================================================
# Scene 1: Moduli Space Introduction
# ============================================================================

class ModuliSpaceIntroScene(Scene):
    """
    Introduces the moduli space concept by showing equivalent networks
    (same function, different parameters).
    """

    def construct(self):
        # Title
        title = Text("The Moduli Space of Neural Networks", font_size=36)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait()

        # Subtitle
        subtitle = Text(
            "Classifying networks by their function, not parameters",
            font_size=24,
            color=GRAY
        )
        subtitle.next_to(title, DOWN, buff=0.3)
        self.play(FadeIn(subtitle))
        self.wait()

        # Show parameter space
        self.show_parameter_space()

        # Show equivalence
        self.show_equivalence()

        # Show quotient
        self.show_moduli_space()

        self.wait(2)

    def show_parameter_space(self):
        """Show the high-dimensional parameter space."""
        text = Text("Parameter Space", font_size=32, color=PARAM_COLOR)
        text.shift(UP * 2)

        formula = MathTex(r"W = \mathbb{R}^{\#E}", font_size=36)
        formula.next_to(text, DOWN, buff=0.5)

        explanation = Text(
            "One point for each possible set of weights",
            font_size=20,
            color=GRAY
        )
        explanation.next_to(formula, DOWN, buff=0.4)

        # Visualize as a space of points
        dots = VGroup(*[
            Dot(point=[
                np.random.uniform(-3, 3),
                np.random.uniform(-1, 1),
                0
            ], color=PARAM_COLOR, radius=0.05)
            for _ in range(100)
        ])
        dots.next_to(explanation, DOWN, buff=0.5)

        self.play(Write(text), Write(formula))
        self.play(FadeIn(explanation))
        self.play(FadeIn(dots, lag_ratio=0.05))
        self.wait()

        # Store for later
        self.param_group = VGroup(text, formula, explanation, dots)

    def show_equivalence(self):
        """Show networks with same function but different parameters."""
        # Fade out parameter space
        self.play(self.param_group.animate.shift(LEFT * 5).fade(0.7))

        # Title
        title = Text("Equivalent Networks", font_size=28, color=EQUIV_COLOR)
        title.to_edge(RIGHT).shift(UP * 2)
        self.play(Write(title))

        # Show two equivalent parameterizations
        network1 = self.create_small_network()
        network1.shift(RIGHT * 2 + UP * 0.5)

        network2 = self.create_small_network()
        network2.shift(RIGHT * 2 + DOWN * 1.2)

        label1 = Text("Weights: [1, 2]", font_size=20)
        label1.next_to(network1, RIGHT)

        label2 = Text("Weights: [2, 4] (scaled)", font_size=20)
        label2.next_to(network2, RIGHT)

        self.play(Create(network1), Write(label1))
        self.play(Create(network2), Write(label2))

        # Arrow showing equivalence
        arrow = DoubleArrow(
            network1.get_bottom() + DOWN * 0.2,
            network2.get_top() + UP * 0.2,
            color=EQUIV_COLOR
        )
        equiv_text = Text("Same Function!", font_size=20, color=EQUIV_COLOR)
        equiv_text.next_to(arrow, RIGHT, buff=0.2)

        self.play(GrowArrow(arrow), Write(equiv_text))
        self.wait()

        # Store for cleanup
        self.equiv_group = VGroup(
            title, network1, network2, label1, label2, arrow, equiv_text
        )

    def show_moduli_space(self):
        """Show the moduli space as quotient."""
        # Clear screen
        self.play(
            FadeOut(self.param_group),
            FadeOut(self.equiv_group)
        )

        # Formula for moduli space
        title = Text("Moduli Space", font_size=36, color=MODULI_COLOR)
        title.shift(UP * 2)

        formula = MathTex(
            r"M = W / \tilde{G}",
            font_size=42
        )
        formula.next_to(title, DOWN, buff=0.5)

        explanation = VGroup(
            Text("Quotient by symmetry group", font_size=24),
            MathTex(r"\tilde{G} = \text{Permutations} \times \text{Scalings}", font_size=28)
        ).arrange(DOWN, buff=0.3)
        explanation.next_to(formula, DOWN, buff=0.8)

        self.play(Write(title), Write(formula))
        self.play(FadeIn(explanation))
        self.wait()

        # Dimension formula
        dim_formula = MathTex(
            r"\text{dim}(M) = \#E - \#\tilde{V}",
            font_size=38,
            color=GREEN
        )
        dim_formula.next_to(explanation, DOWN, buff=1)

        dim_box = SurroundingRectangle(dim_formula, color=GREEN, buff=0.2)

        self.play(Write(dim_formula))
        self.play(Create(dim_box))
        self.wait(2)

    def create_small_network(self):
        """Create a small network visualization."""
        # Create simple 2-node network
        vertices = [1, 2]
        edges = [(1, 2)]

        graph = Graph(
            vertices,
            edges,
            layout="tree",
            root_vertex=1,
            vertex_config={'radius': 0.15},
            layout_scale=0.8
        )
        return graph


# ============================================================================
# Scene 2: Data Representation
# ============================================================================

class DataRepresentationScene(Scene):
    """
    Visualizes the data representation W_x^f and how it defines
    coordinates on the moduli space.
    """

    def construct(self):
        # Title
        title = Text("Data Representation: Coordinates on Moduli Space", font_size=32)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait()

        # Formula
        formula = MathTex(
            r"(W_x^f)_\varepsilon = \frac{W_\varepsilon \cdot a_{s(\varepsilon)}(x)}{\sum_{\delta \in \zeta_{s(\varepsilon)}} W_\delta \cdot a_{s(\delta)}(x)}",
            font_size=32
        )
        formula.next_to(title, DOWN, buff=0.6)
        self.play(Write(formula))
        self.wait()

        # Explanation
        explanation = VGroup(
            Text("For each input x:", font_size=24),
            Text("• Compute activations at each layer", font_size=20),
            Text("• Normalize by total incoming activation", font_size=20),
            Text("• Get coordinate value for each edge", font_size=20),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        explanation.next_to(formula, DOWN, buff=0.6)

        self.play(FadeIn(explanation, shift=DOWN))
        self.wait()

        # Show example
        self.show_coordinate_example()

        self.wait(2)

    def show_coordinate_example(self):
        """Show a concrete example of coordinate computation."""
        # Clear previous
        self.play(*[FadeOut(mob) for mob in self.mobjects[2:]])

        # Network diagram
        network = self.create_example_network()
        network.to_edge(LEFT, buff=1)
        self.play(Create(network))

        # Input
        input_label = MathTex("x = 1.0", font_size=28)
        input_label.next_to(network, UP, buff=0.5)
        self.play(Write(input_label))

        # Weights
        weight_labels = VGroup(
            MathTex("w_1 = 2", font_size=20),
            MathTex("w_2 = 3", font_size=20)
        )
        weight_labels.arrange(DOWN, buff=0.2)
        weight_labels.next_to(network, DOWN, buff=0.5)
        self.play(Write(weight_labels))

        # Computation
        computation = VGroup(
            Text("Coordinate Computation:", font_size=24, color=COORDINATE_COLOR),
            MathTex(r"(W_x^f)_1 = \frac{2 \cdot 1.0}{2 \cdot 1.0 + 3 \cdot 1.0} = \frac{2}{5} = 0.4", font_size=22),
            MathTex(r"(W_x^f)_2 = \frac{3 \cdot 1.0}{2 \cdot 1.0 + 3 \cdot 1.0} = \frac{3}{5} = 0.6", font_size=22),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        computation.to_edge(RIGHT, buff=0.8)

        self.play(FadeIn(computation, shift=LEFT))

        # Highlight normalization
        box = SurroundingRectangle(computation[1:], color=GREEN, buff=0.2)
        norm_text = Text("Normalized: sum = 1.0", font_size=20, color=GREEN)
        norm_text.next_to(box, DOWN, buff=0.2)

        self.play(Create(box), Write(norm_text))
        self.wait()

    def create_example_network(self):
        """Create example network for coordinate computation."""
        vertices = [1, 2]
        edges = [(1, 2), (1, 2)]  # Two edges (simplified)

        graph = Graph(
            [1, 2],
            [(1, 2)],
            layout={1: LEFT * 2, 2: RIGHT * 2},
            vertex_config={'radius': 0.2}
        )
        return graph


# ============================================================================
# Scene 3: Pruning in Moduli Space
# ============================================================================

class PruningVisualizationScene(Scene):
    """
    Shows how pruning works in moduli space by removing
    low-importance coordinates.
    """

    def construct(self):
        # Title
        title = Text("Pruning in Moduli Space", font_size=36)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait()

        # Show importance scores
        coords = self.create_coordinate_display()
        coords.shift(DOWN * 0.5)
        self.play(FadeIn(coords))
        self.wait()

        # Show pruning threshold
        threshold_line = Line(LEFT * 6, RIGHT * 6, color=RED)
        threshold_line.shift(DOWN * 0.5)
        threshold_label = Text("Threshold", font_size=20, color=RED)
        threshold_label.next_to(threshold_line, RIGHT)

        self.play(Create(threshold_line), Write(threshold_label))
        self.wait()

        # Remove low-importance coordinates
        low_importance = coords[5:]  # Bottom 5
        self.play(*[FadeOut(coord, shift=DOWN) for coord in low_importance])

        # Show result
        result_text = Text("Pruned Network: 50% sparsity", font_size=28, color=GREEN)
        result_text.to_edge(DOWN, buff=1)
        self.play(Write(result_text))

        self.wait(2)

    def create_coordinate_display(self):
        """Create visualization of coordinate importance scores."""
        coords = VGroup()

        importances = [0.9, 0.85, 0.75, 0.65, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1]
        colors = [interpolate_color(RED, GREEN, imp) for imp in importances]

        for i, (imp, color) in enumerate(zip(importances, colors)):
            bar = Rectangle(
                width=imp * 5,
                height=0.3,
                fill_color=color,
                fill_opacity=0.8,
                stroke_width=1
            )
            bar.align_to(ORIGIN, LEFT)

            label = Text(f"ε_{i+1}: {imp:.2f}", font_size=16)
            label.next_to(bar, LEFT, buff=0.2)

            coord = VGroup(label, bar)
            coords.add(coord)

        coords.arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        return coords


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("To render these animations, use:")
    print("  manim -pql moduli_space_visualization.py ModuliSpaceIntroScene")
    print("  manim -pql moduli_space_visualization.py DataRepresentationScene")
    print("  manim -pql moduli_space_visualization.py PruningVisualizationScene")
