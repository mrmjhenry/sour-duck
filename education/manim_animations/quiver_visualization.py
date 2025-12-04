"""
Quiver Representation Visualization

This Manim animation demonstrates how neural networks are represented as quivers,
showing the extraction of vertices (neurons) and edges (weights).

Usage:
    manim -pql quiver_visualization.py QuiverExtractionScene
    manim -pqh quiver_visualization.py QuiverNetworkScene
    manim -pql quiver_visualization.py QuiverDimensionScene

Requirements:
    pip install manim manim-ml
"""

from manim import *
import numpy as np

# Try to import ManimML, fall back to custom implementation if not available
try:
    from manim_ml.neural_network import NeuralNetwork, FeedForwardLayer
    HAS_MANIM_ML = True
except ImportError:
    HAS_MANIM_ML = False
    print("ManimML not installed. Using custom network visualization.")


# ============================================================================
# Configuration
# ============================================================================

# Colors
NODE_COLOR = BLUE
EDGE_COLOR = GRAY
HIGHLIGHT_COLOR = YELLOW
INPUT_COLOR = GREEN
OUTPUT_COLOR = RED
HIDDEN_COLOR = BLUE

# Sizes
NODE_RADIUS = 0.2
EDGE_WIDTH = 1
HIGHLIGHT_WIDTH = 3

# Timing
ANIMATION_RUNTIME = 1.5
WAIT_TIME = 0.5


# ============================================================================
# Scene 1: Quiver Extraction from Neural Network
# ============================================================================

class QuiverExtractionScene(Scene):
    """
    Demonstrates the process of extracting a quiver representation from
    a simple neural network.

    Shows:
    1. Network architecture
    2. Vertex identification
    3. Edge extraction
    4. Complete quiver structure
    """

    def construct(self):
        # Title
        title = Text("Extracting Quiver Representation", font_size=40)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(WAIT_TIME)

        # Network architecture
        arch_text = Text("Network: [3, 4, 2]", font_size=28)
        arch_text.next_to(title, DOWN, buff=0.5)
        self.play(FadeIn(arch_text))
        self.wait(WAIT_TIME)

        # Create network visualization
        network = self.create_network([3, 4, 2])
        network.shift(DOWN * 0.5)
        self.play(Create(network), run_time=2)
        self.wait(WAIT_TIME)

        # Step 1: Highlight vertices
        self.show_vertices(network)

        # Step 2: Highlight edges
        self.show_edges(network)

        # Step 3: Show counts
        self.show_counts([3, 4, 2])

        self.wait(2)

    def create_network(self, layers):
        """Create a neural network graph visualization."""
        vertices = []
        edges = []
        partitions = []

        # Create vertices
        c = 0
        for i, layer_size in enumerate(layers):
            partition = list(range(c + 1, c + layer_size + 1))
            partitions.append(partition)
            vertices.extend(partition)
            c += layer_size

        # Create edges (fully connected between layers)
        for i in range(len(layers) - 1):
            start_idx = sum(layers[:i])
            end_idx = sum(layers[:i+1])
            next_start = end_idx
            next_end = sum(layers[:i+2])

            for j in range(start_idx + 1, end_idx + 1):
                for k in range(next_start + 1, next_end + 1):
                    edges.append((j, k))

        # Vertex colors by layer
        vertex_config = {}
        c = 0
        for i, layer_size in enumerate(layers):
            for j in range(layer_size):
                vertex_num = c + j + 1
                if i == 0:
                    vertex_config[vertex_num] = {'fill_color': INPUT_COLOR}
                elif i == len(layers) - 1:
                    vertex_config[vertex_num] = {'fill_color': OUTPUT_COLOR}
                else:
                    vertex_config[vertex_num] = {'fill_color': HIDDEN_COLOR}
            c += layer_size

        # Create graph
        graph = Graph(
            vertices,
            edges,
            layout='partite',
            partitions=partitions,
            layout_scale=3,
            vertex_config={
                'radius': NODE_RADIUS,
                **vertex_config
            },
            edge_config={
                'stroke_width': EDGE_WIDTH,
                'stroke_color': EDGE_COLOR
            }
        )

        return graph

    def show_vertices(self, network):
        """Highlight vertices and explain them."""
        text = Text("Step 1: Identify Vertices (Neurons)", font_size=24)
        text.to_edge(DOWN, buff=1)
        self.play(Write(text))

        # Highlight all vertices
        for vertex in network.vertices.values():
            self.play(
                vertex.animate.set_fill(YELLOW, opacity=0.8).scale(1.3),
                run_time=0.1
            )

        self.wait(WAIT_TIME)

        # Count vertices
        v_count = len(network.vertices)
        count_text = Text(f"Total Vertices: {v_count}", font_size=24, color=YELLOW)
        count_text.next_to(text, DOWN, buff=0.3)
        self.play(Write(count_text))

        self.wait(WAIT_TIME)

        # Reset colors
        for i, vertex in enumerate(network.vertices.values()):
            original_color = list(network.vertices.keys())[i]
            if original_color <= 3:
                color = INPUT_COLOR
            elif original_color >= 8:
                color = OUTPUT_COLOR
            else:
                color = HIDDEN_COLOR
            self.play(
                vertex.animate.set_fill(color, opacity=1).scale(1/1.3),
                run_time=0.1
            )

        self.play(FadeOut(text), FadeOut(count_text))

    def show_edges(self, network):
        """Highlight edges and explain them."""
        text = Text("Step 2: Identify Edges (Weights)", font_size=24)
        text.to_edge(DOWN, buff=1)
        self.play(Write(text))

        # Highlight edges
        edge_count = 0
        for edge in network.edges.values():
            self.play(
                edge.animate.set_color(YELLOW).set_stroke_width(HIGHLIGHT_WIDTH),
                run_time=0.05
            )
            edge_count += 1

        self.wait(WAIT_TIME)

        # Count edges
        count_text = Text(f"Total Edges: {edge_count}", font_size=24, color=YELLOW)
        count_text.next_to(text, DOWN, buff=0.3)
        self.play(Write(count_text))

        self.wait(WAIT_TIME)

        # Reset colors
        for edge in network.edges.values():
            self.play(
                edge.animate.set_color(EDGE_COLOR).set_stroke_width(EDGE_WIDTH),
                run_time=0.05
            )

        self.play(FadeOut(text), FadeOut(count_text))

    def show_counts(self, layers):
        """Show final vertex and edge counts."""
        # Calculate counts
        v_total = sum(layers)
        v_hidden = sum(layers[1:-1])
        e_total = sum(layers[i] * layers[i+1] for i in range(len(layers)-1))

        # Display counts
        summary = VGroup(
            Text("Quiver Statistics:", font_size=28, color=BLUE),
            Text(f"Total Vertices (V): {v_total}", font_size=24),
            Text(f"Hidden Vertices (Ṽ): {v_hidden}", font_size=24, color=BLUE),
            Text(f"Total Edges (E): {e_total}", font_size=24),
            Text(f"Moduli Dimension: {e_total - v_hidden}", font_size=24, color=GREEN)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)

        summary.to_edge(DOWN, buff=0.5)

        self.play(Write(summary))
        self.wait(2)


# ============================================================================
# Scene 2: Quiver Representation with ManimML
# ============================================================================

class QuiverNetworkScene(ThreeDScene):
    """
    Uses ManimML to show a neural network and its quiver representation side by side.
    Demonstrates the natural correspondence.
    """

    def construct(self):
        if not HAS_MANIM_ML:
            # Fallback message
            text = Text("This scene requires ManimML\nInstall with: pip install manim-ml")
            self.play(Write(text))
            self.wait(3)
            return

        # Title
        title = Text("Neural Network = Quiver Representation", font_size=36)
        title.to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title))

        # Create neural network with ManimML
        nn = NeuralNetwork([
            FeedForwardLayer(num_nodes=3, node_color=INPUT_COLOR),
            FeedForwardLayer(num_nodes=5, node_color=HIDDEN_COLOR),
            FeedForwardLayer(num_nodes=2, node_color=OUTPUT_COLOR)
        ], layer_spacing=0.5)

        nn.shift(DOWN * 0.5)

        # Add network
        self.play(Create(nn), run_time=2)
        self.wait()

        # Animate forward pass
        forward_animation = nn.make_forward_pass_animation()
        self.play(forward_animation, run_time=3)

        # Add labels
        quiver_text = Text("Quiver: Vertices + Directed Edges", font_size=24)
        quiver_text.to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(quiver_text)
        self.play(Write(quiver_text))

        self.wait(2)


# ============================================================================
# Scene 3: Dimension Calculation
# ============================================================================

class QuiverDimensionScene(Scene):
    """
    Explains the moduli space dimension formula: dim(M) = #E - #Ṽ
    Shows calculation for different network architectures.
    """

    def construct(self):
        # Title
        title = Text("Moduli Space Dimension", font_size=40)
        title.to_edge(UP)
        self.play(Write(title))

        # Formula
        formula = MathTex(
            r"\text{dim}(M) = \#E - \#\tilde{V}",
            font_size=48
        )
        formula.next_to(title, DOWN, buff=0.8)
        self.play(Write(formula))
        self.wait()

        # Explanation
        explanation = VGroup(
            MathTex(r"\#E = \text{number of edges (weights)}", font_size=32),
            MathTex(r"\#\tilde{V} = \text{number of hidden vertices}", font_size=32)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        explanation.next_to(formula, DOWN, buff=0.8)
        self.play(Write(explanation))
        self.wait(2)

        # Examples
        self.play(FadeOut(explanation))

        examples = [
            ([3, 4, 2], "Simple Network"),
            ([10, 20, 10, 5], "Deeper Network"),
            ([784, 128, 64, 10], "MNIST Classifier")
        ]

        for i, (layers, name) in enumerate(examples):
            self.show_example(layers, name, position=DOWN * (0.5 + i * 2))
            if i < len(examples) - 1:
                self.wait(WAIT_TIME)

    def show_example(self, layers, name, position):
        """Show dimension calculation for a specific architecture."""
        # Calculate values
        e_total = sum(layers[i] * layers[i+1] for i in range(len(layers)-1))
        v_hidden = sum(layers[1:-1])
        moduli_dim = e_total - v_hidden

        # Create display
        example = VGroup(
            Text(f"{name}: {layers}", font_size=28, color=BLUE),
            MathTex(f"\\#E = {e_total}", font_size=24),
            MathTex(f"\\#\\tilde{{V}} = {v_hidden}", font_size=24),
            MathTex(
                f"\\text{{dim}}(M) = {e_total} - {v_hidden} = {moduli_dim}",
                font_size=24,
                color=GREEN
            )
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)

        example.move_to(position)

        # Animate
        self.play(FadeIn(example), run_time=1.5)

        # Highlight the result
        self.play(
            example[-1].animate.scale(1.2).set_color(YELLOW),
            run_time=0.5
        )
        self.play(
            example[-1].animate.scale(1/1.2).set_color(GREEN),
            run_time=0.5
        )


# ============================================================================
# Scene 4: Path Composition
# ============================================================================

class PathCompositionScene(Scene):
    """
    Demonstrates paths in the quiver and their composition,
    showing how information flows through the network.
    """

    def construct(self):
        # Title
        title = Text("Paths in the Quiver", font_size=40)
        title.to_edge(UP)
        self.play(Write(title))

        # Create simple network
        layers = [2, 3, 2]
        network = self.create_simple_network(layers)
        network.shift(UP * 0.5)
        self.play(Create(network))

        # Explanation
        text = Text("Paths = Information Flow", font_size=28)
        text.to_edge(DOWN, buff=1.5)
        self.play(Write(text))

        # Show different paths
        paths = [
            [(1, 3), (3, 6)],  # Input 1 → Hidden 1 → Output 1
            [(1, 4), (4, 7)],  # Input 1 → Hidden 2 → Output 2
            [(2, 5), (5, 6)],  # Input 2 → Hidden 3 → Output 1
        ]

        path_colors = [RED, YELLOW, GREEN]

        for i, (path, color) in enumerate(zip(paths, path_colors)):
            path_text = Text(f"Path {i+1}", font_size=24, color=color)
            path_text.next_to(text, DOWN, buff=0.3)
            self.play(Write(path_text))

            # Highlight path
            for edge in path:
                if edge in network.edges:
                    self.play(
                        network.edges[edge].animate.set_color(color).set_stroke_width(HIGHLIGHT_WIDTH),
                        run_time=0.5
                    )

            self.wait(WAIT_TIME)

            # Reset
            for edge in path:
                if edge in network.edges:
                    self.play(
                        network.edges[edge].animate.set_color(EDGE_COLOR).set_stroke_width(EDGE_WIDTH),
                        run_time=0.3
                    )

            self.play(FadeOut(path_text))

        self.wait(2)

    def create_simple_network(self, layers):
        """Create a simple network for path demonstration."""
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
            layout_scale=3.5,
            vertex_config={'radius': NODE_RADIUS},
            edge_config={'stroke_width': EDGE_WIDTH, 'stroke_color': EDGE_COLOR}
        )

        return graph


# ============================================================================
# Main - Run all scenes
# ============================================================================

if __name__ == "__main__":
    # This allows running the script directly
    # Use: python quiver_visualization.py
    print("To render these animations, use:")
    print("  manim -pql quiver_visualization.py QuiverExtractionScene")
    print("  manim -pql quiver_visualization.py QuiverNetworkScene")
    print("  manim -pql quiver_visualization.py QuiverDimensionScene")
    print("  manim -pql quiver_visualization.py PathCompositionScene")
