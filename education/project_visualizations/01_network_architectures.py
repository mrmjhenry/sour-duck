"""
Network Architecture Visualization

Visualizes the architectures used in moduli space pruning:
- SimpleNet (lightweight CNN)
- ResNet-18 (with residual connections)

Shows the structure, layer counts, and parameter counts.

Usage:
    manim -pql 01_network_architectures.py SimpleNetArchitecture
    manim -pqh 01_network_architectures.py ResNet18Architecture
    manim -pql 01_network_architectures.py ArchitectureComparison

Requirements:
    pip install manim manim-ml
"""

from manim import *
import numpy as np

try:
    from manim_ml.neural_network import NeuralNetwork, FeedForwardLayer, Convolutional2DLayer
    HAS_MANIM_ML = True
except ImportError:
    HAS_MANIM_ML = False


# ============================================================================
# Configuration
# ============================================================================

SIMPLENET_COLOR = BLUE
RESNET_COLOR = GREEN
CONV_COLOR = ORANGE
FC_COLOR = PURPLE
RESIDUAL_COLOR = YELLOW


# ============================================================================
# Scene 1: SimpleNet Architecture
# ============================================================================

class SimpleNetArchitecture(ThreeDScene):
    """
    Visualizes SimpleNet architecture:
    - Conv layers with BatchNorm and ReLU
    - MaxPooling
    - Fully connected classifier
    """

    def construct(self):
        # Title
        title = Text("SimpleNet Architecture", font_size=40, color=SIMPLENET_COLOR)
        title.to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title))
        self.wait()

        # Architecture description
        arch_text = VGroup(
            Text("Lightweight CNN for CIFAR-10", font_size=24),
            Text("Input: 32×32×3", font_size=20, color=GRAY),
            Text("Output: 10 classes", font_size=20, color=GRAY)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        arch_text.to_edge(DOWN, buff=1)
        self.add_fixed_in_frame_mobjects(arch_text)
        self.play(FadeIn(arch_text))

        if HAS_MANIM_ML:
            # Create network with ManimML
            nn = NeuralNetwork([
                Convolutional2DLayer(1, 7, 3, filter_spacing=0.2),
                Convolutional2DLayer(3, 6, 3, filter_spacing=0.2),
                FeedForwardLayer(256, node_color=FC_COLOR),
                FeedForwardLayer(10, node_color=SIMPLENET_COLOR),
            ], layer_spacing=0.3)

            # Add layer labels
            labels = self.create_simplenet_labels()
            labels.next_to(nn, DOWN, buff=0.5)
            self.add_fixed_in_frame_mobjects(labels)

            self.play(Create(nn), run_time=3)
            self.play(FadeIn(labels))

            # Animate forward pass
            forward_pass = nn.make_forward_pass_animation()
            self.play(forward_pass, run_time=3)

        else:
            # Fallback visualization
            self.show_simplenet_diagram()

        # Show statistics
        self.show_simplenet_stats()

        self.wait(2)

    def create_simplenet_labels(self):
        """Create labels for SimpleNet layers."""
        labels = VGroup(
            Text("Conv 32", font_size=16),
            Text("Conv 64", font_size=16),
            Text("FC 256", font_size=16),
            Text("Out 10", font_size=16),
        ).arrange(RIGHT, buff=1.2)
        return labels

    def show_simplenet_diagram(self):
        """Fallback diagram without ManimML."""
        layers = ["Input\n32×32×3", "Conv\n32 filters", "Conv\n64 filters",
                  "FC\n256", "Output\n10"]

        boxes = VGroup()
        for i, layer_text in enumerate(layers):
            box = Rectangle(width=1.5, height=1.0, color=SIMPLENET_COLOR)
            text = Text(layer_text, font_size=14)
            text.move_to(box.get_center())
            layer = VGroup(box, text)
            boxes.add(layer)

        boxes.arrange(RIGHT, buff=0.5)

        # Add arrows
        arrows = VGroup()
        for i in range(len(boxes) - 1):
            arrow = Arrow(
                boxes[i].get_right(),
                boxes[i + 1].get_left(),
                buff=0.1,
                stroke_width=2
            )
            arrows.add(arrow)

        diagram = VGroup(boxes, arrows)
        self.add_fixed_in_frame_mobjects(diagram)
        self.play(Create(diagram), run_time=2)

    def show_simplenet_stats(self):
        """Show parameter statistics."""
        stats = VGroup(
            Text("Parameters:", font_size=24, color=SIMPLENET_COLOR),
            Text("Conv layers: ~100K", font_size=20),
            Text("FC layers: ~65K", font_size=20),
            Text("Total: ~165K", font_size=20, color=GREEN),
            Text("Suitable for: Quick experiments", font_size=18, color=GRAY)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)

        stats.to_corner(UL, buff=0.5)
        self.add_fixed_in_frame_mobjects(stats)
        self.play(FadeIn(stats, shift=RIGHT))
        self.wait()


# ============================================================================
# Scene 2: ResNet-18 Architecture
# ============================================================================

class ResNet18Architecture(ThreeDScene):
    """
    Visualizes ResNet-18 architecture:
    - Residual blocks
    - Skip connections
    - Deep structure
    """

    def construct(self):
        # Title
        title = Text("ResNet-18 Architecture for CIFAR-10", font_size=36, color=RESNET_COLOR)
        title.to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title))
        self.wait()

        # Architecture description
        arch_text = VGroup(
            Text("18-layer Deep Network", font_size=24),
            Text("Input: 32×32×3", font_size=20, color=GRAY),
            Text("4 Residual Layers [2,2,2,2] blocks", font_size=20, color=GRAY),
            Text("Output: 10 classes", font_size=20, color=GRAY)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        arch_text.to_edge(DOWN, buff=0.5)
        self.add_fixed_in_frame_mobjects(arch_text)
        self.play(FadeIn(arch_text))

        # Show residual block concept
        self.show_residual_block()

        # Show full architecture
        self.show_resnet_architecture()

        # Show statistics
        self.show_resnet_stats()

        self.wait(2)

    def show_residual_block(self):
        """Visualize a residual block with skip connection."""
        # Title for this section
        block_title = Text("Residual Block", font_size=28, color=RESIDUAL_COLOR)
        block_title.shift(UP * 2)
        self.add_fixed_in_frame_mobjects(block_title)
        self.play(Write(block_title))

        # Main path
        main_path = VGroup()
        conv1 = Rectangle(width=1.2, height=0.8, color=CONV_COLOR)
        conv1_text = Text("Conv 3×3", font_size=14)
        conv1_text.move_to(conv1)

        relu1 = Rectangle(width=0.8, height=0.6, color=BLUE)
        relu1_text = Text("ReLU", font_size=12)
        relu1_text.move_to(relu1)

        conv2 = Rectangle(width=1.2, height=0.8, color=CONV_COLOR)
        conv2_text = Text("Conv 3×3", font_size=14)
        conv2_text.move_to(conv2)

        add_node = Circle(radius=0.3, color=GREEN)
        add_text = Text("+", font_size=20)
        add_text.move_to(add_node)

        relu2 = Rectangle(width=0.8, height=0.6, color=BLUE)
        relu2_text = Text("ReLU", font_size=12)
        relu2_text.move_to(relu2)

        main_path = VGroup(
            VGroup(conv1, conv1_text),
            VGroup(relu1, relu1_text),
            VGroup(conv2, conv2_text),
            VGroup(add_node, add_text),
            VGroup(relu2, relu2_text)
        ).arrange(RIGHT, buff=0.4)

        # Arrows
        arrows = VGroup()
        for i in range(len(main_path) - 1):
            arrow = Arrow(
                main_path[i].get_right(),
                main_path[i + 1].get_left(),
                buff=0.1
            )
            arrows.add(arrow)

        # Skip connection
        skip_start = main_path[0].get_top()
        skip_end = main_path[3].get_top()

        skip_arrow = CurvedArrow(
            skip_start + UP * 0.2,
            skip_end + UP * 0.2,
            color=RESIDUAL_COLOR,
            angle=-TAU/4
        )
        skip_label = Text("Skip Connection", font_size=16, color=RESIDUAL_COLOR)
        skip_label.next_to(skip_arrow, UP, buff=0.1)

        # Animate
        self.add_fixed_in_frame_mobjects(main_path, arrows)
        self.play(Create(main_path), Create(arrows))
        self.add_fixed_in_frame_mobjects(skip_arrow, skip_label)
        self.play(Create(skip_arrow), Write(skip_label))

        self.wait(2)

        # Fade out for next section
        self.play(
            FadeOut(block_title),
            FadeOut(main_path),
            FadeOut(arrows),
            FadeOut(skip_arrow),
            FadeOut(skip_label)
        )

    def show_resnet_architecture(self):
        """Show simplified ResNet-18 structure."""
        # Layer blocks
        layers = [
            ("Initial Conv", CONV_COLOR),
            ("Layer 1\n2 blocks\n64 ch", RESNET_COLOR),
            ("Layer 2\n2 blocks\n128 ch", RESNET_COLOR),
            ("Layer 3\n2 blocks\n256 ch", RESNET_COLOR),
            ("Layer 4\n2 blocks\n512 ch", RESNET_COLOR),
            ("Avg Pool", BLUE),
            ("FC 10", FC_COLOR),
        ]

        blocks = VGroup()
        for layer_text, color in layers:
            box = Rectangle(width=1.0, height=1.2, color=color, fill_opacity=0.3)
            text = Text(layer_text, font_size=12)
            text.move_to(box.get_center())
            layer = VGroup(box, text)
            blocks.add(layer)

        blocks.arrange(RIGHT, buff=0.3)
        blocks.shift(UP * 0.5)

        # Add arrows
        arrows = VGroup()
        for i in range(len(blocks) - 1):
            arrow = Arrow(
                blocks[i].get_right(),
                blocks[i + 1].get_left(),
                buff=0.05,
                stroke_width=2
            )
            arrows.add(arrow)

        diagram = VGroup(blocks, arrows)
        self.add_fixed_in_frame_mobjects(diagram)
        self.play(Create(diagram), run_time=3)

    def show_resnet_stats(self):
        """Show ResNet-18 parameter statistics."""
        stats = VGroup(
            Text("Parameters:", font_size=24, color=RESNET_COLOR),
            Text("Standard: ~11.2M", font_size=20),
            Text("Small variant: ~2.8M", font_size=20),
            Text("Residual blocks: 8 (2×4 layers)", font_size=18),
            Text("Skip connections: 8", font_size=18, color=RESIDUAL_COLOR),
            Text("Suitable for: Realistic experiments", font_size=18, color=GRAY)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)

        stats.to_corner(UL, buff=0.5)
        self.add_fixed_in_frame_mobjects(stats)
        self.play(FadeIn(stats, shift=RIGHT))
        self.wait()


# ============================================================================
# Scene 3: Architecture Comparison
# ============================================================================

class ArchitectureComparison(Scene):
    """
    Side-by-side comparison of SimpleNet and ResNet-18.
    """

    def construct(self):
        # Title
        title = Text("Architecture Comparison", font_size=40)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait()

        # Create two columns
        self.show_side_by_side_comparison()

        # Show comparative metrics
        self.show_comparison_table()

        self.wait(2)

    def show_side_by_side_comparison(self):
        """Show SimpleNet and ResNet-18 side by side."""
        # Left: SimpleNet
        simplenet_title = Text("SimpleNet", font_size=28, color=SIMPLENET_COLOR)
        simplenet_title.shift(LEFT * 3 + UP * 2)

        simplenet_layers = VGroup(
            self.create_layer_box("Conv 32", CONV_COLOR),
            self.create_layer_box("Conv 64", CONV_COLOR),
            self.create_layer_box("FC 256", FC_COLOR),
            self.create_layer_box("FC 10", FC_COLOR),
        ).arrange(DOWN, buff=0.3)
        simplenet_layers.next_to(simplenet_title, DOWN, buff=0.5)

        # Right: ResNet-18
        resnet_title = Text("ResNet-18", font_size=28, color=RESNET_COLOR)
        resnet_title.shift(RIGHT * 3 + UP * 2)

        resnet_layers = VGroup(
            self.create_layer_box("Initial Conv", CONV_COLOR),
            self.create_layer_box("Layer 1 (×2)", RESNET_COLOR),
            self.create_layer_box("Layer 2 (×2)", RESNET_COLOR),
            self.create_layer_box("Layer 3 (×2)", RESNET_COLOR),
            self.create_layer_box("Layer 4 (×2)", RESNET_COLOR),
            self.create_layer_box("FC 10", FC_COLOR),
        ).arrange(DOWN, buff=0.2)
        resnet_layers.next_to(resnet_title, DOWN, buff=0.5)

        # Animate
        self.play(
            Write(simplenet_title),
            Write(resnet_title)
        )
        self.play(
            FadeIn(simplenet_layers, shift=DOWN),
            FadeIn(resnet_layers, shift=DOWN)
        )

        self.wait()

    def create_layer_box(self, text, color):
        """Create a colored box with text for a layer."""
        box = Rectangle(width=2.0, height=0.5, color=color, fill_opacity=0.2)
        label = Text(text, font_size=16)
        label.move_to(box.get_center())
        return VGroup(box, label)

    def show_comparison_table(self):
        """Show comparison metrics in a table."""
        table_data = [
            ["Metric", "SimpleNet", "ResNet-18"],
            ["Layers", "4", "18"],
            ["Parameters", "~165K", "~11.2M"],
            ["Training Time", "Fast", "Slower"],
            ["Accuracy", "Good", "Better"],
            ["Use Case", "Quick tests", "Realistic"],
        ]

        table = Table(
            table_data,
            include_outer_lines=True,
            line_config={"stroke_width": 1}
        )
        table.scale(0.5)
        table.to_edge(DOWN, buff=0.5)

        # Color headers
        table.get_entries((1, 2)).set_color(SIMPLENET_COLOR)
        table.get_entries((1, 3)).set_color(RESNET_COLOR)

        self.play(Create(table))
        self.wait()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("To render these animations, use:")
    print("  manim -pql 01_network_architectures.py SimpleNetArchitecture")
    print("  manim -pql 01_network_architectures.py ResNet18Architecture")
    print("  manim -pql 01_network_architectures.py ArchitectureComparison")
