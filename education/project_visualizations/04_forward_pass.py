"""
Forward Pass Visualization
===========================

This module creates Manim animations demonstrating how data flows through
a neural network during a forward pass, both before and after pruning.

Scenes:
-------
1. ForwardPassBasicScene: Shows basic forward propagation
   - Input data flows through layers
   - Activations computed at each layer
   - Output prediction produced

2. ForwardPassWithActivationsScene: Shows activation values flowing
   - Visualizes actual activation magnitudes
   - Color intensity represents activation strength
   - Shows how ReLU affects flow

3. BeforeAfterPruningFlowScene: Compares forward pass before/after pruning
   - Side-by-side networks
   - Shows that pruned network maintains similar flow pattern
   - Highlights differences in active paths

4. DataFlowAnimationScene: Animated data flowing through network
   - Shows multiple data points flowing sequentially
   - Different colors for different classes
   - Demonstrates network's learned representation

Usage:
------
Render specific scenes:
    manim -pql 04_forward_pass.py ForwardPassBasicScene
    manim -pql 04_forward_pass.py ForwardPassWithActivationsScene
    manim -pql 04_forward_pass.py BeforeAfterPruningFlowScene
    manim -pql 04_forward_pass.py DataFlowAnimationScene

Render all scenes:
    manim -pql 04_forward_pass.py

High quality:
    manim -pqh 04_forward_pass.py
"""

from manim import *
import numpy as np

# Color scheme for data flow
INPUT_COLOR = GREEN
HIDDEN_COLOR = BLUE
OUTPUT_COLOR = RED
ACTIVE_FLOW = YELLOW
INACTIVE_FLOW = GRAY
PRUNED_EDGE = RED
KEPT_EDGE = GREEN


class ForwardPassBasicScene(Scene):
    """
    Demonstrates basic forward propagation through a neural network.
    Shows step-by-step computation at each layer.
    """

    def construct(self):
        # Title
        title = Text("Forward Pass: Basic Propagation", font_size=40)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)

        # Create network [3, 4, 3, 2]
        layers = [3, 4, 3, 2]
        network = self.create_network(layers)
        network.shift(DOWN * 0.5)

        # Show network
        self.play(Create(network['graph']), Write(network['labels']), run_time=2)
        self.wait(0.5)

        # Create input data visualization
        input_data = self.create_input_visualization([0.8, 0.3, 0.9])
        input_data.scale(0.8).next_to(network['graph'].vertices[0], LEFT, buff=1.2)

        input_label = Text("Input: [0.8, 0.3, 0.9]", font_size=20, color=INPUT_COLOR)
        input_label.next_to(input_data, UP, buff=0.3)

        self.play(FadeIn(input_data), Write(input_label))
        self.wait(0.5)

        # Animate forward pass layer by layer
        self.animate_layer_computation(network, layers, 0, "Input → Hidden 1")
        self.wait(0.5)

        self.animate_layer_computation(network, layers, 1, "Hidden 1 → Hidden 2")
        self.wait(0.5)

        self.animate_layer_computation(network, layers, 2, "Hidden 2 → Output")
        self.wait(0.5)

        # Show final output
        output_viz = self.create_output_visualization([0.73, 0.27])
        output_viz.scale(0.8).next_to(network['graph'].vertices[sum(layers)-1], RIGHT, buff=1.2)

        output_label = Text("Output: [0.73, 0.27]", font_size=20, color=OUTPUT_COLOR)
        output_label.next_to(output_viz, UP, buff=0.3)

        prediction_label = Text("Prediction: Class 0", font_size=22, color=GREEN)
        prediction_label.next_to(output_viz, DOWN, buff=0.5)

        self.play(FadeIn(output_viz), Write(output_label))
        self.wait(0.5)
        self.play(Write(prediction_label))
        self.wait(2)

    def create_network(self, layers):
        """Create neural network graph"""
        vertices = []
        edges = []
        partitions = []

        # Create vertices
        node_idx = 0
        for layer_idx, layer_size in enumerate(layers):
            layer_vertices = []
            for i in range(layer_size):
                vertices.append(node_idx)
                layer_vertices.append(node_idx)
                node_idx += 1
            partitions.append(layer_vertices)

        # Create edges
        for layer_idx in range(len(layers) - 1):
            layer_start = sum(layers[:layer_idx])
            next_layer_start = sum(layers[:layer_idx + 1])

            for i in range(layers[layer_idx]):
                for j in range(layers[layer_idx + 1]):
                    edges.append((layer_start + i, next_layer_start + j))

        # Create graph
        graph = Graph(
            vertices, edges,
            layout='partite',
            partitions=partitions,
            layout_scale=2.5,
            vertex_config={'radius': 0.15, 'fill_color': BLUE, 'fill_opacity': 0.8},
            edge_config={'stroke_width': 1, 'stroke_color': GRAY, 'stroke_opacity': 0.5}
        )

        # Add layer labels
        labels = VGroup()
        layer_names = ['Input', 'Hidden 1', 'Hidden 2', 'Output']
        for partition, name in zip(partitions, layer_names):
            first_node_pos = graph.vertices[partition[0]].get_center()
            label = Text(name, font_size=18, color=GRAY)
            label.next_to(first_node_pos, UP, buff=0.8)
            labels.add(label)

        return {'graph': graph, 'labels': labels, 'layers': layers, 'partitions': partitions}

    def create_input_visualization(self, values):
        """Create bar chart for input values"""
        bars = VGroup()
        for i, val in enumerate(values):
            bar = Rectangle(
                width=0.3, height=val * 1.5,
                fill_color=INPUT_COLOR, fill_opacity=0.8,
                stroke_color=WHITE, stroke_width=1
            )
            bar.shift(DOWN * (1.5 - val * 1.5) / 2)
            bar.shift(RIGHT * i * 0.4)

            # Add value label
            label = Text(f"{val:.1f}", font_size=14, color=WHITE)
            label.next_to(bar, DOWN, buff=0.1)
            bars.add(VGroup(bar, label))

        return bars

    def create_output_visualization(self, values):
        """Create bar chart for output values"""
        bars = VGroup()
        for i, val in enumerate(values):
            bar = Rectangle(
                width=0.4, height=val * 1.5,
                fill_color=OUTPUT_COLOR if i == 0 else GRAY,  # Highlight max
                fill_opacity=0.8,
                stroke_color=WHITE, stroke_width=1
            )
            bar.shift(DOWN * (1.5 - val * 1.5) / 2)
            bar.shift(RIGHT * i * 0.5)

            label = Text(f"{val:.2f}", font_size=14, color=WHITE)
            label.next_to(bar, DOWN, buff=0.1)

            class_label = Text(f"C{i}", font_size=12, color=GRAY)
            class_label.next_to(label, DOWN, buff=0.05)

            bars.add(VGroup(bar, label, class_label))

        return bars

    def animate_layer_computation(self, network, layers, from_layer_idx, description):
        """Animate computation from one layer to the next"""

        # Show description
        desc_text = Text(description, font_size=24, color=YELLOW)
        desc_text.to_edge(DOWN).shift(UP * 0.5)
        self.play(Write(desc_text), run_time=0.5)

        # Get edge indices for this layer transition
        layer_start = sum(layers[:from_layer_idx])
        next_layer_start = sum(layers[:from_layer_idx + 1])

        # Highlight relevant edges
        animations = []
        for i in range(layers[from_layer_idx]):
            for j in range(layers[from_layer_idx + 1]):
                from_node = layer_start + i
                to_node = next_layer_start + j
                edge_mob = network['graph'].edges[(from_node, to_node)]
                animations.append(
                    edge_mob.animate.set_stroke(color=ACTIVE_FLOW, width=2, opacity=1)
                )

        self.play(*animations, run_time=1)

        # Fade back
        fade_animations = []
        for i in range(layers[from_layer_idx]):
            for j in range(layers[from_layer_idx + 1]):
                from_node = layer_start + i
                to_node = next_layer_start + j
                edge_mob = network['graph'].edges[(from_node, to_node)]
                fade_animations.append(
                    edge_mob.animate.set_stroke(color=GRAY, width=1, opacity=0.5)
                )

        self.play(*fade_animations, FadeOut(desc_text), run_time=0.5)


class ForwardPassWithActivationsScene(Scene):
    """
    Shows forward pass with actual activation values visualized.
    Nodes pulse with intensity proportional to activation.
    """

    def construct(self):
        # Title
        title = Text("Forward Pass: Activation Flow", font_size=40)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)

        # Create network
        layers = [3, 5, 4, 2]
        network = self.create_network(layers)
        network.shift(DOWN * 0.5)

        self.play(Create(network['graph']), Write(network['labels']), run_time=2)
        self.wait(0.5)

        # Simulate activations
        np.random.seed(42)
        activations = {
            'input': np.array([0.8, 0.4, 0.9]),
            'hidden1': np.array([0.7, 0.3, 0.85, 0.2, 0.6]),
            'hidden2': np.array([0.65, 0.8, 0.45, 0.3]),
            'output': np.array([0.75, 0.25])
        }

        # Animate activations flowing through network
        self.animate_activation_flow(network, layers, activations)
        self.wait(2)

    def create_network(self, layers):
        """Create network (similar to basic scene)"""
        vertices = []
        edges = []
        partitions = []

        node_idx = 0
        for layer_idx, layer_size in enumerate(layers):
            layer_vertices = []
            for i in range(layer_size):
                vertices.append(node_idx)
                layer_vertices.append(node_idx)
                node_idx += 1
            partitions.append(layer_vertices)

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
            layout_scale=2.5,
            vertex_config={'radius': 0.15, 'fill_color': BLUE, 'fill_opacity': 0.3},
            edge_config={'stroke_width': 1, 'stroke_color': GRAY, 'stroke_opacity': 0.3}
        )

        labels = VGroup()
        layer_names = ['Input', 'Hidden 1', 'Hidden 2', 'Output']
        for partition, name in zip(partitions, layer_names):
            first_node_pos = graph.vertices[partition[0]].get_center()
            label = Text(name, font_size=18, color=GRAY)
            label.next_to(first_node_pos, UP, buff=0.8)
            labels.add(label)

        return {'graph': graph, 'labels': labels, 'layers': layers, 'partitions': partitions}

    def animate_activation_flow(self, network, layers, activations):
        """Animate activations flowing through the network"""

        layer_keys = ['input', 'hidden1', 'hidden2', 'output']

        for layer_idx, (key, act_values) in enumerate(zip(layer_keys, [
            activations['input'],
            activations['hidden1'],
            activations['hidden2'],
            activations['output']
        ])):
            # Get nodes for this layer
            partition = network['partitions'][layer_idx]

            # Animate each node's activation
            animations = []
            for node_idx, activation in zip(partition, act_values):
                vertex = network['graph'].vertices[node_idx]

                # Color intensity based on activation
                color = interpolate_color(BLUE, YELLOW, activation)

                animations.append(
                    vertex.animate.set_fill(color=color, opacity=0.3 + 0.7 * activation)
                )

            self.play(*animations, run_time=0.8)
            self.wait(0.3)


class BeforeAfterPruningFlowScene(Scene):
    """
    Side-by-side comparison of data flow before and after pruning.
    Shows that pruned network maintains similar activation patterns.
    """

    def construct(self):
        # Title
        title = Text("Data Flow: Before vs After Pruning", font_size=38)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)

        # Create two networks side by side
        layers = [3, 5, 4, 2]

        # Before network (left)
        before_label = Text("Before Pruning", font_size=26, color=GRAY)
        before_label.shift(UP * 2.5 + LEFT * 3.5)

        network_before = self.create_network(layers, pruned=False)
        network_before['graph'].scale(0.6).shift(LEFT * 3.5 + DOWN * 0.3)
        network_before['labels'].scale(0.8).shift(LEFT * 3.5 + DOWN * 0.3)

        # After network (right)
        after_label = Text("After Pruning (50%)", font_size=26, color=GREEN)
        after_label.shift(UP * 2.5 + RIGHT * 3.5)

        # Determine which edges to prune (random 50%)
        np.random.seed(42)
        total_edges = sum(layers[i] * layers[i+1] for i in range(len(layers)-1))
        n_prune = total_edges // 2
        all_edge_indices = list(range(total_edges))
        pruned_indices = set(np.random.choice(all_edge_indices, n_prune, replace=False))

        network_after = self.create_network(layers, pruned=True,
                                           pruned_edge_indices=pruned_indices)
        network_after['graph'].scale(0.6).shift(RIGHT * 3.5 + DOWN * 0.3)
        network_after['labels'].scale(0.8).shift(RIGHT * 3.5 + DOWN * 0.3)

        # Show networks
        self.play(
            Write(before_label), Write(after_label),
            Create(network_before['graph']), Write(network_before['labels']),
            Create(network_after['graph']), Write(network_after['labels']),
            run_time=2
        )
        self.wait(1)

        # Simulate forward pass on both networks
        np.random.seed(42)
        activations = {
            'input': np.array([0.8, 0.4, 0.9]),
            'hidden1': np.array([0.7, 0.3, 0.85, 0.2, 0.6]),
            'hidden2': np.array([0.65, 0.8, 0.45, 0.3]),
            'output': np.array([0.75, 0.25])
        }

        # Animate activations on both networks simultaneously
        self.animate_both_networks(network_before, network_after, layers, activations)
        self.wait(1)

        # Show comparison
        comparison_text = Text(
            "Output similar despite 50% sparsity!", font_size=24, color=GREEN
        )
        comparison_text.to_edge(DOWN).shift(UP * 0.5)
        self.play(Write(comparison_text))
        self.wait(2)

    def create_network(self, layers, pruned=False, pruned_edge_indices=None):
        """Create network, optionally with pruned edges"""
        vertices = []
        edges = []
        partitions = []
        edge_config_dict = {}

        node_idx = 0
        for layer_idx, layer_size in enumerate(layers):
            layer_vertices = []
            for i in range(layer_size):
                vertices.append(node_idx)
                layer_vertices.append(node_idx)
                node_idx += 1
            partitions.append(layer_vertices)

        # Create edges with potential pruning
        edge_counter = 0
        for layer_idx in range(len(layers) - 1):
            layer_start = sum(layers[:layer_idx])
            next_layer_start = sum(layers[:layer_idx + 1])

            for i in range(layers[layer_idx]):
                for j in range(layers[layer_idx + 1]):
                    from_node = layer_start + i
                    to_node = next_layer_start + j
                    edges.append((from_node, to_node))

                    # Configure edge appearance
                    if pruned and pruned_edge_indices and edge_counter in pruned_edge_indices:
                        # Pruned edge: very faint or invisible
                        edge_config_dict[(from_node, to_node)] = {
                            'stroke_width': 0.3,
                            'stroke_color': RED,
                            'stroke_opacity': 0.1
                        }
                    else:
                        # Active edge
                        edge_config_dict[(from_node, to_node)] = {
                            'stroke_width': 1,
                            'stroke_color': KEPT_EDGE if pruned else GRAY,
                            'stroke_opacity': 0.6 if pruned else 0.5
                        }

                    edge_counter += 1

        graph = Graph(
            vertices, edges,
            layout='partite',
            partitions=partitions,
            layout_scale=2.5,
            vertex_config={'radius': 0.15, 'fill_color': BLUE, 'fill_opacity': 0.3},
            edge_config=edge_config_dict if pruned else {'stroke_width': 1, 'stroke_color': GRAY}
        )

        labels = VGroup()
        layer_names = ['Input', 'H1', 'H2', 'Out']
        for partition, name in zip(partitions, layer_names):
            first_node_pos = graph.vertices[partition[0]].get_center()
            label = Text(name, font_size=16, color=GRAY)
            label.next_to(first_node_pos, UP, buff=0.6)
            labels.add(label)

        return {'graph': graph, 'labels': labels, 'layers': layers, 'partitions': partitions}

    def animate_both_networks(self, net1, net2, layers, activations):
        """Animate activations on both networks simultaneously"""

        layer_keys = ['input', 'hidden1', 'hidden2', 'output']

        for layer_idx, act_values in enumerate([
            activations['input'],
            activations['hidden1'],
            activations['hidden2'],
            activations['output']
        ]):
            partition = net1['partitions'][layer_idx]

            animations = []
            for node_idx, activation in zip(partition, act_values):
                color = interpolate_color(BLUE, YELLOW, activation)

                # Animate both networks
                vertex1 = net1['graph'].vertices[node_idx]
                vertex2 = net2['graph'].vertices[node_idx]

                animations.append(
                    vertex1.animate.set_fill(color=color, opacity=0.3 + 0.7 * activation)
                )
                animations.append(
                    vertex2.animate.set_fill(color=color, opacity=0.3 + 0.7 * activation)
                )

            self.play(*animations, run_time=0.8)
            self.wait(0.2)


class DataFlowAnimationScene(Scene):
    """
    Shows animated data points flowing through the network.
    Different colors for different input patterns/classes.
    """

    def construct(self):
        # Title
        title = Text("Data Flow Animation: Multiple Inputs", font_size=38)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)

        # Create network
        layers = [3, 6, 5, 3]
        network = self.create_network(layers)
        network.shift(DOWN * 0.5)

        self.play(Create(network['graph']), Write(network['labels']), run_time=2)
        self.wait(0.5)

        # Show multiple data points flowing through
        n_samples = 3
        sample_colors = [GREEN, YELLOW, RED]
        sample_labels = ["Class 0", "Class 1", "Class 2"]

        for sample_idx in range(n_samples):
            sample_text = Text(
                f"Sample {sample_idx + 1}: {sample_labels[sample_idx]}",
                font_size=24, color=sample_colors[sample_idx]
            )
            sample_text.to_edge(DOWN).shift(UP * 0.5)
            self.play(Write(sample_text), run_time=0.5)

            # Animate flow for this sample
            self.animate_sample_flow(network, layers, sample_colors[sample_idx])

            self.play(FadeOut(sample_text), run_time=0.3)
            self.wait(0.3)

        # Final message
        final_text = Text(
            "Each input creates unique activation pattern", font_size=26, color=BLUE
        )
        final_text.to_edge(DOWN).shift(UP * 0.5)
        self.play(Write(final_text))
        self.wait(2)

    def create_network(self, layers):
        """Create network graph"""
        vertices = []
        edges = []
        partitions = []

        node_idx = 0
        for layer_idx, layer_size in enumerate(layers):
            layer_vertices = []
            for i in range(layer_size):
                vertices.append(node_idx)
                layer_vertices.append(node_idx)
                node_idx += 1
            partitions.append(layer_vertices)

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
            layout_scale=2.5,
            vertex_config={'radius': 0.12, 'fill_color': BLUE, 'fill_opacity': 0.2},
            edge_config={'stroke_width': 0.8, 'stroke_color': GRAY, 'stroke_opacity': 0.3}
        )

        labels = VGroup()
        layer_names = ['Input', 'Hidden 1', 'Hidden 2', 'Output']
        for partition, name in zip(partitions, layer_names):
            first_node_pos = graph.vertices[partition[0]].get_center()
            label = Text(name, font_size=18, color=GRAY)
            label.next_to(first_node_pos, UP, buff=0.8)
            labels.add(label)

        return {'graph': graph, 'labels': labels, 'layers': layers, 'partitions': partitions}

    def animate_sample_flow(self, network, layers, flow_color):
        """Animate a single sample flowing through the network"""

        # Generate random activations for this sample
        np.random.seed()
        activations = [np.random.rand(size) * 0.8 + 0.2 for size in layers]

        for layer_idx, act_values in enumerate(activations):
            partition = network['partitions'][layer_idx]

            animations = []
            for node_idx, activation in zip(partition, act_values):
                vertex = network['graph'].vertices[node_idx]

                # Use the sample's color with intensity based on activation
                color = interpolate_color(BLUE, flow_color, activation)

                animations.append(
                    vertex.animate.set_fill(color=color, opacity=0.2 + 0.6 * activation)
                )

            self.play(*animations, run_time=0.6)

        # Fade back
        fade_animations = []
        for partition in network['partitions']:
            for node_idx in partition:
                vertex = network['graph'].vertices[node_idx]
                fade_animations.append(
                    vertex.animate.set_fill(color=BLUE, opacity=0.2)
                )

        self.play(*fade_animations, run_time=0.4)


# Entry point
if __name__ == "__main__":
    pass
