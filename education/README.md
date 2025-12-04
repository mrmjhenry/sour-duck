# Educational Materials: Quiver Representations and Moduli Space Pruning

This directory contains comprehensive educational materials explaining the mathematical foundations of moduli space coordinate pruning for neural networks.

## 📚 Contents

### Theory Documents

1. **[01_quiver_representations.md](01_quiver_representations.md)** - Introduction to quiver representations
   - What are quivers?
   - How neural networks map to quivers
   - Vertices, edges, and their interpretation
   - Mathematical formalism

2. **[02_moduli_space.md](02_moduli_space.md)** - Understanding the moduli space
   - What is a moduli space?
   - Deriving the moduli space from networks
   - Coordinate systems and dimensions
   - Invariance properties

3. **[tutorial.md](tutorial.md)** - Complete walkthrough
   - Step-by-step guide combining theory and practice
   - From neural networks to pruning decisions
   - Interactive examples

### Manim Animations

Visual explanations using ManimML library:

- **[quiver_visualization.py](manim_animations/quiver_visualization.py)** - Visualize quiver extraction
- **[moduli_space_visualization.py](manim_animations/moduli_space_visualization.py)** - Moduli space concepts
- **[forward_pass_animation.py](manim_animations/forward_pass_animation.py)** - Data representation
- **[pruning_visualization.py](manim_animations/pruning_visualization.py)** - Pruning process
- **[complete_demo.py](manim_animations/complete_demo.py)** - Full pipeline visualization

## 🚀 Quick Start

### Prerequisites

Install required packages:
```bash
# Install Manim Community
pip install manim

# Install ManimML for neural network visualizations
pip install manim_ml

# Install project dependencies
pip install torch numpy matplotlib
```

### Running Animations

```bash
# Navigate to animations directory
cd education/manim_animations

# Render a specific animation (low quality for preview)
manim -pql quiver_visualization.py QuiverExtractionScene

# Render high quality
manim -pqh moduli_space_visualization.py ModuliSpaceScene

# Render all scenes in a file
manim -pql complete_demo.py
```

### Rendering Options

- `-pql` - Preview, low quality (fast)
- `-pqm` - Preview, medium quality
- `-pqh` - Preview, high quality
- `-pqk` - Preview, 4K quality
- `-s` - Save last frame as image

## 📖 Learning Path

### Beginner Path

1. Read [01_quiver_representations.md](01_quiver_representations.md)
2. Watch/render `quiver_visualization.py`
3. Read [02_moduli_space.md](02_moduli_space.md)
4. Watch/render `moduli_space_visualization.py`
5. Follow [tutorial.md](tutorial.md) with hands-on examples

### Advanced Path

1. Study the theoretical documents in depth
2. Examine all Manim animation source code
3. Modify animations to explore different architectures
4. Implement custom visualizations for your networks

## 🎯 Key Concepts Explained

### Quiver Representation

A quiver is a directed graph representing the network structure:
- **Vertices**: Neurons/feature maps
- **Edges**: Connections (weights)
- **Path algebra**: Composition of connections

### Moduli Space

The geometric space of network parameters modulo isomorphisms:
- **Dimension**: `d = #E - #Ṽ` (edges minus hidden vertices)
- **Coordinates**: Invariant descriptions of network function
- **Geometry**: Structure preserved under neuron reordering

### Data Representation (W_x^f)

For each input x, we compute its representation:
```
(W_x^f)_ε = W_ε * a_s(ε)(x) / Σ_δ W_δ * a_s(δ)(x)
```

This captures how the network processes specific inputs through each coordinate.

## 🎬 Animation Descriptions

### Quiver Extraction (`quiver_visualization.py`)

Shows the process of converting a neural network into a quiver representation:
1. Display the network architecture
2. Highlight vertices (neurons)
3. Highlight edges (weights)
4. Show the complete quiver structure
5. Demonstrate edge counting and vertex counting

### Moduli Space Visualization (`moduli_space_visualization.py`)

Illustrates the moduli space concept:
1. Show network parameter space
2. Demonstrate isomorphic networks (same function, different params)
3. Visualize the quotient space (moduli space)
4. Show coordinate system
5. Display dimension calculation

### Forward Pass Animation (`forward_pass_animation.py`)

Demonstrates data representation computation:
1. Input data flowing through network
2. Activation values at each layer
3. Coordinate values being computed
4. W_x^f representation visualization

### Pruning Visualization (`pruning_visualization.py`)

Shows the pruning process:
1. Compute coordinate importance scores
2. Highlight high/low importance coordinates
3. Remove low-importance coordinates
4. Show pruned network
5. Compare before/after performance

## 📊 Example Networks

All animations can be configured with different architectures:

```python
# Simple network
layers = [3, 5, 3]

# Deeper network
layers = [10, 20, 15, 10, 5]

# Convolutional network
# See individual scripts for CNN examples
```

## 🔧 Customization

All animation scripts are heavily commented and designed to be modified:

```python
# Change colors
NODE_COLOR = BLUE
EDGE_COLOR = GRAY
HIGHLIGHT_COLOR = YELLOW

# Adjust timing
ANIMATION_RUNTIME = 2.0
WAIT_TIME = 1.0

# Modify layout
LAYER_SPACING = 2.5
NODE_RADIUS = 0.2
```

## 📝 Additional Resources

### From the Main Project

- [../README.md](../README.md) - Project overview
- [../CLAUDE.md](../CLAUDE.md) - Developer guide
- [../moduli_space_pruning.py](../moduli_space_pruning.py) - Implementation
- [../experiments.py](../experiments.py) - Experimental framework

### External References

- **Paper**: "The Representation Theory of Neural Networks" by Armenta & Jodoin (2021)
- **Quiver Theory**: Standard texts on representation theory
- **Manim Documentation**: https://docs.manim.community/
- **ManimML Documentation**: https://github.com/helblazer811/ManimML

## 🤝 Contributing

To add new educational materials:

1. Follow the existing naming convention
2. Include detailed comments in animations
3. Provide clear mathematical explanations
4. Test all animations render correctly
5. Update this README with new content

## 📧 Questions?

If you have questions about the mathematical concepts or visualizations:
- Check the main project README
- Review the theory documents
- Examine the animation source code
- Open an issue on the main repository

## 🎓 Educational Use

These materials are designed for:
- **Students**: Learning about neural network pruning
- **Researchers**: Understanding moduli space methods
- **Educators**: Teaching representation theory
- **Practitioners**: Implementing principled pruning

Feel free to use and adapt these materials for educational purposes!

---

**Created**: 2025-12-04
**Issue**: "Create educational materials for quiver representations and moduli space"
**Status**: Complete educational package with theory and visualizations
