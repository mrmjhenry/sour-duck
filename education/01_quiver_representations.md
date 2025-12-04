# Quiver Representations and Neural Networks

## Introduction

A **quiver** is a fundamental mathematical structure in representation theory consisting of vertices connected by directed edges. This document explains how neural networks can be naturally represented as quivers, providing the foundation for our pruning methodology.

---

## What is a Quiver?

### Mathematical Definition

A **quiver** Q is a directed graph consisting of:
- A set of **vertices** V (also called nodes)
- A set of **edges** E (also called arrows)
- Two functions: **source** s: E → V and **target** t: E → V

Each edge ε has a source vertex s(ε) and a target vertex t(ε), representing a directed connection.

### Example: Simple Quiver

```
    v₁ ─ε₁→ v₂ ─ε₂→ v₃
         ↘ε₃↗
```

This quiver has:
- Vertices: {v₁, v₂, v₃}
- Edges: {ε₁, ε₂, ε₃}
- s(ε₁) = v₁, t(ε₁) = v₂
- s(ε₂) = v₂, t(ε₂) = v₃
- s(ε₃) = v₁, t(ε₃) = v₃

---

## Neural Networks as Quivers

### The Natural Mapping

A feed-forward neural network can be represented as a quiver where:

1. **Vertices** correspond to **neurons** (or feature maps in CNNs)
2. **Edges** correspond to **connections** (weights)

### Example: Simple Neural Network

Consider a network with architecture [2, 3, 2]:

```
Input Layer (2 neurons)  →  Hidden Layer (3 neurons)  →  Output Layer (2 neurons)

    x₁  ────────→  h₁  ────────→  y₁
     ↘  ↘      ↗  ↓  ↘      ↗
       ↘   ↘  ↗    ↓    ↘  ↗
         ↘   h₂  ──┼──────→  y₂
           ↗  ↓  ↘ ↓  ↗
    x₂  ──────────→  h₃
```

**Quiver Representation**:
- **Vertices**: V = {x₁, x₂, h₁, h₂, h₃, y₁, y₂}
- **Edges**: One edge for each weight connection
  - From x₁: {(x₁→h₁), (x₁→h₂), (x₁→h₃)}
  - From x₂: {(x₂→h₁), (x₂→h₂), (x₂→h₃)}
  - From h₁: {(h₁→y₁), (h₁→y₂)}
  - From h₂: {(h₂→y₁), (h₂→y₂)}
  - From h₃: {(h₃→y₁), (h₃→y₂)}
- **Total**: 7 vertices, 12 edges

---

## Quiver Representations with Weights

### Adding Structure: Representations

A **representation** of a quiver Q over a field K (usually ℝ for neural networks) assigns:
- A vector space Vᵢ to each vertex vᵢ
- A linear map W_ε: V_{s(ε)} → V_{t(ε)} to each edge ε

For neural networks:
- Each vertex vᵢ has a vector space ℝ^{dim(vᵢ)}
- Each edge ε has a weight matrix W_ε

### Neural Network as Quiver Representation

For a network with layers of sizes [n₀, n₁, ..., n_L]:

**Vertices**:
- Input vertices: ℝ^{n₀}
- Hidden vertices (layer i): ℝ^{nᵢ} for i = 1, ..., L-1
- Output vertices: ℝ^{n_L}

**Edges**:
- Between layer i and i+1: nᵢ × n_{i+1} weight matrices
- Each entry in the matrix is an edge in the quiver

---

## Detailed Example: Layer-by-Layer Breakdown

### Network Architecture: [3, 4, 2]

**Layer 0 (Input)**: 3 neurons {x₁, x₂, x₃}
**Layer 1 (Hidden)**: 4 neurons {h₁, h₂, h₃, h₄}
**Layer 2 (Output)**: 2 neurons {y₁, y₂}

### Edge Extraction

**From Layer 0 to Layer 1** (Weight matrix W¹: 4×3):
```
         x₁      x₂      x₃
    ┌─────────────────────┐
h₁  │ w¹₁₁    w¹₁₂    w¹₁₃ │
h₂  │ w¹₂₁    w¹₂₂    w¹₂₃ │
h₃  │ w¹₃₁    w¹₃₂    w¹₃₃ │
h₄  │ w¹₄₁    w¹₄₂    w¹₄₃ │
    └─────────────────────┘
```

Creates 12 edges:
- ε₁: (x₁ → h₁, w¹₁₁)
- ε₂: (x₁ → h₂, w¹₂₁)
- ...
- ε₁₂: (x₃ → h₄, w¹₄₃)

**From Layer 1 to Layer 2** (Weight matrix W²: 2×4):
```
         h₁      h₂      h₃      h₄
    ┌───────────────────────────────┐
y₁  │ w²₁₁    w²₁₂    w²₁₃    w²₁₄  │
y₂  │ w²₂₁    w²₂₂    w²₂₃    w²₂₄  │
    └───────────────────────────────┘
```

Creates 8 edges:
- ε₁₃: (h₁ → y₁, w²₁₁)
- ε₁₄: (h₁ → y₂, w²₂₁)
- ...
- ε₂₀: (h₄ → y₂, w²₂₄)

**Total**:
- Vertices: V = 3 + 4 + 2 = 9
- Edges: E = 12 + 8 = 20

---

## Paths in the Quiver

### Definition

A **path** in a quiver is a sequence of composable edges:
```
p = ε_k ∘ ε_{k-1} ∘ ... ∘ ε_2 ∘ ε_1
```
where t(ε_i) = s(ε_{i+1})

### Neural Network Interpretation

Paths correspond to **information flow** through the network:

**Example Path**: x₁ → h₂ → y₁
- Composed of edges: ε_j: (x₁ → h₂) and ε_k: (h₂ → y₁)
- Weight contribution: w¹₂₁ × w²₁₂

### Basis Paths

For ReLU networks, **basis paths** are fundamental:
- Paths from input to output that remain active for some inputs
- Determine the network's representational capacity
- Related to the number of linear regions in the network's function

---

## Key Properties for Pruning

### 1. Vertex Count

**Hidden Vertices (Ṽ)**: All non-input, non-output neurons
```
#Ṽ = Σᵢ₌₁^{L-1} nᵢ
```

For [3, 4, 2]: #Ṽ = 4

### 2. Edge Count

**Total Edges (E)**: All weight connections
```
#E = Σᵢ₌₀^{L-1} nᵢ × n_{i+1}
```

For [3, 4, 2]: #E = 3×4 + 4×2 = 12 + 8 = 20

### 3. Parameter Space Dimension

The network parameter space is ℝ^{#E} (each edge has a weight)

---

## Convolutional Networks as Quivers

### Adaptation for CNNs

For convolutional layers:
- **Vertices**: Feature maps (channels)
- **Edges**: Filters/kernels

**Example**: Conv2D(3 channels → 64 channels, 3×3 kernel)
- Creates 64 vertices (one per output channel)
- Creates 3 × 64 = 192 edge groups (each is a 3×3 filter)
- Can be simplified by treating each filter as a single edge

### Practical Representation

In our implementation (`moduli_space_pruning.py`):

```python
class QuiverRepresentation:
    def _extract_edges(self):
        edges = []
        edge_idx = 0

        for name, module in self.network.named_modules():
            if isinstance(module, nn.Linear):
                # Each weight is an edge
                for i in range(module.in_features):
                    for j in range(module.out_features):
                        edges.append((edge_idx, (i, j), name))
                        edge_idx += 1

            elif isinstance(module, nn.Conv2d):
                # Each filter is treated as an edge
                for i in range(module.out_channels):
                    edges.append((edge_idx, i, name))
                    edge_idx += 1
```

---

## Why Quiver Representations Matter

### 1. Mathematical Rigor

Quiver theory provides:
- Formal language for network structure
- Well-understood algebraic properties
- Connection to representation theory

### 2. Invariance

Quiver structure is invariant under:
- **Neuron permutations**: Reordering neurons within a layer
- **Scaling**: Positive scale transformations (for ReLU)
- **Basis changes**: Linear reparameterizations

### 3. Compositionality

Quivers capture:
- How information flows
- How transformations compose
- How features are constructed hierarchically

### 4. Dimension Analysis

The quiver structure determines:
- Parameter count: #E
- Redundancy: Related to #E - #Ṽ
- Degrees of freedom in the moduli space

---

## Connection to Moduli Space

The quiver representation is the foundation for the moduli space:

1. **Parameter space**: ℝ^{#E} (all possible weights)
2. **Equivalence**: Networks with the same function (up to neuron reordering)
3. **Quotient space**: Parameter space / Symmetries
4. **Moduli space**: M = ℝ^{#E} / G̃ where G̃ is the symmetry group

**Key Insight**: The moduli space dimension is approximately:
```
dim(M) = #E - #Ṽ
```

This reduces redundancy by accounting for permutation symmetries.

---

## Visualization

See `manim_animations/quiver_visualization.py` for animated visualizations showing:
1. Network architecture
2. Vertex identification
3. Edge extraction
4. Complete quiver structure
5. Path composition
6. Dimension calculations

---

## Implementation Details

### From Our Codebase

```python
# moduli_space_pruning.py

class QuiverRepresentation:
    """Represents a neural network as a quiver."""

    def __init__(self, network: nn.Module):
        self.network = network
        self.edges = self._extract_edges()      # List of (idx, info, name)
        self.vertices = self._extract_vertices() # Dict of {name: dim}

    def get_num_edges(self) -> int:
        """Total number of edges (#E)"""
        return len(self.edges)

    def get_num_hidden_vertices(self) -> int:
        """Number of hidden vertices (#Ṽ)"""
        # Count all non-input/output neurons
        return sum(dim for name, dim in self.vertices.items()
                  if 'hidden' in name or 'layer' in name)
```

### Edge Tuple Format

Each edge is a 3-tuple: `(edge_idx, edge_info, layer_name)`
- `edge_idx`: Unique identifier
- `edge_info`: Position in weight matrix (i, j) for Linear, filter index for Conv2d
- `layer_name`: Name of the layer this edge belongs to

---

## Summary

**Key Takeaways**:

1. ✅ Neural networks are naturally represented as quivers
2. ✅ Vertices = neurons, Edges = weights
3. ✅ Paths = information flow through network
4. ✅ Quiver structure determines #E and #Ṽ
5. ✅ Foundation for moduli space construction
6. ✅ Enables principled pruning decisions

**Next**: [02_moduli_space.md](02_moduli_space.md) - Learn how the moduli space is derived from the quiver representation.

---

## Further Reading

- **Armenta & Jodoin (2021)**: "The Representation Theory of Neural Networks"
- **Representation Theory**: Gabriel's theorem on quiver representations
- **Path Algebras**: Composition of linear maps in quivers
- **Neural Network Theory**: Connection to basis paths and linear regions

---

**Related Animations**: `manim_animations/quiver_visualization.py`
**Implementation**: `../moduli_space_pruning.py` - `QuiverRepresentation` class
