# The Moduli Space of Neural Networks

## Introduction

The **moduli space** is a geometric space that classifies neural networks up to isomorphism. This document explains what the moduli space is, how it's derived from neural networks, and why it's crucial for principled pruning.

---

## Motivation: Why Do We Need a Moduli Space?

### The Problem of Redundancy

Consider these two networks with identical function but different parameter values:

**Network A**:
```
x → [w₁=2, w₂=3] → [activation] → [w₃=4, w₄=5] → y
```

**Network B**:
```
x → [w₁=4, w₂=6] → [activation] → [w₃=2, w₄=2.5] → y
```

For ReLU networks, if we scale the first layer by α and the second layer by 1/α, we get the **same function**!

### Multiple Representations, Same Function

Similarly, reordering neurons in a layer doesn't change the network function:
- Network with neurons [h₁, h₂, h₃]
- Network with neurons [h₂, h₃, h₁] (permuted)
- **Same input-output mapping!**

### The Solution: Moduli Space

The **moduli space** identifies all these equivalent networks as a single point, giving us:
- Non-redundant parameterization
- Intrinsic dimension of the function space
- Principled coordinates for pruning

---

## Mathematical Definition

### Parameter Space

Start with the **parameter space** W of all possible weights:
```
W = ℝ^{#E}
```

Each point in W is a complete specification of network weights.

**Example**: For [3, 4, 2] network with 20 edges:
```
W = ℝ²⁰
```

### Group of Symmetries

Define a group **G̃** of transformations that preserve network function:

1. **Neuron Permutations (P)**: Reordering neurons within layers
2. **Scale Transformations (S)**: Positive scaling (for ReLU networks)
3. **Combined**: G̃ = P × S

### Quotient Space

The **moduli space** M is the quotient:
```
M = W / G̃
```

Each point in M represents an **equivalence class** of networks with the same function.

---

## Computing the Moduli Space Dimension

### The Dimension Formula

For a generic neural network:
```
dim(M) = #E - #Ṽ
```

Where:
- **#E**: Number of edges (weight connections)
- **#Ṽ**: Number of hidden vertices (neurons)

### Why This Formula?

**Intuition**:
- Start with #E parameters (dimension of parameter space)
- Each hidden neuron introduces redundancy through permutation/scaling
- Subtract #Ṽ to account for this redundancy
- Result: intrinsic dimension of function space

### Example Calculation

**Network**: [3, 4, 2]
- **Edges**: #E = 3×4 + 4×2 = 20
- **Hidden Vertices**: #Ṽ = 4
- **Moduli Dimension**: dim(M) = 20 - 4 = 16

**Interpretation**: The function space has 16 degrees of freedom, even though we have 20 parameters.

---

## Detailed Construction

### Step 1: Identify Symmetries

#### A. Neuron Permutation

For layer i with nᵢ neurons, we have a permutation group:
```
Pᵢ = S_{nᵢ} (symmetric group)
```

Total permutation group:
```
P = P₁ × P₂ × ... × P_{L-1}
```

**Example**: For [2, 3, 2], hidden layer has S₃ (6 permutations):
- [h₁, h₂, h₃]
- [h₁, h₃, h₂]
- [h₂, h₁, h₃]
- [h₂, h₃, h₁]
- [h₃, h₁, h₂]
- [h₃, h₂, h₁]

#### B. Scale Transformations

For ReLU networks, positive scaling preserves function:
```
ReLU(αx) = α · ReLU(x)  for α > 0
```

Can scale layer i by αᵢ > 0 and layer i+1 by 1/αᵢ:
```
W^{i+1} · ReLU(W^i · x) = (W^{i+1}/αᵢ) · ReLU(αᵢ · W^i · x)
```

Scale group per hidden layer:
```
S = (ℝ₊)^{#Ṽ}
```

### Step 2: Quotient Construction

The group G̃ = P × S acts on parameter space W:
```
g · w = transformed weights
```

Two points w₁, w₂ ∈ W are equivalent if:
```
∃g ∈ G̃ : w₂ = g · w₁
```

The moduli space is:
```
M = W / G̃ = {equivalence classes [w]}
```

---

## Coordinates on the Moduli Space

### Data-Dependent Coordinates

For a data point x, define the **data representation** W_x^f:

```
(W_x^f)_ε = W_ε · a_{s(ε)}(x) / Σ_{δ∈ζ_{s(ε)}} W_δ · a_{s(δ)}(x)
```

Where:
- **W_ε**: Weight of edge ε
- **a_v(x)**: Activation at vertex v for input x
- **ζ_v**: All edges targeting vertex v
- **s(ε)**: Source vertex of edge ε

### Interpretation

This coordinate system:
- **Normalizes** by total incoming activation
- **Captures** relative importance of each edge
- **Is invariant** under the symmetries in G̃
- **Depends on data** x (different coordinates for different inputs)

### Example Computation

Network: x → [w₁, w₂] → h → [w₃] → y

**For input x = 1**:
- Activation at h: a_h(1) = ReLU(w₁ · 1 + w₂ · 1) = ReLU(w₁ + w₂)
- Edge from x to h via w₁:
  ```
  (W_x^f)_{ε₁} = w₁ · 1 / (w₁ · 1 + w₂ · 1) = w₁/(w₁ + w₂)
  ```
- Edge from x to h via w₂:
  ```
  (W_x^f)_{ε₂} = w₂ · 1 / (w₁ · 1 + w₂ · 1) = w₂/(w₁ + w₂)
  ```

Notice: (W_x^f)_{ε₁} + (W_x^f)_{ε₂} = 1 (normalized!)

---

## Properties of the Moduli Space

### 1. Dimension Reduction

**Before**: ℝ^{#E} (high-dimensional, redundant)
**After**: M with dim(M) = #E - #Ṽ (reduced by removing redundancy)

**Example**:
- Network [10, 20, 10, 5]
- Parameters: #E = 10×20 + 20×10 + 10×5 = 450
- Hidden neurons: #Ṽ = 20 + 10 = 30
- Moduli dimension: 450 - 30 = 420 (93% of parameters)

### 2. Invariance

Moduli coordinates are invariant under:
- ✅ **Permuting neurons** within layers
- ✅ **Positive scaling** of layers (ReLU networks)
- ✅ **Any transformation** in G̃

### 3. Geometry

The moduli space inherits geometric structure:
- **Metric**: Distance between network functions
- **Topology**: Continuous deformations
- **Stratification**: Networks with different effective dimensions

### 4. Data Manifold

For a dataset D, we get a **data manifold** in M:
```
Φ(D) = {W_x^f : x ∈ D} ⊂ M
```

This is the "footprint" of the data in the moduli space.

---

## Connection to Pruning

### Why Moduli Space Matters for Pruning

**Key Insight**: Pruning in moduli space is more principled than pruning in parameter space.

#### Parameter Space Pruning (Traditional)

```
Remove weight if |wᵢ| < threshold
```

**Problems**:
- Not invariant (depends on parameterization)
- Doesn't account for correlations
- No notion of "importance" for network function

#### Moduli Space Pruning (Our Method)

```
Remove coordinate if importance(εᵢ) < threshold
```

**Advantages**:
- ✅ Invariant under symmetries
- ✅ Accounts for data distribution
- ✅ Principled importance measure
- ✅ Preserves network function on data manifold

### Coordinate Importance

For each coordinate ε, compute importance across dataset:

**Three Metrics**:

1. **Usage Rate (ρ_ε)**:
   ```
   ρ_ε = |{x ∈ D : (W_x^f)_ε > threshold}| / |D|
   ```

2. **Variance (σ²_ε)**:
   ```
   σ²_ε = Var({(W_x^f)_ε : x ∈ D})
   ```

3. **Information Content (I_ε)**:
   ```
   I_ε = -Σ p(v) log p(v)  [entropy of coordinate values]
   ```

**Combined Importance**:
```
Importance_ε = ρ_ε · √(σ²_ε) · (1 + I_ε)
```

### Pruning Decision

1. Compute importance for all coordinates
2. Sort coordinates by importance
3. Remove lowest k% coordinates
4. Map back to parameter space (set weights to zero)
5. Fine-tune the pruned network

---

## Visualization in Moduli Space

### Conceptual Picture

```
Parameter Space (ℝ^{#E})
    ↓ (quotient by G̃)
Moduli Space (dim = #E - #Ṽ)
    ↑
Data Manifold (coordinates W_x^f)
    ↑
Individual Data Points
```

### Example: 2D Projection

For visualization, we can project to 2D:

```
        Coordinate 2
              ↑
              |     • (high importance)
              |   ••
              | •    •
      --------+--------→ Coordinate 1
              |
              | °°°  (low importance)
              |  °
```

High-importance coordinates cluster around the data manifold.
Low-importance coordinates are sparse or low-variance.

---

## Implementation in Our Codebase

### Computing Data Representation

```python
# moduli_space_pruning.py

class DataRepresentation:
    def compute_representation(self, x):
        """Compute W_x^f for input x"""
        # Forward pass to get activations
        _ = self.network(x)

        representation = {}
        for name, module in self.network.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # Get activations before and after this layer
                input_act = self.pre_activation_cache[name]
                output_act = self.activation_cache[name]

                # Compute normalized representation
                # This implements the formula from the paper
                rep = self._compute_layer_representation(
                    module, input_act, output_act
                )
                representation[name] = rep

        return representation
```

### Analyzing Moduli Coordinates

```python
class ModuliSpaceAnalyzer:
    def analyze_dataset(self, dataloader, max_samples=500):
        """Analyze coordinate usage across dataset"""

        # Collect coordinate values
        coordinate_values = defaultdict(list)

        for batch_idx, (data, _) in enumerate(dataloader):
            if batch_idx * len(data) >= max_samples:
                break

            # Compute representation for each sample
            for sample in data:
                rep = self.data_rep.compute_representation(sample.unsqueeze(0))

                # Store coordinate values
                for edge_idx, _, layer_name in self.quiver.edges:
                    if layer_name in rep:
                        value = self._extract_coordinate_value(
                            rep[layer_name], edge_idx
                        )
                        coordinate_values[edge_idx].append(value)

        # Compute importance for each coordinate
        self._compute_importance_scores(coordinate_values)
```

---

## Advanced Topics

### 1. Stratification

Not all networks are generic:
- Some have **additional symmetries** (e.g., tied weights)
- Some lie on **lower-dimensional strata**
- Moduli space is **stratified** by network structure

### 2. Singularities

Networks with **degenerate behavior** correspond to singularities:
- Dead neurons (always inactive)
- Redundant paths (linearly dependent)
- These are **lower-dimensional subspaces** of M

### 3. Fiber Bundles

More formally, we have a **principal G̃-bundle**:
```
G̃ → W → M
```

Moduli coordinates are sections of this bundle.

### 4. Differential Geometry

The moduli space has:
- **Riemannian metric**: Fisher information metric
- **Geodesics**: Optimal paths in function space
- **Curvature**: Relates to training dynamics

---

## Practical Implications

### For Pruning

1. **Identify redundancy**: Dim reduction #E → #E - #Ṽ
2. **Coordinate importance**: Data-driven, invariant measure
3. **Preserve function**: Keep high-importance coordinates
4. **Efficiency**: Prune in intrinsic dimensions, not raw parameters

### For Training

1. **Reparameterization invariance**: Training should be independent of initialization symmetries
2. **Natural gradient**: Follows geometry of moduli space
3. **Regularization**: Can penalize movement in moduli space

### For Architecture Search

1. **Effective capacity**: Moduli dimension, not parameter count
2. **Comparison**: Networks with different architectures but same moduli dimension have similar capacity
3. **Design principle**: Maximize moduli dimension for given parameter budget

---

## Summary

**Key Concepts**:

1. ✅ **Moduli space** = Parameter space / Symmetries
2. ✅ **Dimension** = #E - #Ṽ (reduces redundancy)
3. ✅ **Coordinates** = Data representations W_x^f
4. ✅ **Invariance** = Robust to neuron permutations and scaling
5. ✅ **Pruning** = Remove low-importance moduli coordinates
6. ✅ **Geometry** = Intrinsic structure of network functions

**Next**: [tutorial.md](tutorial.md) - Hands-on examples combining quiver representations and moduli space for pruning.

---

## Further Reading

- **Armenta & Jodoin (2021)**: Full mathematical treatment
- **Algebraic Geometry**: Moduli spaces in general
- **Differential Geometry**: Riemannian structure
- **Information Geometry**: Fisher metric and natural gradient

---

**Related Animations**: `manim_animations/moduli_space_visualization.py`
**Implementation**: `../moduli_space_pruning.py` - `ModuliSpaceAnalyzer` class
