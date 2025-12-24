# Methods: Moduli Space Coordinate Pruning

## Overview

This document describes the mathematical framework for computing coordinate representations in neural networks and using them for principled pruning decisions. The method is based on **representation theory** and provides mathematically grounded importance scores for network components.

---

## Table of Contents

1. [Coordinate Representation Computation](#coordinate-representation-computation)
2. [The Three Importance Metrics](#the-three-importance-metrics)
3. [Combined Importance Score](#combined-importance-score)
4. [Pruning Information Pathways](#pruning-information-pathways)
5. [Worked Example](#worked-example)
6. [Computational Implementation](#computational-implementation)

---

## Coordinate Representation Computation

### The Core Formula

For a data point **x** and network function **f** with weights **W**, the coordinate representation is:

$$
(W_x^f)_\varepsilon = \frac{W_\varepsilon \cdot a_{s(\varepsilon)}^{(W,f)}(x)}{\sum_{\delta \in \zeta_{s(\varepsilon)}} W_\delta \cdot a_{s(\delta)}^{(W,f)}(x)}
$$

Where:
- **$(W_x^f)_\varepsilon$**: The value of coordinate ε for data point x
- **$W_\varepsilon$**: The weight associated with edge ε
- **$a_{s(\varepsilon)}^{(W,f)}(x)$**: The activation at the source vertex of edge ε
- **$\zeta_{s(\varepsilon)}$**: All edges emanating from the same source vertex
- **Denominator**: Normalization over all edges from the same source

### Intuitive Explanation

This formula captures **how much each edge (connection) contributes to processing a specific data point**:

1. **Numerator**: The contribution of edge ε = (weight × activation)
2. **Denominator**: Total contribution of all edges from the same neuron
3. **Result**: The relative importance of this specific connection for this data point

### Example: Simple Network

Consider a network layer with 3 input neurons connecting to 1 output neuron:

```
Input Layer:        Hidden Layer:
   [x₁]  ─── w₁ ───┐
   [x₂]  ─── w₂ ───├──→ [h₁]
   [x₃]  ─── w₃ ───┘
```

For an input **x = [0.8, 0.3, 0.1]** with weights **w = [0.5, -0.2, 0.9]**:

**Coordinate for edge 1**:
```
Numerator:   w₁ × x₁ = 0.5 × 0.8 = 0.4
Denominator: |w₁ × x₁| + |w₂ × x₂| + |w₃ × x₃|
           = |0.4| + |-0.06| + |0.09| = 0.55

(W_x^f)₁ = 0.4 / 0.55 ≈ 0.727
```

**Interpretation**: Edge 1 contributes ~73% of the total signal for this data point.

---

## The Three Importance Metrics

Each coordinate is analyzed across the entire dataset to compute three complementary metrics that together determine its importance.

### 1. Usage Rate (ρ)

**Definition**: The fraction of data points for which this coordinate is significantly active.

**Formula**:
$$
\rho = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}(|(W_{x_i}^f)_\varepsilon| > \epsilon)
$$

Where:
- **N**: Total number of data points
- **$\mathbb{1}(\cdot)$**: Indicator function (1 if true, 0 if false)
- **ε**: Threshold (typically 10⁻⁶)

**What it measures**: How often this coordinate "fires" across different inputs.

**Example Data**:
```python
# Coordinate values across 10 samples
coordinate_values = [0.73, 0.02, 0.85, 0.0, 0.67, 0.91, 0.0, 0.78, 0.03, 0.82]

# Count active (> 1e-6)
active = [0.73, 0.02, 0.85, 0.67, 0.91, 0.78, 0.03, 0.82]  # 8 out of 10

usage_rate = 8 / 10 = 0.80  # 80% usage
```

**Interpretation**:
- **ρ ≈ 1.0**: Coordinate active for almost all inputs → **highly important**
- **ρ ≈ 0.5**: Coordinate active for half of inputs → **moderately important**
- **ρ ≈ 0.0**: Coordinate rarely active → **candidate for pruning**

### 2. Variance (σ²)

**Definition**: The spread of coordinate values, measuring how much the coordinate's contribution varies across different inputs.

**Formula**:
$$
\sigma^2 = \frac{1}{N} \sum_{i=1}^{N} \left((W_{x_i}^f)_\varepsilon - \mu\right)^2
$$

Where:
- **μ**: Mean coordinate value across all samples
- **N**: Number of samples

**What it measures**: The diversity of responses this coordinate produces.

**Example Data**:
```python
coordinate_values = [0.73, 0.02, 0.85, 0.0, 0.67, 0.91, 0.0, 0.78, 0.03, 0.82]

# Calculate mean
mean = sum(coordinate_values) / 10 = 5.81 / 10 = 0.581

# Calculate variance
deviations² = [(0.73-0.581)², (0.02-0.581)², ..., (0.82-0.581)²]
            = [0.022, 0.315, 0.072, 0.338, 0.008, 0.108, 0.338, 0.040, 0.304, 0.057]

variance = sum(deviations²) / 10 = 1.602 / 10 = 0.160
```

**Interpretation**:
- **High variance (σ² ≫ 0)**: Coordinate responds differently to different inputs → **discriminative**
- **Low variance (σ² ≈ 0)**: Coordinate produces similar values for all inputs → **redundant**

### 3. Information Content (I)

**Definition**: An entropy-based measure of how informative the coordinate's activation pattern is.

**Formula**:
$$
I = -\rho \log(\rho + \epsilon) - (1 - \rho) \log(1 - \rho + \epsilon)
$$

Where:
- **ρ**: Usage rate (computed above)
- **ε**: Small constant for numerical stability (10⁻⁸)

**What it measures**: The uncertainty in whether this coordinate will be active for a random input.

**Example Calculation**:
```python
usage_rate = 0.80

# Information content (binary entropy)
I = -(0.80 × log(0.80)) - (0.20 × log(0.20))
  = -(0.80 × -0.223) - (0.20 × -1.609)
  = 0.178 + 0.322
  = 0.500
```

**Interpretation by Usage Rate**:

| Usage Rate (ρ) | Information (I) | Meaning |
|----------------|-----------------|---------|
| 0.0 or 1.0 | 0.0 | No information (always off or always on) |
| 0.5 | 0.693 | **Maximum information** (unpredictable) |
| 0.1 or 0.9 | ~0.33 | Moderate information |
| 0.8 | 0.50 | Moderate-high information |

**Why this matters**: A coordinate that's active for exactly 50% of inputs (ρ = 0.5) carries the most information because it's most "selective" about which inputs it responds to.

---

## Combined Importance Score

The three metrics are combined into a single importance score:

$$
\text{Importance} = \rho \times \sqrt{\sigma^2} \times (1 + I)
$$

### Why This Combination?

1. **ρ (Usage Rate)**: Ensures we don't keep coordinates that are rarely used
2. **√σ² = σ (Standard Deviation)**: Rewards coordinates with diverse responses
3. **(1 + I) (Scaled Information)**: Adds 0–0.693 bonus for informative patterns

The formula balances three complementary aspects:
- **Frequency**: How often it's used (ρ)
- **Diversity**: How varied its responses are (σ)
- **Selectivity**: How informative its activation pattern is (I)

### Worked Example

Using our example data from above:

```python
# Metrics computed earlier
usage_rate = 0.80
variance = 0.160
information = 0.500

# Combined importance
importance = usage_rate × sqrt(variance) × (1 + information)
           = 0.80 × sqrt(0.160) × (1 + 0.500)
           = 0.80 × 0.400 × 1.500
           = 0.480
```

**Interpretation**:
- This coordinate has an importance score of **0.48**
- On a scale of 0 to 1, this is **moderate-high importance**
- It should likely be **kept** during pruning (if we're pruning the lowest 30–50%)

### Example Comparison

| Coordinate | ρ | σ² | I | Importance | Decision |
|------------|---|----|----|------------|----------|
| A | 0.95 | 0.25 | 0.20 | **0.57** | ✅ **Keep** (high usage, high variance) |
| B | 0.80 | 0.16 | 0.50 | **0.48** | ✅ **Keep** (balanced) |
| C | 0.50 | 0.10 | 0.69 | **0.27** | ⚠️ **Maybe** (moderate) |
| D | 0.15 | 0.05 | 0.34 | **0.05** | ❌ **Prune** (low usage) |
| E | 0.90 | 0.01 | 0.33 | **0.12** | ❌ **Prune** (no variance) |

**Key Insights**:
- **Coordinate A**: High importance due to frequent use and high variance
- **Coordinate D**: Low importance due to rare activation (only 15% usage)
- **Coordinate E**: Low importance despite high usage because it has no variance (redundant)

---

## Pruning Information Pathways

### The Pruning Process

Once importance scores are computed for all coordinates, pruning follows these steps:

#### Step 1: Rank Coordinates by Importance

```
All coordinates sorted (lowest to highest importance):
[0.02, 0.03, 0.05, ..., 0.27, 0.48, 0.57, 0.61, ..., 0.85, 0.92]
 ↑                          ↑                            ↑
 Lowest                   Median                      Highest
```

#### Step 2: Select Pruning Threshold

For a **pruning ratio of 30%**, select the bottom 30% of coordinates:

```python
n_total = len(coordinates)  # e.g., 10,000 coordinates
n_prune = int(n_total * 0.30)  # 3,000 coordinates to prune

# Sort and select
coords_sorted = sorted(coordinates, key=lambda c: c.importance_score)
prunable = coords_sorted[:n_prune]  # Bottom 3,000
kept = coords_sorted[n_prune:]      # Top 7,000
```

#### Step 3: Map Coordinates to Network Weights

Each coordinate corresponds to a specific weight in the network:

```
Coordinate → Edge in Quiver → Weight in Layer
    ε₁₂    →  Layer2.weight[5,3] → Set to 0
    ε₂₅    →  Layer3.weight[8,1] → Set to 0
    ...
```

#### Step 4: Apply Pruning Masks

```python
for coordinate in prunable_coordinates:
    layer_name = coordinate.layer_name
    weight_index = coordinate.weight_index

    # Create mask (1 = keep, 0 = prune)
    mask[layer_name][weight_index] = 0

# Apply masks to network
for layer in network.layers:
    layer.weight *= mask[layer.name]
```

### Information Pathway Visualization

**Before Pruning**:
```
Input → [All connections active] → Hidden → [All connections active] → Output
        [████████████████████]            [████████████████████]
        100% of pathways                  100% of pathways
```

**After 30% Pruning** (removing lowest importance coordinates):
```
Input → [High-importance paths] → Hidden → [High-importance paths] → Output
        [██████████░░░░░░░░]              [██████████░░░░░░░░]
        70% of pathways                    70% of pathways
        (kept strongest)                   (kept strongest)
```

### What Gets Pruned?

**Coordinates pruned** (low importance):
1. **Rarely used connections** (low ρ): Active for <10% of data
2. **Redundant connections** (low σ²): Always produce similar values
3. **Non-selective connections** (low I): Uninformative activation patterns

**Coordinates kept** (high importance):
1. **Frequently used** (high ρ): Active for >80% of data
2. **Discriminative** (high σ²): Produce diverse responses
3. **Selective** (high I): Activate for specific input patterns

### Example: Real Network Pruning

Consider a layer with 1000 weights:

**Before Pruning**:
- Total weights: 1,000
- All connections active: 1,000 pathways
- Many redundant/weak pathways

**Importance Distribution**:
```
Importance Score Distribution:
   │
 # │     ██
 o │    ████
 f │   ██████
   │  ████████
 w │ ██████████
 e │████████████
 i │████████████
 g │████████████
 h │████████████
 t │████████████
 s │████████████
   └────────────────────────
    0.0  0.2  0.4  0.6  0.8  1.0
         Importance Score
```

**After 30% Pruning** (remove 300 lowest importance weights):
- Active weights: 700
- Pruned weights: 300 (set to 0)
- **Key insight**: The 300 pruned weights contributed minimally to accuracy!

---

## Worked Example

### Complete Calculation for One Coordinate

Let's trace through the complete calculation for a single coordinate across a dataset.

#### Dataset
```python
# 5 training samples, 2D input
samples = [
    [0.8, 0.3],  # Sample 1
    [0.2, 0.9],  # Sample 2
    [0.6, 0.1],  # Sample 3
    [0.9, 0.7],  # Sample 4
    [0.1, 0.4],  # Sample 5
]
```

#### Network
```
Simple 2-layer network: [2, 3, 1]
Layer 1 weights:
  w₁₁ = 0.5,  w₁₂ = -0.3
  w₂₁ = 0.8,  w₂₂ = 0.2
  w₃₁ = -0.1, w₃₂ = 0.9

Focus on: Edge ε = (input neuron 1 → hidden neuron 1)
Weight: w₁₁ = 0.5
```

#### Step 1: Compute Coordinate Values

For each sample, compute $(W_x^f)_\varepsilon$:

**Sample 1: x = [0.8, 0.3]**
```
Numerator:   w₁₁ × x₁ = 0.5 × 0.8 = 0.40
Denominator: |w₁₁ × x₁| + |w₁₂ × x₂| = |0.40| + |-0.09| = 0.49
Coordinate value: 0.40 / 0.49 = 0.816
```

**Sample 2: x = [0.2, 0.9]**
```
Numerator:   0.5 × 0.2 = 0.10
Denominator: |0.10| + |-0.27| = 0.37
Coordinate value: 0.10 / 0.37 = 0.270
```

**Sample 3: x = [0.6, 0.1]**
```
Numerator:   0.5 × 0.6 = 0.30
Denominator: |0.30| + |-0.03| = 0.33
Coordinate value: 0.30 / 0.33 = 0.909
```

**Sample 4: x = [0.9, 0.7]**
```
Numerator:   0.5 × 0.9 = 0.45
Denominator: |0.45| + |-0.21| = 0.66
Coordinate value: 0.45 / 0.66 = 0.682
```

**Sample 5: x = [0.1, 0.4]**
```
Numerator:   0.5 × 0.1 = 0.05
Denominator: |0.05| + |-0.12| = 0.17
Coordinate value: 0.05 / 0.17 = 0.294
```

**Coordinate values**: [0.816, 0.270, 0.909, 0.682, 0.294]

#### Step 2: Compute Usage Rate

```python
threshold = 1e-6
active = [True, True, True, True, True]  # All > 1e-6

usage_rate = 5 / 5 = 1.0  # 100% usage
```

#### Step 3: Compute Variance

```python
mean = (0.816 + 0.270 + 0.909 + 0.682 + 0.294) / 5 = 0.594

deviations² = [
    (0.816 - 0.594)² = 0.049,
    (0.270 - 0.594)² = 0.105,
    (0.909 - 0.594)² = 0.099,
    (0.682 - 0.594)² = 0.008,
    (0.294 - 0.594)² = 0.090,
]

variance = (0.049 + 0.105 + 0.099 + 0.008 + 0.090) / 5 = 0.070
```

#### Step 4: Compute Information Content

```python
ρ = 1.0

I = -(1.0 × log(1.0 + 1e-8)) - (0.0 × log(0.0 + 1e-8))
  = -(1.0 × 0) - (0 × large_negative)
  = 0.0
```

**Note**: When usage_rate = 1.0 (always active), information content = 0 because there's no uncertainty.

#### Step 5: Compute Combined Importance

```python
importance = ρ × √σ² × (1 + I)
           = 1.0 × √0.070 × (1 + 0.0)
           = 1.0 × 0.265 × 1.0
           = 0.265
```

#### Step 6: Pruning Decision

```
Importance = 0.265 (moderate)

If pruning threshold is 0.30:
  → KEEP (importance > threshold)

If pruning threshold is 0.20:
  → PRUNE (importance < threshold)
```

---

## Computational Implementation

### Algorithm Pseudocode

```python
def analyze_and_prune(network, dataloader, pruning_ratio=0.3):
    """Complete coordinate analysis and pruning pipeline"""

    # 1. Extract quiver representation
    quiver = extract_quiver(network)
    # Returns: edges = [(edge_idx, weight_info, layer_name), ...]

    # 2. Analyze dataset
    coordinate_values = defaultdict(list)

    for batch in dataloader:
        for sample in batch:
            # Compute data representation for this sample
            representation = compute_representation(network, sample)

            # Extract coordinate value for each edge
            for edge_idx, _, layer_name in quiver.edges:
                if layer_name in representation:
                    coord_value = representation[layer_name].mean()
                    coordinate_values[edge_idx].append(coord_value)

    # 3. Compute importance scores
    importance_scores = {}

    for edge_idx, values in coordinate_values.items():
        # Metric 1: Usage rate
        usage = mean(abs(values) > 1e-6)

        # Metric 2: Variance
        variance = var(values)

        # Metric 3: Information content
        if 0 < usage < 1:
            info = -usage * log(usage) - (1-usage) * log(1-usage)
        else:
            info = 0

        # Combined importance
        importance = usage * sqrt(variance) * (1 + info)
        importance_scores[edge_idx] = importance

    # 4. Select coordinates to prune
    sorted_coords = sorted(importance_scores.items(),
                          key=lambda x: x[1])
    n_prune = int(len(sorted_coords) * pruning_ratio)
    to_prune = [coord for coord, _ in sorted_coords[:n_prune]]

    # 5. Create and apply pruning masks
    masks = create_masks(network, to_prune)
    apply_masks(network, masks)

    return network, masks
```

### Complexity Analysis

**Time Complexity**:
- Coordinate extraction: O(N × P) where N = samples, P = parameters
- Statistics computation: O(P) for each metric
- Sorting: O(P log P)
- **Total**: O(N × P + P log P) ≈ O(N × P)

**Space Complexity**:
- Coordinate values: O(N × P)
- Can be reduced with mini-batch processing

**Optimizations**:
1. **Limit samples**: Analyze 500-1000 samples instead of entire dataset
2. **Batch processing**: Compute statistics incrementally
3. **Layer caching**: Precompute layer values to avoid redundant lookups

---

## Key Insights

### Why This Works

1. **Mathematical Foundation**: Based on representation theory, not heuristics
2. **Global View**: Considers entire dataset, not individual weights
3. **Invariance**: Scores are invariant to weight reparameterization
4. **Multi-faceted**: Combines frequency, diversity, and selectivity

### Advantages Over Traditional Methods

| Method | Basis | Limitations |
|--------|-------|-------------|
| **Magnitude Pruning** | Weight size | Ignores activation patterns |
| **Activation Pruning** | Layer outputs | Ignores weight structure |
| **Random Pruning** | Chance | No principled basis |
| **Moduli Space** | Representation theory | ✅ Principled, global, invariant |

### When to Use This Method

**Best for**:
- Networks where understanding information flow is critical
- Scenarios requiring theoretical guarantees
- Applications where pruning must preserve specific pathways

**Challenges**:
- Requires dataset analysis (computational cost)
- More complex than simple magnitude pruning
- Needs proper coordinate-to-layer mapping (implementation detail)

---

## References

1. Armenta, M. A., & Jodoin, P. M. (2021). "The Representation Theory of Neural Networks". *Machine Learning*.

2. Frankle, J., & Carbin, M. (2019). "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks". *ICLR*.

3. Implementation: `moduli_space_pruning.py:167-317` (ModuliSpaceAnalyzer class)

---

**Document Version**: 1.0
**Last Updated**: 2025-12-24
**Maintainer**: Claude (AI Assistant)
