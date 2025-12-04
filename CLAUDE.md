# CLAUDE.md - AI Assistant Guide for Moduli Space Coordinate Pruning

**Last Updated**: 2025-12-04
**Project**: Neural Network Pruning via Moduli Space Coordinates
**Foundation**: "The Representation Theory of Neural Networks" by Armenta & Jodoin

---

## Project Overview

This repository implements a novel neural network pruning method based on quiver representation theory. Unlike traditional heuristic-based pruning (weight magnitude, activations), this approach uses the geometric structure of the moduli space to make mathematically principled pruning decisions.

**Key Innovation**: Networks are represented as quiver representations, data is mapped to a moduli space, and pruning decisions are made based on coordinate usage patterns in this space - providing invariance guarantees and theoretical foundations.

---

## Codebase Structure

```
sour-duck/
├── Core Implementation
│   ├── moduli_space_pruning.py      # Main implementation (17KB)
│   ├── experiments.py                # Experimental framework (21KB)
│   ├── experiments_fixed.py          # Fixed version with dimension corrections
│   ├── analysis.py                   # Visualization and analysis (19KB)
│   └── demo_simplified.py            # Standalone NumPy demo (14KB)
│
├── Optimizations
│   ├── optimized_analyzer.py         # Performance-optimized analyzer
│   ├── optimized_extraction.py       # Optimized coordinate extraction
│   └── moduli_space_pruning.py      # Contains optimization improvements
│
├── Test Data & Examples
│   ├── simple_test_data.py          # Test data generators (NEW)
│   ├── generate_examples.py         # Generate example datasets (NEW)
│   └── examples/                    # Pre-generated datasets (NEW)
│       ├── README.md                # Dataset documentation
│       ├── tiny_binary.npz          # 50 samples, quick tests
│       ├── small_binary.npz         # 100 samples, standard tests
│       ├── xor.npz                  # Non-linear problem
│       ├── spiral.npz               # Complex 3-class problem
│       └── large_multiclass.npz     # 1000 samples, realistic scale
│
├── Bug Fixes & Tests
│   ├── fix_inhomogeneous_array.py   # Fixes for array dimension issues
│   ├── fix_layer_indexing.py        # Fixes for layer indexing problems
│   ├── fix_unpacking_error.py       # Fixes for unpacking errors
│   ├── test_moduli_pruning.py       # Comprehensive test suite (12KB)
│   ├── test_dimension_fix.py        # Dimension-specific tests
│   └── test_all_fixes.py            # Integration tests for all fixes
│
├── Documentation
│   ├── README.md                     # Complete user documentation (8KB)
│   ├── SUMMARY.md                    # Implementation summary (8KB)
│   ├── OPTIMIZATION_GUIDE.md         # Performance optimization guide
│   ├── TESTING_GUIDE.md              # Test data usage guide (NEW)
│   ├── MANIFEST.txt                  # File manifest
│   └── CLAUDE.md                     # This file
│
└── Generated Assets
    ├── demo_results.json             # Demonstration results
    └── moduli_analysis_demo.png      # Visualization output (104KB)
```

---

## Core Architecture & Components

### 1. **QuiverRepresentation** (`moduli_space_pruning.py`)
Represents the neural network as a directed graph (quiver):
- **Vertices**: Neurons/layers
- **Edges**: Connections (weights)
- **Purpose**: Extract algebraic structure from network

```python
class QuiverRepresentation:
    def _extract_edges(self) -> List[Tuple[int, int, str]]:
        # Returns: (edge_idx, edge_info, layer_name)
        # edge_info is (in_idx, out_idx) for Linear, filter_idx for Conv2d
```

**Key Point**: Each edge tuple MUST have exactly 3 elements: `(idx, info, name)`

### 2. **DataRepresentation** (`moduli_space_pruning.py`)
Computes W_x^f (Equation 5 from paper) - the representation of data point x:

```python
(W_x^f)_ε = W_ε * a(W,f)_s(ε)(x) / Σ_δ∈ζ_s(ε) W_δ * a(W,f)_s(δ)(x)
```

- Captures how the network processes specific inputs
- Encodes both weights and activations
- Uses forward hooks to capture layer activations

### 3. **ModuliSpaceAnalyzer** (`moduli_space_pruning.py`)
Analyzes coordinate usage patterns across dataset:

**Three Key Metrics**:
1. **Usage Rate (ρ)**: Fraction of data points activating this coordinate
2. **Variance (σ²)**: Variability of coordinate values
3. **Information Content (I)**: Entropy-based importance measure

**Combined Importance Score**:
```python
Importance = ρ * √σ² * (1 + I)
```

### 4. **ModuliSpacePruner** (`moduli_space_pruning.py`)
Maps coordinate pruning decisions back to network weights:
- Identifies lowest-importance coordinates
- Maps coordinates → network parameters
- Applies masks to prune weights
- Returns pruned network and pruning masks

### 5. **Experimental Framework** (`experiments.py`)
Compares moduli space pruning against baselines:
- **Moduli Space Pruning**: The proposed method
- **Magnitude Pruning**: Traditional L1/L2 weight magnitude
- **Random Pruning**: Baseline
- **Structured Pruning**: Layer-wise structured pruning

**Architectures Tested**:
- `SimpleNet`: Lightweight CNN for quick experiments
- `VGG11_Small`: Scaled-down VGG for CIFAR-10

---

## Mathematical Foundation

### Moduli Space Dimension
```
d = #E - #Ṽ
```
- `#E`: Number of edges (weights)
- `#Ṽ`: Number of hidden vertices (neurons)
- Represents true degrees of freedom after accounting for isomorphisms

### Invariance Properties
Pruning decisions are invariant under:
1. **Scale transformations**: Positive scaling for ReLU networks
2. **Neuron permutations**: Within-layer reordering
3. **Weight reparameterizations**: That preserve network function

### Theoretical Guarantees
- Preserving high-importance coordinates maintains network function on data manifold
- Coordinate importance correlates with basis paths in ReLU networks
- Connection to network capacity and generalization

---

## Key Development Workflows

### 1. Using Simple Test Data (NEW)

```python
from simple_test_data import load_example_dataset, load_dataset

# Quick start with pre-defined datasets
X, y = load_example_dataset('xor')  # Also: 'tiny', 'small', 'spiral'

# Load with metadata
X, y, metadata = load_dataset('examples/small_binary.npz')
print(f"Dataset: {metadata['description']}")
```

Generate example datasets:
```bash
python generate_examples.py  # Creates all examples in examples/ directory
```

**Available test datasets:**
- `tiny_binary.npz`: 50 samples, 5 features - fastest for unit tests
- `small_binary.npz`: 100 samples, 10 features - standard testing
- `xor.npz`: 200 samples, 2 features - non-linear problem
- `spiral.npz`: 300 samples, 2 features - complex 3-class
- `large_multiclass.npz`: 1000 samples, 50 features - realistic scale

See `TESTING_GUIDE.md` for complete documentation and workflows.

### 2. Running Quick Tests
```bash
# Standalone demo (no PyTorch required)
python demo_simplified.py

# Full experimental suite
python experiments.py

# Run tests
python test_moduli_pruning.py
```

### 3. Analyzing a Network
```python
from moduli_space_pruning import ModuliSpaceAnalyzer, ModuliSpacePruner

# Initialize analyzer
analyzer = ModuliSpaceAnalyzer(network)

# Analyze dataset (typically 500-1000 samples sufficient)
analyzer.analyze_dataset(dataloader, max_samples=500)

# View coordinate statistics
print(f"Total coordinates: {analyzer.total_coordinates}")
print(f"Moduli dimension: {analyzer.moduli_dimension}")
```

### 4. Pruning a Network
```python
# Create pruner
pruner = ModuliSpacePruner(network, analyzer)

# Prune (0.3 = 30% sparsity)
pruned_network, masks = pruner.prune(pruning_ratio=0.3)

# Fine-tune pruned network
# ... standard training loop ...
```

### 5. Visualization & Analysis
```bash
# Generate comprehensive visualizations
python analysis.py

# Creates:
# - coordinate_importance_3d.html (interactive)
# - layer_wise_pruning.html
# - strategy_comparison.html
```

---

## Code Conventions & Style

### Python Style
- **Python 3.8+** required
- Type hints used throughout (typing module)
- Docstrings: Google style
- Line length: ~80-100 characters
- Imports: Standard library → Third party → Local

### Naming Conventions
- **Classes**: `PascalCase` (e.g., `ModuliSpaceAnalyzer`)
- **Functions/Methods**: `snake_case` (e.g., `analyze_dataset`)
- **Constants**: `UPPER_SNAKE_CASE` (rare in this codebase)
- **Private methods**: `_leading_underscore` (e.g., `_extract_edges`)

### Key Patterns

#### 1. Coordinate Storage
```python
# Coordinates stored as defaultdict(list)
coordinate_values = defaultdict(list)
coordinate_values[edge_idx].append(value)
```

#### 2. Layer Representation Access
```python
# Always check if layer exists in representation
if layer_name in representation:
    rep_tensor = representation[layer_name]
```

#### 3. Edge Iteration
```python
# Edge tuples: (edge_idx, edge_info, layer_name)
for edge_idx, edge_info, layer_name in self.quiver.edges:
    # Process edge
```

#### 4. Device Management
```python
# Always handle device placement
device = 'cuda' if torch.cuda.is_available() else 'cpu'
network = network.to(device)
data = data.to(device)
```

---

## Testing Guidelines

### Test Structure (`test_moduli_pruning.py`)
1. **Basic Tests**: Quiver extraction, data representation
2. **Component Tests**: Analyzer, pruner functionality
3. **Integration Tests**: End-to-end workflows
4. **Functionality Tests**: Network behavior preservation

### Running Tests
```bash
# All tests
python test_moduli_pruning.py

# Specific test categories
python test_dimension_fix.py      # Dimension-related tests
python test_all_fixes.py          # All bug fixes
```

### Test Assertions
- Network output shape preservation
- Coordinate dimension correctness
- Importance score ranges [0, 1]
- Pruning mask validity

---

## Common Tasks & Patterns

### Task 1: Adding a New Pruning Strategy
**Location**: `experiments.py`

1. Define pruning function:
```python
def my_pruning_method(model, pruning_ratio):
    # Implement pruning logic
    return pruned_model, masks
```

2. Add to comparison:
```python
methods = ['moduli', 'magnitude', 'random', 'structured', 'my_method']
```

3. Update visualization code

### Task 2: Supporting New Layer Types
**Location**: `moduli_space_pruning.py` → `QuiverRepresentation`

1. Update `_extract_edges()`:
```python
if isinstance(module, (nn.Linear, nn.Conv2d, nn.NewLayerType)):
    # Add edge extraction logic
```

2. Update `_extract_vertices()` similarly

3. Update `DataRepresentation` hooks if needed

### Task 3: Optimizing Performance
**See**: `OPTIMIZATION_GUIDE.md` for detailed strategies

Key optimization: Precompute layer values
```python
# BEFORE (slow)
for edge_idx, edge_info, layer_name in self.quiver.edges:
    coord_value = representation[layer_name].abs().mean().item()

# AFTER (fast)
layer_values = {name: tensor.abs().mean().item()
                for name, tensor in representation.items()}
for edge_idx, edge_info, layer_name in self.quiver.edges:
    coord_value = layer_values[layer_name]
```

**Performance Impact**: 2-10x speedup depending on network structure

### Task 4: Debugging Dimension Mismatches
**Common Issues**:

1. **Inhomogeneous arrays**: Coordinates have different lengths
   - **Cause**: Missing coordinates for some edges
   - **Fix**: Check edge extraction logic, ensure all edges captured

2. **Layer indexing**: Wrong layer accessed
   - **Cause**: Layer names don't match between extraction and analysis
   - **Fix**: Verify layer names in `named_modules()`

3. **Unpacking errors**: Edge tuples have wrong structure
   - **Cause**: Edge tuples not exactly 3 elements
   - **Fix**: Ensure format `(edge_idx, edge_info, layer_name)`

**Debug Files**: `fix_inhomogeneous_array.py`, `fix_layer_indexing.py`, `fix_unpacking_error.py`

---

## Performance Considerations

### Computational Complexity
- **Analysis**: O(n_samples × n_parameters)
- **Pruning decision**: O(n_parameters × log(n_parameters))
- **Memory**: O(n_parameters × n_samples) for full analysis

### Optimization Strategies

1. **Sample Limitation**:
```python
analyzer.analyze_dataset(dataloader, max_samples=500)
# 500 samples usually sufficient for good estimates
```

2. **Batch Processing** (see `optimized_analyzer.py`):
```python
# Process coordinates in batches
coordinate_matrix = np.zeros((n_edges, n_samples))
# Vectorized statistics
usage_rates = np.mean(coord_matrix > 1e-6, axis=1)
```

3. **Layer Value Caching**:
```python
# Compute once per layer, not per edge
layer_values = {name: compute_value(tensor)
                for name, tensor in representation.items()}
```

### Memory Optimization
- Use generators for large datasets
- Process in chunks if dataset > 10K samples
- Clear activation cache between batches

---

## Dependencies & Environment Setup

### Core Dependencies
```bash
# Required
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.20.0

# Visualization
matplotlib>=3.3.0
seaborn>=0.11.0
plotly>=5.0.0

# Utilities
pandas>=1.3.0
scikit-learn>=0.24.0
tqdm>=4.60.0
```

### Environment Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install torch torchvision numpy pandas matplotlib seaborn plotly scikit-learn tqdm
```

### GPU Support
- CUDA recommended for experiments.py
- CPU sufficient for demo_simplified.py
- Device automatically detected: `torch.cuda.is_available()`

---

## Important Notes for AI Assistants

### 1. **Always Read Before Modifying**
- This codebase has evolved through bug fixes
- Read existing implementation before suggesting changes
- Check `experiments.py` vs `experiments_fixed.py` for fixed versions

### 2. **Dimension Awareness**
Key dimension invariants:
- Edge tuples: MUST be exactly 3 elements `(idx, info, name)`
- Coordinate values: Lists must have equal length across edges
- Layer names: Must match between extraction and analysis phases

### 3. **Mathematical Correctness**
- The coordinate computation follows Equation (5) from the paper
- Don't modify the core formula without mathematical justification
- Importance scoring combines three metrics: usage, variance, information

### 4. **Testing is Critical**
- This is research code with subtle mathematical properties
- Always run tests after modifications
- Pay special attention to dimension mismatches
- Verify network functionality preservation after pruning

### 5. **Performance vs Correctness**
- Optimizations should preserve correctness first
- The OPTIMIZATION_GUIDE.md provides vetted optimizations
- Be cautious with approximations - they may break theoretical guarantees

### 6. **Common Pitfalls**

**Pitfall 1**: Modifying edge extraction without updating data representation
- These must stay synchronized

**Pitfall 2**: Changing coordinate computation without understanding invariances
- Breaks mathematical properties

**Pitfall 3**: Over-optimizing at the cost of clarity
- Research code values understanding over pure performance

**Pitfall 4**: Ignoring device placement
- Causes tensor device mismatch errors

**Pitfall 5**: Assuming all layers are Linear
- Code supports Conv2d; be careful generalizing

### 7. **When Helping Users**

**For understanding**: Direct to README.md and SUMMARY.md first

**For bugs**: Check fix_*.py files for similar issues

**For performance**: Reference OPTIMIZATION_GUIDE.md

**For experiments**: Start with demo_simplified.py, then experiments.py

**For theory**: Refer to paper by Armenta & Jodoin (2021)

### 8. **Code Evolution**
This codebase shows typical research code evolution:
- `experiments.py` → `experiments_fixed.py` (dimension fixes)
- Multiple fix_*.py files addressing specific bugs
- Optimization files built on stable foundation

When making changes:
- Check if there's a "fixed" version of the file
- Look for related fix files
- Test thoroughly with existing test suite

### 9. **File Relationships**

```
moduli_space_pruning.py (core)
    ↓ used by
experiments.py (evaluation)
    ↓ analyzed by
analysis.py (visualization)

Parallel:
demo_simplified.py (standalone NumPy version)

Enhancements:
optimized_analyzer.py (performance improvements)
optimized_extraction.py (extraction optimizations)

Fixes:
fix_*.py (address specific bugs)
test_*.py (validate fixes)
```

### 10. **Git Workflow**
- Work on feature branch: `claude/claude-md-*`
- Commit with clear messages describing changes
- Push with: `git push -u origin <branch-name>`
- Branch must start with 'claude/' for proper permissions

---

## Quick Reference

### Most Common Code Locations

| Task | File | Class/Function |
|------|------|----------------|
| Edge extraction | moduli_space_pruning.py | QuiverRepresentation._extract_edges() |
| Coordinate computation | moduli_space_pruning.py | DataRepresentation.compute() |
| Importance scoring | moduli_space_pruning.py | ModuliSpaceAnalyzer._compute_importance() |
| Pruning logic | moduli_space_pruning.py | ModuliSpacePruner.prune() |
| Training loop | experiments.py | train_epoch() / evaluate() |
| Baselines | experiments.py | magnitude_pruning(), random_pruning() |
| Visualizations | analysis.py | ModuliSpaceVisualizer |

### Key Equations

**Moduli Space Dimension**:
```
d = #E - #Ṽ
```

**Data Representation** (Equation 5):
```
(W_x^f)_ε = W_ε * a(W,f)_s(ε)(x) / Σ_δ W_δ * a(W,f)_s(δ)(x)
```

**Importance Score**:
```
Importance = usage_rate × √variance × (1 + information_content)
```

---

## Changelog

**2025-12-04**: Added Simple Test Data Examples (Issue: "Create more simple testing data example")
- Added `simple_test_data.py` with comprehensive data generators
- Created `generate_examples.py` to generate pre-made datasets
- Added `examples/` directory with 7 ready-to-use datasets
- Created `TESTING_GUIDE.md` with complete testing workflows
- Added `examples/README.md` documenting all datasets
- Updated README.md and CLAUDE.md with test data documentation

**2025-12-04**: Initial CLAUDE.md creation
- Comprehensive codebase structure documentation
- Core architectural components explained
- Development workflows outlined
- Common pitfalls and debugging guide added
- Performance optimization guidance included

---

## Additional Resources

- **Paper**: Armenta, M. A., & Jodoin, P. M. (2021). "The Representation Theory of Neural Networks". Machine Learning.
- **Related Work**: Frankle & Carbin (2019) - Lottery Ticket Hypothesis
- **Related Work**: Zheng et al. (2019) - Capacity control via basis-path norm

---

**For questions or issues**: Refer to specific documentation files (README.md, SUMMARY.md, OPTIMIZATION_GUIDE.md) or examine test files for usage examples.
