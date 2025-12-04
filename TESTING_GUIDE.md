# Testing Guide: Simple Test Data Examples

This guide explains how to use the simple test data examples for experimenting with moduli space pruning.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Understanding the Test Data](#understanding-the-test-data)
3. [Common Testing Workflows](#common-testing-workflows)
4. [Integration Examples](#integration-examples)
5. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Installation

No additional dependencies beyond the main project requirements:

```bash
pip install numpy matplotlib  # For data generation
```

### Generate Example Datasets

```bash
# Generate all pre-made examples (requires numpy)
python generate_examples.py
```

This creates:
- 7 different datasets in `examples/` directory
- 2 visualizations (XOR and Spiral)
- Toy network weights

### Load and Use

```python
from simple_test_data import load_dataset, print_dataset_stats

# Load any example
X, y, metadata = load_dataset('examples/xor.npz')

# View statistics
print_dataset_stats(X, y)
```

---

## Understanding the Test Data

### Dataset Selection Guide

| Your Goal | Recommended Dataset | Why |
|-----------|-------------------|-----|
| Quick unit test | `tiny_binary.npz` | Fastest, minimal |
| Debug pruning logic | `small_binary.npz` | Easy to reason about |
| Test multi-class | `medium_multiclass.npz` | 3 balanced classes |
| Test non-linearity | `xor.npz` | Classic non-linear problem |
| Challenging test | `spiral.npz` | Complex decision boundaries |
| Regression testing | `regression.npz` | Continuous targets |
| Performance benchmark | `large_multiclass.npz` | Realistic scale |

### Dataset Characteristics

#### 1. Tiny Binary (`tiny_binary.npz`)
```python
# 50 samples, 5 features, 2 classes
# Perfect for: Rapid iteration, debugging
X, y, meta = load_dataset('examples/tiny_binary.npz')
# Expected time: <1 second for full analysis
```

**Use Cases:**
- Unit testing
- Quick sanity checks
- CI/CD pipelines
- Debugging edge cases

#### 2. XOR Dataset (`xor.npz`)
```python
# 200 samples, 2 features, 2 classes
# Perfect for: Testing non-linear pruning
X, y, meta = load_dataset('examples/xor.npz')
```

**Key Properties:**
- Not linearly separable
- Requires hidden layer capacity
- Tests if pruning preserves non-linear decision boundary
- 2D visualization available

**Expected Behavior:**
- Linear models should fail (~50% accuracy)
- Pruned network should maintain ~90%+ accuracy if essential pathways preserved

#### 3. Spiral Dataset (`spiral.npz`)
```python
# 300 samples, 2 features, 3 classes
# Perfect for: Challenging non-linear multi-class
X, y, meta = load_dataset('examples/spiral.npz')
```

**Key Properties:**
- Three interleaved spirals
- Requires significant network capacity
- Good test for importance scoring
- 2D visualization available

**Expected Behavior:**
- Random pruning should hurt accuracy significantly
- Moduli space pruning should preserve important spiral pathways

---

## Common Testing Workflows

### Workflow 1: Quick Functionality Test

Test that your pruning implementation works at all:

```python
from simple_test_data import load_example_dataset
from demo_simplified import SimpleNetwork, SimplifiedModuliAnalyzer

# Load smallest dataset
X, y = load_example_dataset('tiny')

# Create tiny network
network = SimpleNetwork([5, 10, 2])

# Quick analysis
analyzer = SimplifiedModuliAnalyzer(network)
analyzer.analyze_data(X, num_samples=50)

# Test pruning
masks = analyzer.get_pruning_mask(pruning_ratio=0.3)

print(f"✓ Pruning works! Pruned {len(masks)} layers")
```

**Expected Runtime**: <5 seconds

### Workflow 2: Non-linearity Test

Verify pruning preserves non-linear decision boundaries:

```python
from simple_test_data import load_dataset, visualize_2d_data

# Load XOR
X, y, _ = load_dataset('examples/xor.npz')

# Visualize to understand problem
visualize_2d_data(X, y, "XOR Problem")

# Train network (your training code)
network = train_network(X, y, architecture=[2, 10, 10, 2])
accuracy_before = evaluate(network, X, y)

# Prune
pruned_network = prune_with_moduli_space(network, X, y, ratio=0.5)
accuracy_after = evaluate(pruned_network, X, y)

# Check preservation
print(f"Before: {accuracy_before:.2f}%, After: {accuracy_after:.2f}%")
assert accuracy_after > 0.8, "Pruning hurt non-linear capacity too much"
```

**Expected Result**: Accuracy drop <10% for 50% pruning

### Workflow 3: Importance Scoring Validation

Verify that importance scores correlate with actual importance:

```python
from simple_test_data import load_dataset
from moduli_space_pruning import ModuliSpaceAnalyzer

# Load data
X, y, _ = load_dataset('examples/small_binary.npz')

# Analyze
analyzer = ModuliSpaceAnalyzer(network)
analyzer.analyze_dataset(dataloader, max_samples=100)

# Compare pruning strategies
results = {}
for ratio in [0.3, 0.5, 0.7]:
    # Moduli space pruning
    moduli_acc = test_pruning(network, analyzer, ratio, strategy='moduli')

    # Random pruning (baseline)
    random_acc = test_pruning(network, analyzer, ratio, strategy='random')

    results[ratio] = {'moduli': moduli_acc, 'random': random_acc}

# Moduli should outperform random
for ratio, scores in results.items():
    print(f"{ratio:.0%} pruning: Moduli={scores['moduli']:.2f}%, Random={scores['random']:.2f}%")
    assert scores['moduli'] >= scores['random'], "Moduli should beat random"
```

### Workflow 4: Scalability Test

Test performance on larger datasets:

```python
from simple_test_data import load_dataset
import time

# Load large dataset
X, y, meta = load_dataset('examples/large_multiclass.npz')
print(f"Testing on {meta['n_samples']} samples")

# Time the analysis
start = time.time()
analyzer = ModuliSpaceAnalyzer(network)
analyzer.analyze_dataset(dataloader, max_samples=500)
elapsed = time.time() - start

print(f"Analysis time: {elapsed:.2f}s")
print(f"Samples/second: {500/elapsed:.1f}")
```

**Expected Performance**: >10 samples/second on CPU

### Workflow 5: Regression Testing

Test pruning on regression tasks:

```python
from simple_test_data import load_dataset

# Load regression data
X, y, _ = load_dataset('examples/regression.npz')

# Train regression network
network = train_regression_network(X, y)

# Evaluate with MSE
mse_before = evaluate_mse(network, X, y)

# Prune
pruned_network = prune_with_moduli_space(network, X, y, ratio=0.5)
mse_after = evaluate_mse(pruned_network, X, y)

# Check MSE increase is acceptable
mse_increase = (mse_after - mse_before) / mse_before
print(f"MSE increase: {mse_increase*100:.1f}%")
assert mse_increase < 0.2, "MSE increased too much"
```

---

## Integration Examples

### With PyTorch (Full Implementation)

```python
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from simple_test_data import load_dataset
from moduli_space_pruning import ModuliSpaceAnalyzer, ModuliSpacePruner

# 1. Load test data
X, y, metadata = load_dataset('examples/small_binary.npz')
print(f"Loaded: {metadata['description']}")

# 2. Convert to PyTorch
X_tensor = torch.FloatTensor(X)
y_tensor = torch.LongTensor(y)
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 3. Create and train network
network = nn.Sequential(
    nn.Linear(metadata['n_features'], 20),
    nn.ReLU(),
    nn.Linear(20, 10),
    nn.ReLU(),
    nn.Linear(10, 2)
)

# Train network (your training code here)
# ...

# 4. Analyze with moduli space
analyzer = ModuliSpaceAnalyzer(network)
analyzer.analyze_dataset(dataloader, max_samples=100)

print(f"Moduli dimension: {analyzer.moduli_dimension}")
print(f"Total coordinates: {analyzer.total_coordinates}")

# 5. Prune
pruner = ModuliSpacePruner(network, analyzer)
pruned_network, masks = pruner.prune(pruning_ratio=0.5)

print(f"Pruned {sum(m.sum().item() for m in masks)} weights")

# 6. Fine-tune pruned network
# ... fine-tuning code ...
```

### With NumPy (Simplified Demo)

```python
import numpy as np
from simple_test_data import load_dataset, visualize_2d_data
from demo_simplified import SimpleNetwork, SimplifiedModuliAnalyzer

# 1. Load test data
X, y, metadata = load_dataset('examples/xor.npz')

# 2. Visualize (for 2D data)
visualize_2d_data(X, y, "XOR Dataset Before Pruning")

# 3. Create network
network = SimpleNetwork([2, 10, 10, 2])

# 4. Analyze
analyzer = SimplifiedModuliAnalyzer(network)
analyzer.analyze_data(X, num_samples=len(X))

# 5. Get pruning masks
masks = analyzer.get_pruning_mask(pruning_ratio=0.5)

# 6. Apply masks
for i, (W, mask) in enumerate(zip(network.weights, masks)):
    network.weights[i] = W * mask

print(f"Network sparsity: {network.get_sparsity()*100:.1f}%")

# 7. Visualize coordinate importance
analyzer.visualize_analysis()
```

### Custom Dataset Creation

```python
from simple_test_data import (
    generate_binary_classification,
    save_dataset,
    print_dataset_stats
)

# Create custom dataset with specific properties
X, y = generate_binary_classification(
    n_samples=500,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    random_state=42
)

# Inspect
print_dataset_stats(X, y)

# Save for reuse
save_dataset(
    X, y,
    'examples/my_custom_data.npz',
    metadata={
        'purpose': 'Testing redundant feature pruning',
        'expected_prunable_features': 5,
        'created_by': 'me'
    }
)

# Load later
X_loaded, y_loaded, meta = load_dataset('examples/my_custom_data.npz')
print(meta['purpose'])
```

---

## Troubleshooting

### Issue: "Module not found: numpy"

**Solution:**
```bash
pip install numpy matplotlib
```

### Issue: "File not found: examples/xor.npz"

**Cause**: Example datasets not generated yet

**Solution:**
```bash
python generate_examples.py
```

### Issue: Accuracy drops significantly after pruning

**Diagnosis:**
1. Check pruning ratio - start with 30% instead of 50%+
2. Verify importance scores are computed correctly
3. Test on simpler dataset first (tiny_binary)
4. Check if network was actually trained before pruning

**Example:**
```python
# Add diagnostic code
analyzer = ModuliSpaceAnalyzer(network)
analyzer.analyze_dataset(dataloader, max_samples=100)

# Check importance distribution
importances = [coord.importance_score for coord in analyzer.coordinates]
print(f"Importance range: [{min(importances):.4f}, {max(importances):.4f}]")
print(f"Mean importance: {np.mean(importances):.4f}")

# Should see variation - if all similar, something's wrong
```

### Issue: Analysis takes too long

**Solutions:**

1. **Reduce samples:**
```python
analyzer.analyze_dataset(dataloader, max_samples=100)  # Instead of all data
```

2. **Use optimized version:**
```python
from optimized_analyzer import OptimizedModuliSpaceAnalyzer
analyzer = OptimizedModuliSpaceAnalyzer(network)
```

3. **Start with tiny dataset:**
```python
X, y = load_example_dataset('tiny')  # Only 50 samples
```

### Issue: Out of memory

**Solutions:**

1. Use smaller batch size:
```python
dataloader = DataLoader(dataset, batch_size=16)  # Instead of 32 or 64
```

2. Use smaller dataset:
```python
# Use small or medium instead of large
X, y, _ = load_dataset('examples/small_binary.npz')
```

3. Limit analysis samples:
```python
analyzer.analyze_dataset(dataloader, max_samples=200)
```

---

## Best Practices

### 1. Always Start Small

```python
# ✓ Good
X, y = load_example_dataset('tiny')
# Test your code
# Then scale up to larger datasets

# ✗ Bad
X, y = load_example_dataset('large')  # Might take forever if code has bugs
```

### 2. Visualize 2D Datasets

```python
# For XOR and Spiral
from simple_test_data import visualize_2d_data

X, y, _ = load_dataset('examples/spiral.npz')
visualize_2d_data(X, y, "Understanding the Problem")
```

### 3. Save Intermediate Results

```python
# After expensive analysis
analyzer.save('analyzer_state.pkl')  # If supported

# Or save coordinates manually
import pickle
with open('coordinates.pkl', 'wb') as f:
    pickle.dump(analyzer.coordinates, f)
```

### 4. Compare Against Random Baseline

```python
# Always compare moduli space pruning to random
moduli_result = test_pruning(network, method='moduli', ratio=0.5)
random_result = test_pruning(network, method='random', ratio=0.5)

print(f"Improvement over random: {moduli_result - random_result:.2f}%")
```

### 5. Use Metadata

```python
# Metadata helps track experiments
X, y, meta = load_dataset('examples/xor.npz')

results = {
    'dataset': meta['name'],
    'n_samples': meta['n_samples'],
    'accuracy_before': acc_before,
    'accuracy_after': acc_after,
    # ... more results
}

# Save experiment log
import json
with open('experiment_log.json', 'a') as f:
    json.dump(results, f)
    f.write('\n')
```

---

## Next Steps

1. **Run the demo**: `python simple_test_data.py`
2. **Generate examples**: `python generate_examples.py`
3. **Try integration**: Follow examples in this guide
4. **Read**: `examples/README.md` for dataset details
5. **Experiment**: Create custom datasets for your needs

---

## Additional Resources

- `simple_test_data.py` - All data generation functions
- `generate_examples.py` - Example generation script
- `examples/README.md` - Detailed dataset descriptions
- `demo_simplified.py` - Simple demo without PyTorch
- `experiments.py` - Full experimental framework
