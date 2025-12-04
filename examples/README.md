# Example Test Datasets

This directory contains pre-generated example datasets for testing and demonstrating the moduli space pruning method.

## Quick Start

```python
from simple_test_data import load_dataset

# Load any example dataset
X, y, metadata = load_dataset('examples/xor.npz')
print(f"Loaded {metadata['name']}: {X.shape}")
```

## Available Datasets

### 1. **tiny_binary.npz**
- **Type**: Binary classification
- **Size**: 50 samples, 5 features
- **Use case**: Quick unit tests, debugging
- **Description**: Minimal dataset for rapid testing

### 2. **small_binary.npz**
- **Type**: Binary classification
- **Size**: 100 samples, 10 features
- **Use case**: Small-scale experiments
- **Description**: Standard binary classification with informative and redundant features

### 3. **medium_multiclass.npz**
- **Type**: Multi-class classification (3 classes)
- **Size**: 300 samples, 20 features
- **Use case**: Multi-class pruning experiments
- **Description**: Medium-sized dataset with 3 well-separated classes

### 4. **xor.npz**
- **Type**: Binary classification (non-linear)
- **Size**: 200 samples, 2 features
- **Use case**: Testing non-linear separability
- **Description**: Classic XOR problem requiring non-linear decision boundary
- **Visualization**: `xor_visualization.png`

### 5. **spiral.npz**
- **Type**: Multi-class classification (3 classes, non-linear)
- **Size**: 300 samples, 2 features
- **Use case**: Challenging non-linear classification
- **Description**: Three interleaved spirals, requires complex decision boundaries
- **Visualization**: `spiral_visualization.png`

### 6. **regression.npz**
- **Type**: Regression
- **Size**: 100 samples, 10 features
- **Use case**: Testing pruning on regression tasks
- **Description**: Linear regression with Gaussian noise

### 7. **large_multiclass.npz**
- **Type**: Multi-class classification (5 classes)
- **Size**: 1000 samples, 50 features
- **Use case**: Realistic performance testing
- **Description**: Large-scale dataset for benchmarking

### 8. **toy_network_weights.npz**
- **Type**: Network weights
- **Architecture**: [10, 20, 10, 2]
- **Use case**: Testing pruning without training
- **Description**: Pre-initialized network weights

## Generating Custom Datasets

If you want to regenerate or create new datasets:

```bash
# Generate all example datasets
python generate_examples.py
```

Or create custom datasets programmatically:

```python
from simple_test_data import generate_xor_data, save_dataset

# Generate custom XOR data
X, y = generate_xor_data(n_samples=500, noise=0.2)

# Save it
save_dataset(X, y, 'examples/my_xor.npz',
             metadata={'custom': True, 'noise_level': 0.2})
```

## Using with Moduli Space Pruning

### With PyTorch (Full Implementation)

```python
import torch
from torch.utils.data import TensorDataset, DataLoader
from simple_test_data import load_dataset
from moduli_space_pruning import ModuliSpaceAnalyzer, ModuliSpacePruner

# Load data
X, y, meta = load_dataset('examples/small_binary.npz')

# Convert to PyTorch
X_tensor = torch.FloatTensor(X)
y_tensor = torch.LongTensor(y)
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=32)

# Analyze and prune (assuming you have a trained network)
analyzer = ModuliSpaceAnalyzer(network)
analyzer.analyze_dataset(dataloader, max_samples=100)

pruner = ModuliSpacePruner(network, analyzer)
pruned_network, masks = pruner.prune(pruning_ratio=0.5)
```

### With NumPy (Simplified Demo)

```python
from simple_test_data import load_dataset
from demo_simplified import SimplifiedModuliAnalyzer, SimpleNetwork

# Load data
X, y, meta = load_dataset('examples/xor.npz')

# Create and analyze network
network = SimpleNetwork([2, 10, 2])
analyzer = SimplifiedModuliAnalyzer(network)
analyzer.analyze_data(X, num_samples=len(X))

# Get pruning masks
masks = analyzer.get_pruning_mask(pruning_ratio=0.3)
```

## Dataset Statistics

Quick overview of all datasets:

| Dataset | Samples | Features | Classes | Type | Difficulty |
|---------|---------|----------|---------|------|------------|
| tiny_binary | 50 | 5 | 2 | Binary | Easy |
| small_binary | 100 | 10 | 2 | Binary | Easy |
| medium_multiclass | 300 | 20 | 3 | Multi-class | Medium |
| xor | 200 | 2 | 2 | Binary | Hard (non-linear) |
| spiral | 300 | 2 | 3 | Multi-class | Hard (non-linear) |
| regression | 100 | 10 | - | Regression | Easy |
| large_multiclass | 1000 | 50 | 5 | Multi-class | Medium |

## Visualizations

For 2D datasets (XOR and Spiral), PNG visualizations are provided to help understand the data structure:

- `xor_visualization.png`: Shows the XOR pattern
- `spiral_visualization.png`: Shows the three spirals

## Creating Your Own Test Data

See `simple_test_data.py` for all available generators:

```python
from simple_test_data import (
    generate_binary_classification,
    generate_multiclass_classification,
    generate_regression_data,
    generate_xor_data,
    generate_spiral_data,
    list_example_datasets
)

# See all options
list_example_datasets()

# Generate custom data
X, y = generate_binary_classification(
    n_samples=200,
    n_features=15,
    n_informative=8,
    n_redundant=3,
    random_state=42
)
```

## File Formats

All datasets are saved in NumPy's `.npz` format (compressed):

```python
# Manual loading
import numpy as np
data = np.load('examples/xor.npz', allow_pickle=True)
X = data['X']
y = data['y']
metadata = data['metadata'].item()
```

Or use the convenient loader:

```python
from simple_test_data import load_dataset
X, y, metadata = load_dataset('examples/xor.npz')
```

## Tips

1. **Start Small**: Use `tiny_binary.npz` for initial testing
2. **Non-linear Testing**: Use `xor.npz` or `spiral.npz` to test if your method handles non-linear boundaries
3. **Scalability Testing**: Use `large_multiclass.npz` to test performance on larger datasets
4. **Regression**: Use `regression.npz` if testing beyond classification

## Regenerating Datasets

To regenerate all example datasets with different parameters, modify `generate_examples.py` and run:

```bash
python generate_examples.py
```

This will overwrite existing files in the `examples/` directory.

## Notes

- All datasets use `random_state=42` for reproducibility
- Datasets are normalized/standardized as appropriate
- Class balance is approximately equal in classification datasets
- Regression targets have Gaussian noise added
