# Neural Network Pruning Experiment Framework

A comprehensive, modular framework for evaluating neural network pruning methods. Compare moduli space pruning against state-of-the-art baselines across multiple architectures and datasets.

## Overview

This framework provides:

- **5 Pruning Methods**: Moduli Space, Magnitude, Random, Lottery Ticket, Wanda
- **3 Datasets**: CIFAR-10, CIFAR-100, ImageNet-1k
- **2 Architectures**: ResNet-18, VGG-16
- **Comprehensive Metrics**: Accuracy, memory footprint, FLOPs, inference time
- **Moduli Space Analysis**: Dimension analysis, correlation plots, component ablation
- **YAML Configuration**: Easy experiment customization
- **Automated Visualization**: Publication-ready plots

## Installation

### Requirements

```bash
# Core dependencies
pip install torch torchvision numpy pandas matplotlib seaborn plotly scikit-learn tqdm pyyaml

# Python 3.8+ required
```

### Quick Setup

```bash
# Clone repository
git clone https://github.com/mrmjhenry/sour-duck.git
cd sour-duck

# Install dependencies
pip install -r requirements.txt  # If available

# Verify installation
python experiments/run_experiment.py --help
```

## Quick Start

### 1. Run a Pre-configured Experiment

```bash
# ResNet-18 on CIFAR-10 (quick test)
python experiments/run_experiment.py --config experiments/config/cifar10_resnet18.yaml
```

### 2. Results

The experiment will:
1. Train baseline ResNet-18 (~20 minutes on GPU)
2. Prune with all 5 methods at 5 sparsity levels (10%, 30%, 50%, 70%, 90%)
3. Fine-tune each pruned model
4. Generate visualizations and save results

**Output structure:**
```
experiments/results/
├── baseline_best.pth              # Trained baseline model
├── results.json                   # All metrics in JSON
└── plots/
    ├── accuracy_vs_sparsity.png   # Main comparison plot
    ├── memory_reduction.png
    ├── dimension_vs_prunable.png  # Moduli space correlation
    └── moduli_ablation_study.png  # Component ablation
```

### 3. View Results

```python
import json

with open('experiments/results/results.json') as f:
    results = json.load(f)

# Baseline accuracy
print(f"Baseline: {results['experiment']['baseline_accuracy']:.2f}%")

# Moduli space at 50% sparsity
moduli_50 = results['methods']['moduli_space'][2]  # Index 2 = 50% sparsity
print(f"Moduli Space @ 50%: {moduli_50['test_accuracy']:.2f}%")
```

## Configuration

### YAML Structure

Experiments are configured via YAML files. See `experiments/config/default.yaml` for complete template.

#### Key Sections:

**1. Experiment Metadata**
```yaml
experiment:
  name: "moduli_space_comparison"
  output_dir: "experiments/results"
```

**2. Dataset Configuration**
```yaml
dataset:
  name: "CIFAR10"  # CIFAR10, CIFAR100, or IMAGENET
  data_dir: "./data"
  batch_size: 128
  num_workers: 4
  augmentation: true
  download: true
```

**3. Model Configuration**
```yaml
model:
  name: "ResNet18"  # ResNet18 or VGG16
  pretrained: false
```

**4. Training Configuration**
```yaml
training:
  baseline_epochs: 100
  finetune_epochs: 20
  learning_rate: 0.1
  finetune_lr: 0.01
  momentum: 0.9
  weight_decay: 0.0001
  lr_milestones: [50, 75]
  lr_gamma: 0.1
  retrain_baseline: false  # Set to true to retrain even if baseline exists
```

**5. Pruning Configuration**
```yaml
pruning:
  methods:
    - "moduli_space"
    - "magnitude"
    - "random"
    - "lottery_ticket"
    - "wanda"

  sparsity_levels: [0.1, 0.3, 0.5, 0.7, 0.9]

  # Moduli space specific settings
  moduli_space:
    strategy: "combined"  # combined, usage, variance, information
    max_samples: 1000

    ablation:
      enabled: true
      components:
        - "usage_only"
        - "variance_only"
        - "information_only"
        - "combined"
```

**6. Evaluation Configuration**
```yaml
evaluation:
  compute_flops: true
  measure_inference_time: true
  top5_accuracy: false  # Set to true for ImageNet
```

### Creating Custom Configurations

```bash
# Copy template
cp experiments/config/default.yaml experiments/config/my_experiment.yaml

# Edit configuration
vim experiments/config/my_experiment.yaml

# Run experiment
python experiments/run_experiment.py --config experiments/config/my_experiment.yaml
```

## Pruning Methods

### 1. Moduli Space Pruning (Ours)

**Theory**: Uses representation theory to analyze coordinate importance in the network's moduli space.

**Importance Score**:
```
Importance = usage_rate × √variance × (1 + information_content)
```

**Strategies**:
- `combined`: Full importance formula (default)
- `usage`: Usage rate only
- `variance`: Variance only
- `information`: Information content only

**Configuration**:
```yaml
moduli_space:
  strategy: "combined"
  max_samples: 1000  # Samples for coordinate analysis
```

**Reference**: See `Methods.md` for detailed mathematical explanation

### 2. Magnitude Pruning

**Theory**: Prune weights with smallest L1 magnitude (Han et al., 2015)

**Method**: Global threshold based on absolute weight values

**Reference**: Han et al., "Learning both Weights and Connections for Efficient Neural Networks", NIPS 2015

### 3. Random Pruning

**Theory**: Random baseline - prune weights uniformly at random

**Method**: Stochastic selection without bias

**Purpose**: Lower-bound baseline for comparison

### 4. Lottery Ticket Pruning

**Theory**: Iterative magnitude pruning with weight rewinding (Frankle & Carbin, 2019)

**Method**:
1. Prune smallest magnitude weights
2. Rewind remaining weights to initial values
3. Retrain

**Reference**: Frankle & Carbin, "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks", ICLR 2019

### 5. Wanda Pruning

**Theory**: Pruning by Weights AND activations (Sun et al., 2024)

**Method**: Combines weight magnitude with activation importance

**Reference**: Sun et al., "A Simple and Effective Pruning Approach for Large Language Models", 2024

## Metrics

### Automatic Metrics

All methods are evaluated on:

1. **Test Accuracy** (%)
   - Top-1 accuracy for CIFAR
   - Top-1 and Top-5 for ImageNet

2. **Sparsity** (%)
   - Fraction of pruned (zero) weights
   - Actual vs target sparsity

3. **Model Size** (MB)
   - Memory footprint reduction
   - Only counts non-zero parameters

4. **Active Parameters**
   - Number of remaining (non-pruned) parameters

5. **Inference Time** (ms) *[Optional]*
   - Average forward pass time
   - Measured on 100 iterations

6. **FLOPs** *[Optional]*
   - Floating point operations
   - Structured pruning only

### Moduli Space Specific Metrics

For moduli space pruning only:

7. **Moduli Dimension**
   - Theoretical degrees of freedom: `#E - #Ṽ`
   - Tracked across sparsity levels

8. **Coordinate Statistics**
   - Mean usage rate, variance, information content
   - Per-layer analysis

## Visualizations

### 1. Accuracy vs Sparsity

**File**: `accuracy_vs_sparsity.png`

**Content**: Line plot comparing all methods across sparsity levels

**Interpretation**:
- Y-axis: Test accuracy (%)
- X-axis: Sparsity (%)
- Higher curves = better pruning method
- Look for graceful degradation

### 2. Memory Reduction

**File**: `memory_reduction.png`

**Content**: Model size vs sparsity

**Interpretation**:
- Shows memory savings
- Compare compression ratios

### 3. Moduli Space: Dimension vs Prunable Parameters

**File**: `dimension_vs_prunable.png`

**Content**: Scatter plot with correlation analysis

**Interpretation**:
- Tests theoretical prediction: dimension correlates with prunability
- Correlation coefficient shown
- Trend line fitted

### 4. Moduli Space: Component Ablation

**File**: `moduli_ablation_study.png`

**Content**: Compares usage-only, variance-only, information-only, and combined strategies

**Interpretation**:
- Shows contribution of each component
- Validates combined scoring approach
- Only generated if `ablation.enabled: true`

## Advanced Usage

### Running Ablation Study Only

Modify config to test only moduli space components:

```yaml
pruning:
  methods:
    - "moduli_space"  # Only this method

  sparsity_levels: [0.3, 0.5, 0.7]  # Fewer levels for speed

  moduli_space:
    ablation:
      enabled: true
      components: ["usage_only", "variance_only", "combined"]
```

### Using Pre-trained Baseline

To avoid retraining baseline every time:

```yaml
training:
  retrain_baseline: false  # Will reuse experiments/results/baseline_best.pth if it exists
```

### Quick Experiments (Reduced Training)

For rapid iteration:

```yaml
training:
  baseline_epochs: 20      # Reduced from 100
  finetune_epochs: 5       # Reduced from 20

pruning:
  sparsity_levels: [0.3, 0.5]  # Test only 2 levels

  moduli_space:
    max_samples: 500       # Fewer samples for faster analysis
```

### ImageNet Experiments

**Requirements**:
- ImageNet dataset downloaded to `data_dir`
- Organized as: `{data_dir}/imagenet/train/` and `{data_dir}/imagenet/val/`

**Configuration**:
```yaml
dataset:
  name: "IMAGENET"
  data_dir: "/path/to/imagenet"
  batch_size: 256  # Larger batches for ImageNet
  num_workers: 8

model:
  name: "ResNet18"
  pretrained: true  # Use ImageNet pre-trained weights

training:
  baseline_epochs: 0  # Skip training if using pretrained

evaluation:
  top5_accuracy: true  # Report Top-5 for ImageNet
```

## Extending the Framework

### Adding a New Pruning Method

**1. Create pruner class:**

```python
# experiments/pruning/my_method.py

from .base import BasePruner
import torch
import torch.nn as nn
from typing import Dict, Tuple

class MyMethodPruner(BasePruner):
    """My custom pruning method"""

    def __init__(self, model: nn.Module, **kwargs):
        super().__init__(model)
        self.my_param = kwargs.get('my_param', 0.5)

    def prune(
        self,
        pruning_ratio: float,
        **kwargs
    ) -> Tuple[nn.Module, Dict[str, torch.Tensor]]:
        """
        Implement pruning logic.

        Returns:
            (pruned_model, masks)
        """
        # Get prunable layers
        prunable_layers = self.get_prunable_layers(self.model)

        # Create masks
        masks = {}

        # YOUR PRUNING LOGIC HERE
        for name, module in prunable_layers.items():
            mask = torch.ones_like(module.weight.data)
            # ... compute mask ...
            masks[name] = mask

        # Apply masks
        self.apply_masks(masks)

        return self.model, masks
```

**2. Register in factory:**

```python
# experiments/pruning/__init__.py

from .my_method import MyMethodPruner

def get_pruner(method: str, model, **kwargs):
    # ... existing methods ...
    elif method == 'my_method':
        return MyMethodPruner(model, **kwargs)
```

**3. Add to config:**

```yaml
pruning:
  methods:
    - "moduli_space"
    - "my_method"
```

### Adding a New Dataset

**1. Implement loader:**

```python
# experiments/datasets/data_loaders.py

class DatasetFactory:
    @staticmethod
    def _get_my_dataset(data_dir, batch_size, num_workers, download):
        # Load your dataset
        train_dataset = MyDataset(root=data_dir, train=True, download=download)
        test_dataset = MyDataset(root=data_dir, train=False, download=download)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, ...)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, ...)

        num_classes = len(train_dataset.classes)

        return train_loader, test_loader, num_classes

    @staticmethod
    def get_dataset(name, ...):
        # ... existing datasets ...
        elif name == 'MYDATASET':
            return DatasetFactory._get_my_dataset(...)
```

**2. Use in config:**

```yaml
dataset:
  name: "MYDATASET"
  data_dir: "./data/my_dataset"
```

### Adding a New Model

**1. Implement architecture:**

```python
# experiments/models/my_architecture.py

import torch.nn as nn

def create_my_model(num_classes=10):
    """Create my custom architecture"""
    return nn.Sequential(
        # Your architecture here
    )
```

**2. Register in factory:**

```python
# experiments/models/__init__.py

from .my_architecture import create_my_model

def get_model(model_name, num_classes, **kwargs):
    # ... existing models ...
    elif model_name == 'MyModel':
        return create_my_model(num_classes=num_classes)
```

**3. Use in config:**

```yaml
model:
  name: "MyModel"
```

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**

```yaml
# Reduce batch size
dataset:
  batch_size: 64  # Instead of 128

# Or use CPU
# The framework will automatically use CPU if CUDA is unavailable
```

**2. Slow Moduli Space Analysis**

```yaml
# Reduce samples
moduli_space:
  max_samples: 500  # Instead of 1000
```

**3. Baseline Already Exists**

```bash
# Remove existing baseline to retrain
rm experiments/results/baseline_best.pth

# Or set in config:
training:
  retrain_baseline: true
```

**4. Missing ImageNet Data**

```python
# Verify path structure:
# {data_dir}/imagenet/train/n01440764/...
# {data_dir}/imagenet/val/n01440764/...
```

### Performance Tips

**1. Use GPU**: Experiments are 10-50× faster on GPU

**2. Reuse Baseline**: Set `retrain_baseline: false`

**3. Reduce Samples**: For quick tests, use `max_samples: 500`

**4. Parallel Data Loading**: Increase `num_workers: 8` (if you have CPU cores)

## API Reference

### Main Components

#### `ExperimentOrchestrator`

Main experiment runner.

```python
from experiments.run_experiment import ExperimentOrchestrator
from experiments.config import load_config

config = load_config('experiments/config/my_config.yaml')
orchestrator = ExperimentOrchestrator(config)
orchestrator.run()
```

**Methods**:
- `setup_data()`: Load datasets
- `create_model(num_classes)`: Initialize model
- `train_baseline(model, train_loader, test_loader)`: Train baseline
- `prune_and_finetune(model, method, sparsity, ...)`: Prune and fine-tune
- `run_pruning_experiments(...)`: Run all pruning experiments
- `run_ablation_study(...)`: Run moduli space ablation
- `generate_visualizations()`: Create all plots
- `save_results()`: Save results to JSON
- `run()`: Execute complete pipeline

#### `MetricsTracker`

Compute and track metrics.

```python
from experiments.utils.metrics import MetricsTracker

tracker = MetricsTracker(device='cuda')

# Compute accuracy
accuracy = tracker.compute_accuracy(model, test_loader, top5=False)

# Measure inference time
inf_time = tracker.measure_inference_time(model, input_shape=(1, 3, 32, 32))

# Create comprehensive metrics
metrics = tracker.create_metrics(
    method='moduli_space',
    model=model,
    test_loader=test_loader,
    sparsity=0.5,
    masks=masks
)
```

#### `ExperimentVisualizer`

Generate plots.

```python
from experiments.utils.visualization import ExperimentVisualizer

viz = ExperimentVisualizer(output_dir='experiments/results/plots')

# Accuracy vs sparsity
results = {
    'moduli_space': [(10, 91.2), (30, 89.5), (50, 86.3)],
    'magnitude': [(10, 90.1), (30, 87.2), (50, 82.1)]
}
viz.plot_accuracy_vs_sparsity(results)

# Ablation study
viz.plot_ablation_study(
    components=['usage_only', 'variance_only', 'combined'],
    accuracies={'usage_only': [89, 85, 78], ...},
    sparsities=[10, 30, 50]
)
```

### Configuration Classes

#### `Config`

Main configuration object (dataclass).

```python
from experiments.config import Config, load_config

# Load from YAML
config = load_config('path/to/config.yaml')

# Access nested config
print(config.dataset.name)  # "CIFAR10"
print(config.pruning.sparsity_levels)  # [0.1, 0.3, 0.5, ...]
```

**Nested Configurations**:
- `config.experiment`: `ExperimentConfig`
- `config.dataset`: `DatasetConfig`
- `config.model`: `ModelConfig`
- `config.training`: `TrainingConfig`
- `config.pruning`: `PruningConfig`
- `config.evaluation`: `EvaluationConfig`

### Pruner Base Class

#### `BasePruner`

Abstract base class for all pruning methods.

```python
from experiments.pruning.base import BasePruner

class MyPruner(BasePruner):
    def prune(self, pruning_ratio, **kwargs):
        # Implement pruning
        masks = {}
        # ...
        self.apply_masks(masks)
        return self.model, masks
```

**Methods**:
- `get_prunable_layers(model)`: Get dict of prunable layers
- `get_sparsity(masks)`: Calculate sparsity from masks
- `apply_masks(masks)`: Apply pruning masks to model
- `prune(pruning_ratio, **kwargs)`: Abstract method - implement this

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{armenta2021representation,
  title={The Representation Theory of Neural Networks},
  author={Armenta, Marco A and Jodoin, Pierre-Marc},
  journal={arXiv preprint arXiv:2111.14823},
  year={2021}
}
```

## License

[Include your license here]

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Support

For questions or issues:
- Open an issue on GitHub
- See `Methods.md` for mathematical details
- See `CLAUDE.md` for implementation details

---

**Happy Pruning!** 🎯
