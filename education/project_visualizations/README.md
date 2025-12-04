# Project-Specific Visualizations

This folder contains Manim animations that demonstrate the **moduli space pruning** framework applied to real neural network architectures. These visualizations complement the theoretical materials in the parent `education/` folder by showing practical applications.

## Overview

The visualizations in this folder demonstrate:
1. **Network Architecture**: Structure of SimpleNet and ResNet-18
2. **Pruning Effects**: How pruning changes network connectivity
3. **Coordinate Importance**: Visualization of importance scoring
4. **Forward Pass**: Data flow through networks before/after pruning
5. **Architecture Comparisons**: Side-by-side analysis of different models

## Files

### 1. `01_network_architectures.py`
**Purpose**: Visualize the structure of networks used in the project.

**Scenes**:
- `SimpleNetArchitecture`: Shows SimpleNet (Conv 32 → Conv 64 → FC 256 → FC 10)
  - Displays layer dimensions and parameter counts (~165K parameters)
  - Uses ManimML for professional neural network visualization
  - Shows feature map transformations through convolutions

- `ResNet18Architecture`: Demonstrates ResNet-18 structure
  - Shows residual blocks with skip connections (curved arrows)
  - 4 residual layers: [2, 2, 2, 2] blocks
  - Highlights the identity mappings that enable deep learning
  - Total parameters: ~11M (standard) or ~3M (small variant)

- `ArchitectureComparison`: Side-by-side comparison table
  - Layers count (5 vs 18)
  - Parameter counts (165K vs 11M)
  - Training time comparison (5 min vs 30 min)
  - Accuracy comparison (89% vs 93%)
  - Use case recommendations

**Rendering**:
```bash
# All scenes
manim -pql 01_network_architectures.py

# Specific scene
manim -pql 01_network_architectures.py SimpleNetArchitecture
manim -pql 01_network_architectures.py ResNet18Architecture
manim -pql 01_network_architectures.py ArchitectureComparison

# High quality
manim -pqh 01_network_architectures.py
```

**Key Concepts Demonstrated**:
- Convolutional layers vs fully-connected layers
- Residual connections (skip connections)
- Network depth vs width trade-offs
- Parameter efficiency

---

### 2. `02_pruning_effects.py`
**Purpose**: Demonstrate the pruning process and its effects on network structure.

**Scenes**:
- `PruningProcessScene`: 3-step pruning animation
  1. **Analyze Importance**: Shows importance scores as colored bars
  2. **Identify Prunable**: Highlights low-importance connections in red
  3. **Remove Connections**: Fades out pruned edges, shows sparsity

- `BeforeAfterPruning`: Side-by-side comparison
  - Left: Original network (all connections gray)
  - Right: Pruned network (50% sparsity)
  - Kept connections highlighted in green
  - Pruned connections faded out
  - Shows that network remains functional

- `PruningStatistics`: Quantitative analysis
  - Shows statistics for 30%, 50%, 70% pruning ratios
  - Displays: sparsity, active parameters, compression ratio
  - Plots accuracy vs sparsity trade-off curve
  - Compares moduli space pruning (green) vs random pruning (red)
  - Demonstrates that moduli pruning maintains accuracy better

**Rendering**:
```bash
manim -pql 02_pruning_effects.py
manim -pql 02_pruning_effects.py PruningProcessScene
manim -pql 02_pruning_effects.py BeforeAfterPruning
manim -pql 02_pruning_effects.py PruningStatistics
```

**Key Concepts Demonstrated**:
- Coordinate importance scoring
- Pruning threshold selection
- Sparsity vs accuracy trade-off
- Superiority of informed pruning over random pruning

**Color Scheme**:
- 🟢 **Green**: Connections kept after pruning
- 🔴 **Red**: Connections to be pruned
- ⚪ **Gray**: Neutral/unpruned connections

---

### 3. `03_coordinate_importance.py`
**Purpose**: Visualize how coordinate importance is computed and used.

**Scenes**:
- `ImportanceComputationScene`: Shows the three-component formula
  - **Usage Rate (ρ)**: Fraction of samples activating coordinate (blue)
  - **Variance (σ²)**: Variability of coordinate values (purple)
  - **Information Content (I)**: Entropy-based measure (orange)
  - **Combined Formula**: `Importance = ρ × √σ² × (1 + I)`
  - Includes example calculation with concrete numbers

- `NodeScalingScene`: Scales node size by importance
  - Larger nodes = higher importance
  - Color gradient: Red (low) → Yellow (medium) → Green (high)
  - Shows which neurons are most critical to network function
  - Before/after comparison (uniform vs scaled)

- `ImportanceEvolutionScene`: Shows importance changes during training
  - Multiple neurons tracked over 100 epochs
  - Shows rapid change early, then stabilization (~30-40 epochs)
  - Highlights stabilization region
  - Shows pruning threshold (e.g., 0.4)
  - Identifies neurons that would be pruned

- `ComparisonBeforeAfterScene`: Side-by-side uniform vs importance-scaled
  - Before: All neurons treated equally
  - After: Size and color indicate importance
  - Clearly shows network's critical components

**Rendering**:
```bash
manim -pql 03_coordinate_importance.py
manim -pql 03_coordinate_importance.py ImportanceComputationScene
manim -pql 03_coordinate_importance.py NodeScalingScene
manim -pql 03_coordinate_importance.py ImportanceEvolutionScene
```

**Key Concepts Demonstrated**:
- Three-metric importance scoring
- Visual representation of neuron criticality
- Importance stabilization during training
- Optimal timing for pruning analysis

**Formula**:
```
Importance = ρ × √σ² × (1 + I)

where:
  ρ  = usage rate (how often coordinate is active)
  σ² = variance (how much coordinate values vary)
  I  = information content (entropy measure)
```

---

### 4. `04_forward_pass.py`
**Purpose**: Show how data flows through networks during inference.

**Scenes**:
- `ForwardPassBasicScene`: Basic forward propagation
  - Input data [0.8, 0.3, 0.9] visualized as bar chart
  - Step-by-step layer computation
  - Edges light up during computation (yellow)
  - Output prediction [0.73, 0.27] → Class 0
  - Shows network structure [3, 4, 3, 2]

- `ForwardPassWithActivationsScene`: Shows activation magnitudes
  - Nodes pulse with intensity ∝ activation strength
  - Color intensity represents activation value
  - Demonstrates ReLU effects (zero vs active)
  - Shows layer-by-layer activation propagation

- `BeforeAfterPruningFlowScene`: Compares flow before/after pruning
  - Side-by-side networks (before left, after right)
  - Before: All edges active (gray)
  - After: 50% pruned (red faded), 50% kept (green)
  - Shows both networks produce similar activations
  - Demonstrates that pruned network maintains functionality

- `DataFlowAnimationScene`: Multiple data points flowing
  - 3 samples with different colors (green, yellow, red)
  - Each sample creates unique activation pattern
  - Shows network's learned representation
  - Demonstrates how network responds to different inputs

**Rendering**:
```bash
manim -pql 04_forward_pass.py
manim -pql 04_forward_pass.py ForwardPassBasicScene
manim -pql 04_forward_pass.py ForwardPassWithActivationsScene
manim -pql 04_forward_pass.py BeforeAfterPruningFlowScene
manim -pql 04_forward_pass.py DataFlowAnimationScene
```

**Key Concepts Demonstrated**:
- Forward propagation mechanics
- Activation patterns
- Network inference before/after pruning
- Output preservation despite pruning
- Class-specific activation patterns

**Color Scheme**:
- 🟢 **Green**: Input data / Class 0
- 🟡 **Yellow**: Active flow / Medium activation / Class 1
- 🔴 **Red**: Output layer / Class 2
- 🔵 **Blue**: Hidden layers

---

## Installation

### Prerequisites

```bash
# Python 3.8+
python --version

# Install Manim Community
pip install manim

# Install ManimML (for professional neural network visualizations)
pip install manim-ml

# Additional dependencies
pip install numpy
```

### Verifying Installation

```bash
# Test Manim installation
manim --version

# Render a test scene (should take ~10 seconds)
manim -pql 01_network_architectures.py SimpleNetArchitecture
```

If successful, this will:
1. Render the scene
2. Open the video in your default player
3. Save output to `media/videos/01_network_architectures/480p15/`

---

## Rendering Guide

### Quality Options

| Flag | Resolution | FPS | Use Case |
|------|------------|-----|----------|
| `-ql` | 480p | 15 | Quick preview (fastest) |
| `-qm` | 720p | 30 | Medium quality |
| `-qh` | 1080p | 60 | High quality |
| `-qk` | 4K | 60 | Ultra quality (slowest) |

### Common Commands

```bash
# Quick preview (low quality, fast rendering)
manim -pql script.py SceneName

# High quality (for presentations)
manim -pqh script.py SceneName

# Save without playing
manim -ql script.py SceneName

# Render all scenes in a file
manim -pql script.py

# Render specific scene only
manim -pql script.py SceneName

# Save as GIF
manim -pql --format gif script.py SceneName

# Transparent background (for overlays)
manim -pql --transparent script.py SceneName
```

### Rendering All Visualizations

To generate all project visualizations at once:

```bash
#!/bin/bash
# render_all.sh

echo "Rendering all project visualizations..."

manim -ql 01_network_architectures.py
manim -ql 02_pruning_effects.py
manim -ql 03_coordinate_importance.py
manim -ql 04_forward_pass.py

echo "Done! Videos saved to media/videos/"
```

Make executable and run:
```bash
chmod +x render_all.sh
./render_all.sh
```

**Note**: Full rendering may take 10-30 minutes depending on your system.

---

## Output Structure

After rendering, Manim creates this structure:

```
media/
├── videos/
│   ├── 01_network_architectures/
│   │   └── 480p15/  (or 1080p60, etc.)
│   │       ├── SimpleNetArchitecture.mp4
│   │       ├── ResNet18Architecture.mp4
│   │       └── ArchitectureComparison.mp4
│   ├── 02_pruning_effects/
│   │   └── 480p15/
│   │       ├── PruningProcessScene.mp4
│   │       ├── BeforeAfterPruning.mp4
│   │       └── PruningStatistics.mp4
│   ├── 03_coordinate_importance/
│   │   └── 480p15/
│   │       ├── ImportanceComputationScene.mp4
│   │       ├── NodeScalingScene.mp4
│   │       ├── ImportanceEvolutionScene.mp4
│   │       └── ComparisonBeforeAfterScene.mp4
│   └── 04_forward_pass/
│       └── 480p15/
│           ├── ForwardPassBasicScene.mp4
│           ├── ForwardPassWithActivationsScene.mp4
│           ├── BeforeAfterPruningFlowScene.mp4
│           └── DataFlowAnimationScene.mp4
└── images/  (if you render images instead of videos)
```

---

## Customization

### Modifying Networks

To visualize your own network:

```python
# In any script, change the layers list:
layers = [3, 5, 4, 2]  # [input, hidden1, hidden2, output]

# For example, a deeper network:
layers = [10, 20, 15, 10, 5]
```

### Changing Colors

All scripts use consistent color schemes defined at the top:

```python
# In 02_pruning_effects.py:
KEEP_COLOR = GREEN      # Connections kept
PRUNE_COLOR = RED       # Connections pruned
NEUTRAL_COLOR = GRAY    # Unpruned connections

# Modify as needed:
KEEP_COLOR = BLUE       # Change to blue
PRUNE_COLOR = ORANGE    # Change to orange
```

### Adjusting Timing

To make animations faster/slower:

```python
# Find lines like:
self.play(Transform(...), run_time=2)  # 2 seconds

# Change run_time:
self.play(Transform(...), run_time=1)  # Faster (1 sec)
self.play(Transform(...), run_time=4)  # Slower (4 sec)
```

### Pruning Ratios

To change pruning percentages:

```python
# In 02_pruning_effects.py, BeforeAfterPruning:
n_prune = total_edges // 2  # 50% pruning

# Change to:
n_prune = int(total_edges * 0.3)  # 30% pruning
n_prune = int(total_edges * 0.7)  # 70% pruning
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'manim_ml'"

**Solution**: Install ManimML or use fallback visualization
```bash
pip install manim-ml
```

The scripts will automatically fall back to custom implementations if ManimML is unavailable.

### Issue: "LaTeX Error: File not found"

**Solution**: Install LaTeX (required for math formulas)
```bash
# Ubuntu/Debian
sudo apt-get install texlive texlive-latex-extra

# macOS
brew install --cask mactex

# Windows
# Download and install MiKTeX: https://miktex.org/download
```

### Issue: Slow rendering

**Solutions**:
1. Use lower quality: `-ql` instead of `-qh`
2. Reduce `run_time` values in the code
3. Render specific scenes instead of all scenes
4. Use a more powerful machine or GPU acceleration

### Issue: Video doesn't play automatically

**Solution**: Videos are saved to `media/videos/`. Open manually:
```bash
# Find your video
ls media/videos/01_network_architectures/480p15/

# Play with default player
open media/videos/01_network_architectures/480p15/SimpleNetArchitecture.mp4
```

Or use `-p` flag to play automatically:
```bash
manim -pql 01_network_architectures.py SimpleNetArchitecture
#      ^-- p for "preview"
```

---

## Integration with Main Project

These visualizations demonstrate concepts from:

1. **Core Implementation**: `moduli_space_pruning.py`
   - QuiverRepresentation (network as graph)
   - ModuliSpaceAnalyzer (importance scoring)
   - ModuliSpacePruner (applying pruning)

2. **Experiments**: `experiments.py`, `experiments_resnet18.py`
   - Network architectures (SimpleNet, ResNet-18)
   - Pruning strategies comparison
   - Accuracy vs sparsity trade-offs

3. **Theory**: `education/01_quiver_representations.md`, `education/02_moduli_space.md`
   - Mathematical foundations
   - Quiver theory
   - Moduli space geometry

### Workflow

```
1. Read theory: education/*.md
   ↓
2. Watch animations: education/manim_animations/*.py
   ↓
3. See project application: education/project_visualizations/*.py (THIS FOLDER)
   ↓
4. Run experiments: experiments.py
   ↓
5. Understand implementation: moduli_space_pruning.py
```

---

## Scene Summary Table

| File | Scene | Duration | Concept |
|------|-------|----------|---------|
| 01 | SimpleNetArchitecture | ~30s | CNN structure |
| 01 | ResNet18Architecture | ~40s | Residual blocks |
| 01 | ArchitectureComparison | ~20s | Model comparison |
| 02 | PruningProcessScene | ~35s | 3-step pruning |
| 02 | BeforeAfterPruning | ~25s | Pruning effects |
| 02 | PruningStatistics | ~30s | Trade-off curves |
| 03 | ImportanceComputationScene | ~30s | Importance formula |
| 03 | NodeScalingScene | ~35s | Scaled visualization |
| 03 | ImportanceEvolutionScene | ~40s | Training dynamics |
| 03 | ComparisonBeforeAfterScene | ~25s | Before/after |
| 04 | ForwardPassBasicScene | ~30s | Basic propagation |
| 04 | ForwardPassWithActivationsScene | ~25s | Activation flow |
| 04 | BeforeAfterPruningFlowScene | ~35s | Flow comparison |
| 04 | DataFlowAnimationScene | ~30s | Multiple samples |

**Total**: ~14 scenes, ~7-10 minutes of animation

---

## Learning Path

### For Beginners
1. Start with `01_network_architectures.py` - SimpleNetArchitecture
2. Watch `04_forward_pass.py` - ForwardPassBasicScene
3. View `02_pruning_effects.py` - BeforeAfterPruning
4. Read the parent folder's theory documents

### For Advanced Users
1. Review `03_coordinate_importance.py` - all scenes
2. Study `02_pruning_effects.py` - PruningStatistics
3. Analyze `04_forward_pass.py` - BeforeAfterPruningFlowScene
4. Dive into the main codebase implementation

### For Researchers
1. Understand all scenes in order
2. Modify code to visualize your own networks
3. Compare with other pruning methods
4. Reference the mathematical foundations in theory docs

---

## Contributing

To add new visualizations:

1. **Create new Python file**: `05_your_visualization.py`
2. **Follow naming convention**: Use descriptive scene names
3. **Include docstring**: Explain purpose and usage
4. **Update this README**: Add section describing your scenes
5. **Test rendering**: Ensure all scenes render correctly
6. **Use consistent colors**: Follow established color scheme

**Template**:
```python
"""
Your Visualization Title
========================

Brief description of what this visualizes.

Scenes:
-------
1. SceneName1: Description
2. SceneName2: Description

Usage:
------
manim -pql 05_your_visualization.py
"""

from manim import *

class SceneName1(Scene):
    \"\"\"Detailed docstring\"\"\"
    def construct(self):
        # Your animation code
        pass
```

---

## References

- **Manim Documentation**: https://docs.manim.community/
- **ManimML Documentation**: https://github.com/helblazer811/ManimML
- **Main Project**: See parent folder and root README.md
- **Paper**: Armenta & Jodoin (2021) - "The Representation Theory of Neural Networks"

---

## Quick Start

```bash
# 1. Install dependencies
pip install manim manim-ml numpy

# 2. Render a scene
manim -pql 01_network_architectures.py SimpleNetArchitecture

# 3. View output
# Video will play automatically if -p flag used
# Otherwise find in: media/videos/01_network_architectures/480p15/

# 4. Explore other scenes
manim -pql 02_pruning_effects.py
manim -pql 03_coordinate_importance.py
manim -pql 04_forward_pass.py
```

---

**Last Updated**: 2025-12-04
**Maintainer**: Claude (AI Assistant)
**License**: Same as main project
