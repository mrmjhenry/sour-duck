"""
Configuration Management for Pruning Experiments
================================================

Handles loading and validation of YAML configuration files.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass, field


@dataclass
class ExperimentConfig:
    """Main experiment configuration"""
    name: str
    output_dir: str
    seed: int = 42
    device: str = "auto"


@dataclass
class DatasetConfig:
    """Dataset configuration"""
    name: str
    data_dir: str = "./data"
    batch_size: int = 128
    num_workers: int = 4
    augmentation: bool = True


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    architecture: str
    pretrained: bool = False
    num_classes: int = None  # Auto-set based on dataset


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    epochs: int = 100
    optimizer: str = "SGD"
    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 0.0001
    lr_schedule: str = "cosine"
    warmup_epochs: int = 5


@dataclass
class ModuliSpaceConfig:
    """Moduli space pruning configuration"""
    max_samples: int = 1000
    strategy: str = "combined"
    ablation_enabled: bool = True
    ablation_components: List[str] = field(default_factory=lambda: [
        "usage_only", "variance_only", "information_only", "combined"
    ])


@dataclass
class PruningConfig:
    """Pruning configuration"""
    methods: List[str] = field(default_factory=list)
    sparsity_levels: List[float] = field(default_factory=lambda: [0.1, 0.3, 0.5, 0.7, 0.9])
    finetune_epochs: int = 20
    finetune_lr: float = 0.01
    moduli_space: ModuliSpaceConfig = field(default_factory=ModuliSpaceConfig)


@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "memory_footprint"])
    moduli_analysis_enabled: bool = True
    moduli_plots: List[str] = field(default_factory=lambda: [
        "dimension_vs_architecture",
        "dimension_vs_prunable",
        "importance_distribution"
    ])


@dataclass
class Config:
    """Complete configuration"""
    experiment: ExperimentConfig
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    pruning: PruningConfig
    evaluation: EvaluationConfig


def load_config(config_path: str) -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Config object with all parameters
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Parse nested configs
    experiment = ExperimentConfig(**config_dict.get('experiment', {}))
    dataset = DatasetConfig(**config_dict.get('dataset', {}))
    model = ModelConfig(**config_dict.get('model', {}))
    training = TrainingConfig(**config_dict.get('training', {}))

    # Parse pruning config
    pruning_dict = config_dict.get('pruning', {})
    moduli_dict = pruning_dict.get('moduli_space', {})

    # Extract ablation config
    ablation_dict = moduli_dict.get('ablation', {})
    moduli_space = ModuliSpaceConfig(
        max_samples=moduli_dict.get('max_samples', 1000),
        strategy=moduli_dict.get('strategy', 'combined'),
        ablation_enabled=ablation_dict.get('enabled', True),
        ablation_components=ablation_dict.get('components', [
            "usage_only", "variance_only", "information_only", "combined"
        ])
    )

    pruning = PruningConfig(
        methods=pruning_dict.get('methods', []),
        sparsity_levels=pruning_dict.get('sparsity_levels', [0.1, 0.3, 0.5, 0.7, 0.9]),
        finetune_epochs=pruning_dict.get('finetune_epochs', 20),
        finetune_lr=pruning_dict.get('finetune_lr', 0.01),
        moduli_space=moduli_space
    )

    # Parse evaluation config
    eval_dict = config_dict.get('evaluation', {})
    moduli_analysis = eval_dict.get('moduli_analysis', {})
    evaluation = EvaluationConfig(
        metrics=eval_dict.get('metrics', ['accuracy', 'memory_footprint']),
        moduli_analysis_enabled=moduli_analysis.get('enabled', True),
        moduli_plots=moduli_analysis.get('plots', [
            "dimension_vs_architecture",
            "dimension_vs_prunable"
        ])
    )

    # Auto-set num_classes based on dataset
    if model.num_classes is None:
        dataset_classes = {
            'CIFAR10': 10,
            'CIFAR100': 100,
            'ImageNet': 1000
        }
        model.num_classes = dataset_classes.get(dataset.name, 10)

    return Config(
        experiment=experiment,
        dataset=dataset,
        model=model,
        training=training,
        pruning=pruning,
        evaluation=evaluation
    )


def save_config(config: Config, output_path: str):
    """Save configuration to YAML file"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dict
    config_dict = {
        'experiment': config.experiment.__dict__,
        'dataset': config.dataset.__dict__,
        'model': config.model.__dict__,
        'training': config.training.__dict__,
        'pruning': {
            'methods': config.pruning.methods,
            'sparsity_levels': config.pruning.sparsity_levels,
            'finetune_epochs': config.pruning.finetune_epochs,
            'finetune_lr': config.pruning.finetune_lr,
            'moduli_space': config.pruning.moduli_space.__dict__
        },
        'evaluation': config.evaluation.__dict__
    }

    with open(output_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


def print_config(config: Config):
    """Pretty print configuration"""
    print("=" * 70)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 70)
    print(f"\nExperiment: {config.experiment.name}")
    print(f"Output Dir: {config.experiment.output_dir}")
    print(f"Device: {config.experiment.device}")
    print(f"Seed: {config.experiment.seed}")

    print(f"\nDataset: {config.dataset.name}")
    print(f"Batch Size: {config.dataset.batch_size}")

    print(f"\nModel: {config.model.architecture}")
    print(f"Classes: {config.model.num_classes}")
    print(f"Pretrained: {config.model.pretrained}")

    print(f"\nTraining:")
    print(f"  Epochs: {config.training.epochs}")
    print(f"  LR: {config.training.learning_rate}")
    print(f"  Optimizer: {config.training.optimizer}")

    print(f"\nPruning Methods: {', '.join(config.pruning.methods)}")
    print(f"Sparsity Levels: {config.pruning.sparsity_levels}")
    print(f"Finetune Epochs: {config.pruning.finetune_epochs}")

    if 'moduli_space' in config.pruning.methods:
        print(f"\nModuli Space Config:")
        print(f"  Max Samples: {config.pruning.moduli_space.max_samples}")
        print(f"  Strategy: {config.pruning.moduli_space.strategy}")
        print(f"  Ablation: {config.pruning.moduli_space.ablation_enabled}")

    print(f"\nMetrics: {', '.join(config.evaluation.metrics)}")
    print("=" * 70)


if __name__ == '__main__':
    # Test configuration loading
    config = load_config('experiments/config/default.yaml')
    print_config(config)
