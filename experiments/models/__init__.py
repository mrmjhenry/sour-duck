"""
Model Factory for Pruning Experiments
======================================

Provides ResNet-18 and VGG-16 implementations optimized for different datasets.
"""

import torch
import torch.nn as nn
from typing import Optional


def get_model(
    architecture: str,
    num_classes: int,
    input_size: tuple = (3, 32, 32),
    pretrained: bool = False
) -> nn.Module:
    """
    Get model by architecture name.

    Args:
        architecture: 'ResNet18' or 'VGG16'
        num_classes: Number of output classes
        input_size: Input image size (C, H, W)
        pretrained: Whether to load pretrained weights (ImageNet only)

    Returns:
        PyTorch model
    """
    architecture = architecture.lower()
    _, height, width = input_size

    # Determine if CIFAR-sized or ImageNet-sized
    is_cifar = (height == 32 and width == 32)

    if architecture == 'resnet18':
        from .resnet import ResNet18
        return ResNet18(num_classes=num_classes, cifar=is_cifar)

    elif architecture == 'vgg16':
        from .vgg import VGG16
        return VGG16(num_classes=num_classes, cifar=is_cifar)

    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def get_model_info(model: nn.Module) -> dict:
    """
    Get model statistics.

    Returns:
        Dictionary with: total_params, trainable_params, model_size_mb
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Estimate model size in MB (assuming float32)
    model_size_mb = total_params * 4 / (1024 ** 2)

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': model_size_mb
    }


__all__ = ['get_model', 'get_model_info']
