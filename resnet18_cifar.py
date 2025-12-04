"""
ResNet-18 Architecture for CIFAR-10

Adapted for 32x32 images (CIFAR-10) from the original ResNet designed for ImageNet.
Compatible with moduli space pruning framework.

Based on: "Deep Residual Learning for Image Recognition" (He et al., 2016)
Adapted for: Moduli space coordinate pruning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class BasicBlock(nn.Module):
    """
    Basic residual block for ResNet-18/34.

    Structure:
        3x3 conv -> BN -> ReLU -> 3x3 conv -> BN -> (+residual) -> ReLU
    """
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None
    ):
        super(BasicBlock, self).__init__()

        # First convolution
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Second convolution
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut/residual connection
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)

        # Add residual
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)

        return out


class ResNet18_CIFAR(nn.Module):
    """
    ResNet-18 adapted for CIFAR-10 (32x32 images).

    Differences from ImageNet ResNet-18:
    - Initial conv is 3x3 instead of 7x7 (no max pooling)
    - No initial max pooling layer
    - Smaller stride in first layer
    - Final average pooling adapted for 32x32 input size

    Architecture:
        - Initial: 3x3 conv, BN, ReLU
        - Layer 1: 2 basic blocks (64 channels)
        - Layer 2: 2 basic blocks (128 channels, stride 2)
        - Layer 3: 2 basic blocks (256 channels, stride 2)
        - Layer 4: 2 basic blocks (512 channels, stride 2)
        - Global average pooling
        - Fully connected classifier

    Total layers: 18 (1 + 2*2 + 2*2 + 2*2 + 2*2 + 1 = 18)
    """

    def __init__(self, num_classes: int = 10, initial_channels: int = 64):
        super(ResNet18_CIFAR, self).__init__()

        self.in_channels = initial_channels

        # Initial convolution (adapted for CIFAR-10)
        # 3x3 instead of 7x7, stride 1 instead of 2
        self.conv1 = nn.Conv2d(
            3, initial_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(initial_channels)
        self.relu = nn.ReLU(inplace=True)

        # No max pooling for CIFAR-10 (images are only 32x32)

        # Residual layers
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected classifier
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _make_layer(
        self,
        out_channels: int,
        num_blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        """
        Create a residual layer with multiple blocks.

        Args:
            out_channels: Number of output channels
            num_blocks: Number of residual blocks in this layer
            stride: Stride for the first block (downsampling)

        Returns:
            Sequential container with residual blocks
        """
        downsample = None

        # If dimensions change, need projection shortcut
        if stride != 1 or self.in_channels != out_channels * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * BasicBlock.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

        layers = []

        # First block (may downsample)
        layers.append(BasicBlock(
            self.in_channels,
            out_channels,
            stride,
            downsample
        ))

        # Update in_channels for subsequent blocks
        self.in_channels = out_channels * BasicBlock.expansion

        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(
                self.in_channels,
                out_channels
            ))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize network weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode='fan_out',
                    nonlinearity='relu'
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass through the network."""
        # Initial conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # Classifier
        x = self.fc(x)

        return x

    def get_num_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_layer_info(self) -> List[dict]:
        """Get information about each layer."""
        layer_info = []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                info = {
                    'name': name,
                    'type': type(module).__name__,
                }
                if isinstance(module, nn.Conv2d):
                    info.update({
                        'in_channels': module.in_channels,
                        'out_channels': module.out_channels,
                        'kernel_size': module.kernel_size,
                        'stride': module.stride,
                        'parameters': module.weight.numel()
                    })
                elif isinstance(module, nn.Linear):
                    info.update({
                        'in_features': module.in_features,
                        'out_features': module.out_features,
                        'parameters': module.weight.numel()
                    })
                layer_info.append(info)
        return layer_info


class ResNet18_CIFAR_Small(nn.Module):
    """
    Smaller ResNet-18 variant for faster experimentation.

    Uses half the channels of standard ResNet-18.
    Useful for quick tests and demonstrations.
    """

    def __init__(self, num_classes: int = 10):
        super(ResNet18_CIFAR_Small, self).__init__()

        # Create ResNet-18 with 32 initial channels instead of 64
        self.resnet = ResNet18_CIFAR(
            num_classes=num_classes,
            initial_channels=32
        )

    def forward(self, x):
        return self.resnet(x)

    def get_num_parameters(self):
        return self.resnet.get_num_parameters()

    def get_layer_info(self):
        return self.resnet.get_layer_info()


def create_resnet18_cifar10(num_classes: int = 10, small: bool = False):
    """
    Factory function to create ResNet-18 for CIFAR-10.

    Args:
        num_classes: Number of output classes (default: 10 for CIFAR-10)
        small: If True, use smaller variant with half the channels

    Returns:
        ResNet-18 model

    Example:
        >>> model = create_resnet18_cifar10()
        >>> print(f"Parameters: {model.get_num_parameters():,}")
        >>> x = torch.randn(1, 3, 32, 32)
        >>> y = model(x)
        >>> print(f"Output shape: {y.shape}")
    """
    if small:
        return ResNet18_CIFAR_Small(num_classes)
    else:
        return ResNet18_CIFAR(num_classes)


# Test/demo code
if __name__ == '__main__':
    print("=" * 70)
    print("ResNet-18 for CIFAR-10 - Architecture Test")
    print("=" * 70)

    # Create models
    print("\n1. Standard ResNet-18:")
    model = create_resnet18_cifar10(num_classes=10, small=False)
    print(f"   Total parameters: {model.get_num_parameters():,}")

    print("\n2. Small ResNet-18:")
    model_small = create_resnet18_cifar10(num_classes=10, small=True)
    print(f"   Total parameters: {model_small.get_num_parameters():,}")

    # Test forward pass
    print("\n3. Testing forward pass:")
    x = torch.randn(4, 3, 32, 32)  # Batch of 4 CIFAR-10 images

    model.eval()
    with torch.no_grad():
        output = model(x)

    print(f"   Input shape:  {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Output values range: [{output.min():.2f}, {output.max():.2f}]")

    # Show layer information
    print("\n4. Layer information (first 10 layers):")
    layer_info = model.get_layer_info()
    for i, info in enumerate(layer_info[:10]):
        print(f"   {i+1}. {info['name']:30s} - {info['type']:10s} - {info['parameters']:,} params")

    print(f"\n   ... ({len(layer_info) - 10} more layers)")
    print(f"\n   Total: {len(layer_info)} conv/linear layers")

    print("\n" + "=" * 70)
    print("✓ ResNet-18 architecture test completed successfully")
    print("=" * 70)
