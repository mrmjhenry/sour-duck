"""
ResNet-18 Implementation
========================

Adapted for both CIFAR (32x32) and ImageNet (224x224) inputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Basic residual block for ResNet-18/34"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out, inplace=True)

        return out


class ResNet18(nn.Module):
    """
    ResNet-18 architecture.

    Supports both CIFAR (32x32) and ImageNet (224x224) inputs.
    """

    def __init__(self, num_classes=10, cifar=True, initial_channels=64):
        """
        Args:
            num_classes: Number of output classes
            cifar: If True, use CIFAR-style input (32x32)
                  If False, use ImageNet-style input (224x224)
            initial_channels: Number of channels in first conv layer
        """
        super(ResNet18, self).__init__()

        self.in_channels = initial_channels
        self.cifar = cifar

        if cifar:
            # CIFAR-style: 3x3 conv, no pooling
            self.conv1 = nn.Conv2d(
                3, initial_channels,
                kernel_size=3, stride=1, padding=1, bias=False
            )
            self.bn1 = nn.BatchNorm2d(initial_channels)
            self.maxpool = nn.Identity()  # No pooling for CIFAR
        else:
            # ImageNet-style: 7x7 conv, max pooling
            self.conv1 = nn.Conv2d(
                3, initial_channels,
                kernel_size=7, stride=2, padding=3, bias=False
            )
            self.bn1 = nn.BatchNorm2d(initial_channels)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers: [2, 2, 2, 2] blocks
        self.layer1 = self._make_layer(initial_channels, 2, stride=1)
        self.layer2 = self._make_layer(initial_channels * 2, 2, stride=2)
        self.layer3 = self._make_layer(initial_channels * 4, 2, stride=2)
        self.layer4 = self._make_layer(initial_channels * 8, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(initial_channels * 8, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, out_channels, blocks, stride=1):
        """Create a residual layer with specified number of blocks"""
        downsample = None

        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels

        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.maxpool(x)

        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Classification head
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def get_num_parameters(self):
        """Count total number of parameters"""
        return sum(p.numel() for p in self.parameters())


def create_resnet18_cifar(num_classes=10, small=False):
    """
    Convenience function for CIFAR ResNet-18.

    Args:
        num_classes: Number of output classes
        small: If True, use half channels for faster training
    """
    initial_channels = 32 if small else 64
    return ResNet18(num_classes=num_classes, cifar=True, initial_channels=initial_channels)


def create_resnet18_imagenet(num_classes=1000):
    """Convenience function for ImageNet ResNet-18"""
    return ResNet18(num_classes=num_classes, cifar=False, initial_channels=64)


if __name__ == '__main__':
    # Test CIFAR model
    print("Testing ResNet-18 for CIFAR-10...")
    model_cifar = create_resnet18_cifar(num_classes=10)
    x_cifar = torch.randn(2, 3, 32, 32)
    y_cifar = model_cifar(x_cifar)
    print(f"✓ CIFAR Input: {x_cifar.shape} → Output: {y_cifar.shape}")
    print(f"  Parameters: {model_cifar.get_num_parameters():,}")

    # Test ImageNet model
    print("\nTesting ResNet-18 for ImageNet...")
    model_imagenet = create_resnet18_imagenet(num_classes=1000)
    x_imagenet = torch.randn(2, 3, 224, 224)
    y_imagenet = model_imagenet(x_imagenet)
    print(f"✓ ImageNet Input: {x_imagenet.shape} → Output: {y_imagenet.shape}")
    print(f"  Parameters: {model_imagenet.get_num_parameters():,}")
