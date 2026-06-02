"""
VGG-16 Implementation
=====================

Adapted for both CIFAR (32x32) and ImageNet (224x224) inputs.
"""

import torch
import torch.nn as nn


class VGG16(nn.Module):
    """
    VGG-16 architecture with batch normalization.

    Supports both CIFAR (32x32) and ImageNet (224x224) inputs.
    """

    def __init__(self, num_classes=10, cifar=True):
        """
        Args:
            num_classes: Number of output classes
            cifar: If True, use CIFAR-style architecture (32x32)
                  If False, use ImageNet-style architecture (224x224)
        """
        super(VGG16, self).__init__()

        self.cifar = cifar

        # VGG-16 configuration: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        self.features = self._make_layers([
            64, 64, 'M',
            128, 128, 'M',
            256, 256, 256, 'M',
            512, 512, 512, 'M',
            512, 512, 512, 'M'
        ])

        if cifar:
            # CIFAR: Smaller spatial dimensions after convolutions
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(512, 512),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes),
            )
        else:
            # ImageNet: Standard VGG classifier
            self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(4096, num_classes),
            )

        self._initialize_weights()

    def _make_layers(self, cfg):
        """
        Build convolutional layers from configuration.

        Args:
            cfg: List specifying layer configuration
                 Numbers indicate conv channels, 'M' indicates max pooling
        """
        layers = []
        in_channels = 3

        for v in cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers.append(conv2d)
                layers.append(nn.BatchNorm2d(v))
                layers.append(nn.ReLU(inplace=True))
                in_channels = v

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def get_num_parameters(self):
        """Count total number of parameters"""
        return sum(p.numel() for p in self.parameters())


def create_vgg16_cifar(num_classes=10):
    """Convenience function for CIFAR VGG-16"""
    return VGG16(num_classes=num_classes, cifar=True)


def create_vgg16_imagenet(num_classes=1000):
    """Convenience function for ImageNet VGG-16"""
    return VGG16(num_classes=num_classes, cifar=False)


if __name__ == '__main__':
    # Test CIFAR model
    print("Testing VGG-16 for CIFAR-10...")
    model_cifar = create_vgg16_cifar(num_classes=10)
    x_cifar = torch.randn(2, 3, 32, 32)
    y_cifar = model_cifar(x_cifar)
    print(f"✓ CIFAR Input: {x_cifar.shape} → Output: {y_cifar.shape}")
    print(f"  Parameters: {model_cifar.get_num_parameters():,}")

    # Test ImageNet model
    print("\nTesting VGG-16 for ImageNet...")
    model_imagenet = create_vgg16_imagenet(num_classes=1000)
    x_imagenet = torch.randn(2, 3, 224, 224)
    y_imagenet = model_imagenet(x_imagenet)
    print(f"✓ ImageNet Input: {x_imagenet.shape} → Output: {y_imagenet.shape}")
    print(f"  Parameters: {model_imagenet.get_num_parameters():,}")
