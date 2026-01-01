"""
Dataset Loaders for Pruning Experiments
========================================

Provides unified interface for CIFAR-10, CIFAR-100, and ImageNet datasets.
"""

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
from typing import Tuple, Optional


class DatasetFactory:
    """Factory for creating dataset loaders"""

    @staticmethod
    def get_dataset(
        name: str,
        data_dir: str = './data',
        batch_size: int = 128,
        num_workers: int = 4,
        augmentation: bool = True,
        download: bool = True
    ) -> Tuple[DataLoader, DataLoader, int]:
        """
        Get train and test dataloaders for specified dataset.

        Args:
            name: Dataset name ('CIFAR10', 'CIFAR100', 'ImageNet')
            data_dir: Directory to store/load data
            batch_size: Batch size for training
            num_workers: Number of data loading workers
            augmentation: Whether to apply data augmentation
            download: Whether to download dataset if not present

        Returns:
            (train_loader, test_loader, num_classes)
        """
        name = name.upper()

        if name == 'CIFAR10':
            return DatasetFactory._get_cifar10(
                data_dir, batch_size, num_workers, augmentation, download
            )
        elif name == 'CIFAR100':
            return DatasetFactory._get_cifar100(
                data_dir, batch_size, num_workers, augmentation, download
            )
        elif name == 'IMAGENET':
            return DatasetFactory._get_imagenet(
                data_dir, batch_size, num_workers, augmentation
            )
        else:
            raise ValueError(f"Unknown dataset: {name}")

    @staticmethod
    def _get_cifar10(
        data_dir: str,
        batch_size: int,
        num_workers: int,
        augmentation: bool,
        download: bool
    ) -> Tuple[DataLoader, DataLoader, int]:
        """Get CIFAR-10 dataloaders"""

        # Data augmentation for training
        if augmentation:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2023, 0.1994, 0.2010]
                ),
            ])
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2023, 0.1994, 0.2010]
                ),
            ])

        # Test transform (no augmentation)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            ),
        ])

        # Load datasets
        trainset = torchvision.datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=download,
            transform=transform_train
        )

        testset = torchvision.datasets.CIFAR10(
            root=data_dir,
            train=False,
            download=download,
            transform=transform_test
        )

        # Create dataloaders
        train_loader = DataLoader(
            trainset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

        test_loader = DataLoader(
            testset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        return train_loader, test_loader, 10  # CIFAR-10 has 10 classes

    @staticmethod
    def _get_cifar100(
        data_dir: str,
        batch_size: int,
        num_workers: int,
        augmentation: bool,
        download: bool
    ) -> Tuple[DataLoader, DataLoader, int]:
        """Get CIFAR-100 dataloaders"""

        # Data augmentation for training
        if augmentation:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5071, 0.4867, 0.4408],
                    std=[0.2675, 0.2565, 0.2761]
                ),
            ])
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5071, 0.4867, 0.4408],
                    std=[0.2675, 0.2565, 0.2761]
                ),
            ])

        # Test transform
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761]
            ),
        ])

        # Load datasets
        trainset = torchvision.datasets.CIFAR100(
            root=data_dir,
            train=True,
            download=download,
            transform=transform_train
        )

        testset = torchvision.datasets.CIFAR100(
            root=data_dir,
            train=False,
            download=download,
            transform=transform_test
        )

        # Create dataloaders
        train_loader = DataLoader(
            trainset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

        test_loader = DataLoader(
            testset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        return train_loader, test_loader, 100  # CIFAR-100 has 100 classes

    @staticmethod
    def _get_imagenet(
        data_dir: str,
        batch_size: int,
        num_workers: int,
        augmentation: bool
    ) -> Tuple[DataLoader, DataLoader, int]:
        """Get ImageNet dataloaders"""

        data_path = Path(data_dir) / 'imagenet'
        train_dir = data_path / 'train'
        val_dir = data_path / 'val'

        if not train_dir.exists() or not val_dir.exists():
            raise FileNotFoundError(
                f"ImageNet data not found at {data_path}. "
                f"Please download and extract ImageNet to {data_path} with "
                f"'train' and 'val' subdirectories."
            )

        # ImageNet normalization
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # Training transform
        if augmentation:
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            transform_train = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])

        # Validation transform
        transform_val = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

        # Load datasets
        trainset = torchvision.datasets.ImageFolder(
            root=str(train_dir),
            transform=transform_train
        )

        valset = torchvision.datasets.ImageFolder(
            root=str(val_dir),
            transform=transform_val
        )

        # Create dataloaders
        train_loader = DataLoader(
            trainset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

        val_loader = DataLoader(
            valset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        return train_loader, val_loader, 1000  # ImageNet has 1000 classes


def get_dataset_info(name: str) -> dict:
    """Get dataset metadata"""
    info = {
        'CIFAR10': {
            'num_classes': 10,
            'input_size': (3, 32, 32),
            'mean': [0.4914, 0.4822, 0.4465],
            'std': [0.2023, 0.1994, 0.2010],
        },
        'CIFAR100': {
            'num_classes': 100,
            'input_size': (3, 32, 32),
            'mean': [0.5071, 0.4867, 0.4408],
            'std': [0.2675, 0.2565, 0.2761],
        },
        'IMAGENET': {
            'num_classes': 1000,
            'input_size': (3, 224, 224),
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
        },
    }
    return info.get(name.upper(), {})


if __name__ == '__main__':
    # Test dataset loading
    print("Testing CIFAR-10 loading...")
    train_loader, test_loader, num_classes = DatasetFactory.get_dataset(
        'CIFAR10',
        batch_size=128,
        download=True
    )
    print(f"✓ CIFAR-10 loaded: {num_classes} classes")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    # Get sample batch
    images, labels = next(iter(train_loader))
    print(f"  Batch shape: {images.shape}")
    print(f"  Labels shape: {labels.shape}")
