"""Data loading and preprocessing utilities."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.transforms import v2 as transforms_v2


class CIFAR10DataModule:
    """Data module for CIFAR-10 semi-supervised learning."""
    
    def __init__(
        self,
        data_dir: str = "data/raw",
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        download: bool = True,
        augment: bool = True,
        augmentation_strength: str = "medium",
        **kwargs
    ):
        """Initialize CIFAR-10 data module.
        
        Args:
            data_dir: Directory to store data.
            batch_size: Batch size for data loaders.
            num_workers: Number of worker processes.
            pin_memory: Whether to pin memory.
            download: Whether to download dataset.
            augment: Whether to apply data augmentation.
            augmentation_strength: Strength of augmentation ('light', 'medium', 'strong').
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.download = download
        self.augment = augment
        self.augmentation_strength = augmentation_strength
        
        # Setup transforms
        self._setup_transforms()
        
        # Initialize datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def _setup_transforms(self) -> None:
        """Setup data transforms."""
        # Base transforms
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            )
        ])
        
        # Augmentation transforms
        if self.augment:
            if self.augmentation_strength == "light":
                self.augment_transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.4914, 0.4822, 0.4465],
                        std=[0.2023, 0.1994, 0.2010]
                    )
                ])
            elif self.augmentation_strength == "medium":
                self.augment_transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomCrop(32, padding=4),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.4914, 0.4822, 0.4465],
                        std=[0.2023, 0.1994, 0.2010]
                    )
                ])
            elif self.augmentation_strength == "strong":
                self.augment_transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomCrop(32, padding=4),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.4914, 0.4822, 0.4465],
                        std=[0.2023, 0.1994, 0.2010]
                    )
                ])
            else:
                raise ValueError(f"Unknown augmentation strength: {self.augmentation_strength}")
        else:
            self.augment_transform = self.base_transform
    
    def prepare_data(self) -> None:
        """Download and prepare data."""
        # Download datasets
        datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=self.download,
            transform=self.base_transform
        )
        datasets.CIFAR10(
            root=self.data_dir,
            train=False,
            download=self.download,
            transform=self.base_transform
        )
    
    def setup(self, stage: Optional[str] = None) -> None:
        """Setup datasets for training/validation/testing."""
        if stage == "fit" or stage is None:
            # Training dataset
            self.train_dataset = datasets.CIFAR10(
                root=self.data_dir,
                train=True,
                download=False,
                transform=self.augment_transform
            )
            
            # Validation dataset (subset of training)
            val_size = int(0.1 * len(self.train_dataset))
            train_size = len(self.train_dataset) - val_size
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                self.train_dataset, [train_size, val_size]
            )
            
            # Apply base transform to validation set
            self.val_dataset.dataset.transform = self.base_transform
        
        if stage == "test" or stage is None:
            # Test dataset
            self.test_dataset = datasets.CIFAR10(
                root=self.data_dir,
                train=False,
                download=False,
                transform=self.base_transform
            )
    
    def create_semi_supervised_loaders(
        self,
        labeled_samples: int = 1000,
        seed: Optional[int] = None
    ) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
        """Create semi-supervised data loaders.
        
        Args:
            labeled_samples: Number of labeled samples.
            seed: Random seed for reproducibility.
            
        Returns:
            Tuple of (labeled_loader, unlabeled_loader, val_loader, test_loader).
        """
        if self.train_dataset is None:
            self.setup("fit")
        
        # Create labeled/unlabeled splits
        labeled_indices, unlabeled_indices = self._create_splits(
            len(self.train_dataset), labeled_samples, seed
        )
        
        # Create subsets
        labeled_dataset = Subset(self.train_dataset, labeled_indices)
        unlabeled_dataset = Subset(self.train_dataset, unlabeled_indices)
        
        # Create data loaders
        labeled_loader = DataLoader(
            labeled_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        
        unlabeled_loader = DataLoader(
            unlabeled_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        
        return labeled_loader, unlabeled_loader, val_loader, test_loader
    
    def _create_splits(
        self,
        dataset_size: int,
        labeled_samples: int,
        seed: Optional[int] = None
    ) -> Tuple[List[int], List[int]]:
        """Create labeled and unlabeled splits."""
        if seed is not None:
            np.random.seed(seed)
        
        all_indices = list(range(dataset_size))
        labeled_indices = np.random.choice(
            all_indices,
            size=min(labeled_samples, dataset_size),
            replace=False
        ).tolist()
        unlabeled_indices = [i for i in all_indices if i not in labeled_indices]
        
        return labeled_indices, unlabeled_indices
    
    def get_class_names(self) -> List[str]:
        """Get CIFAR-10 class names."""
        return [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ]
