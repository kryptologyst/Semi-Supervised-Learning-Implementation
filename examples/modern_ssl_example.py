#!/usr/bin/env python3
"""
Modern implementation of the original semi-supervised learning example.
This demonstrates the same functionality as the original 0976.py but with
modern, clean, and extensible code.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import logging

from src.models import SimpleCNN
from src.models.ssl_methods import PseudoLabeling
from src.data import CIFAR10DataModule
from src.train import BasicTrainer
from src.utils import set_seed, setup_logging, get_device


def main():
    """Main function demonstrating modern SSL implementation."""
    # Setup logging and reproducibility
    setup_logging("INFO")
    set_seed(42)
    
    print("🚀 Modern Semi-Supervised Learning Implementation")
    print("=" * 60)
    
    # Setup device
    device = get_device()
    print(f"Using device: {device}")
    
    # Setup data module (replaces the original manual data loading)
    print("\n📊 Setting up CIFAR-10 dataset...")
    data_module = CIFAR10DataModule(
        data_dir='./data',
        batch_size=32,
        num_workers=2,
        download=True,
        augment=True,
        augmentation_strength="medium"
    )
    
    # Prepare and setup data
    data_module.prepare_data()
    data_module.setup("fit")
    
    # Create semi-supervised splits (replaces original random sampling)
    labeled_samples = 1000
    labeled_loader, unlabeled_loader, val_loader, test_loader = data_module.create_semi_supervised_loaders(
        labeled_samples=labeled_samples,
        seed=42
    )
    
    print(f"Labeled samples: {len(labeled_loader.dataset)}")
    print(f"Unlabeled samples: {len(unlabeled_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Setup model (replaces original SimpleCNN)
    print("\n🏗️ Setting up model...")
    model = SimpleCNN(
        input_channels=3,
        num_classes=10,
        hidden_dim=128,
        dropout=0.2
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup SSL method (replaces original pseudo-labeling logic)
    print("\n🎯 Setting up SSL method...")
    ssl_method = PseudoLabeling(
        confidence_threshold=0.95,
        rampup_epochs=10,
        rampup_type="sigmoid"
    )
    
    # Setup optimizer and scheduler (replaces original Adam setup)
    print("\n⚙️ Setting up optimizer...")
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    
    # Setup trainer (replaces original training loop)
    print("\n🏃 Setting up trainer...")
    trainer = BasicTrainer(
        model=model,
        ssl_method=ssl_method,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        max_epochs=5,
        gradient_clip_val=1.0
    )
    
    # Training (replaces original manual training loop)
    print("\n🎓 Starting training...")
    print("-" * 40)
    
    history = trainer.fit(
        labeled_loader=labeled_loader,
        unlabeled_loader=unlabeled_loader,
        val_loader=val_loader,
        save_dir="outputs/modern_example"
    )
    
    # Evaluation (replaces original manual evaluation)
    print("\n📈 Evaluating on test set...")
    test_metrics = trainer.evaluate(test_loader)
    
    # Display results (similar to original but more comprehensive)
    print("\n" + "=" * 60)
    print("📊 FINAL RESULTS")
    print("=" * 60)
    print(f"Test Accuracy: {test_metrics['accuracy']:.2f}%")
    print(f"Test F1 (macro): {test_metrics['f1_macro']:.4f}")
    print(f"Test F1 (weighted): {test_metrics['f1_weighted']:.4f}")
    print(f"Test Precision (macro): {test_metrics['precision_macro']:.4f}")
    print(f"Test Recall (macro): {test_metrics['recall_macro']:.4f}")
    
    if 'roc_auc_macro' in test_metrics:
        print(f"Test ROC AUC (macro): {test_metrics['roc_auc_macro']:.4f}")
    
    print(f"Expected Calibration Error: {test_metrics['expected_calibration_error']:.4f}")
    
    # Training history summary
    print("\n📈 Training Summary:")
    print(f"Final training loss: {history['train_losses'][-1]:.4f}")
    print(f"Final validation loss: {history['val_losses'][-1]:.4f}")
    print(f"Final validation accuracy: {history['val_accuracies'][-1]:.4f}")
    print(f"Best validation accuracy: {max(history['val_accuracies']):.4f}")
    
    print("\n" + "=" * 60)
    print("✅ Modern implementation completed successfully!")
    print("🔗 This demonstrates the same functionality as the original")
    print("   but with modern, clean, and extensible code.")
    print("⚠️  For research and educational purposes only.")
    print("👨‍💻 Author: kryptologyst")
    print("🔗 GitHub: https://github.com/kryptologyst")
    print("=" * 60)


if __name__ == "__main__":
    main()
