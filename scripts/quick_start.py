#!/usr/bin/env python3
"""Quick start script for semi-supervised learning experiments."""

import argparse
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.models import SimpleCNN
from src.models.ssl_methods import PseudoLabeling
from src.data import CIFAR10DataModule
from src.train import BasicTrainer
from src.utils import set_seed, setup_logging, get_device


def main():
    """Quick start training script."""
    parser = argparse.ArgumentParser(description="Quick start SSL training")
    parser.add_argument("--labeled_samples", type=int, default=1000, help="Number of labeled samples")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--ssl_method", type=str, default="pseudo_labeling", 
                       choices=["pseudo_labeling", "consistency_regularization", "mixmatch", "fixmatch"],
                       help="SSL method to use")
    parser.add_argument("--output_dir", type=str, default="quick_start_output", help="Output directory")
    
    args = parser.parse_args()
    
    # Setup
    setup_logging("INFO")
    set_seed(42)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Starting quick start training with {args.ssl_method}")
    logging.info(f"Labeled samples: {args.labeled_samples}")
    logging.info(f"Epochs: {args.epochs}")
    
    # Setup device
    device = get_device()
    
    # Setup data
    data_module = CIFAR10DataModule(
        data_dir="data/raw",
        batch_size=args.batch_size,
        num_workers=2,
        download=True
    )
    data_module.prepare_data()
    data_module.setup("fit")
    
    # Create semi-supervised data loaders
    labeled_loader, unlabeled_loader, val_loader, test_loader = data_module.create_semi_supervised_loaders(
        labeled_samples=args.labeled_samples,
        seed=42
    )
    
    logging.info(f"Labeled samples: {len(labeled_loader.dataset)}")
    logging.info(f"Unlabeled samples: {len(unlabeled_loader.dataset)}")
    
    # Setup model
    model = SimpleCNN()
    
    # Setup SSL method
    if args.ssl_method == "pseudo_labeling":
        ssl_method = PseudoLabeling(confidence_threshold=0.95)
    elif args.ssl_method == "consistency_regularization":
        from src.models.ssl_methods import ConsistencyRegularization
        ssl_method = ConsistencyRegularization(consistency_weight=1.0)
    elif args.ssl_method == "mixmatch":
        from src.models.ssl_methods import MixMatch
        ssl_method = MixMatch(alpha=0.75, lambda_u=75.0)
    elif args.ssl_method == "fixmatch":
        from src.models.ssl_methods import FixMatch
        ssl_method = FixMatch(confidence_threshold=0.95, lambda_u=1.0)
    
    # Setup optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Setup trainer
    trainer = BasicTrainer(
        model=model,
        ssl_method=ssl_method,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        max_epochs=args.epochs
    )
    
    # Training
    logging.info("Starting training...")
    history = trainer.fit(
        labeled_loader=labeled_loader,
        unlabeled_loader=unlabeled_loader,
        val_loader=val_loader,
        save_dir=str(output_dir)
    )
    
    # Evaluation
    logging.info("Evaluating on test set...")
    test_metrics = trainer.evaluate(test_loader)
    
    # Save results
    results = {
        "history": history,
        "test_metrics": test_metrics,
        "config": vars(args)
    }
    
    torch.save(results, output_dir / "results.pt")
    
    # Log final results
    logging.info("=" * 50)
    logging.info("FINAL RESULTS")
    logging.info("=" * 50)
    logging.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    logging.info(f"Test F1 (macro): {test_metrics['f1_macro']:.4f}")
    logging.info(f"Test F1 (weighted): {test_metrics['f1_weighted']:.4f}")
    if 'roc_auc_macro' in test_metrics:
        logging.info(f"Test ROC AUC (macro): {test_metrics['roc_auc_macro']:.4f}")
    logging.info(f"Expected Calibration Error: {test_metrics['expected_calibration_error']:.4f}")
    logging.info("=" * 50)
    
    logging.info(f"Quick start completed. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
