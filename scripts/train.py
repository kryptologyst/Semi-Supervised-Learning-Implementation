#!/usr/bin/env python3
"""Main training script for semi-supervised learning experiments."""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import hydra
from omegaconf import DictConfig, OmegaConf

from src.models import SimpleCNN, WideResNet
from src.models.ssl_methods import PseudoLabeling, ConsistencyRegularization, MixMatch, FixMatch
from src.data import CIFAR10DataModule
from src.train import BasicTrainer
from src.utils import set_seed, setup_logging, get_device


def get_ssl_method(method_name: str, **kwargs) -> Any:
    """Get SSL method by name.
    
    Args:
        method_name: Name of the SSL method.
        **kwargs: Additional arguments for the method.
        
    Returns:
        SSL method instance.
    """
    methods = {
        "pseudo_labeling": PseudoLabeling,
        "consistency_regularization": ConsistencyRegularization,
        "mixmatch": MixMatch,
        "fixmatch": FixMatch,
    }
    
    if method_name not in methods:
        raise ValueError(f"Unknown SSL method: {method_name}")
    
    return methods[method_name](**kwargs)


def get_model(model_name: str, **kwargs) -> nn.Module:
    """Get model by name.
    
    Args:
        model_name: Name of the model.
        **kwargs: Additional arguments for the model.
        
    Returns:
        Model instance.
    """
    models = {
        "simple_cnn": SimpleCNN,
        "wideresnet": WideResNet,
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")
    
    return models[model_name](**kwargs)


def get_optimizer(model: nn.Module, optimizer_name: str, **kwargs) -> optim.Optimizer:
    """Get optimizer by name.
    
    Args:
        model: Model to optimize.
        optimizer_name: Name of the optimizer.
        **kwargs: Additional arguments for the optimizer.
        
    Returns:
        Optimizer instance.
    """
    optimizers = {
        "adam": optim.Adam,
        "sgd": optim.SGD,
        "adamw": optim.AdamW,
    }
    
    if optimizer_name not in optimizers:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    return optimizers[optimizer_name](model.parameters(), **kwargs)


def get_scheduler(optimizer: optim.Optimizer, scheduler_name: str, **kwargs) -> Any:
    """Get scheduler by name.
    
    Args:
        optimizer: Optimizer to schedule.
        scheduler_name: Name of the scheduler.
        **kwargs: Additional arguments for the scheduler.
        
    Returns:
        Scheduler instance.
    """
    schedulers = {
        "cosine": CosineAnnealingLR,
        "step": optim.lr_scheduler.StepLR,
        "exponential": optim.lr_scheduler.ExponentialLR,
    }
    
    if scheduler_name not in schedulers:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    return schedulers[scheduler_name](optimizer, **kwargs)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function."""
    # Setup
    setup_logging(cfg.logging.level)
    set_seed(cfg.seed)
    
    # Create output directory
    output_dir = Path("outputs") / cfg.experiment.name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)
    
    logging.info(f"Starting experiment: {cfg.experiment.name}")
    logging.info(f"Output directory: {output_dir}")
    
    # Setup device
    device = get_device()
    
    # Setup data
    data_module = CIFAR10DataModule(**cfg.data)
    data_module.prepare_data()
    data_module.setup("fit")
    
    # Create semi-supervised data loaders
    labeled_loader, unlabeled_loader, val_loader, test_loader = data_module.create_semi_supervised_loaders(
        labeled_samples=cfg.ssl.labeled_samples,
        seed=cfg.seed
    )
    
    logging.info(f"Labeled samples: {len(labeled_loader.dataset)}")
    logging.info(f"Unlabeled samples: {len(unlabeled_loader.dataset)}")
    logging.info(f"Validation samples: {len(val_loader.dataset)}")
    logging.info(f"Test samples: {len(test_loader.dataset)}")
    
    # Setup model
    model = get_model(**cfg.model)
    logging.info(f"Model: {cfg.model._target_.split('.')[-1]}")
    
    # Setup SSL method
    ssl_method = get_ssl_method(**cfg.ssl_method)
    logging.info(f"SSL Method: {cfg.ssl_method._target_.split('.')[-1]}")
    
    # Setup optimizer
    optimizer = get_optimizer(model, **cfg.trainer)
    logging.info(f"Optimizer: {cfg.trainer.optimizer}")
    
    # Setup scheduler
    scheduler = None
    if hasattr(cfg.trainer, 'scheduler') and cfg.trainer.scheduler:
        scheduler = get_scheduler(optimizer, cfg.trainer.scheduler, T_max=cfg.trainer.max_epochs)
        logging.info(f"Scheduler: {cfg.trainer.scheduler}")
    
    # Setup trainer
    trainer = BasicTrainer(
        model=model,
        ssl_method=ssl_method,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        **cfg.trainer
    )
    
    # Training
    logging.info("Starting training...")
    history = trainer.fit(
        labeled_loader=labeled_loader,
        unlabeled_loader=unlabeled_loader,
        val_loader=val_loader,
        save_dir=str(output_dir),
        save_every_n_epochs=cfg.logging.checkpoint_every_n_epochs
    )
    
    # Evaluation
    logging.info("Evaluating on test set...")
    test_metrics = trainer.evaluate(test_loader)
    
    # Save results
    results = {
        "history": history,
        "test_metrics": test_metrics,
        "config": OmegaConf.to_container(cfg, resolve=True)
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
    
    logging.info(f"Experiment completed. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
