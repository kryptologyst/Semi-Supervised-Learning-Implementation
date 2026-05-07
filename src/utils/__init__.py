"""Utility functions for semi-supervised learning project."""

import random
import logging
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device (CUDA, MPS, or CPU).
    
    Returns:
        torch.device: The best available device.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("Using MPS device (Apple Silicon)")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU device")
    
    return device


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model.
        
    Returns:
        int: Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size(model: nn.Module) -> Dict[str, Any]:
    """Get model size information.
    
    Args:
        model: PyTorch model.
        
    Returns:
        Dict containing model size information.
    """
    param_count = count_parameters(model)
    
    # Estimate model size in MB
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    
    return {
        "parameters": param_count,
        "size_mb": size_all_mb,
        "param_size_mb": param_size / 1024**2,
        "buffer_size_mb": buffer_size / 1024**2,
    }


def create_semi_supervised_split(
    dataset_size: int,
    labeled_samples: int,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Create labeled and unlabeled splits for semi-supervised learning.
    
    Args:
        dataset_size: Total size of the dataset.
        labeled_samples: Number of labeled samples.
        seed: Random seed for reproducibility.
        
    Returns:
        Tuple of (labeled_indices, unlabeled_indices).
    """
    if seed is not None:
        np.random.seed(seed)
    
    all_indices = np.arange(dataset_size)
    labeled_indices = np.random.choice(
        all_indices, 
        size=min(labeled_samples, dataset_size), 
        replace=False
    )
    unlabeled_indices = np.setdiff1d(all_indices, labeled_indices)
    
    return labeled_indices, unlabeled_indices


def calculate_rampup_weight(epoch: int, rampup_epochs: int, rampup_type: str = "sigmoid") -> float:
    """Calculate rampup weight for semi-supervised learning.
    
    Args:
        epoch: Current epoch.
        rampup_epochs: Number of epochs for rampup.
        rampup_type: Type of rampup ('sigmoid' or 'linear').
        
    Returns:
        float: Rampup weight between 0 and 1.
    """
    if epoch >= rampup_epochs:
        return 1.0
    
    if rampup_type == "sigmoid":
        # Sigmoid rampup
        return float(np.exp(-5.0 * (1.0 - epoch / rampup_epochs) ** 2))
    elif rampup_type == "linear":
        # Linear rampup
        return epoch / rampup_epochs
    else:
        raise ValueError(f"Unknown rampup type: {rampup_type}")


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    metrics: Dict[str, float],
    filepath: str
) -> None:
    """Save model checkpoint.
    
    Args:
        model: PyTorch model.
        optimizer: Optimizer.
        epoch: Current epoch.
        loss: Current loss.
        metrics: Dictionary of metrics.
        filepath: Path to save checkpoint.
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "metrics": metrics,
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    filepath: str
) -> Dict[str, Any]:
    """Load model checkpoint.
    
    Args:
        model: PyTorch model.
        optimizer: Optimizer.
        filepath: Path to checkpoint file.
        
    Returns:
        Dict containing checkpoint information.
    """
    checkpoint = torch.load(filepath, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    return {
        "epoch": checkpoint["epoch"],
        "loss": checkpoint["loss"],
        "metrics": checkpoint["metrics"],
    }


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration.
    
    Args:
        level: Logging level.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
