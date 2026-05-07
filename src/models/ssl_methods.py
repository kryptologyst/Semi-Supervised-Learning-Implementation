"""Semi-supervised learning methods."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..utils import calculate_rampup_weight


class SSLMethod(ABC):
    """Abstract base class for semi-supervised learning methods."""
    
    def __init__(self, **kwargs):
        """Initialize SSL method."""
        self.kwargs = kwargs
    
    @abstractmethod
    def compute_loss(
        self,
        model: nn.Module,
        labeled_batch: Tuple[torch.Tensor, torch.Tensor],
        unlabeled_batch: torch.Tensor,
        epoch: int,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Compute SSL loss.
        
        Args:
            model: Neural network model.
            labeled_batch: Tuple of (inputs, targets) for labeled data.
            unlabeled_batch: Unlabeled inputs.
            epoch: Current epoch.
            
        Returns:
            Dictionary containing loss components.
        """
        pass


class PseudoLabeling(SSLMethod):
    """Pseudo-labeling semi-supervised learning method."""
    
    def __init__(
        self,
        confidence_threshold: float = 0.95,
        rampup_epochs: int = 10,
        rampup_type: str = "sigmoid",
        ema_decay: float = 0.999,
        **kwargs
    ):
        """Initialize pseudo-labeling method.
        
        Args:
            confidence_threshold: Minimum confidence for pseudo-labels.
            rampup_epochs: Number of epochs for rampup.
            rampup_type: Type of rampup ('sigmoid' or 'linear').
            ema_decay: EMA decay for teacher model.
        """
        super().__init__(**kwargs)
        self.confidence_threshold = confidence_threshold
        self.rampup_epochs = rampup_epochs
        self.rampup_type = rampup_type
        self.ema_decay = ema_decay
        
        logging.info(f"Initialized PseudoLabeling with threshold={confidence_threshold}, "
                    f"rampup_epochs={rampup_epochs}")
    
    def compute_loss(
        self,
        model: nn.Module,
        labeled_batch: Tuple[torch.Tensor, torch.Tensor],
        unlabeled_batch: torch.Tensor,
        epoch: int,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Compute pseudo-labeling loss."""
        labeled_inputs, labeled_targets = labeled_batch
        
        # Supervised loss on labeled data
        labeled_outputs = model(labeled_inputs)
        supervised_loss = F.cross_entropy(labeled_outputs, labeled_targets)
        
        # Pseudo-labeling on unlabeled data
        unlabeled_outputs = model(unlabeled_batch)
        pseudo_labels = torch.argmax(unlabeled_outputs, dim=1)
        confidence = torch.max(F.softmax(unlabeled_outputs, dim=1), dim=1)[0]
        
        # Mask for high-confidence predictions
        mask = confidence > self.confidence_threshold
        
        if mask.sum() > 0:
            # Compute unsupervised loss only on high-confidence samples
            unsupervised_loss = F.cross_entropy(
                unlabeled_outputs[mask], 
                pseudo_labels[mask], 
                reduction='mean'
            )
            
            # Calculate rampup weight
            rampup_weight = calculate_rampup_weight(epoch, self.rampup_epochs, self.rampup_type)
            
            # Total loss
            total_loss = supervised_loss + rampup_weight * unsupervised_loss
            
            return {
                "total_loss": total_loss,
                "supervised_loss": supervised_loss,
                "unsupervised_loss": unsupervised_loss,
                "rampup_weight": torch.tensor(rampup_weight),
                "pseudo_labeled_samples": torch.tensor(mask.sum().item()),
                "total_unlabeled_samples": torch.tensor(len(unlabeled_batch))
            }
        else:
            # No high-confidence predictions
            return {
                "total_loss": supervised_loss,
                "supervised_loss": supervised_loss,
                "unsupervised_loss": torch.tensor(0.0),
                "rampup_weight": torch.tensor(0.0),
                "pseudo_labeled_samples": torch.tensor(0),
                "total_unlabeled_samples": torch.tensor(len(unlabeled_batch))
            }


class ConsistencyRegularization(SSLMethod):
    """Consistency regularization semi-supervised learning method."""
    
    def __init__(
        self,
        consistency_weight: float = 1.0,
        rampup_epochs: int = 10,
        rampup_type: str = "sigmoid",
        noise_std: float = 0.1,
        **kwargs
    ):
        """Initialize consistency regularization method.
        
        Args:
            consistency_weight: Weight for consistency loss.
            rampup_epochs: Number of epochs for rampup.
            rampup_type: Type of rampup.
            noise_std: Standard deviation of noise for augmentation.
        """
        super().__init__(**kwargs)
        self.consistency_weight = consistency_weight
        self.rampup_epochs = rampup_epochs
        self.rampup_type = rampup_type
        self.noise_std = noise_std
        
        logging.info(f"Initialized ConsistencyRegularization with weight={consistency_weight}, "
                    f"rampup_epochs={rampup_epochs}")
    
    def compute_loss(
        self,
        model: nn.Module,
        labeled_batch: Tuple[torch.Tensor, torch.Tensor],
        unlabeled_batch: torch.Tensor,
        epoch: int,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Compute consistency regularization loss."""
        labeled_inputs, labeled_targets = labeled_batch
        
        # Supervised loss on labeled data
        labeled_outputs = model(labeled_inputs)
        supervised_loss = F.cross_entropy(labeled_outputs, labeled_targets)
        
        # Consistency loss on unlabeled data
        # First forward pass
        unlabeled_outputs1 = model(unlabeled_batch)
        
        # Add noise and second forward pass
        noise = torch.randn_like(unlabeled_batch) * self.noise_std
        unlabeled_outputs2 = model(unlabeled_batch + noise)
        
        # Consistency loss (MSE between predictions)
        consistency_loss = F.mse_loss(
            F.softmax(unlabeled_outputs1, dim=1),
            F.softmax(unlabeled_outputs2, dim=1)
        )
        
        # Calculate rampup weight
        rampup_weight = calculate_rampup_weight(epoch, self.rampup_epochs, self.rampup_type)
        
        # Total loss
        total_loss = supervised_loss + self.consistency_weight * rampup_weight * consistency_loss
        
        return {
            "total_loss": total_loss,
            "supervised_loss": supervised_loss,
            "consistency_loss": consistency_loss,
            "rampup_weight": torch.tensor(rampup_weight),
            "unlabeled_samples": torch.tensor(len(unlabeled_batch))
        }


class MixMatch(SSLMethod):
    """MixMatch semi-supervised learning method."""
    
    def __init__(
        self,
        alpha: float = 0.75,
        lambda_u: float = 75.0,
        rampup_epochs: int = 10,
        rampup_type: str = "sigmoid",
        **kwargs
    ):
        """Initialize MixMatch method.
        
        Args:
            alpha: Beta distribution parameter for mixing.
            lambda_u: Weight for unlabeled loss.
            rampup_epochs: Number of epochs for rampup.
            rampup_type: Type of rampup.
        """
        super().__init__(**kwargs)
        self.alpha = alpha
        self.lambda_u = lambda_u
        self.rampup_epochs = rampup_epochs
        self.rampup_type = rampup_type
        
        logging.info(f"Initialized MixMatch with alpha={alpha}, lambda_u={lambda_u}, "
                    f"rampup_epochs={rampup_epochs}")
    
    def compute_loss(
        self,
        model: nn.Module,
        labeled_batch: Tuple[torch.Tensor, torch.Tensor],
        unlabeled_batch: torch.Tensor,
        epoch: int,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Compute MixMatch loss."""
        labeled_inputs, labeled_targets = labeled_batch
        
        # Generate pseudo-labels for unlabeled data
        with torch.no_grad():
            unlabeled_outputs = model(unlabeled_batch)
            pseudo_labels = F.softmax(unlabeled_outputs, dim=1)
        
        # Mix labeled and unlabeled data
        batch_size = labeled_inputs.size(0)
        unlabeled_size = unlabeled_batch.size(0)
        
        # Create mixed inputs
        mixed_inputs = torch.cat([labeled_inputs, unlabeled_batch], dim=0)
        
        # Generate mixing weights
        lam = np.random.beta(self.alpha, self.alpha)
        lam = max(lam, 1 - lam)
        
        # Random permutation for mixing
        index = torch.randperm(mixed_inputs.size(0))
        mixed_inputs = lam * mixed_inputs + (1 - lam) * mixed_inputs[index]
        
        # Forward pass on mixed inputs
        mixed_outputs = model(mixed_inputs)
        
        # Split outputs
        labeled_outputs = mixed_outputs[:batch_size]
        unlabeled_outputs = mixed_outputs[batch_size:]
        
        # Supervised loss
        supervised_loss = F.cross_entropy(labeled_outputs, labeled_targets)
        
        # Unsupervised loss (consistency with mixed pseudo-labels)
        mixed_pseudo_labels = lam * pseudo_labels + (1 - lam) * pseudo_labels[index]
        unsupervised_loss = F.mse_loss(
            F.softmax(unlabeled_outputs, dim=1),
            mixed_pseudo_labels
        )
        
        # Calculate rampup weight
        rampup_weight = calculate_rampup_weight(epoch, self.rampup_epochs, self.rampup_type)
        
        # Total loss
        total_loss = supervised_loss + self.lambda_u * rampup_weight * unsupervised_loss
        
        return {
            "total_loss": total_loss,
            "supervised_loss": supervised_loss,
            "unsupervised_loss": unsupervised_loss,
            "rampup_weight": torch.tensor(rampup_weight),
            "mixing_weight": torch.tensor(lam),
            "unlabeled_samples": torch.tensor(unlabeled_size)
        }


class FixMatch(SSLMethod):
    """FixMatch semi-supervised learning method."""
    
    def __init__(
        self,
        confidence_threshold: float = 0.95,
        lambda_u: float = 1.0,
        rampup_epochs: int = 10,
        rampup_type: str = "sigmoid",
        **kwargs
    ):
        """Initialize FixMatch method.
        
        Args:
            confidence_threshold: Minimum confidence for pseudo-labels.
            lambda_u: Weight for unlabeled loss.
            rampup_epochs: Number of epochs for rampup.
            rampup_type: Type of rampup.
        """
        super().__init__(**kwargs)
        self.confidence_threshold = confidence_threshold
        self.lambda_u = lambda_u
        self.rampup_epochs = rampup_epochs
        self.rampup_type = rampup_type
        
        logging.info(f"Initialized FixMatch with threshold={confidence_threshold}, "
                    f"lambda_u={lambda_u}, rampup_epochs={rampup_epochs}")
    
    def compute_loss(
        self,
        model: nn.Module,
        labeled_batch: Tuple[torch.Tensor, torch.Tensor],
        unlabeled_batch: torch.Tensor,
        epoch: int,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Compute FixMatch loss."""
        labeled_inputs, labeled_targets = labeled_batch
        
        # Supervised loss on labeled data
        labeled_outputs = model(labeled_inputs)
        supervised_loss = F.cross_entropy(labeled_outputs, labeled_targets)
        
        # Generate weak and strong augmentations for unlabeled data
        # For simplicity, we'll use the same data twice (in practice, you'd apply different augmentations)
        weak_outputs = model(unlabeled_batch)
        strong_outputs = model(unlabeled_batch)  # In practice, apply strong augmentation
        
        # Generate pseudo-labels from weak predictions
        with torch.no_grad():
            pseudo_labels = torch.argmax(weak_outputs, dim=1)
            confidence = torch.max(F.softmax(weak_outputs, dim=1), dim=1)[0]
        
        # Mask for high-confidence predictions
        mask = confidence > self.confidence_threshold
        
        if mask.sum() > 0:
            # Compute unsupervised loss only on high-confidence samples
            unsupervised_loss = F.cross_entropy(
                strong_outputs[mask],
                pseudo_labels[mask],
                reduction='mean'
            )
            
            # Calculate rampup weight
            rampup_weight = calculate_rampup_weight(epoch, self.rampup_epochs, self.rampup_type)
            
            # Total loss
            total_loss = supervised_loss + self.lambda_u * rampup_weight * unsupervised_loss
            
            return {
                "total_loss": total_loss,
                "supervised_loss": supervised_loss,
                "unsupervised_loss": unsupervised_loss,
                "rampup_weight": torch.tensor(rampup_weight),
                "pseudo_labeled_samples": torch.tensor(mask.sum().item()),
                "total_unlabeled_samples": torch.tensor(len(unlabeled_batch))
            }
        else:
            # No high-confidence predictions
            return {
                "total_loss": supervised_loss,
                "supervised_loss": supervised_loss,
                "unsupervised_loss": torch.tensor(0.0),
                "rampup_weight": torch.tensor(0.0),
                "pseudo_labeled_samples": torch.tensor(0),
                "total_unlabeled_samples": torch.tensor(len(unlabeled_batch))
            }
