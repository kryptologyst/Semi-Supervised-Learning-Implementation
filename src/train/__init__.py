"""Training utilities for semi-supervised learning."""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.ssl_methods import SSLMethod
from ..metrics import MetricsCalculator
from ..utils import get_device, save_checkpoint, load_checkpoint


class BasicTrainer:
    """Basic trainer for semi-supervised learning."""
    
    def __init__(
        self,
        model: nn.Module,
        ssl_method: SSLMethod,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: Optional[torch.device] = None,
        max_epochs: int = 50,
        gradient_clip_val: float = 1.0,
        accumulate_grad_batches: int = 1,
        precision: int = 32,
        **kwargs
    ):
        """Initialize trainer.
        
        Args:
            model: Neural network model.
            ssl_method: SSL method to use.
            optimizer: Optimizer.
            scheduler: Learning rate scheduler.
            device: Device to use for training.
            max_epochs: Maximum number of epochs.
            gradient_clip_val: Gradient clipping value.
            accumulate_grad_batches: Number of batches to accumulate gradients.
            precision: Training precision (16 or 32).
        """
        self.model = model
        self.ssl_method = ssl_method
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or get_device()
        self.max_epochs = max_epochs
        self.gradient_clip_val = gradient_clip_val
        self.accumulate_grad_batches = accumulate_grad_batches
        self.precision = precision
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize metrics calculator
        self.metrics_calculator = MetricsCalculator()
        
        # Training state
        self.current_epoch = 0
        self.best_val_accuracy = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
        logging.info(f"Initialized trainer with device: {self.device}")
        logging.info(f"Model has {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def train_epoch(
        self,
        labeled_loader: DataLoader,
        unlabeled_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            labeled_loader: DataLoader for labeled data.
            unlabeled_loader: DataLoader for unlabeled data.
            epoch: Current epoch number.
            
        Returns:
            Dictionary containing training metrics.
        """
        self.model.train()
        
        total_loss = 0.0
        supervised_loss = 0.0
        unsupervised_loss = 0.0
        num_batches = 0
        
        # Create iterators
        labeled_iter = iter(labeled_loader)
        unlabeled_iter = iter(unlabeled_loader)
        
        # Progress bar
        pbar = tqdm(
            range(max(len(labeled_loader), len(unlabeled_loader))),
            desc=f"Epoch {epoch+1}/{self.max_epochs}",
            leave=False
        )
        
        for batch_idx in pbar:
            try:
                # Get batches
                labeled_batch = next(labeled_iter)
                unlabeled_batch = next(unlabeled_iter)
            except StopIteration:
                # Reset iterators if one is exhausted
                labeled_iter = iter(labeled_loader)
                unlabeled_iter = iter(unlabeled_loader)
                labeled_batch = next(labeled_iter)
                unlabeled_batch = next(unlabeled_iter)
            
            # Move to device
            labeled_batch = (labeled_batch[0].to(self.device), labeled_batch[1].to(self.device))
            unlabeled_batch = unlabeled_batch.to(self.device)
            
            # Compute SSL loss
            loss_dict = self.ssl_method.compute_loss(
                self.model, labeled_batch, unlabeled_batch, epoch
            )
            
            loss = loss_dict["total_loss"]
            
            # Normalize loss by accumulation steps
            loss = loss / self.accumulate_grad_batches
            
            # Backward pass
            loss.backward()
            
            # Accumulate gradients
            if (batch_idx + 1) % self.accumulate_grad_batches == 0:
                # Gradient clipping
                if self.gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip_val
                    )
                
                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # Update metrics
            total_loss += loss_dict["total_loss"].item()
            supervised_loss += loss_dict["supervised_loss"].item()
            if "unsupervised_loss" in loss_dict:
                unsupervised_loss += loss_dict["unsupervised_loss"].item()
            
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Sup": f"{loss_dict['supervised_loss'].item():.4f}",
                "Unsup": f"{loss_dict.get('unsupervised_loss', 0).item():.4f}"
            })
        
        # Average metrics
        avg_total_loss = total_loss / num_batches
        avg_supervised_loss = supervised_loss / num_batches
        avg_unsupervised_loss = unsupervised_loss / num_batches
        
        return {
            "total_loss": avg_total_loss,
            "supervised_loss": avg_supervised_loss,
            "unsupervised_loss": avg_unsupervised_loss,
        }
    
    def validate(
        self,
        val_loader: DataLoader
    ) -> Dict[str, float]:
        """Validate the model.
        
        Args:
            val_loader: DataLoader for validation data.
            
        Returns:
            Dictionary containing validation metrics.
        """
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", leave=False):
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = nn.CrossEntropyLoss()(outputs, targets)
                
                # Get predictions and probabilities
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                total_loss += loss.item()
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_probabilities = np.array(all_probabilities)
        
        # Compute metrics
        metrics = self.metrics_calculator.compute_classification_metrics(
            all_predictions, all_targets, all_probabilities
        )
        
        metrics["val_loss"] = total_loss / len(val_loader)
        
        return metrics
    
    def fit(
        self,
        labeled_loader: DataLoader,
        unlabeled_loader: DataLoader,
        val_loader: DataLoader,
        save_dir: Optional[str] = None,
        save_every_n_epochs: int = 10
    ) -> Dict[str, List[float]]:
        """Train the model.
        
        Args:
            labeled_loader: DataLoader for labeled data.
            unlabeled_loader: DataLoader for unlabeled data.
            val_loader: DataLoader for validation data.
            save_dir: Directory to save checkpoints.
            save_every_n_epochs: Save checkpoint every N epochs.
            
        Returns:
            Dictionary containing training history.
        """
        logging.info(f"Starting training for {self.max_epochs} epochs")
        
        for epoch in range(self.max_epochs):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch(labeled_loader, unlabeled_loader, epoch)
            
            # Validation
            val_metrics = self.validate(val_loader)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Store metrics
            self.train_losses.append(train_metrics["total_loss"])
            self.val_losses.append(val_metrics["val_loss"])
            self.val_accuracies.append(val_metrics["accuracy"])
            
            # Log metrics
            logging.info(
                f"Epoch {epoch+1}/{self.max_epochs}: "
                f"Train Loss: {train_metrics['total_loss']:.4f}, "
                f"Val Loss: {val_metrics['val_loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}"
            )
            
            # Save best model
            if val_metrics["accuracy"] > self.best_val_accuracy:
                self.best_val_accuracy = val_metrics["accuracy"]
                if save_dir:
                    best_path = os.path.join(save_dir, "best_model.pt")
                    save_checkpoint(
                        self.model, self.optimizer, epoch, 
                        val_metrics["val_loss"], val_metrics, best_path
                    )
            
            # Save checkpoint
            if save_dir and (epoch + 1) % save_every_n_epochs == 0:
                checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pt")
                save_checkpoint(
                    self.model, self.optimizer, epoch,
                    val_metrics["val_loss"], val_metrics, checkpoint_path
                )
        
        logging.info(f"Training completed. Best validation accuracy: {self.best_val_accuracy:.4f}")
        
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_accuracies": self.val_accuracies,
        }
    
    def evaluate(
        self,
        test_loader: DataLoader
    ) -> Dict[str, Any]:
        """Evaluate the model on test data.
        
        Args:
            test_loader: DataLoader for test data.
            
        Returns:
            Dictionary containing evaluation metrics.
        """
        logging.info("Evaluating model on test data")
        
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_probabilities = np.array(all_probabilities)
        
        # Compute comprehensive metrics
        metrics = self.metrics_calculator.compute_classification_metrics(
            all_predictions, all_targets, all_probabilities
        )
        
        # Add calibration metrics
        calibration_metrics = self.metrics_calculator.compute_confidence_calibration(
            all_predictions, all_targets, all_probabilities
        )
        metrics.update(calibration_metrics)
        
        # Create classification report
        report = self.metrics_calculator.create_classification_report(
            all_predictions, all_targets
        )
        metrics["classification_report"] = report
        
        logging.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
        logging.info(f"Test F1 (macro): {metrics['f1_macro']:.4f}")
        
        return metrics
