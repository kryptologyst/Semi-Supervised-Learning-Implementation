"""Unit tests for semi-supervised learning components."""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch

from src.models import SimpleCNN, WideResNet
from src.models.ssl_methods import PseudoLabeling, ConsistencyRegularization, MixMatch, FixMatch
from src.data import CIFAR10DataModule
from src.metrics import MetricsCalculator
from src.utils import set_seed, get_device, count_parameters, calculate_rampup_weight


class TestModels:
    """Test model architectures."""
    
    def test_simple_cnn_initialization(self):
        """Test SimpleCNN initialization."""
        model = SimpleCNN(input_channels=3, num_classes=10, hidden_dim=128)
        
        assert isinstance(model, nn.Module)
        assert model.input_channels == 3
        assert model.num_classes == 10
        assert model.hidden_dim == 128
        
        # Test forward pass
        x = torch.randn(2, 3, 32, 32)
        output = model(x)
        assert output.shape == (2, 10)
    
    def test_wideresnet_initialization(self):
        """Test WideResNet initialization."""
        model = WideResNet(depth=28, widen_factor=2, num_classes=10)
        
        assert isinstance(model, nn.Module)
        assert model.depth == 28
        assert model.widen_factor == 2
        assert model.num_classes == 10
        
        # Test forward pass
        x = torch.randn(2, 3, 32, 32)
        output = model(x)
        assert output.shape == (2, 10)
    
    def test_model_parameter_count(self):
        """Test parameter counting."""
        model = SimpleCNN()
        param_count = count_parameters(model)
        assert param_count > 0
        assert isinstance(param_count, int)


class TestSSLMethods:
    """Test SSL method implementations."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.model = SimpleCNN()
        self.labeled_batch = (torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,)))
        self.unlabeled_batch = torch.randn(8, 3, 32, 32)
        self.epoch = 5
    
    def test_pseudo_labeling(self):
        """Test pseudo-labeling method."""
        ssl_method = PseudoLabeling(confidence_threshold=0.9)
        
        loss_dict = ssl_method.compute_loss(
            self.model, self.labeled_batch, self.unlabeled_batch, self.epoch
        )
        
        assert "total_loss" in loss_dict
        assert "supervised_loss" in loss_dict
        assert "unsupervised_loss" in loss_dict
        assert "rampup_weight" in loss_dict
        
        assert isinstance(loss_dict["total_loss"], torch.Tensor)
        assert loss_dict["total_loss"].requires_grad
    
    def test_consistency_regularization(self):
        """Test consistency regularization method."""
        ssl_method = ConsistencyRegularization(consistency_weight=1.0)
        
        loss_dict = ssl_method.compute_loss(
            self.model, self.labeled_batch, self.unlabeled_batch, self.epoch
        )
        
        assert "total_loss" in loss_dict
        assert "supervised_loss" in loss_dict
        assert "consistency_loss" in loss_dict
        assert "rampup_weight" in loss_dict
        
        assert isinstance(loss_dict["total_loss"], torch.Tensor)
        assert loss_dict["total_loss"].requires_grad
    
    def test_mixmatch(self):
        """Test MixMatch method."""
        ssl_method = MixMatch(alpha=0.75, lambda_u=75.0)
        
        loss_dict = ssl_method.compute_loss(
            self.model, self.labeled_batch, self.unlabeled_batch, self.epoch
        )
        
        assert "total_loss" in loss_dict
        assert "supervised_loss" in loss_dict
        assert "unsupervised_loss" in loss_dict
        assert "mixing_weight" in loss_dict
        
        assert isinstance(loss_dict["total_loss"], torch.Tensor)
        assert loss_dict["total_loss"].requires_grad
    
    def test_fixmatch(self):
        """Test FixMatch method."""
        ssl_method = FixMatch(confidence_threshold=0.95, lambda_u=1.0)
        
        loss_dict = ssl_method.compute_loss(
            self.model, self.labeled_batch, self.unlabeled_batch, self.epoch
        )
        
        assert "total_loss" in loss_dict
        assert "supervised_loss" in loss_dict
        assert "unsupervised_loss" in loss_dict
        assert "rampup_weight" in loss_dict
        
        assert isinstance(loss_dict["total_loss"], torch.Tensor)
        assert loss_dict["total_loss"].requires_grad


class TestDataModule:
    """Test data loading functionality."""
    
    def test_cifar10_data_module_initialization(self):
        """Test CIFAR10DataModule initialization."""
        data_module = CIFAR10DataModule(
            data_dir="test_data",
            batch_size=16,
            num_workers=0,  # Use 0 for testing
            download=False
        )
        
        assert data_module.batch_size == 16
        assert data_module.num_workers == 0
        assert data_module.data_dir == "test_data"
    
    def test_data_module_transforms(self):
        """Test data module transforms."""
        data_module = CIFAR10DataModule(augment=True, augmentation_strength="medium")
        
        assert data_module.augment is True
        assert data_module.augmentation_strength == "medium"
        assert data_module.augment_transform is not None
        assert data_module.base_transform is not None
    
    def test_class_names(self):
        """Test class names retrieval."""
        data_module = CIFAR10DataModule()
        class_names = data_module.get_class_names()
        
        assert len(class_names) == 10
        assert "airplane" in class_names
        assert "automobile" in class_names
        assert "bird" in class_names


class TestMetrics:
    """Test metrics calculation."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.metrics_calculator = MetricsCalculator(num_classes=10)
        self.predictions = np.array([0, 1, 2, 0, 1])
        self.targets = np.array([0, 1, 2, 1, 1])
        self.probabilities = np.random.rand(5, 10)
        # Normalize probabilities
        self.probabilities = self.probabilities / self.probabilities.sum(axis=1, keepdims=True)
    
    def test_classification_metrics(self):
        """Test classification metrics calculation."""
        metrics = self.metrics_calculator.compute_classification_metrics(
            self.predictions, self.targets, self.probabilities
        )
        
        assert "accuracy" in metrics
        assert "f1_macro" in metrics
        assert "f1_weighted" in metrics
        assert "precision_macro" in metrics
        assert "recall_macro" in metrics
        assert "confusion_matrix" in metrics
        
        assert isinstance(metrics["accuracy"], float)
        assert 0 <= metrics["accuracy"] <= 1
        assert metrics["confusion_matrix"].shape == (10, 10)
    
    def test_ssl_metrics(self):
        """Test SSL-specific metrics."""
        labeled_predictions = np.array([0, 1, 2])
        labeled_targets = np.array([0, 1, 2])
        unlabeled_predictions = np.array([0, 1, 2, 0])
        pseudo_labels = np.array([0, 1, 2])
        confidence_scores = np.array([0.9, 0.8, 0.95])
        
        metrics = self.metrics_calculator.compute_ssl_metrics(
            labeled_predictions, labeled_targets, unlabeled_predictions,
            pseudo_labels, confidence_scores
        )
        
        assert "labeled" in metrics
        assert "unlabeled_samples" in metrics
        assert "pseudo_labeled_samples" in metrics
        assert "pseudo_labeling_rate" in metrics
        assert "mean_confidence" in metrics
    
    def test_efficiency_metrics(self):
        """Test efficiency metrics calculation."""
        metrics = self.metrics_calculator.compute_efficiency_metrics(
            model_size_mb=10.5,
            inference_time_ms=5.2,
            memory_usage_mb=256.0,
            flops=1000000
        )
        
        assert "model_size_mb" in metrics
        assert "inference_time_ms" in metrics
        assert "memory_usage_mb" in metrics
        assert "throughput_samples_per_sec" in metrics
        assert "flops" in metrics
        
        assert metrics["model_size_mb"] == 10.5
        assert metrics["throughput_samples_per_sec"] > 0


class TestUtils:
    """Test utility functions."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        
        # Test that seeds are set (can't directly test, but ensure no errors)
        assert True
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ["cpu", "cuda", "mps"]
    
    def test_calculate_rampup_weight(self):
        """Test rampup weight calculation."""
        # Test sigmoid rampup
        weight = calculate_rampup_weight(5, 10, "sigmoid")
        assert 0 <= weight <= 1
        
        # Test linear rampup
        weight = calculate_rampup_weight(5, 10, "linear")
        assert weight == 0.5
        
        # Test completed rampup
        weight = calculate_rampup_weight(15, 10, "sigmoid")
        assert weight == 1.0
        
        # Test invalid rampup type
        with pytest.raises(ValueError):
            calculate_rampup_weight(5, 10, "invalid")


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_ssl_training(self):
        """Test end-to-end SSL training (simplified)."""
        # Create model
        model = SimpleCNN()
        
        # Create SSL method
        ssl_method = PseudoLabeling()
        
        # Create dummy data
        labeled_batch = (torch.randn(2, 3, 32, 32), torch.randint(0, 10, (2,)))
        unlabeled_batch = torch.randn(4, 3, 32, 32)
        
        # Test loss computation
        loss_dict = ssl_method.compute_loss(model, labeled_batch, unlabeled_batch, 0)
        
        assert "total_loss" in loss_dict
        assert loss_dict["total_loss"].requires_grad
        
        # Test backward pass
        loss_dict["total_loss"].backward()
        
        # Check that gradients are computed
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
                break
    
    def test_metrics_integration(self):
        """Test metrics integration with model outputs."""
        model = SimpleCNN()
        metrics_calculator = MetricsCalculator()
        
        # Create dummy predictions
        predictions = np.array([0, 1, 2, 0, 1])
        targets = np.array([0, 1, 2, 1, 1])
        probabilities = np.random.rand(5, 10)
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
        
        # Compute metrics
        metrics = metrics_calculator.compute_classification_metrics(
            predictions, targets, probabilities
        )
        
        # Verify metrics are reasonable
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["f1_macro"] <= 1
        assert metrics["confusion_matrix"].sum() == len(predictions)


if __name__ == "__main__":
    pytest.main([__file__])
