"""Evaluation metrics for semi-supervised learning."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)


class MetricsCalculator:
    """Calculator for various evaluation metrics."""
    
    def __init__(self, num_classes: int = 10, class_names: Optional[List[str]] = None):
        """Initialize metrics calculator.
        
        Args:
            num_classes: Number of classes.
            class_names: Names of classes.
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        
    def compute_classification_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        probabilities: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Compute classification metrics.
        
        Args:
            predictions: Predicted class labels.
            targets: True class labels.
            probabilities: Predicted class probabilities.
            
        Returns:
            Dictionary containing computed metrics.
        """
        metrics = {}
        
        # Basic classification metrics
        metrics["accuracy"] = accuracy_score(targets, predictions)
        metrics["f1_macro"] = f1_score(targets, predictions, average="macro")
        metrics["f1_weighted"] = f1_score(targets, predictions, average="weighted")
        metrics["precision_macro"] = precision_score(targets, predictions, average="macro", zero_division=0)
        metrics["recall_macro"] = recall_score(targets, predictions, average="macro", zero_division=0)
        
        # Per-class metrics
        metrics["f1_per_class"] = f1_score(targets, predictions, average=None)
        metrics["precision_per_class"] = precision_score(targets, predictions, average=None, zero_division=0)
        metrics["recall_per_class"] = recall_score(targets, predictions, average=None, zero_division=0)
        
        # Confusion matrix
        metrics["confusion_matrix"] = confusion_matrix(targets, predictions)
        
        # ROC AUC and PR AUC (if probabilities available)
        if probabilities is not None:
            try:
                if self.num_classes == 2:
                    # Binary classification
                    metrics["roc_auc"] = roc_auc_score(targets, probabilities[:, 1])
                    metrics["pr_auc"] = average_precision_score(targets, probabilities[:, 1])
                else:
                    # Multi-class classification
                    metrics["roc_auc_macro"] = roc_auc_score(
                        targets, probabilities, multi_class="ovr", average="macro"
                    )
                    metrics["roc_auc_weighted"] = roc_auc_score(
                        targets, probabilities, multi_class="ovr", average="weighted"
                    )
                    metrics["pr_auc_macro"] = average_precision_score(
                        targets, probabilities, average="macro"
                    )
            except Exception as e:
                logging.warning(f"Could not compute AUC metrics: {e}")
        
        return metrics
    
    def compute_ssl_metrics(
        self,
        labeled_predictions: np.ndarray,
        labeled_targets: np.ndarray,
        unlabeled_predictions: np.ndarray,
        pseudo_labels: Optional[np.ndarray] = None,
        confidence_scores: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Compute SSL-specific metrics.
        
        Args:
            labeled_predictions: Predictions on labeled data.
            labeled_targets: True labels for labeled data.
            unlabeled_predictions: Predictions on unlabeled data.
            pseudo_labels: Generated pseudo-labels.
            confidence_scores: Confidence scores for predictions.
            
        Returns:
            Dictionary containing SSL metrics.
        """
        metrics = {}
        
        # Labeled data performance
        labeled_metrics = self.compute_classification_metrics(
            labeled_predictions, labeled_targets
        )
        metrics["labeled"] = labeled_metrics
        
        # Unlabeled data statistics
        metrics["unlabeled_samples"] = len(unlabeled_predictions)
        
        if pseudo_labels is not None:
            # Pseudo-labeling statistics
            metrics["pseudo_labeled_samples"] = len(pseudo_labels)
            metrics["pseudo_labeling_rate"] = len(pseudo_labels) / len(unlabeled_predictions)
            
            # Consistency between predictions and pseudo-labels
            consistency = (unlabeled_predictions == pseudo_labels).mean()
            metrics["pseudo_label_consistency"] = consistency
        
        if confidence_scores is not None:
            # Confidence statistics
            metrics["mean_confidence"] = confidence_scores.mean()
            metrics["std_confidence"] = confidence_scores.std()
            metrics["min_confidence"] = confidence_scores.min()
            metrics["max_confidence"] = confidence_scores.max()
            
            # High confidence predictions
            high_conf_mask = confidence_scores > 0.9
            metrics["high_confidence_rate"] = high_conf_mask.mean()
            metrics["high_confidence_samples"] = high_conf_mask.sum()
        
        return metrics
    
    def compute_efficiency_metrics(
        self,
        model_size_mb: float,
        inference_time_ms: float,
        memory_usage_mb: float,
        flops: Optional[int] = None
    ) -> Dict[str, Any]:
        """Compute model efficiency metrics.
        
        Args:
            model_size_mb: Model size in MB.
            inference_time_ms: Inference time per sample in ms.
            memory_usage_mb: Memory usage in MB.
            flops: Floating point operations.
            
        Returns:
            Dictionary containing efficiency metrics.
        """
        metrics = {
            "model_size_mb": model_size_mb,
            "inference_time_ms": inference_time_ms,
            "memory_usage_mb": memory_usage_mb,
            "throughput_samples_per_sec": 1000.0 / inference_time_ms if inference_time_ms > 0 else 0,
        }
        
        if flops is not None:
            metrics["flops"] = flops
            metrics["flops_per_sample"] = flops
        
        return metrics
    
    def create_classification_report(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        output_dict: bool = False
    ) -> str:
        """Create detailed classification report.
        
        Args:
            predictions: Predicted class labels.
            targets: True class labels.
            output_dict: Whether to return as dictionary.
            
        Returns:
            Classification report.
        """
        return classification_report(
            targets, predictions,
            target_names=self.class_names,
            output_dict=output_dict
        )
    
    def compute_confidence_calibration(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        probabilities: np.ndarray,
        num_bins: int = 10
    ) -> Dict[str, Any]:
        """Compute confidence calibration metrics.
        
        Args:
            predictions: Predicted class labels.
            targets: True class labels.
            probabilities: Predicted class probabilities.
            num_bins: Number of bins for calibration.
            
        Returns:
            Dictionary containing calibration metrics.
        """
        # Get confidence scores (max probability)
        confidence_scores = np.max(probabilities, axis=1)
        
        # Compute calibration error
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        accuracies = []
        confidences = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidence_scores > bin_lower) & (confidence_scores <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = (predictions[in_bin] == targets[in_bin]).mean()
                avg_confidence_in_bin = confidence_scores[in_bin].mean()
                
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
                accuracies.append(accuracy_in_bin)
                confidences.append(avg_confidence_in_bin)
        
        return {
            "expected_calibration_error": ece,
            "bin_accuracies": accuracies,
            "bin_confidences": confidences,
            "num_bins": num_bins
        }


def compute_ssl_leaderboard(
    results: Dict[str, Dict[str, Any]],
    metric_names: List[str] = None
) -> Dict[str, Any]:
    """Compute SSL leaderboard from multiple experiment results.
    
    Args:
        results: Dictionary of experiment results.
        metric_names: List of metrics to include in leaderboard.
        
    Returns:
        Dictionary containing leaderboard.
    """
    if metric_names is None:
        metric_names = [
            "accuracy", "f1_macro", "f1_weighted",
            "pseudo_labeling_rate", "pseudo_label_consistency"
        ]
    
    leaderboard = {}
    
    for metric in metric_names:
        metric_values = []
        experiment_names = []
        
        for exp_name, exp_results in results.items():
            if metric in exp_results:
                metric_values.append(exp_results[metric])
                experiment_names.append(exp_name)
        
        if metric_values:
            leaderboard[metric] = {
                "values": metric_values,
                "experiments": experiment_names,
                "best": max(metric_values) if metric != "expected_calibration_error" else min(metric_values),
                "worst": min(metric_values) if metric != "expected_calibration_error" else max(metric_values),
                "mean": np.mean(metric_values),
                "std": np.std(metric_values)
            }
    
    return leaderboard
