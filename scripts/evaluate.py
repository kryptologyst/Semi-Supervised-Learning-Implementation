#!/usr/bin/env python3
"""Comprehensive evaluation script for semi-supervised learning experiments."""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd

from src.models import SimpleCNN, WideResNet
from src.models.ssl_methods import PseudoLabeling, ConsistencyRegularization, MixMatch, FixMatch
from src.data import CIFAR10DataModule
from src.metrics import MetricsCalculator, compute_ssl_leaderboard
from src.utils import set_seed, setup_logging, get_device, load_checkpoint


def get_ssl_method(method_name: str, **kwargs) -> Any:
    """Get SSL method by name."""
    methods = {
        "pseudo_labeling": PseudoLabeling,
        "consistency_regularization": ConsistencyRegularization,
        "mixmatch": MixMatch,
        "fixmatch": FixMatch,
    }
    
    if method_name not in methods:
        raise ValueError(f"Unknown SSL method: {method_name}")
    
    return methods[method_name](**kwargs)


def get_model(model_name: str, **kwargs) -> torch.nn.Module:
    """Get model by name."""
    models = {
        "simple_cnn": SimpleCNN,
        "wideresnet": WideResNet,
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")
    
    return models[model_name](**kwargs)


def extract_features(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, device: torch.device) -> np.ndarray:
    """Extract features from the model."""
    model.eval()
    features = []
    
    with torch.no_grad():
        for batch in data_loader:
            inputs, _ = batch
            inputs = inputs.to(device)
            
            if hasattr(model, 'get_features'):
                feat = model.get_features(inputs)
            else:
                # Use the last layer before classification
                feat = model(inputs)
                feat = feat[:, :-1]  # Remove last classification layer
            
            features.append(feat.cpu().numpy())
    
    return np.vstack(features)


def plot_training_curves(history: Dict[str, List[float]], save_path: str) -> None:
    """Plot training curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    epochs = range(1, len(history['train_losses']) + 1)
    
    # Training and validation loss
    axes[0].plot(epochs, history['train_losses'], label='Train Loss', color='blue')
    axes[0].plot(epochs, history['val_losses'], label='Val Loss', color='red')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Validation accuracy
    axes[1].plot(epochs, history['val_accuracies'], label='Val Accuracy', color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    # Learning rate (if available)
    if 'learning_rates' in history:
        axes[2].plot(epochs, history['learning_rates'], label='Learning Rate', color='orange')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_title('Learning Rate Schedule')
        axes[2].legend()
        axes[2].grid(True)
    else:
        axes[2].text(0.5, 0.5, 'Learning Rate\nNot Available', 
                    ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_title('Learning Rate Schedule')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(confusion_matrix: np.ndarray, class_names: List[str], save_path: str) -> None:
    """Plot confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_feature_visualization(features: np.ndarray, labels: np.ndarray, class_names: List[str], save_path: str) -> None:
    """Plot feature visualization using t-SNE."""
    # Reduce dimensionality using PCA first
    pca = PCA(n_components=50)
    features_pca = pca.fit_transform(features)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_tsne = tsne.fit_transform(features_pca)
    
    # Plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter)
    plt.title('Feature Visualization (t-SNE)')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    
    # Add class labels
    for i, class_name in enumerate(class_names):
        class_points = features_tsne[labels == i]
        if len(class_points) > 0:
            plt.annotate(class_name, 
                        (class_points[:, 0].mean(), class_points[:, 1].mean()),
                        ha='center', va='center', fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_calibration_curve(probabilities: np.ndarray, predictions: np.ndarray, targets: np.ndarray, save_path: str) -> None:
    """Plot calibration curve."""
    from sklearn.calibration import calibration_curve
    
    # Get confidence scores
    confidence_scores = np.max(probabilities, axis=1)
    
    # Compute calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        targets, confidence_scores, n_bins=10
    )
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, 'o-', label='Model')
    plt.plot([0, 1], [0, 1], '--', label='Perfect Calibration')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_results_table(results: Dict[str, Any], save_path: str) -> None:
    """Create a results table."""
    # Extract key metrics
    metrics = results['test_metrics']
    
    # Create DataFrame
    data = {
        'Metric': [
            'Accuracy',
            'F1 (Macro)',
            'F1 (Weighted)',
            'Precision (Macro)',
            'Recall (Macro)',
            'ROC AUC (Macro)',
            'PR AUC (Macro)',
            'Expected Calibration Error'
        ],
        'Value': [
            f"{metrics.get('accuracy', 0):.4f}",
            f"{metrics.get('f1_macro', 0):.4f}",
            f"{metrics.get('f1_weighted', 0):.4f}",
            f"{metrics.get('precision_macro', 0):.4f}",
            f"{metrics.get('recall_macro', 0):.4f}",
            f"{metrics.get('roc_auc_macro', 0):.4f}",
            f"{metrics.get('pr_auc_macro', 0):.4f}",
            f"{metrics.get('expected_calibration_error', 0):.4f}"
        ]
    }
    
    df = pd.DataFrame(data)
    
    # Save as CSV
    df.to_csv(save_path, index=False)
    
    # Also create a markdown table
    markdown_path = save_path.replace('.csv', '.md')
    with open(markdown_path, 'w') as f:
        f.write("# Results Summary\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n")


def evaluate_experiment(experiment_dir: str, output_dir: str) -> None:
    """Evaluate a single experiment."""
    experiment_path = Path(experiment_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load results
    results_file = experiment_path / "results.pt"
    if not results_file.exists():
        logging.error(f"Results file not found: {results_file}")
        return
    
    results = torch.load(results_file, map_location='cpu')
    config = results['config']
    
    logging.info(f"Evaluating experiment: {experiment_path.name}")
    
    # Setup device
    device = get_device()
    
    # Setup data
    data_module = CIFAR10DataModule(**config['data'])
    data_module.prepare_data()
    data_module.setup("test")
    
    # Create test loader
    _, _, _, test_loader = data_module.create_semi_supervised_loaders(
        labeled_samples=config['ssl']['labeled_samples'],
        seed=config['seed']
    )
    
    # Setup model
    model = get_model(**config['model'])
    model.to(device)
    
    # Load best model
    best_model_path = experiment_path / "best_model.pt"
    if best_model_path.exists():
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info("Loaded best model checkpoint")
    else:
        logging.warning("Best model checkpoint not found, using current model")
    
    # Extract features and predictions
    logging.info("Extracting features...")
    features = extract_features(model, test_loader, device)
    
    # Get predictions
    model.eval()
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in test_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_probabilities = np.array(all_probabilities)
    
    # Get class names
    class_names = data_module.get_class_names()
    
    # Create visualizations
    logging.info("Creating visualizations...")
    
    # Training curves
    if 'history' in results:
        plot_training_curves(results['history'], str(output_path / "training_curves.png"))
    
    # Confusion matrix
    confusion_matrix = results['test_metrics']['confusion_matrix']
    plot_confusion_matrix(confusion_matrix, class_names, str(output_path / "confusion_matrix.png"))
    
    # Feature visualization
    plot_feature_visualization(features, all_targets, class_names, str(output_path / "feature_visualization.png"))
    
    # Calibration curve
    plot_calibration_curve(all_probabilities, all_predictions, all_targets, str(output_path / "calibration_curve.png"))
    
    # Results table
    create_results_table(results, str(output_path / "results_summary.csv"))
    
    logging.info(f"Evaluation completed. Results saved to {output_path}")


def compare_experiments(experiment_dirs: List[str], output_dir: str) -> None:
    """Compare multiple experiments."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    # Load all results
    for exp_dir in experiment_dirs:
        exp_path = Path(exp_dir)
        results_file = exp_path / "results.pt"
        
        if results_file.exists():
            results = torch.load(results_file, map_location='cpu')
            all_results[exp_path.name] = results['test_metrics']
        else:
            logging.warning(f"Results file not found: {results_file}")
    
    if not all_results:
        logging.error("No valid results found for comparison")
        return
    
    # Create comparison table
    comparison_data = []
    for exp_name, metrics in all_results.items():
        row = {'Experiment': exp_name}
        row.update({
            'Accuracy': f"{metrics.get('accuracy', 0):.4f}",
            'F1 (Macro)': f"{metrics.get('f1_macro', 0):.4f}",
            'F1 (Weighted)': f"{metrics.get('f1_weighted', 0):.4f}",
            'ROC AUC (Macro)': f"{metrics.get('roc_auc_macro', 0):.4f}",
            'ECE': f"{metrics.get('expected_calibration_error', 0):.4f}"
        })
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    
    # Save comparison table
    df.to_csv(output_path / "experiment_comparison.csv", index=False)
    
    # Create markdown table
    with open(output_path / "experiment_comparison.md", 'w') as f:
        f.write("# Experiment Comparison\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n")
    
    # Create leaderboard
    leaderboard = compute_ssl_leaderboard(all_results)
    
    # Save leaderboard
    torch.save(leaderboard, output_path / "leaderboard.pt")
    
    logging.info(f"Comparison completed. Results saved to {output_path}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate semi-supervised learning experiments")
    parser.add_argument("--experiment_dir", type=str, help="Path to experiment directory")
    parser.add_argument("--experiment_dirs", nargs="+", help="Paths to multiple experiment directories")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Output directory")
    parser.add_argument("--compare", action="store_true", help="Compare multiple experiments")
    
    args = parser.parse_args()
    
    setup_logging("INFO")
    
    if args.compare and args.experiment_dirs:
        compare_experiments(args.experiment_dirs, args.output_dir)
    elif args.experiment_dir:
        evaluate_experiment(args.experiment_dir, args.output_dir)
    else:
        logging.error("Please provide either --experiment_dir or --experiment_dirs with --compare")


if __name__ == "__main__":
    main()
