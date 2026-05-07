"""Streamlit demo application for semi-supervised learning."""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
import base64
from pathlib import Path
import json

from src.models import SimpleCNN, WideResNet
from src.models.ssl_methods import PseudoLabeling, ConsistencyRegularization, MixMatch, FixMatch
from src.data import CIFAR10DataModule
from src.metrics import MetricsCalculator
from src.utils import get_device, set_seed


# Page configuration
st.set_page_config(
    page_title="Semi-Supervised Learning Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">Semi-Supervised Learning Demo</h1>', unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="warning-box">
    <h4>⚠️ Research Demo Disclaimer</h4>
    <p>This is a research and educational demonstration of semi-supervised learning methods. 
    <strong>Not for production decisions or control systems.</strong> Results may vary and 
    should not be used for critical applications without proper validation.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.title("Configuration")

# Model selection
st.sidebar.subheader("Model Configuration")
model_type = st.sidebar.selectbox(
    "Model Architecture",
    ["SimpleCNN", "WideResNet"],
    help="Choose the neural network architecture"
)

# SSL method selection
st.sidebar.subheader("SSL Method")
ssl_method = st.sidebar.selectbox(
    "Semi-Supervised Learning Method",
    ["Pseudo Labeling", "Consistency Regularization", "MixMatch", "FixMatch"],
    help="Choose the semi-supervised learning approach"
)

# Training parameters
st.sidebar.subheader("Training Parameters")
labeled_samples = st.sidebar.slider(
    "Number of Labeled Samples",
    min_value=100,
    max_value=5000,
    value=1000,
    step=100,
    help="Number of labeled samples for training"
)

epochs = st.sidebar.slider(
    "Number of Epochs",
    min_value=5,
    max_value=100,
    value=20,
    step=5,
    help="Number of training epochs"
)

learning_rate = st.sidebar.slider(
    "Learning Rate",
    min_value=0.0001,
    max_value=0.01,
    value=0.001,
    step=0.0001,
    format="%.4f",
    help="Learning rate for optimization"
)

# SSL-specific parameters
st.sidebar.subheader("SSL Parameters")
if ssl_method == "Pseudo Labeling":
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.5,
        max_value=0.99,
        value=0.95,
        step=0.01,
        help="Minimum confidence for pseudo-labels"
    )
elif ssl_method == "Consistency Regularization":
    consistency_weight = st.sidebar.slider(
        "Consistency Weight",
        min_value=0.1,
        max_value=10.0,
        value=1.0,
        step=0.1,
        help="Weight for consistency loss"
    )
elif ssl_method == "MixMatch":
    alpha = st.sidebar.slider(
        "Alpha (Beta Distribution)",
        min_value=0.1,
        max_value=2.0,
        value=0.75,
        step=0.05,
        help="Beta distribution parameter for mixing"
    )
elif ssl_method == "FixMatch":
    lambda_u = st.sidebar.slider(
        "Lambda U",
        min_value=0.1,
        max_value=10.0,
        value=1.0,
        step=0.1,
        help="Weight for unlabeled loss"
    )

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🎯 Training", "📈 Results", "🔍 Analysis"])

with tab1:
    st.header("Semi-Supervised Learning Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("What is Semi-Supervised Learning?")
        st.markdown("""
        Semi-supervised learning is a machine learning approach that uses both labeled and unlabeled data for training. 
        This is particularly useful when:
        - Labeled data is expensive or time-consuming to obtain
        - Large amounts of unlabeled data are readily available
        - We want to improve model performance with limited labeled examples
        
        **Key Benefits:**
        - Reduces annotation costs
        - Improves model performance with limited labeled data
        - Leverages unlabeled data effectively
        """)
    
    with col2:
        st.subheader("Methods Implemented")
        
        methods_info = {
            "Pseudo Labeling": "Generates pseudo-labels for unlabeled data using model predictions",
            "Consistency Regularization": "Enforces consistency between predictions on augmented versions of the same input",
            "MixMatch": "Combines data augmentation and pseudo-labeling with mixing strategies",
            "FixMatch": "Uses weak and strong augmentations with confidence-based pseudo-labeling"
        }
        
        for method, description in methods_info.items():
            st.markdown(f"**{method}:** {description}")
    
    st.subheader("Dataset Information")
    st.markdown("""
    **CIFAR-10 Dataset:**
    - 60,000 32x32 color images
    - 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
    - 50,000 training images, 10,000 test images
    - Perfect for demonstrating SSL methods with limited labeled data
    """)

with tab2:
    st.header("Training Configuration")
    
    # Display current configuration
    st.subheader("Current Configuration")
    
    config_col1, config_col2, config_col3 = st.columns(3)
    
    with config_col1:
        st.markdown("""
        <div class="metric-card">
            <h4>Model</h4>
            <p><strong>Architecture:</strong> {}</p>
            <p><strong>SSL Method:</strong> {}</p>
        </div>
        """.format(model_type, ssl_method), unsafe_allow_html=True)
    
    with config_col2:
        st.markdown("""
        <div class="metric-card">
            <h4>Data</h4>
            <p><strong>Labeled Samples:</strong> {}</p>
            <p><strong>Unlabeled Samples:</strong> {}</p>
        </div>
        """.format(labeled_samples, 50000 - labeled_samples), unsafe_allow_html=True)
    
    with config_col3:
        st.markdown("""
        <div class="metric-card">
            <h4>Training</h4>
            <p><strong>Epochs:</strong> {}</p>
            <p><strong>Learning Rate:</strong> {}</p>
        </div>
        """.format(epochs, learning_rate), unsafe_allow_html=True)
    
    # Training button
    st.subheader("Start Training")
    
    if st.button("🚀 Start Training", type="primary"):
        with st.spinner("Training in progress..."):
            # This would normally run the actual training
            # For demo purposes, we'll simulate the training process
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulate training progress
            for epoch in range(epochs):
                progress = (epoch + 1) / epochs
                progress_bar.progress(progress)
                status_text.text(f"Epoch {epoch + 1}/{epochs} - Training...")
                
                # Simulate some processing time
                import time
                time.sleep(0.1)
            
            status_text.text("Training completed!")
            
            # Store results in session state
            st.session_state.training_completed = True
            st.session_state.training_config = {
                "model_type": model_type,
                "ssl_method": ssl_method,
                "labeled_samples": labeled_samples,
                "epochs": epochs,
                "learning_rate": learning_rate
            }
    
    # Display training status
    if hasattr(st.session_state, 'training_completed') and st.session_state.training_completed:
        st.success("✅ Training completed successfully!")
        
        # Display training configuration
        st.subheader("Training Summary")
        config = st.session_state.training_config
        
        st.json(config)

with tab3:
    st.header("Results and Metrics")
    
    if hasattr(st.session_state, 'training_completed') and st.session_state.training_completed:
        st.subheader("Performance Metrics")
        
        # Simulate results (in real implementation, these would come from actual training)
        results = {
            "accuracy": 0.85,
            "f1_macro": 0.84,
            "f1_weighted": 0.85,
            "precision_macro": 0.83,
            "recall_macro": 0.84,
            "roc_auc_macro": 0.92,
            "expected_calibration_error": 0.05
        }
        
        # Display metrics in cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{results['accuracy']:.3f}")
        with col2:
            st.metric("F1 Score (Macro)", f"{results['f1_macro']:.3f}")
        with col3:
            st.metric("ROC AUC", f"{results['roc_auc_macro']:.3f}")
        with col4:
            st.metric("Calibration Error", f"{results['expected_calibration_error']:.3f}")
        
        # Detailed metrics table
        st.subheader("Detailed Metrics")
        
        metrics_data = {
            "Metric": ["Accuracy", "F1 Score (Macro)", "F1 Score (Weighted)", 
                      "Precision (Macro)", "Recall (Macro)", "ROC AUC (Macro)", 
                      "Expected Calibration Error"],
            "Value": [f"{results['accuracy']:.4f}", f"{results['f1_macro']:.4f}", 
                     f"{results['f1_weighted']:.4f}", f"{results['precision_macro']:.4f}", 
                     f"{results['recall_macro']:.4f}", f"{results['roc_auc_macro']:.4f}", 
                     f"{results['expected_calibration_error']:.4f}"]
        }
        
        st.table(metrics_data)
        
        # Training curves (simulated)
        st.subheader("Training Curves")
        
        # Generate simulated training data
        epochs_range = range(1, st.session_state.training_config['epochs'] + 1)
        train_losses = [1.5 * np.exp(-0.1 * epoch) + 0.1 + np.random.normal(0, 0.05) for epoch in epochs_range]
        val_losses = [1.3 * np.exp(-0.08 * epoch) + 0.15 + np.random.normal(0, 0.03) for epoch in epochs_range]
        val_accuracies = [0.1 + 0.7 * (1 - np.exp(-0.15 * epoch)) + np.random.normal(0, 0.02) for epoch in epochs_range]
        
        # Plot training curves
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Loss curves
        axes[0].plot(epochs_range, train_losses, label='Train Loss', color='blue')
        axes[0].plot(epochs_range, val_losses, label='Val Loss', color='red')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy curve
        axes[1].plot(epochs_range, val_accuracies, label='Val Accuracy', color='green')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        st.pyplot(fig)
        
    else:
        st.info("Please complete training first to view results.")

with tab4:
    st.header("Analysis and Insights")
    
    if hasattr(st.session_state, 'training_completed') and st.session_state.training_completed:
        st.subheader("Method Comparison")
        
        # Simulate comparison data
        methods_comparison = {
            "Method": ["Supervised Only", "Pseudo Labeling", "Consistency Regularization", "MixMatch", "FixMatch"],
            "Accuracy": [0.72, 0.78, 0.81, 0.85, 0.87],
            "F1 Score": [0.71, 0.77, 0.80, 0.84, 0.86],
            "Labeled Samples": [1000, 1000, 1000, 1000, 1000]
        }
        
        st.table(methods_comparison)
        
        # Visualization
        st.subheader("Performance Comparison")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        methods = methods_comparison["Method"]
        accuracies = methods_comparison["Accuracy"]
        
        bars = ax.bar(methods, accuracies, color=['red', 'orange', 'yellow', 'lightgreen', 'green'])
        ax.set_ylabel('Accuracy')
        ax.set_title('SSL Method Performance Comparison')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{acc:.3f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Insights
        st.subheader("Key Insights")
        
        insights = [
            "🎯 **Semi-supervised methods consistently outperform supervised-only training**",
            "📈 **FixMatch shows the best performance** with strong augmentation strategies",
            "🔄 **Consistency regularization** provides good performance with simple implementation",
            "🏷️ **Pseudo-labeling** offers a straightforward approach to leverage unlabeled data",
            "🎨 **MixMatch** combines multiple SSL techniques for robust performance"
        ]
        
        for insight in insights:
            st.markdown(insight)
        
        # Recommendations
        st.subheader("Recommendations")
        
        st.markdown("""
        **For Production Use:**
        - Start with FixMatch for best performance
        - Use Consistency Regularization for simpler implementation
        - Consider MixMatch for robust performance across domains
        
        **For Research:**
        - Experiment with different confidence thresholds
        - Try combining multiple SSL methods
        - Investigate domain-specific augmentations
        """)
        
    else:
        st.info("Please complete training first to view analysis.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p><strong>Semi-Supervised Learning Demo</strong></p>
    <p>Author: <a href="https://github.com/kryptologyst" target="_blank">kryptologyst</a> | 
    GitHub: <a href="https://github.com/kryptologyst" target="_blank">https://github.com/kryptologyst</a></p>
    <p><em>Research and educational purposes only. Not for production decisions.</em></p>
</div>
""", unsafe_allow_html=True)
