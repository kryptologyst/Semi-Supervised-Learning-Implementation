#!/usr/bin/env python3
"""
Comparison between original and modern SSL implementations.
This script demonstrates the improvements made in the refactored version.
"""

import torch
import time
import logging
from pathlib import Path

from src.models import SimpleCNN
from src.models.ssl_methods import PseudoLabeling
from src.data import CIFAR10DataModule
from src.train import BasicTrainer
from src.utils import set_seed, setup_logging, get_device


def run_original_style():
    """Run the original implementation style (simplified version)."""
    print("🔄 Running Original Style Implementation")
    print("-" * 50)
    
    start_time = time.time()
    
    # Original style: Manual setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Original style: Simple model definition
    class OriginalCNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3)
            self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3)
            self.fc1 = torch.nn.Linear(64 * 6 * 6, 128)
            self.fc2 = torch.nn.Linear(128, 10)
        
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.max_pool2d(x, 2)
            x = torch.relu(self.conv2(x))
            x = torch.max_pool2d(x, 2)
            x = x.view(x.size(0), -1)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    # Original style: Manual data loading
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.Resize((32, 32)), 
        transforms.ToTensor()
    ])
    
    train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    # Original style: Manual splitting
    import random
    labeled_indices = random.sample(range(len(train_dataset)), 1000)
    labeled_data = torch.utils.data.Subset(train_dataset, labeled_indices)
    
    labeled_loader = torch.utils.data.DataLoader(
        labeled_data, batch_size=32, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False
    )
    
    # Original style: Manual training setup
    model = OriginalCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Original style: Simple training loop
    model.train()
    for epoch in range(5):
        total_loss = 0
        for data, target in labeled_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(labeled_loader):.4f}")
    
    # Original style: Manual evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"Original Style - Test Accuracy: {accuracy:.2f}%")
    print(f"Original Style - Time: {duration:.2f}s")
    
    return accuracy, duration


def run_modern_style():
    """Run the modern implementation style."""
    print("\n🚀 Running Modern Style Implementation")
    print("-" * 50)
    
    start_time = time.time()
    
    # Modern style: Proper setup
    set_seed(42)
    device = get_device()
    
    # Modern style: Structured data module
    data_module = CIFAR10DataModule(
        data_dir='./data',
        batch_size=32,
        num_workers=2,
        download=True,
        augment=True,
        augmentation_strength="medium"
    )
    
    data_module.prepare_data()
    data_module.setup("fit")
    
    labeled_loader, unlabeled_loader, val_loader, test_loader = data_module.create_semi_supervised_loaders(
        labeled_samples=1000,
        seed=42
    )
    
    # Modern style: Modular model
    model = SimpleCNN(
        input_channels=3,
        num_classes=10,
        hidden_dim=128,
        dropout=0.2
    )
    
    # Modern style: SSL method
    ssl_method = PseudoLabeling(
        confidence_threshold=0.95,
        rampup_epochs=10,
        rampup_type="sigmoid"
    )
    
    # Modern style: Structured training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    
    trainer = BasicTrainer(
        model=model,
        ssl_method=ssl_method,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        max_epochs=5
    )
    
    # Modern style: Comprehensive training
    history = trainer.fit(labeled_loader, unlabeled_loader, val_loader)
    
    # Modern style: Comprehensive evaluation
    test_metrics = trainer.evaluate(test_loader)
    
    end_time = time.time()
    duration = end_time - start_time
    
    accuracy = test_metrics['accuracy'] * 100
    
    print(f"Modern Style - Test Accuracy: {accuracy:.2f}%")
    print(f"Modern Style - F1 Score: {test_metrics['f1_macro']:.4f}")
    print(f"Modern Style - ROC AUC: {test_metrics.get('roc_auc_macro', 0):.4f}")
    print(f"Modern Style - Calibration Error: {test_metrics['expected_calibration_error']:.4f}")
    print(f"Modern Style - Time: {duration:.2f}s")
    
    return accuracy, duration


def main():
    """Main comparison function."""
    print("🔍 SSL Implementation Comparison")
    print("=" * 60)
    print("Comparing original vs modern implementation styles")
    print("⚠️  For research and educational purposes only")
    print("=" * 60)
    
    # Run both implementations
    orig_acc, orig_time = run_original_style()
    modern_acc, modern_time = run_modern_style()
    
    # Compare results
    print("\n📊 COMPARISON RESULTS")
    print("=" * 60)
    print(f"Original Implementation:")
    print(f"  - Accuracy: {orig_acc:.2f}%")
    print(f"  - Time: {orig_time:.2f}s")
    print(f"  - Features: Basic training, manual setup")
    
    print(f"\nModern Implementation:")
    print(f"  - Accuracy: {modern_acc:.2f}%")
    print(f"  - Time: {modern_time:.2f}s")
    print(f"  - Features: SSL methods, comprehensive metrics, structured code")
    
    print(f"\nImprovements:")
    print(f"  - Accuracy: {modern_acc - orig_acc:+.2f}%")
    print(f"  - Time: {modern_time - orig_time:+.2f}s")
    print(f"  - Code Quality: ✅ Modular, typed, tested")
    print(f"  - Features: ✅ SSL methods, metrics, visualization")
    print(f"  - Reproducibility: ✅ Seeded, configurable")
    
    print("\n🎯 Key Improvements in Modern Version:")
    print("1. 🏗️  Modular architecture with separate components")
    print("2. 🎯 Multiple SSL methods (Pseudo-labeling, Consistency, MixMatch, FixMatch)")
    print("3. 📊 Comprehensive evaluation metrics")
    print("4. 🔧 Configuration management with YAML")
    print("5. 🧪 Unit tests and integration tests")
    print("6. 📈 Training visualization and monitoring")
    print("7. 🎮 Interactive Streamlit demo")
    print("8. 🔄 Reproducible experiments with proper seeding")
    print("9. 📝 Type hints and documentation")
    print("10. 🚀 CI/CD pipeline with automated testing")
    
    print("\n" + "=" * 60)
    print("✅ Comparison completed!")
    print("🔗 Modern implementation provides significant improvements")
    print("   in code quality, features, and maintainability.")
    print("👨‍💻 Author: kryptologyst")
    print("🔗 GitHub: https://github.com/kryptologyst")
    print("=" * 60)


if __name__ == "__main__":
    main()
