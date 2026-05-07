#!/usr/bin/env bash

# Semi-Supervised Learning Project Setup Script

set -e

echo "🚀 Setting up Semi-Supervised Learning Project..."

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p data/{raw,processed}
mkdir -p outputs
mkdir -p assets
mkdir -p logs
mkdir -p checkpoints

# Create .gitkeep files
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch assets/.gitkeep

echo "✅ Directories created successfully"

# Check Python version
echo "🐍 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo "✅ Python $python_version is compatible (requires >= $required_version)"
else
    echo "❌ Python $python_version is not compatible (requires >= $required_version)"
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "✅ Dependencies installed from requirements.txt"
else
    echo "❌ requirements.txt not found"
    exit 1
fi

# Install development dependencies if available
if [ -f "pyproject.toml" ]; then
    echo "🔧 Installing development dependencies..."
    pip install -e ".[dev]"
    echo "✅ Development dependencies installed"
fi

# Setup pre-commit hooks
echo "🪝 Setting up pre-commit hooks..."
if command -v pre-commit &> /dev/null; then
    pre-commit install
    echo "✅ Pre-commit hooks installed"
else
    echo "⚠️  pre-commit not found, skipping hook installation"
fi

# Download CIFAR-10 dataset (optional)
echo "📊 Setting up CIFAR-10 dataset..."
python3 -c "
import torch
from torchvision import datasets, transforms
print('Downloading CIFAR-10 dataset...')
datasets.CIFAR10(root='data/raw', train=True, download=True, transform=transforms.ToTensor())
datasets.CIFAR10(root='data/raw', train=False, download=True, transform=transforms.ToTensor())
print('✅ CIFAR-10 dataset downloaded successfully')
"

# Run basic tests
echo "🧪 Running basic tests..."
if [ -f "tests/test_ssl.py" ]; then
    python3 -m pytest tests/test_ssl.py -v --tb=short
    echo "✅ Basic tests passed"
else
    echo "⚠️  Test file not found, skipping tests"
fi

# Create example configuration
echo "⚙️  Creating example configuration..."
cat > configs/example_config.yaml << 'EOF'
# Example configuration for semi-supervised learning
experiment:
  name: "example_experiment"
  tags: ["example", "ssl"]

train:
  epochs: 10
  batch_size: 32
  learning_rate: 0.001

ssl:
  labeled_samples: 1000
  pseudo_label_threshold: 0.95
  rampup_epochs: 5

model:
  _target_: src.models.simple_cnn.SimpleCNN
  input_channels: 3
  num_classes: 10
  hidden_dim: 128

data:
  _target_: src.data.cifar10.CIFAR10DataModule
  data_dir: "data/raw"
  batch_size: 32
  download: true

ssl_method:
  _target_: src.models.ssl_methods.PseudoLabeling
  confidence_threshold: 0.95
  rampup_epochs: 5

trainer:
  _target_: src.train.trainer.BasicTrainer
  max_epochs: 10
  optimizer: "adam"
  scheduler: "cosine"
EOF

echo "✅ Example configuration created"

# Create run script
echo "📝 Creating run script..."
cat > run_example.sh << 'EOF'
#!/bin/bash
echo "🚀 Running example SSL experiment..."
python3 scripts/quick_start.py \
    --labeled_samples 1000 \
    --epochs 10 \
    --batch_size 32 \
    --lr 0.001 \
    --ssl_method pseudo_labeling \
    --output_dir example_output

echo "✅ Example experiment completed!"
echo "📊 Results saved to example_output/"
echo "🎯 To view results: python3 scripts/evaluate.py --experiment_dir example_output"
EOF

chmod +x run_example.sh
echo "✅ Run script created"

# Create demo script
echo "🎮 Creating demo script..."
cat > run_demo.sh << 'EOF'
#!/bin/bash
echo "🎮 Starting interactive demo..."
echo "📱 Open your browser and go to: http://localhost:8501"
streamlit run demo/app.py
EOF

chmod +x run_demo.sh
echo "✅ Demo script created"

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "   1. Run example experiment: ./run_example.sh"
echo "   2. Start interactive demo: ./run_demo.sh"
echo "   3. Explore configurations in configs/"
echo "   4. Check README.md for detailed usage"
echo ""
echo "🔗 Useful commands:"
echo "   - Quick start: python3 scripts/quick_start.py --help"
echo "   - Full training: python3 scripts/train.py"
echo "   - Evaluation: python3 scripts/evaluate.py --help"
echo "   - Tests: python3 -m pytest tests/"
echo ""
echo "⚠️  Remember: This is for research and educational purposes only!"
echo "   Not for production decisions or control systems."
echo ""
echo "👨‍💻 Author: kryptologyst"
echo "🔗 GitHub: https://github.com/kryptologyst"
