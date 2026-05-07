"""Simple CNN model for CIFAR-10 classification."""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """Simple CNN architecture for CIFAR-10 classification."""
    
    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 10,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        """Initialize SimpleCNN.
        
        Args:
            input_channels: Number of input channels.
            num_classes: Number of output classes.
            hidden_dim: Hidden dimension size.
            dropout: Dropout probability.
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        # Initialize weights
        self._initialize_weights()
        
        logging.info(f"Initialized SimpleCNN with {self._count_parameters()} parameters")
    
    def _initialize_weights(self) -> None:
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def _count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width).
            
        Returns:
            Output tensor of shape (batch_size, num_classes).
        """
        # First conv block
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        # Second conv block
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        # Third conv block
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        # Flatten and fully connected layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout_layer(x)
        x = self.fc2(x)
        
        return x
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before the final classification layer.
        
        Args:
            x: Input tensor.
            
        Returns:
            Feature tensor.
        """
        # Forward pass until fc1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        
        return x


class WideResNet(nn.Module):
    """Wide ResNet architecture for better performance."""
    
    def __init__(
        self,
        depth: int = 28,
        widen_factor: int = 2,
        num_classes: int = 10,
        dropout: float = 0.3,
    ):
        """Initialize WideResNet.
        
        Args:
            depth: Depth of the network.
            widen_factor: Width factor.
            num_classes: Number of output classes.
            dropout: Dropout probability.
        """
        super().__init__()
        
        self.depth = depth
        self.widen_factor = widen_factor
        self.num_classes = num_classes
        self.dropout = dropout
        
        # Calculate number of layers per block
        assert (depth - 4) % 6 == 0, "Depth must be 6n+4"
        n = (depth - 4) // 6
        
        # First layer
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Residual blocks
        self.layer1 = self._make_layer(16, 16 * widen_factor, n, stride=1)
        self.layer2 = self._make_layer(16 * widen_factor, 32 * widen_factor, n, stride=2)
        self.layer3 = self._make_layer(32 * widen_factor, 64 * widen_factor, n, stride=2)
        
        # Batch normalization and dropout
        self.bn1 = nn.BatchNorm2d(64 * widen_factor)
        self.dropout_layer = nn.Dropout(dropout)
        
        # Final classification layer
        self.fc = nn.Linear(64 * widen_factor, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
        logging.info(f"Initialized WideResNet-{depth}x{widen_factor} with {self._count_parameters()} parameters")
    
    def _make_layer(self, in_channels: int, out_channels: int, blocks: int, stride: int) -> nn.Module:
        """Make a layer with residual blocks."""
        layers = []
        
        # First block with potential downsampling
        layers.append(
            ResidualBlock(in_channels, out_channels, stride, dropout=self.dropout)
        )
        
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(
                ResidualBlock(out_channels, out_channels, 1, dropout=self.dropout)
            )
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self) -> None:
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def _count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.conv1(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = F.relu(self.bn1(x))
        x = F.avg_pool2d(x, 8)
        x = x.view(x.size(0), -1)
        x = self.dropout_layer(x)
        x = self.fc(x)
        
        return x


class ResidualBlock(nn.Module):
    """Residual block for WideResNet."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dropout: float = 0.0,
    ):
        """Initialize residual block."""
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.dropout = nn.Dropout(dropout)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        residual = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        
        out += residual
        out = F.relu(out)
        
        return out
