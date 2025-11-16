"""
heads.py - Neural heads for classification and segmentation tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ClassificationHead(nn.Module):
    """
    MLP head for image-level classification
    Takes concatenated features from dual backbones and outputs binary classification
    """

    def __init__(
        self,
        input_dim: int = 1536,  # DINOv2 (768) + Forensic (768)
        hidden_dims: Tuple[int, ...] = (512, 256),
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True
    ):
        """
        Args:
            input_dim: Input feature dimension (concatenated backbone features)
            hidden_dims: Hidden layer dimensions
            dropout_rate: Dropout probability
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm

        # Build MLP layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # Batch normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            # Activation
            layers.append(nn.ReLU(inplace=True))

            # Dropout
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

            prev_dim = hidden_dim

        # Final classification layer
        layers.append(nn.Linear(prev_dim, 1))

        self.classifier = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier/He initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        Args:
            features: Concatenated backbone features (B, input_dim)
        Returns:
            Classification logits (B, 1)
        """
        return self.classifier(features)




# Utility functions
def create_classification_head(input_dim: int = 1536, **kwargs) -> ClassificationHead:
    """Create classification head with default parameters"""
    return ClassificationHead(input_dim=input_dim, **kwargs)