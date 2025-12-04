"""
backbones/base_backbone.py - Abstract base class for all feature extraction backbones

This basebackbone class has been debugged with the assistance of Claude AI, All suggestions were reviewed critically and modified as needed. 

"""



from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn





class BaseBackbone(nn.Module, ABC):
    """
    Abstract base class for all feature extraction backbones
    
    This class defines the interface that all backbones must implement,
    ensuring consistency across different architectures.
    """
    
    def __init__(self, device: Optional[str] = None):
        super().__init__()
        self.device = self._get_device(device)
        
        # These should be set by subclasses
        self.feature_dim: int = None
        self.input_size: Tuple[int, int] = (224, 224)
        self.normalization_type: str = None  # 'clip', 'imagenet', etc.
        self.architecture_family: str = None  # 'clip', 'dinov2', etc.
        self.model_name: str = None
        
        # Model components (to be initialized by subclasses)
        self.model = None
        self.preprocess = None
    
    def _get_device(self, device: Optional[str] = None) -> torch.device:
        """Get appropriate device"""
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    @abstractmethod
    def _load_model(self):
        """Load the underlying model. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract features from preprocessed images.
        Must be implemented by subclasses.
        Args:
            images: Preprocessed image tensor (N, 3, H, W)    
        Returns:
            Feature tensor (N, feature_dim)
        """
        pass
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the backbone
        Args:
            images: Image tensor (N, 3, H, W) with appropriate normalization 
        Returns:
            Feature tensor (N, feature_dim)
        """
        images = images.to(self.device)
        
        with torch.no_grad():
            features = self._extract_features(images)
        
        # Normalize features for consistency
        features = torch.nn.functional.normalize(features, dim=1)
        
        return features.detach().cpu()
    
    def get_preprocessing_transform(self):
        """
        Get the preprocessing transform for this backbone
        Returns:
            Transform function or torchvision.transforms.Compose
        """
        if self.preprocess is not None:
            return self.preprocess
        
        # Default preprocessing based on normalization type
        from torchvision import transforms
        
        transform_list = [
            transforms.Resize(self.input_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ]
        
        if self.normalization_type == 'clip':
            transform_list.append(transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            ))
        elif self.normalization_type == 'imagenet':
            transform_list.append(transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ))
        
        return transforms.Compose(transform_list)
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about this backbone
        
        Returns:
            Dictionary with backbone information
        """
        return {
            'model_name': self.model_name,
            'architecture_family': self.architecture_family,
            'feature_dim': self.feature_dim,
            'input_size': self.input_size,
            'normalization_type': self.normalization_type,
            'device': str(self.device),
            'parameters': sum(p.numel() for p in self.parameters()) if self.model else 0,
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad) if self.model else 0
        }
    
    def freeze(self):
        """Freeze all parameters"""
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        """Unfreeze all parameters"""
        for param in self.parameters():
            param.requires_grad = True
    
    def to_device(self, device: str):
        """Move model to device"""
        self.device = torch.device(device)
        if self.model is not None:
            self.model = self.model.to(self.device)
        return self
    
    def eval(self):
        """Set to evaluation mode"""
        if self.model is not None:
            self.model.eval()
        return super().eval()
    
    def train(self, mode: bool = True):
        """Set training mode"""
        if self.model is not None:
            self.model.train(mode)
        return super().train(mode)
    
    def __repr__(self):
        info = self.get_info()
        return (f"{self.__class__.__name__}("
                f"model_name='{info['model_name']}', "
                f"feature_dim={info['feature_dim']}, "
                f"device='{info['device']}')")

