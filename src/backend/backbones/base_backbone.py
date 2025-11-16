"""
backbones/base_backbone.py - Abstract base class for all feature extraction backbones
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


class MultiBackbone(nn.Module):
    """
    Wrapper for using multiple backbones together
    This can be useful for ensemble methods or multimodal approaches
    """
    
    def __init__(self, backbones: Dict[str, BaseBackbone], fusion_method: str = 'concat'):
        """
        Args:
            backbones: Dictionary mapping names to backbone instances
            fusion_method: How to combine features ('concat', 'add', 'mean', 'max')
        """
        super().__init__()
        self.backbones = nn.ModuleDict(backbones)
        self.fusion_method = fusion_method
        
        # Calculate combined feature dimension
        if fusion_method == 'concat':
            self.feature_dim = sum(bb.feature_dim for bb in backbones.values())
        else:
            # For other fusion methods, all backbones must have same feature dim
            feature_dims = [bb.feature_dim for bb in backbones.values()]
            if len(set(feature_dims)) > 1:
                raise ValueError(f"All backbones must have same feature_dim for fusion_method='{fusion_method}'")
            self.feature_dim = feature_dims[0]
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all backbones and fuse features
        Args:
            images: Image tensor (N, 3, H, W) 
        Returns:
            Fused feature tensor (N, combined_feature_dim)
        """
        features = []
        
        for name, backbone in self.backbones.items():
            backbone_features = backbone(images)
            features.append(backbone_features)
        
        # Fuse features
        if self.fusion_method == 'concat':
            return torch.cat(features, dim=1)
        elif self.fusion_method == 'add':
            return torch.stack(features, dim=0).sum(dim=0)
        elif self.fusion_method == 'mean':
            return torch.stack(features, dim=0).mean(dim=0)
        elif self.fusion_method == 'max':
            return torch.stack(features, dim=0).max(dim=0)[0]
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about all backbones"""
        info = {
            'fusion_method': self.fusion_method,
            'combined_feature_dim': self.feature_dim,
            'backbones': {}
        }
        
        for name, backbone in self.backbones.items():
            info['backbones'][name] = backbone.get_info()
        
        return info
    
    def freeze_backbone(self, name: str):
        """Freeze specific backbone"""
        if name in self.backbones:
            self.backbones[name].freeze()
    
    def unfreeze_backbone(self, name: str):
        """Unfreeze specific backbone"""
        if name in self.backbones:
            self.backbones[name].unfreeze()
    
    def freeze_all(self):
        """Freeze all backbones"""
        for backbone in self.backbones.values():
            backbone.freeze()
    
    def unfreeze_all(self):
        """Unfreeze all backbones"""
        for backbone in self.backbones.values():
            backbone.unfreeze()