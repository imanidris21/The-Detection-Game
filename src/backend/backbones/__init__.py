"""
Backbones module for AI detection model
"""

from .base_backbone import BaseBackbone, MultiBackbone
from .dinov3_backbone import DINOv3Backbone
from .forensic_backbone import ForensicBackbone

# Registry of available backbones
BACKBONE_REGISTRY = {
    'dinov2_vitb14': DINOv3Backbone,
    'dinov2_vitl14': DINOv3Backbone,
    'dinov2_vitg14': DINOv3Backbone,
    'dinov3_vitb16': DINOv3Backbone,
    'dinov3_vitl16': DINOv3Backbone,
    'forensic': ForensicBackbone,
    'frequency': ForensicBackbone,
}

def create_backbone(backbone_name, device=None, **kwargs):
    """Create a backbone instance"""
    if backbone_name not in BACKBONE_REGISTRY:
        raise ValueError(f"Unknown backbone: {backbone_name}. Available: {list(BACKBONE_REGISTRY.keys())}")

    backbone_class = BACKBONE_REGISTRY[backbone_name]

    # Pass model name to the backbone for proper initialization
    if backbone_name.startswith('dinov'):
        kwargs['model_name'] = backbone_name

    return backbone_class(device=device, **kwargs)

def get_available_backbones():
    """Get list of available backbone configs"""
    return list(BACKBONE_REGISTRY.keys())

def list_backbone_names():
    """Get list of available backbone names"""
    return list(BACKBONE_REGISTRY.keys())

# Simple backbone manager
class BackboneManager:
    def __init__(self):
        self.registry = BACKBONE_REGISTRY

    def get_backbone(self, name):
        return self.registry.get(name)

    def list_backbones(self):
        return list(self.registry.keys())

backbone_manager = BackboneManager()

__all__ = [
    'BaseBackbone',
    'MultiBackbone',
    'DINOv3Backbone',
    'ForensicBackbone',
    'create_backbone',
    'get_available_backbones',
    'list_backbone_names',
    'backbone_manager'
]