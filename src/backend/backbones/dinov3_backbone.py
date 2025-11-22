"""
backbones/dinov3_backbone.py - DINOv3 backbone implementation

DINOv3 is the latest self-supervised vision transformer from Meta AI,
trained on 1.7B web images.
"""

import torch
import torch.nn as nn
from typing import Optional
from pathlib import Path
from .base_backbone import BaseBackbone


class DINOv3Backbone(BaseBackbone):
    """DINOv3-based backbone for feature extraction (identical structure to DINOv2)"""

    def __init__(
        self,
        model_name: str = "dinov3_vitb16",
        device: Optional[str] = None,
        use_cls_token: bool = True,
        freeze_backbone: bool = False,
        extract_multiscale: bool = False
    ):
        """
        Args:
            model_name: DINOv3 model variant ('dinov3_vits16', 'dinov3_vitb16', etc.)
            device: Device gpu or cpu
            use_cls_token: Use [CLS] token features (recommended for classification)
            freeze_backbone: Freeze backbone weights (False for neural training)
            extract_multiscale: Extract multi-scale features for segmentation tasks
        """
        super().__init__(device)

        self.model_name = model_name
        self.use_cls_token = use_cls_token
        self.freeze_backbone = freeze_backbone
        self.extract_multiscale = extract_multiscale
        self.architecture_family = 'dinov3'
        self.normalization_type = 'imagenet'
        self.input_size = (224, 224)

        self._load_model()
        self._setup_feature_extraction()

    def _load_model(self):
        """Load DINOv3 model from manually downloaded checkpoint"""
        print(f"Loading DINOv3 model: {self.model_name} on {self.device}")

        # Mapping of model names to checkpoint filenames and architecture configs
        checkpoint_files = {
            'dinov3_vits16': 'dinov3_vits16_pretrain_lvd1689m-08c60483.pth',
            'dinov3_vitb16': 'dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth',
            'dinov3_vitl16': 'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth',
        }

        # Model architecture configurations (matching DINOv3 hub defaults with all features)
        model_configs = {
            'dinov3_vits16': {
                'compact_arch_name': 'vits', #small
                'embed_dim': 384,
                'depth': 12,            
                'num_heads': 6,
                'n_storage_tokens': 4,  # DINOv3 uses 4 storage tokens
                'mask_k_bias': True,    # Masked attention biases
                'layerscale_init': 1e-5,  # LayerScale initialization
            },
            'dinov3_vitb16': {
                'compact_arch_name': 'vitb', # base
                'embed_dim': 768,
                'depth': 12,
                'num_heads': 12,
                'n_storage_tokens': 4,  # DINOv3 uses 4 storage tokens
                'mask_k_bias': True,
                'layerscale_init': 1e-5,
            },
            'dinov3_vitl16': {
                'compact_arch_name': 'vitl', # large
                'embed_dim': 1024,
                'depth': 24,
                'num_heads': 16,
                'n_storage_tokens': 4,  # DINOv3 uses 4 storage tokens
                'mask_k_bias': True,
                'layerscale_init': 1e-5,
            },
        }

        if self.model_name not in checkpoint_files:
            raise ValueError(f"Model {self.model_name} not supported. "
                           f"Available models: {list(checkpoint_files.keys())}")

        # Get checkpoint path - check torch hub cache (standard location)
        checkpoint_name = checkpoint_files[self.model_name]

        # Standard torch hub cache location (same as DINOv2) to get pretrained weights
        hub_checkpoint = Path.home() / '.cache' / 'torch' / 'hub' / 'checkpoints' / checkpoint_name

        # Fallback: project checkpoints folder
        project_checkpoint = Path(__file__).parent.parent.parent / 'checkpoints' / checkpoint_name

        if hub_checkpoint.exists():
            checkpoint_path = hub_checkpoint
            print(f"Found checkpoint in torch hub cache: {checkpoint_name}")
        elif project_checkpoint.exists():
            checkpoint_path = project_checkpoint
            print(f"Found checkpoint in project folder: {checkpoint_name}")
        else:
            # Try to download from Google Drive if available
            try:
                from ..model_downloader import ensure_dinov3_pretrain_model
                project_root = Path(__file__).parent.parent.parent.parent  # Go to project root
                checkpoint_path = ensure_dinov3_pretrain_model(str(project_root))
                print(f"Downloaded checkpoint from Google Drive: {checkpoint_name}")
            except Exception as e:
                raise FileNotFoundError(
                    f"Checkpoint not found and failed to download. Searched:\n"
                    f"  - {hub_checkpoint}\n"
                    f"  - {project_checkpoint}\n"
                    f"  - Google Drive download failed: {e}\n"
                    f"Please place the checkpoint in: {hub_checkpoint}"
                )

        try:
            # Import the DINOv3 model builder directly from cached hub
            import sys
            hub_dir = Path.home() / '.cache' / 'torch' / 'hub' / 'facebookresearch_dinov3_main'
            if str(hub_dir) not in sys.path:
                sys.path.insert(0, str(hub_dir))

            # Add dinov3 package to path for import
            import sys
            # Go from src/backend/backbones/ -> src/ -> dinov3/
            src_dir = Path(__file__).parent.parent.parent
            dinov3_path = src_dir / "dinov3"
            if str(dinov3_path) not in sys.path:
                sys.path.insert(0, str(dinov3_path))

            # load model architecture/builder
            from dinov3.hub.backbones import _make_dinov3_vit

            config = model_configs[self.model_name]

            # Create model without downloading (pretrained=False)
            print(f"Creating DINOv3 model architecture...")
            self.model = _make_dinov3_vit(
                pretrained=False,  # Don't try to download - we'll load manually
                **config
            )

            # Load the manually downloaded checkpoint
            print(f"Loading weights from {checkpoint_name}...")
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            self.model.load_state_dict(state_dict, strict=True)
            print("✓ DINOv3 checkpoint loaded successfully")

        except Exception as e:
            import traceback
            print(f"\nError details:\n{traceback.format_exc()}")
            raise RuntimeError(f"Failed to load DINOv3 model: {e}")

        self.model = self.model.to(self.device)
        self.model.eval()

        if self.freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        # Get feature dimension
        self._determine_feature_dim()

    def _determine_feature_dim(self):
        """Determine the feature dimension of the model"""
        # Try to get from model attributes
        if hasattr(self.model, 'embed_dim'):
            self.feature_dim = self.model.embed_dim
        elif hasattr(self.model, 'num_features'):
            self.feature_dim = self.model.num_features
        else:
            # Fallback: pass a dummy input through the model for testing
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
                dummy_output = self.model(dummy_input)

                if self.use_cls_token:
                    if len(dummy_output.shape) == 2:  # Already (N, D)
                        self.feature_dim = dummy_output.shape[-1]
                    elif len(dummy_output.shape) == 3:  # (N, num_patches+1, D)
                        self.feature_dim = dummy_output.shape[-1]
                else:
                    # Using patch tokens
                    if len(dummy_output.shape) == 3:
                        self.feature_dim = dummy_output.shape[-1]
                    else:
                        self.feature_dim = dummy_output.shape[-1]

        print(f"DINOv3 feature dimension: {self.feature_dim}")

    def _setup_feature_extraction(self):
        """Setup feature extraction method"""
        # DINOv3 doesn't need special hook setup like CLIP
        # Feature extraction is handled in _extract_features
        pass




    def _extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features using DINOv3 (supports neural mode and multi-scale)"""
        # Apply normalization conversion if needed
        if images.max() <= 1.0:
            # Convert from CLIP normalization to ImageNet normalization
            clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(images.device)
            clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(images.device)

            # Check if input is already in CLIP format
            if torch.allclose(images.mean(dim=(0, 2, 3)), torch.tensor([0.0, 0.0, 0.0]).to(images.device), atol=0.5):
                # Denormalize from CLIP
                images = images * clip_std + clip_mean

            # Apply ImageNet normalization
            imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
            imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
            images = (images - imagenet_mean) / imagenet_std

        # Extract features (support both frozen and trainable modes)
        if self.freeze_backbone and not self.training:
            with torch.no_grad():
                features = self.model(images)
        else:
            # Neural mode: allow gradients to flow
            features = self.model(images)

        if self.extract_multiscale:
            # Return patch tokens reshaped as feature maps for segmentation
            if len(features.shape) == 3:  # (N, num_patches+1, D)
                batch_size, num_tokens, feature_dim = features.shape
                # Assume square patch grid (e.g., 14x14 = 196 patches for 224x224 input with patch size 16)
                num_patches = num_tokens - 1  # Exclude CLS token
                patch_size = int(num_patches ** 0.5)

                if patch_size ** 2 == num_patches:
                    # Reshape patch tokens to spatial feature map
                    patch_features = features[:, 1:]  # Skip CLS token (N, num_patches, D)
                    spatial_features = patch_features.view(
                        batch_size, patch_size, patch_size, feature_dim
                    ).permute(0, 3, 1, 2)  # (N, D, H, W)

                    if self.use_cls_token:
                        # Also return CLS token for classification
                        cls_features = features[:, 0]  # (N, D)
                        return spatial_features, cls_features
                    else:
                        return spatial_features
                else:
                    # Fallback: create minimal spatial map
                    if self.use_cls_token:
                        cls_features = features[:, 0]
                        spatial_features = cls_features.view(
                            batch_size, feature_dim, 1, 1
                        ).expand(-1, -1, 14, 14)
                        return spatial_features, cls_features
                    else:
                        avg_features = features[:, 1:].mean(dim=1)
                        spatial_features = avg_features.view(
                            batch_size, feature_dim, 1, 1
                        ).expand(-1, -1, 14, 14)
                        return spatial_features
            else:
                # Features already in 2D format, create spatial representation
                batch_size, feature_dim = features.shape
                spatial_features = features.view(
                    batch_size, feature_dim, 1, 1
                ).expand(-1, -1, 14, 14)
                return spatial_features
        else:
            # Standard classification mode
            if self.use_cls_token:
                # For newer DINOv3 models, features might already be [CLS] token
                if len(features.shape) == 2:  # Already (N, D)
                    pass
                elif len(features.shape) == 3:  # (N, num_patches+1, D)
                    features = features[:, 0]  # Take [CLS] token
            else:
                # Use average of patch tokens (excluding CLS if present)
                if len(features.shape) == 3:
                    features = features[:, 1:].mean(dim=1)  # Skip CLS token, average patches

            return features



    def get_patch_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """
        Get patch tokens (DINOv3-specific functionality)

        Args:
            images: Image tensor (N, 3, H, W)

        Returns:
            Patch tokens (N, num_patches, feature_dim)
        """
        images = images.to(self.device)

        with torch.no_grad():
            features = self.model(images)

            if len(features.shape) == 3:  # (N, num_patches+1, D)
                # Return patch tokens (excluding CLS)
                return features[:, 1:]
            else:
                raise RuntimeError("Cannot extract patch tokens from this model output shape")



    def get_info(self):
        """Get DINOv3-specific information"""
        info = super().get_info()
        info.update({
            'use_cls_token': self.use_cls_token,
            'freeze_backbone': self.freeze_backbone,
            'extract_multiscale': self.extract_multiscale,
            'supports_text': False,
            'supports_patches': True,
            'supports_segmentation': self.extract_multiscale,
            'dinov3_model_name': self.model_name,
            'training_data': 'LVD-1689M (1.7B web images)'
        })
        return info


def create_dinov3_backbone(model_name: str = "dinov3_vitb16", **kwargs):
    """Convenience function to create DINOv3 backbone"""
    return DINOv3Backbone(model_name=model_name, **kwargs)


# Model configurations
DINOV3_CONFIGS = {
    'dinov3_vits16': {'model_name': 'dinov3_vits16', 'feature_dim': 384},
    'dinov3_vits16_plus': {'model_name': 'dinov3_vits16_plus', 'feature_dim': 384},
    'dinov3_vitb16': {'model_name': 'dinov3_vitb16', 'feature_dim': 768},
    'dinov3_vitl16': {'model_name': 'dinov3_vitl16', 'feature_dim': 1024},
    'dinov3_vith16_plus': {'model_name': 'dinov3_vith16_plus', 'feature_dim': 1280},
}
