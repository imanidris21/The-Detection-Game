"""
model.py -  AI Image Detector with Multiple Backbones and Classifiers
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import numpy as np
from typing import Optional, Tuple, Union, Dict, Any
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import pickle
from pathlib import Path


from .backbones import (
    create_backbone,
    get_available_backbones,
    list_backbone_names,
    backbone_manager,
)
from .heads import ClassificationHead


# -------------------------------------------------------------------------
# Identify device type for torch
# -------------------------------------------------------------------------
def _get_device(device: Optional[str] = None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)





# -------------------------------------------------------------------------
# Main AI Image Detector
# -------------------------------------------------------------------------


class AIImageDetector(nn.Module):
    """
    AI Image Detector with dual-branch architecture
    Supports both neural training (MLP with fine-tuning) and feature-based training (SVM/Logistic)

    Architecture:
    - Semantic branch: DINOv2/CLIP for high-level semantic features
    - Forensic branch: SRM/DCT/Frequency for low-level forensic artifacts
    - Classification head: MLP for binary classification (real vs AI-generated)

    Modes:
    - Neural mode (default): End-to-end training with gradient descent, supports fine-tuning
    - Feature-based mode: Extract features + train SVM/Logistic classifier
    """

    def __init__(
        self,
        backbone_name: str = "dinov2_vitb14",
        classifier_type: str = "svm",
        device: Optional[str] = None,
        backbone_kwargs: Optional[Dict] = None,
        use_frequency: bool = True,
        frequency_backbone: str = "forensic",
        frequency_kwargs: Optional[Dict] = None,
        neural_mode: bool = True,  # Default: neural mode (MLP) instead of feature-based
        freeze_backbone: bool = False,  # Default: allow fine-tuning
        head_kwargs: Optional[Dict] = None
    ):
        """
        Args:
            backbone_name: Name of semantic backbone to use (e.g., 'dinov2_vitb14')
            classifier_type: 'svm' (LinearSVC) or 'logistic' (used in feature-based mode)
            device: Device to run on
            backbone_kwargs: Additional arguments for backbone initialization
            use_frequency: Enable forensic branch (dual-branch architecture)
            frequency_backbone: Type of forensic backbone ('forensic', 'frequency', etc.)
            frequency_kwargs: Additional arguments for forensic backbone
            neural_mode: Enable neural training with MLP head (default True)
                        Set to False to use feature-based SVM/Logistic training
            freeze_backbone: Freeze backbone weights during training (default False for fine-tuning)
            head_kwargs: Additional arguments for neural classification head
        """
        super().__init__()

        # Store configuration
        self.device = _get_device(device)
        self.backbone_name = backbone_name
        self.classifier_type = classifier_type
        self.use_frequency = use_frequency
        self.frequency_backbone_name = frequency_backbone if use_frequency else None
        self.neural_mode = neural_mode
        self.freeze_backbone = freeze_backbone and not neural_mode  # Only applies in feature-based mode

        # Legacy compatibility
        self.classifier = None
        self.is_trained = False

        # Initialize semantic backbone
        backbone_kwargs = backbone_kwargs or {}
        if neural_mode:
            # Neural mode: allow fine-tuning
            backbone_kwargs.update({
                'freeze_backbone': False
            })
        else:
            # Feature-based mode: use freeze setting
            backbone_kwargs.setdefault('freeze_backbone', self.freeze_backbone)

        self.backbone = create_backbone(
            backbone_name,
            device=str(self.device),
            **backbone_kwargs
        )

        # Initialize forensic backbone if enabled
        if self.use_frequency:
            frequency_kwargs = frequency_kwargs or {}
            # Match forensic feature dim to semantic feature dim for balance
            if 'feature_dim' not in frequency_kwargs:
                frequency_kwargs['feature_dim'] = self.backbone.feature_dim

            self.forensic_backbone = create_backbone(
                frequency_backbone,
                device=str(self.device),
                **frequency_kwargs
            )
            self.total_feature_dim = self.backbone.feature_dim + self.forensic_backbone.feature_dim
        else:
            self.forensic_backbone = None
            self.total_feature_dim = self.backbone.feature_dim

        # Initialize classification head for neural mode
        if self.neural_mode:
            head_kwargs = head_kwargs or {}
            # Concatenated input dimension: semantic + forensic (if enabled)
            input_dim = self.total_feature_dim if self.use_frequency else self.backbone.feature_dim
            self.heads = ClassificationHead(
                input_dim=input_dim,
                **head_kwargs
            )
        else:
            self.heads = None

        # Move to device
        self.to(self.device)

        # Print initialization info
        mode_str = "NEURAL (MLP)" if self.neural_mode else "FEATURE-BASED (SVM/Logistic)"
        if self.use_frequency:
            print(f"Initialized {mode_str} DUAL-BRANCH detector:")
            print(f"  Semantic: {backbone_name} ({self.backbone.feature_dim}D)")
            print(f"  Forensic: {frequency_backbone} ({self.forensic_backbone.feature_dim}D)")
            print(f"  Total feature dimension: {self.total_feature_dim}")
            if self.neural_mode:
                print(f"  Head: ClassificationHead (MLP)")
        else:
            print(f"Initialized {mode_str} detector with {backbone_name} backbone")
            print(f"Feature dimension: {self.backbone.feature_dim}")
            if self.neural_mode:
                print(f"  Head: ClassificationHead (MLP)")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass (neural mode only)
        Args:
            images: Input images (B, 3, H, W)
        Returns:
            Classification logits (B, 1) for binary classification
        Raises:
            RuntimeError: If not in neural mode
        """
        if not self.neural_mode:
            raise RuntimeError("Forward pass only available in neural mode. Use extract_features() for feature-based mode.")

        images = images.to(self.device)

        # Extract features from semantic backbone
        semantic_features = self.backbone(images)

        # Flatten if needed
        if semantic_features.dim() > 2:
            semantic_features = F.adaptive_avg_pool2d(semantic_features, (1, 1)).flatten(1)

        # Extract features from forensic backbone if enabled
        if self.use_frequency:
            forensic_features = self.forensic_backbone(images)

            # Flatten if needed
            if forensic_features.dim() > 2:
                forensic_features = F.adaptive_avg_pool2d(forensic_features, (1, 1)).flatten(1)

            # Ensure both features are on the same device before concatenation
            semantic_features = semantic_features.to(self.device)
            forensic_features = forensic_features.to(self.device)

            # Concatenate semantic and forensic features
            combined_features = torch.cat([semantic_features, forensic_features], dim=1)
            return self.heads(combined_features)
        else:
            # Single backbone case
            return self.heads(semantic_features)



    def extract_features(self, dataloader) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features and labels from a dataloader (supports dual-branch)"""
        all_features = []
        all_labels = []
        self.backbone.eval()
        if self.use_frequency:
            self.forensic_backbone.eval()

        total_batches = len(dataloader)
        branch_type = "DUAL-BRANCH (spatial+frequency)" if self.use_frequency else "single-branch (spatial)"
        print(f"🔄 Starting {branch_type} feature extraction: {total_batches} batches to process...")

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader, 1):
                if batch_idx % 10 == 0 or batch_idx == 1:
                    print(f"   Processing batch {batch_idx}/{total_batches}...")

                images = batch["image"]
                labels_t = batch["label"]
                labels = labels_t.detach().cpu().numpy()

                # Convert images to torch if they are numpy arrays
                if isinstance(images, np.ndarray):
                    images = torch.from_numpy(images).float()
                images = images.to(self.device)

                labels = labels_t.detach().cpu().numpy() if isinstance(labels_t, torch.Tensor) else np.array(labels_t)

                # Extract spatial features
                spatial_feats = self.backbone(images)

                # Handle multi-scale features for neural mode
                if isinstance(spatial_feats, tuple):
                    # Multi-scale output from backbone
                    spatial_feats = spatial_feats[1] if len(spatial_feats) > 1 else spatial_feats[0]

                # Ensure features are flattened for feature-based training
                if spatial_feats.dim() > 2:
                    spatial_feats = F.adaptive_avg_pool2d(spatial_feats, (1, 1)).flatten(1)

                # Extract forensic features if enabled
                if self.use_frequency:
                    forensic_feats = self.forensic_backbone(images)
                    if forensic_feats.dim() > 2:
                        forensic_feats = F.adaptive_avg_pool2d(forensic_feats, (1, 1)).flatten(1)
                    # Concatenate semantic and forensic features
                    feats = torch.cat([spatial_feats, forensic_feats], dim=1)
                else:
                    feats = spatial_feats

                # always detach + move to CPU before NumPy
                all_features.append(feats.detach().cpu().numpy())
                all_labels.append(labels)

        print(f"✅ Feature extraction complete! Processed {total_batches} batches.")
        X = np.vstack(all_features)
        y = np.concatenate(all_labels)
        print(f"📊 Final feature shape: {X.shape}, Labels shape: {y.shape}")
        return X, y




    def train_classifier(self, train_features: np.ndarray, train_labels: np.ndarray, C: float = 1.0, **kwargs):
        """
        Train the feature-based classifier (SVM/Logistic)

        Args:
            train_features: (N, D) array
            train_labels: (N,) array with {0=real, 1=fake}
            C: Regularization parameter
            **kwargs: Additional classifier parameters
        """
        if self.classifier_type == "svm":
            self.classifier = svm.LinearSVC(C=C, max_iter=10000, random_state=42, **kwargs)
        elif self.classifier_type == "logistic":
            self.classifier = LogisticRegression(C=C, max_iter=1000, random_state=42, **kwargs)
        else:
            raise ValueError(f"Unknown classifier: {self.classifier_type}")

        print(f"Training {self.classifier_type} classifier on {train_features.shape[0]} samples...")
        # print(f"Feature dimension: {train_features.shape[1]}")
        self.classifier.fit(train_features, train_labels)
        self.is_trained = True
        print("Training complete.")




    def predict(self, images: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Predict labels for input images
        Supports both neural mode and feature-based mode
        """
        if self.neural_mode:
            # Neural mode prediction
            if not self.is_trained:
                raise RuntimeError("Neural detector not trained yet")

            if isinstance(images, np.ndarray):
                images = torch.from_numpy(images).float()

            self.eval()
            with torch.no_grad():
                output = self.forward(images)
                if isinstance(output, tuple):
                    # Classification + segmentation output
                    class_logits, _ = output
                else:
                    # Classification only
                    class_logits = output

                # Convert logits to predictions
                predictions = torch.sigmoid(class_logits).cpu().numpy()
                return (predictions > 0.5).astype(int).flatten()
        else:
            # Feature-based mode prediction
            if not self.is_trained or self.classifier is None:
                raise RuntimeError("Feature-based detector not trained yet")

            if isinstance(images, np.ndarray):
                images = torch.from_numpy(images).float()

            with torch.no_grad():
                # Extract features using the same method as training
                spatial_feats = self.backbone(images.to(self.device))

                # Handle multi-scale features
                if isinstance(spatial_feats, tuple):
                    spatial_feats = spatial_feats[1] if len(spatial_feats) > 1 else spatial_feats[0]
                if spatial_feats.dim() > 2:
                    spatial_feats = F.adaptive_avg_pool2d(spatial_feats, (1, 1)).flatten(1)

                if self.use_frequency:
                    forensic_feats = self.forensic_backbone(images.to(self.device))
                    if forensic_feats.dim() > 2:
                        forensic_feats = F.adaptive_avg_pool2d(forensic_feats, (1, 1)).flatten(1)
                    features = torch.cat([spatial_feats, forensic_feats], dim=1)
                else:
                    features = spatial_feats

                features_np = features.detach().cpu().numpy()
                return self.classifier.predict(features_np)




    def predict_proba(self, images: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Predict class probabilities for input images
        Supports both neural mode and feature-based mode
        """
        if self.neural_mode:
            # Neural mode prediction
            if not self.is_trained:
                raise RuntimeError("Neural detector not trained yet")

            if isinstance(images, np.ndarray):
                images = torch.from_numpy(images).float()

            self.eval()
            with torch.no_grad():
                output = self.forward(images)
                if isinstance(output, tuple):
                    # Classification + segmentation output
                    class_logits, _ = output
                else:
                    # Classification only
                    class_logits = output

                # Convert logits to probabilities
                probs = torch.sigmoid(class_logits).cpu().numpy().flatten()
                # Return as [prob_real, prob_fake] format
                return np.column_stack([1 - probs, probs])
        else:
            # Feature-based mode prediction
            if not self.is_trained or self.classifier is None:
                raise RuntimeError("Feature-based detector not trained yet")

            if isinstance(images, np.ndarray):
                images = torch.from_numpy(images).float()

            with torch.no_grad():
                # Extract features using the same method as training
                spatial_feats = self.backbone(images.to(self.device))

                # Handle multi-scale features
                if isinstance(spatial_feats, tuple):
                    spatial_feats = spatial_feats[1] if len(spatial_feats) > 1 else spatial_feats[0]
                if spatial_feats.dim() > 2:
                    spatial_feats = F.adaptive_avg_pool2d(spatial_feats, (1, 1)).flatten(1)

                if self.use_frequency:
                    forensic_feats = self.forensic_backbone(images.to(self.device))
                    if forensic_feats.dim() > 2:
                        forensic_feats = F.adaptive_avg_pool2d(forensic_feats, (1, 1)).flatten(1)
                    features = torch.cat([spatial_feats, forensic_feats], dim=1)
                else:
                    features = spatial_feats

                features_np = features.detach().cpu().numpy()

            # Use predict_proba if available
            if hasattr(self.classifier, "predict_proba"):
                return self.classifier.predict_proba(features_np)
            # For LinearSVC or classifiers without predict_proba
            if hasattr(self.classifier, "decision_function"):
                scores = self.classifier.decision_function(features_np)
                probs = 1.0 / (1.0 + np.exp(-scores))
                probs = np.clip(probs, 1e-6, 1 - 1e-6)
                return np.vstack([1 - probs, probs]).T
            raise RuntimeError("Classifier does not support probability prediction")


    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance from classifier (if available)"""
        if not self.is_trained or self.classifier is None:
            return None
        if hasattr(self.classifier, 'coef_'):
            return self.classifier.coef_[0]
        else:
            return None



    def save(self, path: Union[str, Path], include_backbone: bool = False):
        """
        Save trained detector
        Args:
            path: Save path
            include_backbone: Whether to save backbone weights
        """

        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare data to save
        save_data = {
            "classifier": self.classifier,
            "classifier_type": self.classifier_type,
            "backbone_name": self.backbone_name,
            "backbone_info": self.backbone.get_info(),
            "feature_dim": self.backbone.feature_dim,
            "is_trained": self.is_trained
        }

        # Optionally save backbone state
        if include_backbone:
            save_data["backbone_state_dict"] = self.backbone.state_dict()
        with open(path, "wb") as f:
            pickle.dump(save_data, f)

        print(f"Model saved to {path}")




    def load(self, path: str, device: Optional[str] = None):
        """Load detector from file"""
        with open(path, "rb") as f:
            data = pickle.load(f)

        # Load classifier
        self.classifier = data["classifier"]
        self.classifier_type = data["classifier_type"]
        self.is_trained = data.get("is_trained", True)

        # Recreate backbone if different from current
        loaded_backbone_name = data["backbone_name"]
        if loaded_backbone_name != self.backbone_name:
            print(f"Switching backbone from {self.backbone_name} to {loaded_backbone_name}")
            self.backbone_name = loaded_backbone_name
            self.backbone = create_backbone(loaded_backbone_name, device=device or str(self.device))

        # Load backbone state if saved
        if "backbone_state_dict" in data:
            self.backbone.load_state_dict(data["backbone_state_dict"])

        print(f"Model loaded from {path}")




    def get_info(self) -> Dict[str, Any]:
        """Get comprehensive detector information"""
        info = {
            "detector_type": "AIImageDetector",
            "backbone_name": self.backbone_name,
            "backbone_info": self.backbone.get_info(),
            "classifier_type": self.classifier_type,
            "is_trained": self.is_trained,
            "device": str(self.device),
        }

        if self.is_trained and hasattr(self.classifier, 'coef_'):
            info["classifier_info"] = {
                "n_features": len(self.classifier.coef_[0]),
                "n_support_vectors": getattr(self.classifier, 'n_support_', None)
            }

        return info




    def switch_backbone(self, new_backbone_name: str, **kwargs):
        """
        Switch to a different backbone

        Args:
            new_backbone_name: Name of new backbone
            **kwargs: Additional backbone arguments
        """
        print(f"Switching backbone from {self.backbone_name} to {new_backbone_name}")

        self.backbone_name = new_backbone_name
        self.backbone = create_backbone(new_backbone_name, device=str(self.device), **kwargs)

        # Reset training status
        self.is_trained = False
        self.classifier = None

        print(f"Backbone switched. New feature dimension: {self.backbone.feature_dim}")
        # print("Note: Classifier needs to be retrained with new backbone.")



# -------------------------------------------------------------------------
# Helpers for loading/ saving models
# -------------------------------------------------------------------------


# Convenience functions for backward compatibility and easy usage
def create_detector(backbone_name: str = "clip_vit_b32", **kwargs) -> AIImageDetector:
    """Create detector with specified backbone"""
    return AIImageDetector(backbone_name=backbone_name, **kwargs)



def load_neural_detector(model_path: str, device: Optional[str] = None) -> AIImageDetector:
    """Load a neural network detector from .pth file"""
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Extract config
    config = checkpoint.get('config', {})
    backbone_name = config.get('backbone_name', 'dinov2_vitb14')
    frequency_backbone = config.get('frequency_backbone', 'forensic')
    forensic_config = config.get('forensic_config', {})

    # If forensic_config is empty, try check model architecture and checks the actual shape of the saved weights
    # If there are 32 input channels → SRM only
    # If there are 224 input channels → SRM + DCT
    if not forensic_config:
        # Check the shape of conv layers to determine configuration
        state_dict = checkpoint['model_state_dict']
        if 'forensic_backbone.conv_layers.0.weight' in state_dict:
            conv_weight_shape = state_dict['forensic_backbone.conv_layers.0.weight'].shape
            num_input_channels = conv_weight_shape[1]

            # Infer configuration from number of input channels
            # 32 = SRM only, 192 = DCT only, 224 = SRM+DCT
            if num_input_channels == 32:
                forensic_config = {'use_srm': True, 'use_dct': False}
            elif num_input_channels == 192:
                forensic_config = {'use_srm': False, 'use_dct': True}
            elif num_input_channels == 224:
                forensic_config = {'use_srm': True, 'use_dct': True}

            print(f"Inferred forensic config from model architecture: {num_input_channels} input channels")

    print(f"Loading NEURAL (MLP) DUAL-BRANCH detector:")
    print(f"  Semantic: {backbone_name}")
    print(f"  Forensic: {frequency_backbone}")
    print(f"  Forensic config: SRM={forensic_config.get('use_srm', True)}, "
          f"DCT={forensic_config.get('use_dct', True)}")

    # Create detector with same architecture
    detector = AIImageDetector(
        backbone_name=backbone_name,
        neural_mode=True,
        use_frequency=True,
        frequency_backbone=frequency_backbone,
        frequency_kwargs=forensic_config,
        freeze_backbone=False,
        device=device
    )

    # Load weights
    detector.load_state_dict(checkpoint['model_state_dict'])
    detector.eval()

    # Mark as trained
    detector.is_trained = True

    return detector


def load_detector(model_path: str, device: Optional[str] = None) -> AIImageDetector:
    """Load detector from file with automatic type detection (.pth for neural, .pkl for classical)"""
    model_path_str = str(model_path)

    # Check file extension to determine model type
    if model_path_str.endswith('.pth'):
        # Neural network model
        return load_neural_detector(model_path, device)
    elif model_path_str.endswith('.pkl'):
        # Classical ML model (SVM/Logistic)
        temp_detector = AIImageDetector()
        temp_detector.load(model_path, device)
        return temp_detector
    else:
        # Try to detect based on file content
        try:
            return load_neural_detector(model_path, device)
        except:
            temp_detector = AIImageDetector()
            temp_detector.load(model_path, device)
            return temp_detector


def load_pretrained_detector(model_path: str, device: Optional[str] = None) -> AIImageDetector:
    """Alias for load_detector, kept for compatibility with evaluate.py and __init__.py"""
    return load_detector(model_path, device)


# alias for evaluate.py compatibility
def list_available_backbones():
    """List all available backbones"""
    return list_backbone_names()


def compare_backbones():
    """Compare all available backbones"""
    from backbones import compare_backbones as compare_bb
    return compare_bb()


# -------------------------------------------------------------------------
# Legacy wrappers for backward compatibility
# -------------------------------------------------------------------------

class CLIPDetector(AIImageDetector):
    """Legacy CLIP detector for backward compatibility"""
    def __init__(self, model_name: str = "ViT-B/32", **kwargs):
        # Map CLIP model names to backbone names
        clip_mapping = {
            "ViT-B/32": "clip_vit_b32",
            "ViT-B/16": "clip_vit_b16",
            "ViT-L/14": "clip_vit_l14",
            "RN50": "clip_rn50"
        }
        backbone_name = clip_mapping.get(model_name, "clip_vit_b32")
        super().__init__(backbone_name=backbone_name, **kwargs)


class DINOv2Detector(AIImageDetector):
    """Legacy DINOv2 detector for backward compatibility"""
    def __init__(self, model_name: str = "dinov2_vitb14", **kwargs):
        super().__init__(backbone_name=model_name, **kwargs)


class CLIPFeatureExtractor(nn.Module):
    """Extract features using CLIP vision encoder."""
    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: Optional[str] = None,
        use_penultimate: bool = True,
        jit: bool = False,  # must be False to allow forward hooks
    ):
        """
        Args:
            model_name: CLIP model variant ('ViT-B/32', 'ViT-L/14', etc.)
            device: Device to run on (e.g. 'cuda', 'cuda:0', 'cpu')
            use_penultimate: Use next-to-last layer features (often better for detection)
            jit: Set to False so we can register forward hooks
        """
        super().__init__()

        self.device = _get_device(device)
        self.model_name = model_name
        self.use_penultimate = use_penultimate

        # Load CLIP model (jit=False so hooks work)
        print(f"Loading CLIP model: {model_name} (jit={jit}) on {self.device}")
        self.model, self.preprocess = clip.load(model_name, device=str(self.device), jit=jit)
        self.model.eval()

        self.features = None  # populated by hook when use_penultimate=True
        if use_penultimate:
            self._register_hook()




    def _register_hook(self):
        """Register hook to extract penultimate layer features."""
        def hook_fn(module, input, output):
            self.features = output

        # Safety: make sure visual has expected attrs
        if not hasattr(self.model, "visual"):
            raise RuntimeError("CLIP model has no .visual module; cannot register hook.")

        if "ViT" in self.model_name:
            # Vision Transformer
            blocks = self.model.visual.transformer.resblocks
            n_layers = len(blocks)
            if n_layers < 2:
                raise RuntimeError(f"Expected at least 2 transformer blocks, found {n_layers}.")
            # hook second-to-last block output
            blocks[n_layers - 2].register_forward_hook(hook_fn)
        else:
            # ResNet backbone: hook last stage (before pooling/proj)
            if not hasattr(self.model.visual, "layer4"):
                raise RuntimeError("Expected ResNet-style visual.layer4 to exist.")
            self.model.visual.layer4.register_forward_hook(hook_fn)




    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract features from already-preprocessed images tensor (N,3,H,W).
        Returns a 2D tensor (N, D).
        """
        images = images.to(self.device)

        if self.use_penultimate:
            # reset to avoid stale values
            self.features = None
            _ = self.model.encode_image(images)  # triggers hook

            if self.features is None:
                raise RuntimeError("Penultimate feature hook did not fire; check hook registration.")

            features = self.features

            # shapes:
            # - ViT block output: (N, seq_len, width) -> mean pool over tokens
            # - ResNet layer4 output: (N, C, H, W) -> global average pool to (N, C)
            if features.dim() == 3:
                features = features.mean(dim=1)  # (N, D)
            elif features.dim() == 4:
                features = features.mean(dim=(2, 3))  # GAP -> (N, C)
            else:
                raise RuntimeError(f"Unexpected feature dims from hook: {features.shape}")
        else:
            # Use CLIP's projected image embeddings (already pooled)
            features = self.model.encode_image(images)  # (N, D)

        # (optional) L2 normalize for stability across backbones
        features = torch.nn.functional.normalize(features, dim=1)

        return features.detach().cpu()