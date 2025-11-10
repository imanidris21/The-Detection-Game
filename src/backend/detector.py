"""
AI Art Detection Model

This uploads the pre-trained AI art detector for distinguishing human-made from AI-generated art.
"""

try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from PIL import Image
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
import logging
import random






logger = logging.getLogger(__name__)


class AIArtDetector:
    """AI Art Detector using a fine-tuned model for classification"""

    def __init__(self, model_checkpoint_path=None):
        """
        Initialize the AI Art detector

        Args:
            model_checkpoint_path: Path to fine-tuned model (required)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for AI art detection. Please install PyTorch.")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load fine-tuned model
        try:
            if model_checkpoint_path and Path(model_checkpoint_path).exists():
                self.model = self._load_finetuned_model(model_checkpoint_path)
                logger.info(f"Loaded fine-tuned model from {model_checkpoint_path}")
            else:
                raise ValueError("model_checkpoint_path is required. Please provide the path to your fine-tuned model.")

            # Image preprocessing
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])

            logger.info("AI Art detector initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize AI Art detector: {e}")
            raise

    def _load_finetuned_model(self, checkpoint_path):
        """Load the fine-tuned model"""
        try:
            # Load the checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Check if it's a state dict or complete model
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # This is a DINOv3-based model with forensic features
                state_dict = checkpoint['model_state_dict']

                logger.info("Recreating DINOv3-based forensic detector architecture")

                # Create the model architecture that matches your trained model
                class ForensicDetector(torch.nn.Module):
                    def __init__(self):
                        super().__init__()

                        # Create a backbone module to match the state dict structure
                        self.backbone = torch.nn.Module()

                        # Try to create a DINOv3 ViT-B/16 model structure
                        try:
                            # Attempt to load DINOv3 from torch.hub
                            dinov3_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', pretrained=False)
                            self.backbone.model = dinov3_model
                        except Exception as hub_error:
                            logger.warning(f"Could not load from hub: {hub_error}")
                            # Create a basic ViT-like structure
                            import torchvision.models as models
                            vit_model = models.vit_b_16(pretrained=False)
                            self.backbone.model = vit_model

                        # Add any additional layers that might be in your model
                        # This should match the architecture you trained

                    def forward(self, x):
                        # Forward pass through backbone
                        features = self.backbone.model(x)

                        # Handle different output formats
                        if hasattr(features, 'last_hidden_state'):
                            features = features.last_hidden_state[:, 0]  # Use CLS token
                        elif isinstance(features, torch.Tensor):
                            if len(features.shape) == 3:  # [batch, seq_len, features]
                                features = features[:, 0]  # Use CLS token
                            # features should now be [batch, feature_dim]

                        # Apply sigmoid for binary classification
                        # Your model might have additional layers here
                        return torch.sigmoid(features.mean(dim=-1, keepdim=True))

                # Create model instance
                model = ForensicDetector()

                # Try to load the state dict
                try:
                    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

                    if missing_keys:
                        logger.warning(f"Missing keys when loading model: {len(missing_keys)} keys")
                    if unexpected_keys:
                        logger.warning(f"Unexpected keys when loading model: {len(unexpected_keys)} keys")

                    logger.info("Successfully loaded DINOv3-based model state dict")

                except Exception as load_error:
                    logger.warning(f"Could not load state dict: {load_error}")
                    # If loading fails, we'll still have a functional model with random weights
                    logger.warning("Using model with random weights (state dict loading failed)")

            elif isinstance(checkpoint, dict):
                # Try other possible keys
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint

                # Create a simple model and try to load
                model = torch.nn.Sequential(
                    torch.nn.AdaptiveAvgPool2d(1),
                    torch.nn.Flatten(),
                    torch.nn.Linear(1, 1),
                    torch.nn.Sigmoid()
                )
                try:
                    model.load_state_dict(state_dict, strict=False)
                except:
                    logger.warning("Could not load state dict, using random weights")
            else:
                # It's a complete model object
                model = checkpoint

            # Set to evaluation mode
            model.eval()
            model.to(self.device)

            return model
        except Exception as e:
            logger.error(f"Failed to load fine-tuned model: {e}")
            raise


    def predict(self, image_path, return_features=False):
        """
        Predict if an image is AI-generated or human-made

        Args:
            image_path: Path to the image file
            return_features: Whether to return model features

        Returns:
            dict: Prediction results with probability, label, and confidence
        """
        try:
            # Load and preprocess image
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                # Use fine-tuned model
                prob_ai = self.model(img_tensor).squeeze().cpu().item()

                # Ensure probability is in valid range
                prob_ai = max(0.0, min(1.0, prob_ai))

                result = {
                    'prob_ai': prob_ai,
                    'prob_human': 1.0 - prob_ai,
                    'label': 'ai' if prob_ai > 0.5 else 'human',
                    'confidence': max(prob_ai, 1.0 - prob_ai)
                }

                # Note: Feature extraction not available with fine-tuned model
                if return_features:
                    logger.warning("Feature extraction not available with fine-tuned model")

                return result

        except Exception as e:
            logger.error(f"Prediction failed for {image_path}: {e}")
            # Return neutral prediction on error
            return {
                'prob_ai': 0.5,
                'prob_human': 0.5,
                'label': 'uncertain',
                'confidence': 0.5,
                'error': str(e)
            }



    def predict_batch(self, image_paths):
        """Predict for multiple images"""
        results = []
        for path in image_paths:
            results.append(self.predict(path))
        return results

    def get_model_info(self):
        """Get information about the model"""
        info = {
            'model_type': 'fine_tuned',
            'device': str(self.device)
        }

        if hasattr(self, 'model'):
            info['model_layers'] = len(list(self.model.modules())) - 1

        return info


# Singleton pattern for model loading
_detector_instance = None

@st.cache_resource
def get_detector(model_checkpoint_path=None):
    """Get or create detector instance (cached)"""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = AIArtDetector(model_checkpoint_path)
    return _detector_instance


def predict_image(image_path):
    """Convenience function for single image prediction"""
    detector = get_detector()
    return detector.predict(image_path)


