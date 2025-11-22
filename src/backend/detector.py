"""
AI Art Detection Model

This uploads the AI art detector for distinguishing human-made from AI-generated art.
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
        """Load the fine-tuned neural detector using the actual training architecture"""
        logger.info(f"Loading fine-tuned model from {checkpoint_path}")

        import sys
        from pathlib import Path

        # Add the project src to path to import training classes
        project_root = Path(checkpoint_path).parent.parent.parent
        src_path = project_root / "src"
        if src_path.exists() and str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))

        # Import the actual loading function - NO FALLBACKS ALLOWED
        from backend.model import load_neural_detector

        logger.info("Successfully imported load_neural_detector from training code")

        # Load the model using the ONLY correct method
        detector = load_neural_detector(checkpoint_path, device=str(self.device))

        logger.info("Successfully loaded trained neural detector with proper architecture")

        # Mark as trained and set evaluation mode
        detector.is_trained = True
        detector.eval()
        detector.to(self.device)

        logger.info("Model ready for inference")
        return detector


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
                # Use fine-tuned model - output is logits, need sigmoid for probability
                logits = self.model(img_tensor).squeeze()
                prob_ai = torch.sigmoid(logits).cpu().item()

                # Ensure probability is in valid range (should be automatic with sigmoid)
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


