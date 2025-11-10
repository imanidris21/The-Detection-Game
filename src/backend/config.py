"""
Configuration settings for the AI Art Detection Platform 
"""

import os
from pathlib import Path

class Config:
    """Main configuration class with environment-based settings"""

    # Base paths
    BASE_DIR = Path(__file__).parent.parent.parent  # Go up from backend/config.py to project root
    DATA_DIR = BASE_DIR / "data"
    IMAGES_DIR = DATA_DIR / "art_testset"
    UPLOAD_DIR = DATA_DIR / "uploads"
    MODELS_DIR = BASE_DIR / "models"

    # Ensure essential directories exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Database configuration
    DB_PATH = DATA_DIR / "results.db"

    # Experiment settings
    NUM_TRIALS = int(os.getenv("NUM_TRIALS", "10"))

    # Data files
    IMAGES_CSV = DATA_DIR / "art_testset_metadata.csv"
    DETECTOR_CSV = DATA_DIR / "detector_preds.csv"

    # Feature flags
    ENABLE_REAL_TIME_FEEDBACK = os.getenv("ENABLE_REAL_TIME_FEEDBACK", "false").lower() == "true"


# Global config instance
config = Config()