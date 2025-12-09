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

    # Environment detection - Updated for Streamlit Cloud
    IS_STREAMLIT_CLOUD = (
        os.getenv("STREAMLIT_CLOUD", "false").lower() == "true" or
        os.getenv("HOME") == "/home/appuser" or  # New Streamlit Cloud HOME
        os.getenv("HOME") == "/mount/src" or     # Legacy Streamlit Cloud HOME
        os.path.exists("/mount/src") or          # Check if running in Streamlit Cloud filesystem
        "appuser" in os.getenv("HOME", "")       # Additional check for Streamlit Cloud user
    )
    IS_LOCAL_DEV = not IS_STREAMLIT_CLOUD

    # Experiment settings
    NUM_TRIALS = int(os.getenv("NUM_TRIALS", "5"))

    # Data files
    IMAGES_CSV = DATA_DIR / "art_testset_metadata.csv"
    DETECTOR_CSV = DATA_DIR / "detector_preds.csv"

    # Feature flags
    ENABLE_REAL_TIME_FEEDBACK = os.getenv("ENABLE_REAL_TIME_FEEDBACK", "false").lower() == "true"


# Global config instance
config = Config()