"""
Configuration settings for the AI Art Detection Platform

This module centralizes all configuration settings, making it easy to manage
different environments (development, testing, production) and feature flags.
"""

import os
from pathlib import Path

class Config:
    """Main configuration class with environment-based settings"""

    # Base paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    IMAGES_DIR = DATA_DIR / "images_resized"
    UPLOAD_DIR = DATA_DIR / "uploads"
    MODELS_DIR = BASE_DIR / "models"

    # Fallback to current directory structure if data dir doesn't exist
    if not IMAGES_DIR.exists():
        IMAGES_DIR = BASE_DIR / "images_resized"

    # Database configuration
    DB_PATH = DATA_DIR / "results.db" if DATA_DIR.exists() else BASE_DIR / "results.db"

    # Experiment settings
    NUM_TRIALS = int(os.getenv("NUM_TRIALS", "40"))
    ADMIN_SECRET = os.getenv("ADMIN_SECRET", "letmein_admin")

    # Model settings
    MODEL_CHECKPOINT = os.getenv("MODEL_CHECKPOINT", str(MODELS_DIR / "dinov3_finetuned.pth"))
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))

    # Data files
    IMAGES_CSV = "images_metadata.csv"
    DETECTOR_CSV = "detector_preds.csv"

    # Feature flags
    USE_GLAZE_DETECTION = os.getenv("USE_GLAZE_DETECTION", "false").lower() == "true"
    ENABLE_DIFFICULTY_MODES = os.getenv("ENABLE_DIFFICULTY_MODES", "true").lower() == "true"
    ENABLE_REAL_TIME_FEEDBACK = os.getenv("ENABLE_REAL_TIME_FEEDBACK", "false").lower() == "true"
    ENABLE_AB_TESTING = os.getenv("ENABLE_AB_TESTING", "false").lower() == "true"

    # UI settings
    MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10"))
    SHOW_PROGRESS_BAR = os.getenv("SHOW_PROGRESS_BAR", "true").lower() == "true"

    # Performance settings
    CACHE_TTL_HOURS = int(os.getenv("CACHE_TTL_HOURS", "1"))
    PRELOAD_IMAGES = os.getenv("PRELOAD_IMAGES", "false").lower() == "true"

    # Logging settings
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_TO_FILE = os.getenv("LOG_TO_FILE", "true").lower() == "true"
    LOG_FILE_PATH = os.getenv("LOG_FILE_PATH", "app.log")

    # Security settings
    ANONYMIZE_IPS = os.getenv("ANONYMIZE_IPS", "true").lower() == "true"
    SESSION_TIMEOUT_MINUTES = int(os.getenv("SESSION_TIMEOUT_MINUTES", "30"))

    # Difficulty mode configuration
    DIFFICULTY_MODES = {
        "easy": {
            "detector_confidence_min": 0.8,
            "human_agreement_min": 0.7,
            "description": "Clear cases where both AI and humans agree"
        },
        "medium": {
            "detector_confidence_min": 0.6,
            "detector_confidence_max": 0.8,
            "human_agreement_min": 0.4,
            "human_agreement_max": 0.7,
            "description": "Moderate difficulty cases"
        },
        "hard": {
            "detector_confidence_max": 0.6,
            "human_agreement_max": 0.4,
            "description": "Challenging cases where detection is difficult"
        },
        "mixed": {
            "description": "Random selection from all difficulty levels"
        },
        "adaptive": {
            "description": "Difficulty adjusts based on user performance"
        }
    }

    # A/B Testing experiments
    AB_EXPERIMENTS = {
        "ui_variant": {
            "control": "Standard UI",
            "enhanced_feedback": "Enhanced feedback UI"
        },
        "instruction_type": {
            "minimal": "Minimal instructions",
            "detailed": "Detailed instructions with examples"
        },
        "confidence_scale": {
            "slider": "Confidence slider (0-100)",
            "likert": "5-point Likert scale"
        }
    }

    @classmethod
    def get_db_url(cls):
        """Get database URL with proper formatting"""
        return f"sqlite:///{cls.DB_PATH}"

    @classmethod
    def ensure_directories(cls):
        """Ensure all required directories exist"""
        directories = [cls.DATA_DIR, cls.IMAGES_DIR, cls.UPLOAD_DIR, cls.MODELS_DIR]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def is_production(cls):
        """Check if running in production environment"""
        return os.getenv("ENVIRONMENT", "development").lower() == "production"

    @classmethod
    def get_log_config(cls):
        """Get logging configuration"""
        return {
            "level": cls.LOG_LEVEL,
            "to_file": cls.LOG_TO_FILE,
            "file_path": cls.LOG_FILE_PATH,
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }


# Development configuration
class DevelopmentConfig(Config):
    """Development-specific configuration"""
    NUM_TRIALS = 10
    ENABLE_REAL_TIME_FEEDBACK = True
    LOG_LEVEL = "DEBUG"


# Production configuration
class ProductionConfig(Config):
    """Production-specific configuration"""
    NUM_TRIALS = 100
    ADMIN_SECRET = os.getenv("ADMIN_SECRET", "CHANGE_ME_IN_PRODUCTION")
    ANONYMIZE_IPS = True
    LOG_LEVEL = "WARNING"


# Test configuration
class TestConfig(Config):
    """Testing-specific configuration"""
    NUM_TRIALS = 5
    DB_PATH = ":memory:"  # In-memory database for tests
    LOG_LEVEL = "DEBUG"


def get_config():
    """Get the appropriate configuration based on environment"""
    env = os.getenv("ENVIRONMENT", "development").lower()

    if env == "production":
        return ProductionConfig()
    elif env == "test":
        return TestConfig()
    else:
        return DevelopmentConfig()


# Global config instance
config = get_config()