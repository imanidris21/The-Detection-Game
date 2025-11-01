# utils.py
import os
import logging
from datetime import datetime, timezone
import pandas as pd
from sqlalchemy import create_engine, text
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ========== CONFIG ==========
from config import config

IMAGES_DIR = str(config.IMAGES_DIR)
IMAGES_CSV = config.IMAGES_CSV
DETECTOR_CSV = config.DETECTOR_CSV
DB_PATH = str(config.DB_PATH)
NUM_TRIALS = config.NUM_TRIALS
ADMIN_SECRET = config.ADMIN_SECRET

# ========== ERROR HANDLING ==========
def safe_db_operation(func):
    """Decorator for database operations"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Database error in {func.__name__}: {e}")
            raise
    return wrapper

# ========== DB / Engine ==========
def get_engine():
    return create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})

@safe_db_operation
def init_db():
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS participants (
            participant_id TEXT PRIMARY KEY,
            started_at TEXT,
            finished_at TEXT,
            device_type TEXT,
            user_group TEXT,
            discipline TEXT,
            years_experience TEXT,
            confidence_self TEXT,
            seen_training TEXT,
            cues TEXT,
            difficulty_mode TEXT
        )"""))
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS votes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            participant_id TEXT,
            image_id TEXT,
            true_label TEXT,
            human_choice TEXT,
            confidence REAL,
            response_time_ms INTEGER,
            timestamp_utc TEXT,
            detector_pred TEXT,
            detector_confidence REAL
        )"""))
    return engine

# ========== Data loaders ==========
def load_images_meta():
    if not os.path.exists(IMAGES_CSV):
        # return empty frame with columns
        return pd.DataFrame(columns=["image_id","image_filename","true_label","style"])
    return pd.read_csv(IMAGES_CSV, dtype=str)

def load_detector_preds():
    if not os.path.exists(DETECTOR_CSV):
        return pd.DataFrame(columns=["image_id","prob_ai","label_ai","saliency_path"]).set_index("image_id")
    df = pd.read_csv(DETECTOR_CSV, dtype=str)
    if "prob_ai" in df.columns:
        df["prob_ai"] = df["prob_ai"].astype(float)
    return df.set_index("image_id")

# ========== Utilities ==========
def now_utc_iso():
    return datetime.now(timezone.utc).isoformat()

def make_pid():
    return os.urandom(6).hex()

@safe_db_operation
def register_participant(engine, pid, info: dict):
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT OR REPLACE INTO participants
            (participant_id, started_at, device_type, user_group, discipline, years_experience, confidence_self, seen_training, cues, difficulty_mode)
            VALUES (:participant_id, :started_at, :device_type, :user_group, :discipline, :years_experience, :confidence_self, :seen_training, :cues, :difficulty_mode)
        """), [{
            "participant_id": pid,
            "started_at": now_utc_iso(),
            "device_type": info.get("device_type"),
            "user_group": info.get("user_group"),
            "discipline": info.get("discipline"),
            "years_experience": info.get("years_experience"),
            "confidence_self": info.get("confidence_self"),
            "seen_training": info.get("seen_training"),
            "cues": ",".join(info.get("cues", [])) if info.get("cues") else "",
            "difficulty_mode": info.get("difficulty_mode", "mixed")
        }])

@safe_db_operation
def mark_finished(engine, pid):
    with engine.begin() as conn:
        conn.execute(text("UPDATE participants SET finished_at = :finished_at WHERE participant_id = :participant_id"),
                     [{"finished_at": now_utc_iso(), "participant_id": pid}])

@safe_db_operation
def save_vote(engine, rec: dict):
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO votes (participant_id, image_id, true_label, human_choice, confidence, response_time_ms, timestamp_utc, detector_pred, detector_confidence)
            VALUES (:participant_id, :image_id, :true_label, :human_choice, :confidence, :response_time_ms, :timestamp_utc, :detector_pred, :detector_confidence)
        """), [rec])

# ========== DIFFICULTY MODES ==========
def categorize_difficulty(detector_confidence, human_agreement_rate):
    """Categorize images by difficulty based on detector confidence and human performance"""
    if detector_confidence > 0.8 and human_agreement_rate > 0.7:
        return "easy"
    elif detector_confidence < 0.6 or human_agreement_rate < 0.4:
        return "hard"
    else:
        return "medium"

def get_trial_images(images_meta, difficulty_mode="mixed", num_trials=None):
    """Get images for trial based on difficulty mode"""
    if num_trials is None:
        num_trials = NUM_TRIALS

    if not config.ENABLE_DIFFICULTY_MODES or difficulty_mode == "mixed":
        # Random selection from all images
        return images_meta.sample(frac=1, random_state=42).head(num_trials)

    if difficulty_mode == "easy":
        if 'difficulty' in images_meta.columns:
            filtered = images_meta[images_meta['difficulty'] == 'easy']
        else:
            # Fallback: assume images with high detector confidence are easier
            filtered = images_meta.sample(frac=1, random_state=42)
        return filtered.head(num_trials)

    elif difficulty_mode == "hard":
        if 'difficulty' in images_meta.columns:
            filtered = images_meta[images_meta['difficulty'] == 'hard']
        else:
            # Fallback: random selection
            filtered = images_meta.sample(frac=1, random_state=42)
        return filtered.head(num_trials)

    elif difficulty_mode == "medium":
        if 'difficulty' in images_meta.columns:
            filtered = images_meta[images_meta['difficulty'] == 'medium']
        else:
            # Fallback: random selection
            filtered = images_meta.sample(frac=1, random_state=42)
        return filtered.head(num_trials)

    elif difficulty_mode == "adaptive":
        # Start with easy images, will be adjusted during the test
        return get_trial_images(images_meta, "easy", num_trials)

    else:
        # Default to mixed
        return images_meta.sample(frac=1, random_state=42).head(num_trials)

def update_difficulty_based_on_performance(current_accuracy, current_difficulty="medium"):
    """Adjust difficulty based on user performance (for adaptive mode)"""
    if current_accuracy > 0.8:
        return "hard"
    elif current_accuracy < 0.5:
        return "easy"
    else:
        return "medium"
