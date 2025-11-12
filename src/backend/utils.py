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

# CONFIG
from .config import config

IMAGES_DIR = str(config.IMAGES_DIR)
IMAGES_CSV = config.IMAGES_CSV
DETECTOR_CSV = config.DETECTOR_CSV
DB_PATH = str(config.DB_PATH)
NUM_TRIALS = config.NUM_TRIALS

# ERROR HANDLING 
def safe_db_operation(func):
    """Decorator for database operations"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Database error in {func.__name__}: {e}")
            raise
    return wrapper

# DB / Engine
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

            -- Pre-survey data
            pre_confidence TEXT,
            pre_training TEXT,

            -- Background information
            user_type TEXT,
            years_experience TEXT,
            art_mediums TEXT,
            ai_familiarity TEXT,
            ai_frequency TEXT,

            -- Detection experience
            difficulty TEXT,
            visual_cues TEXT,
            hardest_styles TEXT,

            -- Platform experience & concerns
            labeling_importance TEXT,
            encountered_unlabeled TEXT,
            concerns TEXT,
            detection_value TEXT,
            visibility_impact TEXT,

            -- Reflection
            emotions TEXT,
            additional_comments TEXT
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
            detector_confidence REAL,
            reasoning TEXT,
            generator_model TEXT,
            art_style TEXT,
            order_shown INTEGER
        )"""))

        # Migration: Add new columns if they don't exist
        new_columns = [
            "reasoning TEXT",
            "generator_model TEXT",
            "art_style TEXT",
            "order_shown INTEGER"
        ]

        for column in new_columns:
            try:
                conn.execute(text(f"ALTER TABLE votes ADD COLUMN {column}"))
            except Exception:
                # Column already exists, ignore the error
                pass
    return engine

# Data loaders
def load_images_meta():
    if not os.path.exists(IMAGES_CSV):
        # return empty frame with new columns including generator_model and art_style
        return pd.DataFrame(columns=["image_id","image_filename","true_label","generator_model","art_style","subfolder","original_filename"])
    df = pd.read_csv(IMAGES_CSV, dtype=str)

    # Handle backward compatibility with old metadata format
    if "generator_model" not in df.columns:
        df["generator_model"] = df["true_label"].map({"human": "human", "ai": "unknown"})
    if "art_style" not in df.columns:
        df["art_style"] = "unknown"
    if "subfolder" not in df.columns:
        df["subfolder"] = "unknown"
    if "original_filename" not in df.columns:
        df["original_filename"] = df["image_filename"]

    return df

def load_detector_preds():
    if not os.path.exists(DETECTOR_CSV):
        return pd.DataFrame(columns=["image_id","prob_ai","label_ai","saliency_path"]).set_index("image_id")
    df = pd.read_csv(DETECTOR_CSV, dtype=str)
    if "prob_ai" in df.columns:
        df["prob_ai"] = df["prob_ai"].astype(float)
    return df.set_index("image_id")

# Utilities
def now_utc_iso():
    return datetime.now(timezone.utc).isoformat()

def make_pid():
    return os.urandom(6).hex()

@safe_db_operation
def register_participant(engine, pid, info: dict):
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT OR REPLACE INTO participants
            (participant_id, started_at,
             pre_confidence, pre_training, user_type, years_experience, art_mediums,
             ai_familiarity, ai_frequency, difficulty, visual_cues, hardest_styles,
             labeling_importance, encountered_unlabeled, concerns, detection_value,
             visibility_impact, emotions, additional_comments)
            VALUES (:participant_id, :started_at,
                    :pre_confidence, :pre_training, :user_type, :years_experience, :art_mediums,
                    :ai_familiarity, :ai_frequency, :difficulty, :visual_cues, :hardest_styles,
                    :labeling_importance, :encountered_unlabeled, :concerns, :detection_value,
                    :visibility_impact, :emotions, :additional_comments)
        """), [{
            "participant_id": pid,
            "started_at": now_utc_iso(),

            # Comprehensive survey fields
            "pre_confidence": info.get("pre_confidence"),
            "pre_training": info.get("pre_training"),
            "user_type": info.get("user_type"),
            "years_experience": info.get("years_experience"),
            "art_mediums": ",".join(info.get("art_mediums", [])) if info.get("art_mediums") else "",
            "ai_familiarity": info.get("ai_familiarity"),
            "ai_frequency": info.get("ai_frequency"),
            "difficulty": info.get("difficulty"),
            "visual_cues": ",".join(info.get("visual_cues", [])) if info.get("visual_cues") else "",
            "hardest_styles": ",".join(info.get("hardest_styles", [])) if info.get("hardest_styles") else "",
            "labeling_importance": info.get("labeling_importance"),
            "encountered_unlabeled": info.get("encountered_unlabeled"),
            "concerns": ",".join(info.get("concerns", [])) if info.get("concerns") else "",
            "detection_value": info.get("detection_value"),
            "visibility_impact": info.get("visibility_impact"),
            "emotions": ",".join(info.get("emotions", [])) if info.get("emotions") else "",
            "additional_comments": info.get("additional_comments")
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
            INSERT INTO votes (participant_id, image_id, true_label, human_choice, confidence, response_time_ms, timestamp_utc, detector_pred, detector_confidence, reasoning, generator_model, art_style, order_shown)
            VALUES (:participant_id, :image_id, :true_label, :human_choice, :confidence, :response_time_ms, :timestamp_utc, :detector_pred, :detector_confidence, :reasoning, :generator_model, :art_style, :order_shown)
        """), [rec])

def get_trial_images(images_meta, difficulty_mode="mixed", num_trials=None):
    """Get random selection of images for trial"""
    if num_trials is None:
        num_trials = NUM_TRIALS

    # Always return random selection of images
    return images_meta.sample(frac=1).head(num_trials)
