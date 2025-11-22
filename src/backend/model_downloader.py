"""
AI Detection Models Downloader: Downloads large model files from Google Drive 
"""

import os
import streamlit as st
from pathlib import Path
import hashlib
import time
from typing import Optional, Dict

import gdown









def download_from_google_drive(file_id: str, destination: str, expected_size: Optional[int] = None, show_progress: bool = True) -> bool:
    """
    Download a file from Google Drive using gdown

    Args:
        file_id: Google Drive file ID
        destination: Local path to save the file
        expected_size: Expected file size in bytes (for validation)
        show_progress: Whether to show Streamlit progress bar

    Returns:
        bool: True if download successful, False otherwise
    """

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(destination), exist_ok=True)

    # Use gdown for Google Drive downloads
    if show_progress:
        progress_bar = st.progress(0, "Starting download with gdown...")

    try:
        url = f"https://drive.google.com/uc?id={file_id}"

        # Download the file
        if show_progress:
            progress_bar.progress(0.1, "Downloading from Google Drive...")

        success = gdown.download(url, destination, quiet=False)

        if show_progress:
            progress_bar.progress(0.9, "Validating download...")

        if success and os.path.exists(destination):
            # Validate file size if expected
            if expected_size:
                actual_size = os.path.getsize(destination)
                if abs(actual_size - expected_size) > (expected_size * 0.05):  # 5% tolerance
                    st.error(f"Downloaded file size mismatch. Expected: {expected_size}, Got: {actual_size}")
                    os.remove(destination)
                    return False

            if show_progress:
                progress_bar.progress(1.0, "Download complete!")

            return True
        else:
            if show_progress:
                st.error("gdown failed to download the file")
            return False

    except Exception as e:
        st.error(f"gdown download failed: {str(e)}")
        return False



def get_file_hash(file_path: str) -> str:
    """Get MD5 hash of file for integrity checking"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()



# Model configurations with Google Drive file IDs
MODEL_CONFIGS = {
    "neural_detector_dinov3_vitb16_forensic_best.pth": {
        "file_id": "1KsqO_ULB-KCU8QRjcCdB0oFRcemKIvLu",
        "size_mb": 363,
        "expected_size": 363 * 1024 * 1024,  # bytes
        "path": "models/neural_art_80k_dinov3B_SRM_DCT/neural_detector_dinov3_vitb16_forensic_best.pth"
    },
    "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth": {
        "file_id": "18ejBs90kXNhlJeLaKlCKLwJ7T4J6ZnVS",
        "size_mb": 343,
        "expected_size": 343 * 1024 * 1024,  # bytes
        "path": "src/backend/checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
    }
}



@st.cache_data  # cache permanently (until app restart)
def check_model_exists(model_name: str, project_root: str) -> bool:
    """Check if model file exists and has correct size"""
    if model_name not in MODEL_CONFIGS:
        return False

    config = MODEL_CONFIGS[model_name]
    model_path = os.path.join(project_root, config["path"])

    if not os.path.exists(model_path):
        return False

    # Check file size
    actual_size = os.path.getsize(model_path)
    expected_size = config["expected_size"]

    # Allow 1% tolerance for file size differences
    tolerance = expected_size * 0.01
    return abs(actual_size - expected_size) < tolerance



def ensure_model_downloaded(model_name: str, project_root: str, force_redownload: bool = False) -> str:
    """
    Ensure model is downloaded and available

    Args:
        model_name: Name of the model (key in MODEL_CONFIGS)
        project_root: Project root directory
        force_redownload: Force redownload even if file exists

    Returns:
        str: Path to the downloaded model file
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")

    config = MODEL_CONFIGS[model_name]
    model_path = os.path.join(project_root, config["path"])

    # Check if we need to download
    needs_download = (
        force_redownload or
        not check_model_exists(model_name, project_root)
    )

    if needs_download:
        file_id = config["file_id"]
        if not file_id:
            st.error(f"Google Drive file ID not configured for {model_name}")
            st.info("Please upload the model to Google Drive and update MODEL_CONFIGS with the file ID")
            st.stop()

        # Create placeholder for download messages that can be cleared
        download_placeholder = st.empty()

        with download_placeholder.container():
            st.info(f"Downloading {model_name} ({config['size_mb']}MB)...")
            

            # Download the model
            success = download_from_google_drive(
                file_id=file_id,
                destination=model_path,
                expected_size=config["expected_size"],
                show_progress=True
            )

            if not success:
                st.error(f"Failed to download {model_name}")
                st.stop()

            st.success(f"✅ {model_name} downloaded successfully!")

        # Clear messages after a brief delay
        import time
        time.sleep(2)  # Show success message for 2 seconds
        download_placeholder.empty()  # Clear all download messages

    return model_path



def update_model_config(model_name: str, file_id: str):
    """
    Update the Google Drive file ID for a model
    Note: In production, you'd store this in a config file
    """
    if model_name in MODEL_CONFIGS:
        MODEL_CONFIGS[model_name]["file_id"] = file_id
        st.success(f"Updated file ID for {model_name}")
    else:
        st.error(f"Unknown model: {model_name}")


# check if the modles are downloaded and used the cached version if available
def ensure_neural_detector_model(project_root: str) -> str:
    """Ensure the main neural detector model is available"""
    return ensure_model_downloaded(
        "neural_detector_dinov3_vitb16_forensic_best.pth",
        project_root
    )

def ensure_dinov3_pretrain_model(project_root: str) -> str:
    """Ensure the DINOv3 pretrained model is available"""
    return ensure_model_downloaded(
        "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
        project_root
    )