# src/components/navbar.py

# Simple navigation bar component for Human vs AI Detection Experiment

import streamlit as st

# Map labels to Streamlit page targets (filenames must match your files)
PAGES = {
    "Home": "Home.py",
    "Take the Test": "pages/1_Take_the_Test.py",
    "About": "pages/2_About.py",
    "Leaderboard": "pages/3_Leaderboard.py",
    "Admin": "pages/4_Admin.py",
}


def top_nav(active_label: str, show_progress: bool = False, progress_text: str = ""):
    """
    Simple top navigation with title and working buttons
    """
    # Add navigation styling
    st.markdown("""
    <style>
    .nav-header {
        background: transparent;
        padding: 1rem;
        margin: -1rem -1rem 1rem -1rem;
        color: #333;
        text-align: center;
        border-radius: 0;
        border-bottom: none;
        box-shadow: none;
    }
    .nav-title {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 700;
        color: #495057;
    }
    .nav-progress {
        font-size: 0.9rem;
        margin-top: 0.5rem;
        opacity: 0.8;
        color: #6c757d;
    }
    .stButton > button {
        background: white;
        color: #495057;
        border: 2px solid #dee2e6;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: #f8f9fa;
        border-color: #adb5bd;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }

    </style>
    """, unsafe_allow_html=True)

    # Header with title
    progress_html = f'<div class="nav-progress">{progress_text}</div>' if show_progress and progress_text else ''
    st.markdown(f"""
    <div class="nav-header">
        <h1 class="nav-title">The Art Detection Game</h1>
        {progress_html}
    </div>
    """, unsafe_allow_html=True)

    # Navigation buttons (no heading)
    cols = st.columns(len(PAGES))
    for i, (label, target) in enumerate(PAGES.items()):
        with cols[i]:
            button_label = f"{'● ' if label == active_label else ''}{label}"
            if st.button(button_label, key=f"nav_{label}", use_container_width=True):
                try:
                    st.switch_page(target)
                except Exception:
                    st.session_state["__pending_page__"] = target
                    st.rerun()

def handle_fallback_switch():
    """Handle navigation fallback for older Streamlit versions"""
    target = st.session_state.pop("__pending_page__", None)
    if target:
        st.info(f"Navigate to: {target} (Please update Streamlit for automatic routing)")

def show_test_progress(current: int, total: int):
    """Helper function to format test progress"""
    percentage = (current / total) * 100 if total > 0 else 0
    return f"Progress: {current}/{total} ({percentage:.0f}%)"
