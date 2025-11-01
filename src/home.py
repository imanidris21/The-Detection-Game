# Home.py
# streamlit run src/home.py
import streamlit as st
from utils import init_db, load_images_meta
from components.navbar import top_nav, handle_fallback_switch

st.set_page_config(page_title="Human vs AI — Home", layout="wide", initial_sidebar_state="collapsed")


# Initialize database and load data
init_db()
top_nav("Home")
handle_fallback_switch()

# Main content
st.title("Detecting the Diffused — Human or Machine?")

st.markdown("""
### Welcome to the Human vs AI Detection Experiment

This project compares human ability to identify AI-generated artworks with an automated detector.
Participate in our study to test your skills against machine learning algorithms!

**How it works:**
1. Take the test to classify artworks as human-made or AI-generated
2. View your results compared to our AI detector
3. See how you rank against other participants on the leaderboard

Use the navigation bar above to get started.
""")

# Display dataset information
meta = load_images_meta()
if not meta.empty:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Images", len(meta))

    with col2:
        human_count = len(meta[meta['true_label'] == 'human'])
        st.metric("Human Artworks", human_count)

    with col3:
        ai_count = len(meta[meta['true_label'] == 'ai'])
        st.metric("AI Artworks", ai_count)

    # Show style breakdown if available
    if 'style' in meta.columns:
        st.markdown("### Dataset Breakdown by Style")
        style_counts = meta['style'].value_counts()
        st.bar_chart(style_counts)

else:
    st.warning("No image metadata found. Please ensure images_metadata.csv is properly configured.")
