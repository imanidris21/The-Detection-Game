# pages/2_About.py
import streamlit as st

from components.navbar import top_nav, handle_fallback_switch
from datetime import datetime, timezone

st.set_page_config(page_title="About", layout="wide", initial_sidebar_state="collapsed")
top_nav("About")
handle_fallback_switch()


st.title("About / FAQ")
st.markdown("""
This experiment compares human ability to identify AI-generated artworks with an automated detector.

**What we collect:** anonymous participant id, questionnaire responses, per-image choice, confidence, response time.

**How images are handled:** images are anonymised thumbnails, metadata stripped. If you want to withdraw, provide your participant id and contact the researcher.

**FAQ**
- Q: How many images? A: Controlled by NUM_TRIALS (utils.py).
- Q: Can I stop anytime? A: Yes; completed trials are recorded.
""")
