# pages/4_Admin.py
import streamlit as st


from components.navbar import top_nav, handle_fallback_switch
from datetime import datetime, timezone

from utils import init_db, get_engine, load_images_meta, load_detector_preds, ADMIN_SECRET
import pandas as pd



st.set_page_config(page_title="Admin", layout="wide", initial_sidebar_state="collapsed")
top_nav("Admin")
handle_fallback_switch()

init_db()
engine = get_engine()
images_meta = load_images_meta()
detector_preds = load_detector_preds()

st.title("Admin")

secret = st.text_input("Admin secret", type="password")
if secret != ADMIN_SECRET:
    st.warning("Enter admin secret to proceed (set EXPRIMENT_ADMIN_SECRET env var in production).")
    st.stop()

with engine.begin() as conn:
    votes = pd.read_sql("SELECT * FROM votes", conn)
    participants = pd.read_sql("SELECT * FROM participants", conn)

st.write(f"Total votes: {len(votes)} — Participants: {len(participants)}")

if not votes.empty:
    merged = votes.merge(images_meta, on="image_id", how="left")
    merged["correct"] = merged["human_choice"] == merged["true_label"]
    style_acc = merged.groupby("style").agg(total=("correct","size"), acc=("correct","mean")).reset_index().sort_values("acc", ascending=False)
    st.subheader("Per-style accuracy")
    st.dataframe(style_acc)
    st.download_button("Download votes CSV", merged.to_csv(index=False), "votes.csv", mime="text/csv")
else:
    st.info("No votes yet.")

if not detector_preds.empty:
    st.subheader("Detector predictions (sample)")
    st.dataframe(detector_preds.reset_index().head(200))
