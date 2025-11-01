# pages/3_Leaderboard.py
import streamlit as st
import pandas as pd

from components.navbar import top_nav, handle_fallback_switch
from datetime import datetime, timezone

from utils import init_db, get_engine, load_images_meta, load_detector_preds


st.set_page_config(page_title="Leaderboard", layout="wide", initial_sidebar_state="collapsed")
top_nav("Leaderboard")
handle_fallback_switch()

init_db()
engine = get_engine()
images_meta = load_images_meta()
detector_preds = load_detector_preds()

st.title("Leaderboard — Humans vs AI")

with engine.begin() as conn:
    votes = pd.read_sql("SELECT * FROM votes", conn)
    participants = pd.read_sql("SELECT * FROM participants", conn)

if votes.empty:
    st.info("No votes yet. Run the test to generate data.")
    st.stop()

merged = votes.merge(images_meta[["image_id","true_label","style"]], on="image_id", how="left")
merged["correct"] = merged["human_choice"] == merged["true_label"]

part_acc = merged.groupby("participant_id").agg(n=("correct","size"), acc=("correct","mean")).reset_index().sort_values("acc", ascending=False)
part_acc = part_acc.merge(participants[["participant_id","discipline"]], on="participant_id", how="left")
st.dataframe(part_acc.head(50))

st.markdown("---")
st.write("Aggregate human accuracy (all votes):")
st.metric("Human accuracy", f"{merged['correct'].mean():.2%}")

# AI detector baseline (dataset-level)
if not detector_preds.empty:
    preds = detector_preds.reset_index()[["image_id","label_ai"]]
    img_acc = images_meta.merge(preds, on="image_id", how="left")
    img_acc["label_ai"].fillna("human", inplace=True)
    img_acc["ai_correct"] = img_acc["label_ai"] == img_acc["true_label"]
    st.write(f"AI detector accuracy (dataset-level): **{img_acc['ai_correct'].mean():.2%}**")
