# pages/1_Take_the_Test.py
import streamlit as st
from PIL import Image
import time, os
from components.navbar import top_nav, handle_fallback_switch, show_test_progress
from components.feedback import show_immediate_feedback, show_progress_summary
from datetime import datetime, timezone
from utils import (load_images_meta, load_detector_preds, init_db, get_engine,
                   make_pid, register_participant, save_vote, mark_finished, IMAGES_DIR, NUM_TRIALS,
                   get_trial_images)
from detector import get_detector


st.set_page_config(page_title="Take the Test", layout="wide", initial_sidebar_state="collapsed")

# Show progress in navbar if test is started
progress_text = ""
if st.session_state.get("trial_order") and st.session_state.get("consented"):
    idx = st.session_state.get("trial_index", 0)
    total = len(st.session_state.trial_order)
    progress_text = show_test_progress(idx + 1, total)

top_nav("Take the Test", show_progress=bool(progress_text), progress_text=progress_text)
handle_fallback_switch()

init_db()
engine = get_engine()
images_meta = load_images_meta()
detector_preds = load_detector_preds()

st.title("Take the Test — Human vs AI")
st.markdown("Consent is required. Your responses are anonymous.")

# Session state
if "consented" not in st.session_state:
    st.session_state.consented = False
if "participant_id" not in st.session_state:
    st.session_state.participant_id = None
if "questionnaire_done" not in st.session_state:
    st.session_state.questionnaire_done = False
if "trial_order" not in st.session_state:
    st.session_state.trial_order = []
if "trial_index" not in st.session_state:
    st.session_state.trial_index = 0
if "start_time_shown" not in st.session_state:
    st.session_state.start_time_shown = None
if "participant_info" not in st.session_state:
    st.session_state.participant_info = {}
if "show_feedback" not in st.session_state:
    st.session_state.show_feedback = False
if "last_feedback_data" not in st.session_state:
    st.session_state.last_feedback_data = None

# Consent form
if not st.session_state.consented:
    with st.form("consent"):
        device = st.selectbox("Device type", ["desktop","tablet","kiosk","other"])
        user_group = st.selectbox("Are you an artist / student / general public?", ["general","artist","student","other"])

        # Difficulty mode selection (if enabled)
        from config import config
        if config.ENABLE_DIFFICULTY_MODES:
            difficulty_mode = st.selectbox(
                "Test difficulty",
                ["mixed", "easy", "medium", "hard", "adaptive"],
                help="Mixed: Random selection, Easy: Clear cases, Hard: Challenging cases, Adaptive: Adjusts based on your performance"
            )
        else:
            difficulty_mode = "mixed"

        accept = st.checkbox("I consent to participate in this anonymous study")
        submitted = st.form_submit_button("Continue")
        if submitted and accept:
            st.session_state.consented = True
            pid = make_pid()
            st.session_state.participant_id = pid
            st.session_state.participant_info.update({
                "device_type": device,
                "user_group": user_group,
                "difficulty_mode": difficulty_mode
            })
            register_participant(engine, pid, st.session_state.participant_info)
            st.rerun()
        elif submitted and not accept:
            st.warning("You must consent to continue.")
    st.stop()

# Questionnaire
if not st.session_state.questionnaire_done:
    with st.form("q"):
        discipline = st.selectbox("Primary creative discipline", ["Digital Artist", "Photographer", "Designer", "Traditional Artist", "3D Artist", "Other"])
        years = st.selectbox("Years experience", ["<1","1-3","4-7","8-15","15+","Prefer not"])
        conf_self = st.selectbox("How confident are you in identifying AI images?", ["Very", "Somewhat", "Not much", "Not at all"])
        seen_training = st.radio("Have you done training about detecting AI images?", ["No","Yes - small","Yes - extensive"])
        cues = st.multiselect("Which visual cues help you identify AI content?", ["Lighting","Hands/fingers","Textures","Symmetry","Faces","Perspective","Colors","Patterns","Text artifacts","Other"])
        submit = st.form_submit_button("Save & Start test")
        if submit:
            st.session_state.participant_info.update({
                "discipline": discipline, "years_experience": years, "confidence_self": conf_self, "seen_training": seen_training, "cues": cues
            })
            register_participant(engine, st.session_state.participant_id, st.session_state.participant_info)
            st.session_state.questionnaire_done = True
            st.rerun()
    st.stop()

# Build trial order if missing
if not st.session_state.trial_order:
    if images_meta.empty:
        st.error("No images metadata found.")
        st.stop()

    # Get difficulty mode from participant info
    difficulty_mode = st.session_state.participant_info.get("difficulty_mode", "mixed")
    selected_images = get_trial_images(images_meta, difficulty_mode, NUM_TRIALS)
    st.session_state.trial_order = selected_images["image_id"].tolist()
    st.session_state.trial_index = 0
    st.session_state.difficulty_mode = difficulty_mode

# If finished
if st.session_state.trial_index >= len(st.session_state.trial_order):
    st.success("🎉 Test complete — thank you!")
    mark_finished(engine, st.session_state.participant_id)

    # Show detailed progress summary
    show_progress_summary(engine, st.session_state.participant_id, len(st.session_state.trial_order))

    st.markdown("---")
    st.info("You may close this window or navigate to other pages.")
    st.write("Your participant ID:", st.session_state.participant_id)
    st.stop()

# Show trial
idx = st.session_state.trial_index
image_id = st.session_state.trial_order[idx]
meta = images_meta.set_index("image_id").loc[image_id]
img_path = os.path.join(IMAGES_DIR, meta["image_filename"])

# Get AI detector prediction
detector = get_detector()
detector_result = detector.predict(img_path)

try:
    pil = Image.open(img_path)
    st.image(pil, use_column_width=True)
except Exception as e:
    st.error("Image load error")
    st.stop()

if st.session_state.start_time_shown is None:
    st.session_state.start_time_shown = time.time()

conf = st.slider("Confidence (%)", 0, 100, 50)
seen_before = st.checkbox("I have seen this image before (skip)")

# Show feedback if we just made a choice
if st.session_state.show_feedback and st.session_state.last_feedback_data:
    st.markdown("---")
    show_immediate_feedback(**st.session_state.last_feedback_data)

    if st.button("Continue to next image"):
        st.session_state.show_feedback = False
        st.session_state.last_feedback_data = None
        st.session_state.trial_index += 1
        st.session_state.start_time_shown = None
        st.rerun()
    st.stop()

col1, col2 = st.columns(2)
with col1:
    if st.button("Human-made"):
        rt = int((time.time() - st.session_state.start_time_shown) * 1000)
        choice = "human" if not seen_before else "seen"
        rec = {
            "participant_id": st.session_state.participant_id,
            "image_id": image_id,
            "true_label": meta["true_label"],
            "human_choice": choice,
            "confidence": float(conf)/100.0,
            "response_time_ms": rt,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "detector_pred": detector_result["label"],
            "detector_confidence": detector_result["confidence"]
        }
        save_vote(engine, rec)

        # Store feedback data
        if choice != "seen":
            st.session_state.last_feedback_data = {
                "choice": choice,
                "true_label": meta["true_label"],
                "detector_pred": detector_result["label"],
                "detector_confidence": detector_result["confidence"],
                "participant_id": st.session_state.participant_id,
                "engine": engine,
                "trial_number": idx + 1
            }
            st.session_state.show_feedback = True
        else:
            st.session_state.trial_index += 1
            st.session_state.start_time_shown = None
        st.rerun()
with col2:
    if st.button("AI-generated"):
        rt = int((time.time() - st.session_state.start_time_shown) * 1000)
        choice = "ai" if not seen_before else "seen"
        rec = {
            "participant_id": st.session_state.participant_id,
            "image_id": image_id,
            "true_label": meta["true_label"],
            "human_choice": choice,
            "confidence": float(conf)/100.0,
            "response_time_ms": rt,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "detector_pred": detector_result["label"],
            "detector_confidence": detector_result["confidence"]
        }
        save_vote(engine, rec)

        # Store feedback data
        if choice != "seen":
            st.session_state.last_feedback_data = {
                "choice": choice,
                "true_label": meta["true_label"],
                "detector_pred": detector_result["label"],
                "detector_confidence": detector_result["confidence"],
                "participant_id": st.session_state.participant_id,
                "engine": engine,
                "trial_number": idx + 1
            }
            st.session_state.show_feedback = True
        else:
            st.session_state.trial_index += 1
            st.session_state.start_time_shown = None
        st.rerun()

st.progress((idx+1)/len(st.session_state.trial_order))
