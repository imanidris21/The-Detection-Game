# pages/1_Take_the_Test.py

import streamlit as st
from PIL import Image
import time, os
from backend.feedback import show_progress_summary, show_test_progress
from datetime import datetime, timezone
from backend.utils import (load_images_meta, load_detector_preds, init_db, get_engine,
                   make_pid, register_participant, save_vote, mark_finished, IMAGES_DIR, NUM_TRIALS,
                   get_trial_images)
from backend.detector import get_detector


st.set_page_config(page_title="Take the Test", layout="wide", initial_sidebar_state="collapsed")

# Show progress in navbar if test is started
progress_text = ""
if st.session_state.get("trial_order") and st.session_state.get("consented"):
    idx = st.session_state.get("trial_index", 0)
    total = len(st.session_state.trial_order)
    progress_text = show_test_progress(idx + 1, total)

    
init_db()
engine = get_engine()
images_meta = load_images_meta()
detector_preds = load_detector_preds()

# Only show main title during consent and test stages
if st.session_state.get("test_stage") in ["consent", "test"]:
    st.title("The Detection Game: Can You Spot AI-Generated Art?")


# Session state for new streamlined flow
if "test_stage" not in st.session_state:
    st.session_state.test_stage = "consent"  # consent -> pre_survey -> test -> post_survey -> results
if "consented" not in st.session_state:
    st.session_state.consented = False
if "participant_id" not in st.session_state:
    st.session_state.participant_id = None
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
if "survey_completed" not in st.session_state:
    st.session_state.survey_completed = False



# =====================================================
# STEP 1: CONSENT FORM
# =====================================================

if st.session_state.test_stage == "consent":
    # CSS for consent form styling
    st.markdown("""
    <style>
    .consent-container {
        max-width: 800px;
        margin: 2rem auto;
        padding: 2rem;
        background: white;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    }
    .consent-header {
        text-align: center;
        color: #2c3e50;
        margin-bottom: 2rem;
    }
    .consent-section {
        margin-bottom: 1.5rem;
    }
    .consent-section h3 {
        color: #34495e;
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.3rem;
    }
    .consent-section p, .consent-section ul {
        color: #555;
        line-height: 1.6;
        margin-bottom: 1rem;
    }
    .consent-section ul {
        padding-left: 1.5rem;
    }
    .consent-checkbox {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 2px solid #3498db;
        margin: 1.5rem 0;
    }

    /* Remove form container border */
    .stForm {
        border: none !important;
        padding: 0 !important;
    }

    .stForm > div {
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("Consent form")
    # st.subheader("Consent form")
    st.markdown("Consent is required. Your responses are anonymous.")
    st.markdown("---")

    # Purpose
    st.subheader("Purpose")
    st.write("You are invited to participate in a research study comparing how humans and AI systems detect artificially generated digital artworks.")

    # why you
    st.subheader("Why have you been approached?")
    st.write("You have an interest in art, creativity, and/or AI, and we are recruiting both artists and members of the public to take part. Your input will help us understand how humans perform compared to machine learning detectors.")

    # What You'll Do
    st.subheader("If I agree to participate, what will I be required to do?")
    st.write("- View a series of images (a mix of human-created and AI-generated artworks).")
    st.write("- Decide if each is human-made or AI-generated")
    st.write("- Rate your confidence in each decision and explain your reasoning")
    st.write("- Complete a brief post-test survey")
    st.write("- Takes approximately 5-10 minutes")

    # risks
    st.subheader("What are the possible risks or disadvantages?")
    st.write("You may feel frustrated if you misclassify images, but please note mistakes are expected and part of the study.")


    # Your Data
    st.subheader("Your Data")
    st.write("- Responses are anonymous and stored securely on UAL-managed systems.(no names or emails collected)")
    st.write("- We record: your decisions, confidence levels, reasoning behind the decisions, response times, and survey answers")
    st.write("- Data will be used for academic research and may be published")
    st.write("- You can withdraw at any time by closing the browser")




    # Your rights as a participant?
    st.subheader("Your rights as a participant?")
    st.write("- The right to withdraw from participation at any time during the game")
    st.write("- The right to request that any recording cease")
    st.write("- Due to the anonymization of the data upon task completion, it will not be possible to withdraw your data after submission.")



    # Privacy
    st.subheader("Privacy Notice")
    st.write("Your personal data will be processed by UAL on its managed systems for research purposes with your explicit consent. Your personal data will be anonymised and deleted on your request. You can find more information about UAL and your privacy rights at www.arts.ac.uk/privacy-information.")


    # Consent
    st.subheader("Consent")
    st.write("By clicking 'I Agree' below, you confirm:")
    st.write("(a) I understand that my participation is voluntary and that I am free to withdraw from the project at any time during the game")
    st.write("(b) The project is for the purpose of research. It may not be of direct benefit to me.")
    st.write("(c) The security of the research data will be protected during and after completion of the study.  The data collected during the study may be published. Any information which will identify me will not be used.")

    st.markdown("---")

    # Consent checkbox outside the form so it can control button state
    consent_agreed = st.checkbox("✓ I agree to participate in this study",
                               help="You must agree to participate to continue")

    # Consent form inputs
    with st.form("consent_form"):
        # Consent form - no additional data collection needed

        # Submit button
        st.markdown(" ")
        submitted = st.form_submit_button("Start the Game!", type="primary", disabled=not consent_agreed)

        if submitted and consent_agreed:
            st.session_state.consented = True
            st.session_state.test_stage = "pre_survey"
            pid = make_pid()
            st.session_state.participant_id = pid
            # No additional info collected at consent stage
            register_participant(engine, pid, st.session_state.participant_info)
            st.rerun()
        elif submitted and not consent_agreed:
            st.error("Please agree to participate by checking the consent box above.")
    st.stop()




# =====================================================
# STEP 2: PRE-TEST SURVEY
# =====================================================

elif st.session_state.test_stage == "pre_survey":
    st.title("Before We Begin")
    st.write("Can you answer these two quick questions please.")

    with st.form("pre_test_survey"):
        st.subheader("Pre-Test Questions")

        # Question 1: Confidence
        confidence = st.radio(
            "How confident are you in identifying AI-generated art?",
            [
                "Very confident - I can usually tell immediately",
                "Somewhat confident",
                "Not very confident",
                "Not confident at all - I can rarely tell"
            ],
            index=None
        )

        st.markdown("")  # Add some spacing

        # Question 2: Training
        training = st.radio(
            "Have you had any training in detecting AI-generated content?",
            [
                "Yes, formal training",
                "Yes, informal (videos, articles, social media)",
                "No"
            ],
            index=None
        )

        submitted = st.form_submit_button("Start the Test!", type="primary")

        if submitted:
            # Validate that both questions are answered
            if confidence is None:
                st.error("Please select your confidence level for identifying AI-generated art.")
            elif training is None:
                st.error("Please select whether you have had training in detecting AI-generated content.")
            else:
                # Save pre-test data
                pre_survey_data = {
                    "pre_confidence": confidence,
                    "pre_training": training
                }
                st.session_state.participant_info.update(pre_survey_data)
                register_participant(engine, st.session_state.participant_id, st.session_state.participant_info)

                st.session_state.test_stage = "test"
                st.rerun()

    st.stop()




# =====================================================
# STEP 3: TEST (10 IMAGES)
# =====================================================

elif st.session_state.test_stage == "test":
    # Build trial order if missing
    if not st.session_state.trial_order:
        if images_meta.empty:
            st.error("No images metadata found.")
            st.stop()

        # Get random selection of images
        selected_images = get_trial_images(images_meta, num_trials=NUM_TRIALS)
        st.session_state.trial_order = selected_images["image_id"].tolist()
        st.session_state.trial_index = 0

    # Check if test is finished
    if st.session_state.trial_index >= len(st.session_state.trial_order):
        st.session_state.test_stage = "survey"
        st.rerun()

    # Show trial
    idx = st.session_state.trial_index
    image_id = st.session_state.trial_order[idx]
    meta = images_meta.set_index("image_id").loc[image_id]
    img_path = os.path.join(IMAGES_DIR, meta["image_filename"])

    # Get AI detector prediction with trained model
    import os
    # Get project root: go up from src/pages/ to project root
    current_file = os.path.abspath(__file__)  # src/pages/1_Take_the_Test.py
    src_dir = os.path.dirname(os.path.dirname(current_file))  # src/
    project_root = os.path.dirname(src_dir)  # project root
    model_path = os.path.join(project_root, "models", "neural_art_80k_dinov3B_SRM_DCT", "neural_detector_dinov3_vitb16_forensic_best.pth")

    # Load AI detector with trained model
    detector = get_detector(model_checkpoint_path=model_path)
    detector_result = detector.predict(img_path)

    # Add custom CSS for compact viewport layout
    st.markdown("""
    <style>
    /* Make the main container more compact */
    .main .block-container {
        padding-top: 2rem !important;
        padding-bottom: 1rem !important;
        max-width: 100% !important;
    }

    /* Artwork image height */
    .stImage > img {
        max-height: 10vh !important;
        object-fit: contain !important;
    }

    /* Style the control panel: The box that says image 1 of 10 */

    /* Make buttons styling */
    .stButton > button {
        height: 2rem !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        background-color: white !important;
        color: #262730 !important;
        border: 1px solid #d1d5db !important;
    }

    .stButton > button:hover {
        border-color: #0066cc !important;
        color: #0066cc !important;
    }

    /* Style for primary (selected) buttons */
    .stButton > button[kind="primary"] {
        background-color: #0066cc !important;
        border-color: #0066cc !important;
        color: white !important;
    }

    .stButton > button[kind="primary"]:hover {
        background-color: #0052a3 !important;
        border-color: #0052a3 !important;
        color: white !important;
    }

    /* Compact spacing */
    .stSlider {
        margin: 0.2rem 0 !important;
    }

    /* Make slider smaller */
    .stSlider > div > div > div {
        height: 1rem !important;
    }

    .stTextArea {
        margin: 0.2rem 0 !important;
    }

    .stTextArea textarea {
        font-size: 0.8rem !important;
        line-height: 1.3 !important;
        min-height: 60px !important;
        max-height: 80px !important;
    }

    .stCheckbox {
        margin: 0.5rem 0 !important;
    }

    .stProgress {
        margin: 0.5rem 0 !important;
    }

    /* Hide unnecessary margins */
    .stMarkdown hr {
        margin: 0.5rem 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    if st.session_state.start_time_shown is None:
        st.session_state.start_time_shown = time.time()




    # Answers for current image

    # Show Answers and feedback if we just made a choice
    if st.session_state.show_feedback and st.session_state.last_feedback_data:
        # Create same two-column layout as test
        col_image_fb, col_controls_fb = st.columns([3, 2])

        # Left column: Keep image and progress
        with col_image_fb:
            current_trial = idx + 1
            total_trials = len(st.session_state.trial_order)
            st.write(f"**Image {current_trial} of {total_trials}**")

            try:
                pil = Image.open(img_path)
                st.image(pil, width=450)
            except Exception as e:
                st.error("Image load error")

        # Right column: show answers feedback
        with col_controls_fb:
            data = st.session_state.last_feedback_data
            choice = data["choice"]
            true_label = data["true_label"]
            detector_pred = data["detector_pred"]
            detector_confidence = data["detector_confidence"]

            # Calculate correctness
            user_correct = choice == true_label
            detector_correct = detector_pred == true_label

            st.markdown("### Results")

            # Your answer
            st.markdown("**Your Answer:**")
            if user_correct:
                st.success(f"✓ Your choice ({choice.title()}) is correct")
            else:
                st.error(f"✗ Your choice ({choice.title()}) is incorrect")

            # AI answer
            st.markdown("**AI Detector Answer:**")
            if detector_correct:
                st.success(f"✓ Our AI detector choice ({detector_pred.title()}) is right")
            else:
                st.error(f"✗ Our AI detector choice ({detector_pred.title()}) is wrong")

            # Correct answer
            st.markdown("**The Correct Answer:**")
            if true_label == "ai":
                st.info(f"This artwork is **AI-generated**")
            else:
                st.info(f"This artwork is **Human-made**")

            # Agreement
            if choice == detector_pred:
                st.success("You and our AI detector agree")
            else:
                st.warning("You and our AI detector disagree")

            # Continue button
            if st.button("Continue to next image", type="primary", use_container_width=True):
                st.session_state.show_feedback = False
                st.session_state.last_feedback_data = None
                st.session_state.trial_index += 1
                st.session_state.start_time_shown = None
                st.rerun()
        st.stop()

    # Create two-column layout for image and controls
    col_image, col_controls = st.columns([3, 2])

    # Left column: Image with progress on top
    with col_image:
        # Progress indicator on top of image
        current_trial = idx + 1
        total_trials = len(st.session_state.trial_order)
        st.write(f"**Image {current_trial} of {total_trials}**")

        try:
            pil = Image.open(img_path)
            st.image(pil, width=450)
        except Exception as e:
            st.error("Image load error")
            st.stop()

        # Skip option under the image - automatically skip when checked
        # Use unique key for each image to prevent state persistence
        skip_key = f"skip_{image_id}"
        seen_before = st.checkbox("I've seen this image before", key=skip_key)

        if seen_before:
            # Automatically skip when checkbox is checked
            rt = int((time.time() - st.session_state.start_time_shown) * 1000)
            rec = {
                "participant_id": st.session_state.participant_id,
                "image_id": image_id,
                "true_label": meta["true_label"],
                "human_choice": "seen",
                "confidence": 0.5,
                "response_time_ms": rt,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "detector_pred": detector_result["label"],
                "detector_confidence": detector_result["confidence"],
                "reasoning": "Skipped - image seen before",
                "generator_model": meta.get("generator_model", "unknown"),
                "art_style": meta.get("art_style", "unknown"),
                "order_shown": idx + 1
            }
            save_vote(engine, rec)

            # Add replacement image to maintain 10 classification trials
            # Get images not yet used in this trial
            used_images = set(st.session_state.trial_order)
            available_images = images_meta[~images_meta['image_id'].isin(used_images)]

            if not available_images.empty:
                # Randomly select a replacement image
                replacement_image = available_images.sample(n=1)
                replacement_id = replacement_image['image_id'].iloc[0]

                # Add replacement to the end of trial order
                st.session_state.trial_order.append(replacement_id)

            st.session_state.trial_index += 1
            st.session_state.start_time_shown = None
            st.rerun()



    # Right column: Controls for decision, confidence, reasoning, and submit

    with col_controls:
        # Initialize choice state for this image if not exists
        choice_key = f"choice_{image_id}"
        if choice_key not in st.session_state:
            st.session_state[choice_key] = None

        # 1. Main question
        st.markdown("### Is this artwork AI-generated or Human-made?")

        # 2. Decision buttons side by side (small)
        col_btn1, col_btn2 = st.columns(2)

        with col_btn1:
            # Highlight if this choice is selected
            btn_type = "primary" if st.session_state[choice_key] == "human" else "secondary"
            if st.button("Human-made", type=btn_type, use_container_width=True):
                st.session_state[choice_key] = "human"
                st.rerun()

        with col_btn2:
            # Highlight if this choice is selected
            btn_type = "primary" if st.session_state[choice_key] == "ai" else "secondary"
            if st.button("AI-generated", type=btn_type, use_container_width=True):
                st.session_state[choice_key] = "ai"
                st.rerun()


        # 3. Reasoning box
        st.markdown("**Why do you think so?**")
        reasoning = st.text_area(
            "",
            placeholder="What visual cues made you decide?",
            height=40,
            label_visibility="collapsed"
        )

        # 4. Confidence slider
        st.markdown("**How confident are you?**")
        conf = st.slider("", 0, 100, 50,
                       label_visibility="collapsed",
                       help="Rate your confidence level")

        # 5. Show answer button
        show_answer_disabled = st.session_state[choice_key] is None
        if st.button("Show Answer", type="primary", use_container_width=True, disabled=show_answer_disabled):
            rt = int((time.time() - st.session_state.start_time_shown) * 1000)
            choice = st.session_state[choice_key]

            rec = {
                "participant_id": st.session_state.participant_id,
                "image_id": image_id,
                "true_label": meta["true_label"],
                "human_choice": choice,
                "confidence": float(conf)/100.0,
                "response_time_ms": rt,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "detector_pred": detector_result["label"],
                "detector_confidence": detector_result["confidence"],
                "reasoning": reasoning.strip() if reasoning else "",
                "generator_model": meta.get("generator_model", "unknown"),
                "art_style": meta.get("art_style", "unknown"),
                "order_shown": idx + 1
            }
            save_vote(engine, rec)

            # Store feedback data and show answer
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
            # Clear choice for next image
            del st.session_state[choice_key]
            st.rerun()




# =====================================================
# STAGE 3: POST-TEST SURVEY
# =====================================================


elif st.session_state.test_stage == "survey":
    st.title("Post-Test Survey")
    st.write("Thank you for completing the test! Please answer a few questions about your experience.")

    with st.form("post_test_survey"):
        # Section 1: Background
        st.subheader("Section 1: Background")

        col1, col2 = st.columns(2)
        with col1:
            # Question 1: How would you describe yourself?
            user_type = st.selectbox(
                "1. How would you describe yourself?",
                ["Professional artist", "Hobbyist / art student", "Professional designer",
                 "AI researcher or developer", "Other"],
                index=None
            )

            # Question 3: Art mediums (optional for artists/designers)
            art_mediums = st.multiselect(
                "3. [If artist/designer] What art mediums do you work with?",
                ["Digital art", "Traditional painting", "Sketching", "Photography",
                 "Mixed media", "None/Not applicable", "Other"]
            )

            # Add text input for "Other" art mediums
            other_art_medium = ""
            if "Other" in art_mediums:
                other_art_medium = st.text_input("Please specify other art mediums:", key="other_art_medium")

        with col2:
            # Question 2: Years of experience (optional for artists/designers)
            years_experience = st.selectbox(
                "2. [If artist/designer] How many years of creative experience do you have?",
                ["Less than 1 year", "1-3 years", "4-7 years", "8-15 years", "15+ years"],
                index=None
            )

            # Question 4: AI familiarity
            ai_familiarity = st.selectbox(
                "4. How familiar are you with AI art tools (e.g., Midjourney, DALL-E)?",
                ["Never used AI art tools", "Used them a few times", "Regular user"],
                index=None
            )

        # Question 5: Frequency of seeing AI art
        ai_frequency = st.select_slider(
            "5. How often do you see AI-generated art online?",
            options=["Never", "Rarely", "Sometimes", "Often", "Very frequently"]
        )

        # Section 2: Detection
        st.subheader("Section 2: Detection")

        col3, col4 = st.columns(2)
        with col3:
            # Question 6: Difficulty
            difficulty = st.select_slider(
                "6. How difficult did you find distinguishing AI vs human artworks?",
                options=["Very easy", "Easy", "Moderate", "Difficult", "Very difficult"]
            )

        with col4:
            # Question 8: Hardest styles
            hardest_styles = st.multiselect(
                "8. Which art styles were hardest to classify? (Select all that apply)",
                ["Abstract art", "Paintings", "Anime/illustrations", "Sketches"]
            )

        # Question 7: Visual cues
        visual_cues = st.multiselect(
            "7. What visual cues help you identify AI-generated content? (Select all that apply)",
            ["Content/Subject matter", "Unusual lighting or shadows", "Anatomy issues (e.g., hands, faces)",
             "Texture/brushstrokes", "Composition or perspective", "Unnatural colours / saturation",
             "Repetitive patterns", "Text/writing artifacts", "Technical skill displayed",
             "Level of detail/smoothness", "Overall 'uncanny' feeling", "Other"]
        )

        # Add text input for "Other" visual cues
        other_visual_cue = ""
        if "Other" in visual_cues:
            other_visual_cue = st.text_input("Please specify other visual cues:", key="other_visual_cue")

        # Section 3: Platform Experience & Concerns
        st.subheader("Section 3: Platform Experience & Concerns")

        col5, col6 = st.columns(2)
        with col5:
            # Question 9: Importance of labeling
            labeling_importance = st.selectbox(
                "9. How important is it that AI artworks be clearly labelled?",
                ["Extremely important", "Very important", "Somewhat important",
                 "Not very important", "Not important at all"],
                index=None
            )

            # Question 13: Encountered unlabeled AI
            encountered_unlabeled = st.selectbox(
                "13. Have you encountered unlabelled AI artworks presented as human-created?",
                ["Yes, frequently", "Yes, occasionally", "Yes, rarely", "No, never", "Unsure"],
                index=None
            )

        with col6:
            # Question 15: Value of detection
            detection_value = st.selectbox(
                "15. If online platforms had reliable AI detection and labeling, how valuable would this be?",
                ["Very valuable", "Somewhat valuable", "Not valuable", "Unsure"],
                index=None
            )

        # Question 14: Concerns
        concerns = st.multiselect(
            "14. What concerns do you have about AI-generated art? (Select top 3)",
            ["AI training on work without permission", "Difficulty proving originality",
             "Copyright concerns", "Devaluation of human creativity",
             "AI content flooding platforms", "Other"]
        )

        # Add text input for "Other" concerns
        other_concern = ""
        if "Other" in concerns:
            other_concern = st.text_input("Please specify other concerns:", key="other_concern")

        # Section 4: Ethics & Impact (Optional for Artists)
        st.subheader("Section 4: Ethics & Impact")

        col7, col8 = st.columns(2)
        with col7:
            # Question 16: Impact on visibility (optional for artists/designers)
            if user_type in ["Professional artist", "Hobbyist / art student", "Professional designer"]:
                visibility_impact = st.selectbox(
                    "16. [If artist/designer] How has AI affected your work's visibility on social media?",
                    ["Significantly decreased", "Somewhat decreased", "No change",
                     "Somewhat increased", "Significantly increased", "Unsure"],
                    index=None
                )
            else:
                visibility_impact = "Not applicable"

        with col8:
            # Placeholder for layout balance
            st.write("")

        # Section 5: Reflection
        st.subheader("Section 5: Reflection")

        # Question 18: Emotions
        emotions = st.multiselect(
            "18. What emotions did you feel realizing some images were AI-generated?",
            ["Surprise", "Frustration", "Curiosity", "Indifference", "Other"]
        )

        # Add text input for "Other" emotions
        other_emotion = ""
        if "Other" in emotions:
            other_emotion = st.text_input("Please specify other emotions:", key="other_emotion")

        # Question 19: Additional comments
        additional_comments = st.text_area(
            "19. Do you have any additional comments you want to add? (Optional)",
            placeholder="Feel free to share any thoughts, observations, or feedback..."
        )

        submitted = st.form_submit_button("Complete Survey", type="primary")

        if submitted:
            # Validate required fields
            validation_errors = []

            if user_type is None:
                validation_errors.append("Please select how you would describe yourself.")

            # Note: years_experience is optional for artist/designer types

            if ai_familiarity is None:
                validation_errors.append("Please select your familiarity with AI art tools.")

            if ai_frequency is None:
                validation_errors.append("Please select how often you see AI-generated art online.")

            if difficulty is None:
                validation_errors.append("Please select how difficult you found distinguishing AI vs human artworks.")

            if labeling_importance is None:
                validation_errors.append("Please select how important AI artwork labeling is to you.")

            if encountered_unlabeled is None:
                validation_errors.append("Please select whether you have encountered unlabelled AI artworks.")

            if detection_value is None:
                validation_errors.append("Please select how valuable reliable AI detection would be.")

            # Note: visibility_impact is optional for artist/designer types

            # Show validation errors if any exist
            if validation_errors:
                st.error("Please complete all required fields:")
                for error in validation_errors:
                    st.error(f"• {error}")
            else:
                # Process "Other" responses by combining them with the selections

                # Handle user type
                final_user_type = user_type

                # Handle art mediums - replace "Other" with custom text
                final_art_mediums = art_mediums.copy() if isinstance(art_mediums, list) else [art_mediums]
                if "Other" in final_art_mediums and other_art_medium:
                    final_art_mediums.remove("Other")
                    final_art_mediums.append(other_art_medium)

                # Handle visual cues - replace "Other" with custom text
                final_visual_cues = visual_cues.copy() if isinstance(visual_cues, list) else [visual_cues]
                if "Other" in final_visual_cues and other_visual_cue:
                    final_visual_cues.remove("Other")
                    final_visual_cues.append(other_visual_cue)

                # Handle concerns - replace "Other" with custom text
                final_concerns = concerns.copy() if isinstance(concerns, list) else [concerns]
                if "Other" in final_concerns and other_concern:
                    final_concerns.remove("Other")
                    final_concerns.append(other_concern)

                # Handle emotions - replace "Other" with custom text
                final_emotions = emotions.copy() if isinstance(emotions, list) else [emotions]
                if "Other" in final_emotions and other_emotion:
                    final_emotions.remove("Other")
                    final_emotions.append(other_emotion)

                # Update participant info with comprehensive survey data
                survey_data = {
                    "user_type": final_user_type,
                    "years_experience": years_experience,
                    "art_mediums": final_art_mediums,
                    "ai_familiarity": ai_familiarity,
                    "ai_frequency": ai_frequency,
                    "difficulty": difficulty,
                    "visual_cues": final_visual_cues,
                    "hardest_styles": hardest_styles,
                    "labeling_importance": labeling_importance,
                    "encountered_unlabeled": encountered_unlabeled,
                    "concerns": final_concerns,
                    "detection_value": detection_value,
                    "visibility_impact": visibility_impact,
                    "emotions": final_emotions,
                    "additional_comments": additional_comments
                }
                st.session_state.participant_info.update(survey_data)
                register_participant(engine, st.session_state.participant_id, st.session_state.participant_info)
                st.session_state.survey_completed = True
                st.session_state.test_stage = "results"
                st.rerun()

    st.stop()



# =====================================================
# STEP 4: RESULTS
# =====================================================

elif st.session_state.test_stage == "results":
    st.title("Thank You for Participating!")

    # Mark participant as finished
    mark_finished(engine, st.session_state.participant_id)

    st.subheader("Here are Your Results")

    # Show detailed progress summary
    show_progress_summary(engine, st.session_state.participant_id, len(st.session_state.trial_order))

    st.markdown("---")

    # Contact information section
    st.subheader("Want to Learn More?")
    st.write("If you want to know more about this research, please email **imanidris21@gmail.com**")

    st.markdown("---")

    col3, col4, col5 = st.columns([1, 2, 1])
    with col4:
        if st.button("View Leaderboard", type="primary", use_container_width=True):
            st.switch_page("pages/3_Leaderboard.py")



    st.stop()




