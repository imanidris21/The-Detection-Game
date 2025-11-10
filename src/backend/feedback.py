
"""
Real-time Performance Feedback Module

This provides immediate feedback to users after each classification,
showing their accuracy compared to the AI detector.
"""

import streamlit as st
import pandas as pd
from sqlalchemy import text
from backend.config import config






def show_progress_summary(engine, participant_id, total_trials):
    """Show a summary of progress at the end"""
    try:
        with engine.begin() as conn:
            # Get user's votes
            votes_df = pd.read_sql(text("""
                SELECT * FROM votes
                WHERE participant_id = :participant_id
                AND human_choice != 'seen'
                ORDER BY timestamp_utc
            """), conn, params={"participant_id": participant_id})

            if votes_df.empty:
                return

            # Calculate metrics
            user_accuracy = (votes_df['human_choice'] == votes_df['true_label']).mean()
            detector_accuracy = (votes_df['detector_pred'] == votes_df['true_label']).mean()

            # Performance by type
            ai_performance = votes_df[votes_df['true_label'] == 'ai']
            human_performance = votes_df[votes_df['true_label'] == 'human']

            ai_accuracy = (ai_performance['human_choice'] == 'ai').mean() if not ai_performance.empty else 0
            human_accuracy = (human_performance['human_choice'] == 'human').mean() if not human_performance.empty else 0

            # Count correct classifications (excluding skips)
            classification_votes = votes_df[votes_df['human_choice'] != 'seen']
            correct_count = (classification_votes['human_choice'] == classification_votes['true_label']).sum()
            total_count = len(classification_votes)

            # Display simplified results
            # st.markdown("## Your Performance Summary")

            st.info(f"You classified **{correct_count}/{total_count}** images correctly")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Your accuracy in detecting AI art", f"{ai_accuracy:.1%}")

            with col2:
                st.metric("Your accuracy in detecting human art", f"{human_accuracy:.1%}")

            with col3:
                st.metric("Your Overall Accuracy", f"{user_accuracy:.1%}")

            with col4:
                st.metric("Our AI Detector Accuracy", f"{detector_accuracy:.1%}")

    except Exception as e:
        st.error(f"Error generating summary: {e}")




def show_test_progress(current: int, total: int):
    """Helper function to format test progress"""
    if total <= 0:
        return "Progress: 0%"

    percentage = (current / total) * 100
    return f"Progress: {current}/{total} ({percentage:.0f}%)"

