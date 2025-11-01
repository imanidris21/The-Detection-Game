"""
Real-time Performance Feedback Component

This module provides immediate feedback to users after each classification,
showing their accuracy compared to the AI detector and other participants.
"""

import streamlit as st
import pandas as pd
from sqlalchemy import text
from config import config


def show_immediate_feedback(choice, true_label, detector_pred, detector_confidence,
                          participant_id, engine, trial_number):
    """
    Show feedback after each vote

    Args:
        choice: User's choice ('human' or 'ai')
        true_label: Actual label ('human' or 'ai')
        detector_pred: AI detector prediction
        detector_confidence: AI detector confidence
        participant_id: Current participant ID
        engine: Database engine
        trial_number: Current trial number
    """
    if not config.ENABLE_REAL_TIME_FEEDBACK:
        return

    # Calculate if user and detector were correct
    user_correct = choice == true_label
    detector_correct = detector_pred == true_label

    # Show feedback in columns
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Your Answer")
        if user_correct:
            st.success(f"✅ Correct! ({choice})")
        else:
            st.error(f"❌ Incorrect ({choice})")
            st.info(f"Correct answer: {true_label}")

    with col2:
        st.markdown("### AI Detector")
        if detector_correct:
            st.success(f"✅ AI got it right ({detector_pred})")
        else:
            st.error(f"❌ AI was wrong ({detector_pred})")

        # Show confidence with color coding
        if detector_confidence > 0.8:
            st.success(f"Confidence: {detector_confidence:.1%} (High)")
        elif detector_confidence > 0.6:
            st.warning(f"Confidence: {detector_confidence:.1%} (Medium)")
        else:
            st.error(f"Confidence: {detector_confidence:.1%} (Low)")

    with col3:
        st.markdown("### Your Progress")

        # Get user's accuracy so far
        user_accuracy = get_user_accuracy(engine, participant_id)
        if user_accuracy is not None:
            st.metric("Your Accuracy", f"{user_accuracy:.1%}")

        # Get average human performance for this image (if available)
        human_avg = get_human_average_for_image(engine, true_label)
        if human_avg is not None:
            st.metric("Human Average", f"{human_avg:.1%}")

    # Agreement indicator
    st.markdown("---")
    if choice == detector_pred:
        st.info("🤝 You and the AI agree!")
    else:
        st.warning("🤔 You and the AI disagree")

    # Show difficulty context if available
    show_difficulty_context(true_label, detector_confidence, human_avg)


def show_difficulty_context(true_label, detector_confidence, human_avg):
    """Show context about the difficulty of the current image"""
    if detector_confidence is None:
        return

    # Determine difficulty
    if detector_confidence > 0.8 and (human_avg is None or human_avg > 0.7):
        difficulty = "Easy"
        color = "success"
        icon = "🟢"
    elif detector_confidence < 0.6 or (human_avg is not None and human_avg < 0.4):
        difficulty = "Hard"
        color = "error"
        icon = "🔴"
    else:
        difficulty = "Medium"
        color = "warning"
        icon = "🟡"

    if color == "success":
        st.success(f"{icon} Difficulty: {difficulty} - Most people get this right")
    elif color == "error":
        st.error(f"{icon} Difficulty: {difficulty} - This is a challenging case")
    else:
        st.warning(f"{icon} Difficulty: {difficulty} - Moderate difficulty")


def get_user_accuracy(engine, participant_id):
    """Get the current user's accuracy so far"""
    try:
        with engine.begin() as conn:
            result = conn.execute(text("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN human_choice = true_label THEN 1 ELSE 0 END) as correct
                FROM votes
                WHERE participant_id = :participant_id
                AND human_choice != 'seen'
            """), {"participant_id": participant_id})

            row = result.fetchone()
            if row and row[0] > 0:
                return row[1] / row[0]
    except Exception as e:
        st.error(f"Error calculating accuracy: {e}")

    return None


def get_human_average_for_image(engine, true_label):
    """Get average human performance for this type of image"""
    try:
        with engine.begin() as conn:
            result = conn.execute(text("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN human_choice = true_label THEN 1 ELSE 0 END) as correct
                FROM votes
                WHERE true_label = :true_label
                AND human_choice != 'seen'
                AND human_choice IS NOT NULL
            """), {"true_label": true_label})

            row = result.fetchone()
            if row and row[0] > 5:  # Only show if we have enough data
                return row[1] / row[0]
    except Exception as e:
        # Don't show error to user, just log it
        pass

    return None


def show_progress_summary(engine, participant_id, total_trials):
    """Show a summary of progress at the end"""
    try:
        with engine.begin() as conn:
            # Get user's votes
            votes_df = pd.read_sql("""
                SELECT * FROM votes
                WHERE participant_id = ?
                AND human_choice != 'seen'
                ORDER BY timestamp_utc
            """, conn, params=[participant_id])

            if votes_df.empty:
                return

            # Calculate metrics
            user_accuracy = (votes_df['human_choice'] == votes_df['true_label']).mean()
            detector_accuracy = (votes_df['detector_pred'] == votes_df['true_label']).mean()
            agreement_rate = (votes_df['human_choice'] == votes_df['detector_pred']).mean()

            # Performance by type
            ai_performance = votes_df[votes_df['true_label'] == 'ai']
            human_performance = votes_df[votes_df['true_label'] == 'human']

            ai_accuracy = (ai_performance['human_choice'] == 'ai').mean() if not ai_performance.empty else 0
            human_accuracy = (human_performance['human_choice'] == 'human').mean() if not human_performance.empty else 0

            # Display results
            st.markdown("## 📊 Your Performance Summary")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Overall Accuracy", f"{user_accuracy:.1%}")
                st.metric("AI Detection", f"{ai_accuracy:.1%}")
                st.metric("Human Detection", f"{human_accuracy:.1%}")

            with col2:
                st.metric("AI Detector Accuracy", f"{detector_accuracy:.1%}")
                st.metric("Agreement with AI", f"{agreement_rate:.1%}")
                avg_confidence = votes_df['confidence'].mean()
                st.metric("Avg Confidence", f"{avg_confidence:.1%}")

            with col3:
                # Compare to others
                overall_human_avg = get_overall_human_average(engine)
                if overall_human_avg:
                    if user_accuracy > overall_human_avg:
                        st.success(f"🎉 Above average! (avg: {overall_human_avg:.1%})")
                    else:
                        st.info(f"Keep practicing! (avg: {overall_human_avg:.1%})")

                # Response time stats
                avg_time = votes_df['response_time_ms'].mean() / 1000
                st.metric("Avg Response Time", f"{avg_time:.1f}s")

            # Show improvement over time
            if len(votes_df) >= 10:
                show_improvement_chart(votes_df)

    except Exception as e:
        st.error(f"Error generating summary: {e}")


def get_overall_human_average(engine):
    """Get overall human accuracy across all participants"""
    try:
        with engine.begin() as conn:
            result = conn.execute(text("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN human_choice = true_label THEN 1 ELSE 0 END) as correct
                FROM votes
                WHERE human_choice != 'seen'
                AND human_choice IS NOT NULL
            """))

            row = result.fetchone()
            if row and row[0] > 50:  # Only show if we have enough data
                return row[1] / row[0]
    except Exception:
        pass

    return None


def show_improvement_chart(votes_df):
    """Show a chart of accuracy improvement over trials"""
    try:
        # Calculate rolling accuracy
        votes_df['correct'] = votes_df['human_choice'] == votes_df['true_label']
        votes_df['trial_num'] = range(1, len(votes_df) + 1)

        # Rolling average (window of 5)
        window_size = min(5, len(votes_df) // 2)
        votes_df['rolling_accuracy'] = votes_df['correct'].rolling(window=window_size, min_periods=1).mean()

        # Create simple line chart
        st.markdown("### 📈 Your Learning Progress")
        st.line_chart(votes_df.set_index('trial_num')['rolling_accuracy'])

        # Show trend
        if len(votes_df) >= 10:
            first_half = votes_df.iloc[:len(votes_df)//2]['correct'].mean()
            second_half = votes_df.iloc[len(votes_df)//2:]['correct'].mean()

            if second_half > first_half + 0.1:
                st.success("📈 You're improving over time!")
            elif second_half < first_half - 0.1:
                st.warning("📉 Performance declined - you might be getting fatigued")
            else:
                st.info("📊 Consistent performance throughout")

    except Exception as e:
        # Don't break the experience if charting fails
        pass