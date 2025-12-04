# pages/4_Research_Dashboard.py

# The data analysis in the research dashboard is not the actual analysis but rather a data display that helped a lot during the launch phase.
# A clean and legit data analysis is in Data_Analysis.ipynb and model_data_analysis.ipynb 

# Some of the code in this script is generated with the assistance of Claude code AI. all suggestions were reviewed critically and modified as needed. 

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sqlite3
from datetime import datetime, timedelta
from backend.utils import init_db, get_engine, load_images_meta, load_detector_preds

# Page configuration
st.set_page_config(page_title="Scientific Dashboard", layout="wide", initial_sidebar_state="collapsed")

# PASSWORD PROTECTION
st.title("Scientific Research Dashboard")
st.markdown("*Restricted Access*")

# Check if user is authenticated
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.markdown("### Admin Access Required")
    st.markdown("Enter password to access the research dashboard:")

    password = st.text_input("Password", type="password", key="admin_password")

    if st.button("Login", type="primary"):
        try:
            # Get password from Streamlit secrets
            admin_password = st.secrets["admin_password"]

            if password == admin_password:
                st.session_state.authenticated = True
                st.success("Access granted! Reloading dashboard...")
                st.rerun()
            else:
                st.error("Invalid password. Please contact iman for access.")
        except KeyError:
            st.error("Admin password not configured. Please contact the administrator.")
            st.info("**For developers**: Add `admin_password` to your Streamlit secrets.")

    st.stop()

# Add logout button for authenticated users
col1, col2 = st.columns([1, 9])
with col1:
    if st.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()

st.markdown("---")

init_db()
engine = get_engine()
images_meta = load_images_meta()
detector_preds = load_detector_preds()

st.markdown("*Analytics for AI art detection research*")


# Add refresh button and database info
col_refresh, col_info = st.columns([1, 3])
with col_refresh:
    if st.button("Refresh Data", help="Force reload data from database"):
        st.cache_data.clear()  # Clear any cached data
        st.rerun()

with col_info:
    # Show database connection info
    db_info = f"Connected to: {engine.dialect.name} • Engine: {str(engine.url).split('@')[1] if '@' in str(engine.url) else 'Local SQLite'}"
    st.caption(db_info)

# Load completely fresh data every time - NO CACHING WHATSOEVER
def load_and_process_data():
    # Force fresh database connection each time
    fresh_engine = get_engine()
    with fresh_engine.begin() as conn:
        votes = pd.read_sql("SELECT * FROM votes", conn)
        participants = pd.read_sql("SELECT * FROM participants", conn)

    if votes.empty:
        return votes, participants, pd.DataFrame(), {}

    # Data preprocessing
    votes['correct'] = votes['human_choice'] == votes['true_label']
    votes['timestamp'] = pd.to_datetime(votes['timestamp_utc'])
    participant_accuracies = votes.groupby('participant_id')['correct'].agg(['mean', 'count']).reset_index()
    participant_accuracies.columns = ['participant_id', 'accuracy', 'n_responses']

    # Calculate key metrics
    metrics = {
        'n_participants': len(participant_accuracies),
        'n_responses': len(votes[votes['human_choice'] != 'seen']),
        'median_accuracy': participant_accuracies['accuracy'].median() if len(participant_accuracies) > 0 else 0,
        'overall_accuracy': participant_accuracies['accuracy'].mean() if len(participant_accuracies) > 0 else 0,
        'ai_accuracy': (votes['detector_pred'] == votes['true_label']).mean() if 'detector_pred' in votes.columns else 0,
        'response_times': votes['response_time_ms'] / 1000 if 'response_time_ms' in votes.columns else pd.Series([])
    }

    return votes, participants, participant_accuracies, metrics

# Force completely fresh data load every single time
import time
current_time = time.time()
votes, participants, participant_accuracies, metrics = load_and_process_data()

# Debug info - show exact counts and sample IDs with timestamp
if not participants.empty:
    from datetime import datetime
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.info(f"🔍 **Debug Info** ({timestamp}): Found {len(participants)} participants, {len(votes)} votes. "
            f"Participant IDs: {', '.join(participants['participant_id'].head(3).tolist())}"
            f"{'...' if len(participants) > 3 else ''}")

if votes.empty:
    st.warning("No research data available yet.")
    st.stop()

# EXECUTIVE SUMMARY
st.markdown("---")
st.subheader("Executive Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Participants", metrics['n_participants'])

with col2:
    st.metric("Valid Responses", metrics['n_responses'])

with col3:
    st.metric("Human Accuracy", f"{metrics['overall_accuracy']:.1%}", delta=f"Median: {metrics['median_accuracy']:.1%}")

with col4:
    performance_gap = metrics['ai_accuracy'] - metrics['overall_accuracy']
    st.metric("AI Detector Accuracy", f"{metrics['ai_accuracy']:.1%}", delta=f"{performance_gap:+.1%} vs humans")

# STATISTICAL ANALYSIS
st.markdown("---")
st.subheader("Statistical Analysis")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Human Performance Distribution (all users)**")

    # Descriptive statistics
    stats_data = {
        "Metric": ["Mean", "Median", "Std Dev", "Min", "Max", "IQR"],
        "Value": [
            f"{participant_accuracies['accuracy'].mean():.3f}",
            f"{participant_accuracies['accuracy'].median():.3f}",
            f"{participant_accuracies['accuracy'].std():.3f}",
            f"{participant_accuracies['accuracy'].min():.3f}",
            f"{participant_accuracies['accuracy'].max():.3f}",
            f"{participant_accuracies['accuracy'].quantile(0.75) - participant_accuracies['accuracy'].quantile(0.25):.3f}"
        ]
    }
    st.dataframe(pd.DataFrame(stats_data), hide_index=True)

with col2:
    st.markdown("**Sample Characteristics (all users)**")

    # Sample size analysis
    total_images = votes['image_id'].nunique()
    avg_responses_per_image = len(votes) / total_images if total_images > 0 else 0
    completion_rate = len(participants[participants['finished_at'].notna()]) / len(participants) if len(participants) > 0 else 0

    sample_data = {
        "Characteristic": ["Images in Test", "Avg Responses/Image", "Completion Rate", "With Reasoning"],
        "Value": [
            f"{total_images}",
            f"{avg_responses_per_image:.1f}",
            f"{completion_rate:.1%}",
            f"{(votes['reasoning'].notna() & (votes['reasoning'] != '')).sum()}/{len(votes)}"
        ]
    }
    st.dataframe(pd.DataFrame(sample_data), hide_index=True)

# DETAILED VISUALIZATIONS
st.markdown("---")
st.subheader("Research Visualizations")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Performance Analysis", "Response Patterns", "Generator Analysis", "Qualitative Analysis", "Data Export", "Style Analysis", "Detector Data"])

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        # Histogram of participant accuracies
        fig_hist = px.histogram(
            participant_accuracies,
            x='accuracy',
            nbins=20,
            title="Distribution of Participant Accuracies",
            labels={'accuracy': 'Accuracy', 'count': 'Number of Participants'}
        )
        fig_hist.update_xaxes(tickformat='.0%')
        fig_hist.add_vline(x=metrics['median_accuracy'], line_dash="dash", line_color="red",
                          annotation_text=f"Median: {metrics['median_accuracy']:.1%}")
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        # Human vs AI accuracy by image type
        accuracy_by_type = votes.groupby('true_label').agg({
            'correct': 'mean',
            'detector_pred': lambda x: (x == votes.loc[x.index, 'true_label']).mean()
        }).round(3)

        fig_comparison = go.Figure()
        fig_comparison.add_trace(go.Bar(
            name='Human',
            x=['AI Images', 'Human Images'],
            y=[accuracy_by_type.loc['ai', 'correct'], accuracy_by_type.loc['human', 'correct']],
            marker_color='lightblue'
        ))
        fig_comparison.add_trace(go.Bar(
            name='AI Detector',
            x=['AI Images', 'Human Images'],
            y=[accuracy_by_type.loc['ai', 'detector_pred'], accuracy_by_type.loc['human', 'detector_pred']],
            marker_color='lightcoral'
        ))
        fig_comparison.update_layout(
            title="Accuracy by Image Type",
            yaxis_title="Accuracy",
            yaxis=dict(tickformat='.0%')
        )
        st.plotly_chart(fig_comparison, use_container_width=True)

with tab2:
    col1, col2 = st.columns(2)

    with col1:
        # Response times analysis
        response_times = votes['response_time_ms'] / 1000  # Convert to seconds
        fig_rt = px.box(y=response_times, title="Response Time Distribution")
        fig_rt.update_yaxes(title="Response Time (seconds)")
        st.plotly_chart(fig_rt, use_container_width=True)

        st.metric("Median Response Time", f"{response_times.median():.1f}s")

    with col2:
        # Confidence analysis
        fig_conf = px.histogram(
            votes[votes['human_choice'] != 'seen'],
            x='confidence',
            color='correct',
            title="Confidence vs Accuracy",
            labels={'confidence': 'Confidence Level', 'count': 'Number of Responses'}
        )
        st.plotly_chart(fig_conf, use_container_width=True)

        # Confidence correlation
        conf_corr = votes[votes['human_choice'] != 'seen'].groupby('correct')['confidence'].mean()
        st.write("**Confidence by Accuracy:**")
        st.write(f"Correct answers: {conf_corr[True]:.1%}")
        st.write(f"Incorrect answers: {conf_corr[False]:.1%}")

with tab3:
    st.markdown("**AI Generator Model Analysis**")

    # Check for new metadata fields
    has_generator_data = 'generator_model' in votes.columns and votes['generator_model'].notna().any()
    has_style_data = 'art_style' in votes.columns and votes['art_style'].notna().any()

    if has_generator_data:
        # Filter for votes with generator model data
        gen_votes = votes[votes['generator_model'].notna() & (votes['human_choice'] != 'seen')].copy()

        if len(gen_votes) > 0:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Detection Accuracy by AI Generator**")

                # Calculate accuracy by generator
                gen_accuracy = gen_votes.groupby('generator_model').agg({
                    'correct': ['count', 'mean'],
                    'confidence': 'mean'
                }).round(3)

                gen_accuracy.columns = ['Total_Votes', 'Human_Accuracy', 'Avg_Confidence']
                gen_accuracy = gen_accuracy.reset_index()
                gen_accuracy = gen_accuracy.sort_values('Human_Accuracy', ascending=False)

                # Show table
                display_df = gen_accuracy.copy()
                display_df['Human_Accuracy'] = display_df['Human_Accuracy'].apply(lambda x: f"{x:.1%}")
                display_df['Avg_Confidence'] = display_df['Avg_Confidence'].apply(lambda x: f"{x:.1%}")
                st.dataframe(display_df, hide_index=True)

                # Show hardest vs easiest generators
                if len(gen_accuracy) > 1:
                    hardest = gen_accuracy.iloc[-1]['generator_model']
                    easiest = gen_accuracy.iloc[0]['generator_model']
                    st.success(f"**Easiest to detect**: {easiest}")
                    st.error(f"**Hardest to detect**: {hardest}")

            with col2:
                # Visualization
                if len(gen_accuracy) > 1:
                    fig_gen = px.bar(
                        gen_accuracy,
                        x='generator_model',
                        y='Human_Accuracy',
                        title="Human Detection Accuracy by AI Generator",
                        labels={'Human_Accuracy': 'Human Accuracy', 'generator_model': 'AI Generator'},
                        color='Human_Accuracy',
                        color_continuous_scale='RdYlGn'
                    )
                    fig_gen.update_layout(
                        yaxis=dict(tickformat='.0%'),
                        xaxis_tickangle=45
                    )
                    st.plotly_chart(fig_gen, use_container_width=True)

                # Generator distribution
                gen_dist = gen_votes['generator_model'].value_counts()
                st.markdown("**Sample Distribution:**")
                for gen, count in gen_dist.items():
                    st.write(f"• **{gen}**: {count} images")

            # Art Style Analysis (if available)
            if has_style_data:
                st.markdown("---")
                st.markdown("**Art Style Analysis**")

                style_votes = gen_votes[gen_votes['art_style'].notna()].copy()

                if len(style_votes) > 0:
                    col1, col2 = st.columns(2)

                    with col1:
                        # Accuracy by art style
                        style_accuracy = style_votes.groupby('art_style').agg({
                            'correct': ['count', 'mean']
                        }).round(3)
                        style_accuracy.columns = ['Total_Votes', 'Human_Accuracy']
                        style_accuracy = style_accuracy.reset_index()
                        style_accuracy = style_accuracy.sort_values('Human_Accuracy', ascending=False)

                        display_df = style_accuracy.copy()
                        display_df['Human_Accuracy'] = display_df['Human_Accuracy'].apply(lambda x: f"{x:.1%}")
                        st.dataframe(display_df, hide_index=True)

                    with col2:
                        # Heatmap: Generator vs Style
                        if len(style_votes) > 10:  # Only if enough data
                            heatmap_data = style_votes.groupby(['generator_model', 'art_style'])['correct'].mean().unstack(fill_value=0)

                            if len(heatmap_data) > 1 and len(heatmap_data.columns) > 1:
                                fig_heatmap = px.imshow(
                                    heatmap_data,
                                    title="Detection Accuracy: Generator × Art Style",
                                    labels=dict(x="Art Style", y="Generator Model", color="Human Accuracy"),
                                    color_continuous_scale="RdYlGn"
                                )
                                st.plotly_chart(fig_heatmap, use_container_width=True)

            # Export enhanced data
            st.markdown("---")
            csv = gen_votes.to_csv(index=False)
            st.download_button(
                "Download Generator Analysis Data",
                csv,
                "generator_analysis_data.csv",
                mime="text/csv"
            )
    else:
        st.info(" **Generator model data will appear here once you start using the new secure dataset**")
        st.markdown("""
        The new enhanced dataset includes:
        - **6 AI Generators**: DALL-E, Stable Diffusion, Midjourney, etc.
        - **Art Style Metadata**: Detailed style classifications
        - **Balanced Samples**: Equal representation across generators

        This analysis will show which generators are hardest for humans to detect!
        """)

with tab4:
    st.markdown("**Reasoning Analysis**")

    # Get responses with reasoning
    reasoning_votes = votes[(votes['reasoning'].notna()) & (votes['reasoning'] != '') & (votes['reasoning'] != 'Skipped - image seen before')]

    if not reasoning_votes.empty:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("**Sample Reasoning Responses:**")

            # Show recent reasoning responses
            sample_reasoning = reasoning_votes.tail(10)[['image_id', 'human_choice', 'correct', 'reasoning']].copy()
            sample_reasoning['status'] = sample_reasoning['correct'].map({True: 'Correct', False: 'Incorrect'})

            for _, row in sample_reasoning.iterrows():
                st.write(f"**{row['status']}** ({row['human_choice']}): *{row['reasoning']}*")

        with col2:
            st.markdown("**Reasoning Statistics:**")
            st.metric("Responses with Reasoning", f"{len(reasoning_votes)}/{len(votes)}")

            # Reasoning length analysis
            reasoning_votes['reasoning_length'] = reasoning_votes['reasoning'].str.len()
            avg_length = reasoning_votes['reasoning_length'].mean()
            st.metric("Average Reasoning Length", f"{avg_length:.0f} chars")

            # Accuracy with vs without reasoning
            with_reasoning_acc = reasoning_votes['correct'].mean()
            without_reasoning_acc = votes[votes['reasoning'].isin(['', 'Skipped - image seen before']) | votes['reasoning'].isna()]['correct'].mean()

            st.write("**Accuracy Comparison:**")
            st.write(f"With reasoning: {with_reasoning_acc:.1%}")
            st.write(f"Without reasoning: {without_reasoning_acc:.1%}")
    else:
        st.info("No reasoning data available yet.")

with tab5:
    st.markdown("**Research Data Export**")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Summary Statistics (Copy for Papers):**")
        export_stats = f"""

Research Dataset Summary:
- Participants: N = {metrics['n_participants']}
- Valid responses: N = {metrics['n_responses']}
- Human median accuracy: {metrics['median_accuracy']:.1%} (IQR: {participant_accuracies['accuracy'].quantile(0.25):.1%}-{participant_accuracies['accuracy'].quantile(0.75):.1%})
- AI detector accuracy: {metrics['ai_accuracy']:.1%}
- Performance gap: {metrics['median_accuracy'] - metrics['ai_accuracy']:+.1%}
- Response time median: {metrics['response_times'].median():.1f}s
- Reasoning responses: {len(reasoning_votes)}/{len(votes)} ({len(reasoning_votes)/len(votes):.1%})

Data collection period: {votes['timestamp'].min().date()} to {votes['timestamp'].max().date()}
        """
        st.code(export_stats)

    with col2:
        st.markdown("**Data Quality Checks:**")

        # Data quality metrics
        skipped_responses = len(votes[votes['human_choice'] == 'seen'])
        invalid_responses = len(votes[votes['human_choice'].isin(['', None])])

        quality_data = {
            "Check": [
                "Total responses",
                "Valid responses",
                "Skipped (seen before)",
                "Invalid responses",
                "Participants with >5 responses",
                "Images with >1 response"
            ],
            "Count": [
                len(votes),
                len(votes[votes['human_choice'] != 'seen']),
                skipped_responses,
                invalid_responses,
                len(participant_accuracies[participant_accuracies['n_responses'] > 5]),
                votes['image_id'].value_counts().sum()
            ],
            "Status": [
                "Info",
                "Good" if metrics['n_responses'] > 50 else "Warning",
                "Info",
                "Good" if invalid_responses == 0 else "Warning",
                "Good" if len(participant_accuracies[participant_accuracies['n_responses'] > 5]) > 3 else "Warning",
                "Good"
            ]
        }
        st.dataframe(pd.DataFrame(quality_data), hide_index=True)

with tab6:
    st.markdown("**Per-Style Accuracy Analysis**")

    if not votes.empty and not images_meta.empty:
        # Merge votes with image metadata to get style information
        merged = votes.merge(images_meta, on="image_id", how="left", suffixes=('', '_meta'))
        # Use the true_label from votes (not from metadata)
        merged["correct"] = merged["human_choice"] == merged["true_label"]

        if 'art_style' in merged.columns:
            # Calculate per-style accuracy
            style_acc = merged.groupby("art_style").agg(
                total=("correct", "size"),
                acc=("correct", "mean")
            ).reset_index().sort_values("acc", ascending=False)

            col1, col2 = st.columns([1, 1])

            with col1:
                st.dataframe(style_acc, hide_index=True)

                # Download button for style accuracy data
                csv = merged.to_csv(index=False)
                st.download_button(
                    "Download votes with styles CSV",
                    csv,
                    "votes_with_styles.csv",
                    mime="text/csv"
                )

            with col2:
                # Create bar chart of per-style accuracy
                if len(style_acc) > 0:
                    fig_style = px.bar(
                        style_acc,
                        x='art_style',
                        y='acc',
                        title="Accuracy by Art Style",
                        labels={'acc': 'Accuracy', 'art_style': 'Art Style'}
                    )
                    fig_style.update_yaxes(tickformat='.0%')
                    fig_style.update_xaxes(tickangle=45)
                    st.plotly_chart(fig_style, use_container_width=True)
        else:
            st.info("Style information not available in image metadata.")
    else:
        st.info("No data available for style analysis.")

with tab7:
    st.markdown("**AI Detector Predictions**")

    col1, col2 = st.columns([2, 1])

    with col1:
        if 'detector_pred' in votes.columns and votes['detector_pred'].notna().any():
            st.markdown("**Sample Detector Predictions:**")
            # Create a summary table from votes data
            detector_summary = votes[['image_id', 'true_label', 'detector_pred', 'detector_confidence']].copy()
            detector_summary['detector_correct'] = detector_summary['detector_pred'] == detector_summary['true_label']
            st.dataframe(detector_summary.head(200), hide_index=True)

            # Download button for detector predictions
            csv = detector_summary.to_csv(index=False)
            st.download_button(
                "Download detector predictions CSV",
                csv,
                "detector_predictions.csv",
                mime="text/csv"
            )
        else:
            st.info("No detector predictions available.")

    with col2:
        if 'detector_pred' in votes.columns and votes['detector_pred'].notna().any():
            st.markdown("**Detector Statistics:**")
            detector_data = votes[votes['detector_pred'].notna()]
            st.metric("Total Predictions", len(detector_data))

            if 'detector_confidence' in detector_data.columns and detector_data['detector_confidence'].notna().any():
                avg_confidence = detector_data['detector_confidence'].mean()
                st.metric("Average Confidence", f"{avg_confidence:.1%}")

            ai_pred_count = (detector_data['detector_pred'] == 'ai').sum()
            human_pred_count = (detector_data['detector_pred'] == 'human').sum()
            st.metric("AI Predictions", ai_pred_count)
            st.metric("Human Predictions", human_pred_count)
        else:
            st.info("No detector statistics available.")




# PARTICIPANT DETAILS EXPORT

st.markdown("---")
st.subheader("Participant Details")

if st.checkbox("Show individual participant data"):
    # Detailed participant data
    participant_columns = ['participant_id', 'started_at', 'finished_at',
                          'pre_confidence', 'pre_training', 'user_type', 'years_experience', 'art_mediums',
                          'ai_familiarity', 'ai_frequency', 'difficulty', 'visual_cues', 'hardest_styles',
                          'labeling_importance', 'encountered_unlabeled', 'concerns', 'detection_value',
                          'visibility_impact', 'emotions', 'additional_comments']

    # Only include columns that exist in the participants dataframe
    available_columns = ['participant_id'] + [col for col in participant_columns[1:] if col in participants.columns]

    detailed = participant_accuracies.merge(
        participants[available_columns],
        on='participant_id',
        how='left'
    )

    st.dataframe(detailed, hide_index=True)

    if st.button("Export detailed participant data as CSV"):
        csv = detailed.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"participant_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

# VOTES TABLE EXPORT

st.markdown("---")
st.subheader("Votes Data Export")

if st.checkbox("Show votes table and export options"):
    st.markdown("**All Votes Data:**")
    st.dataframe(votes, hide_index=True)

    if st.button("Export all votes data as CSV"):
        csv = votes.to_csv(index=False)
        st.download_button(
            label="Download Votes CSV",
            data=csv,
            file_name=f"votes_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

st.markdown("---")
st.caption("Scientific Dashboard - Confidential Research Data")