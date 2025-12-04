# pages/3_Leaderboard.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from backend.utils import init_db, get_engine

# Page configuration
st.set_page_config(page_title="Leaderboard", layout="wide", initial_sidebar_state="collapsed")

# Initialize database
init_db()
engine = get_engine()

st.title("Leaderboard")
st.markdown("See how humans perform against AI in detecting AI-generated art!")

# Add CSS for center-aligned table
st.markdown("""
<style>
.stDataFrame table {
    text-align: center !important;
}

.stDataFrame th {
    text-align: center !important;
}

.stDataFrame td {
    text-align: center !important;
}
</style>
""", unsafe_allow_html=True)

with engine.begin() as conn:
    votes = pd.read_sql("SELECT * FROM votes", conn)
    participants = pd.read_sql("SELECT * FROM participants", conn)

if votes.empty:
    st.info("No scores yet! Be the first to take the test and see your results here.")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button(" Take the Test", type="primary", use_container_width=True):
            st.switch_page("pages/1_Take_the_Test.py")
    st.stop()

# Calculate performance metrics
votes['correct'] = votes['human_choice'] == votes['true_label']

# Remove 'seen' responses for accuracy calculation
valid_votes = votes[votes['human_choice'] != 'seen']

# Human performance
participant_accuracies = valid_votes.groupby('participant_id')['correct'].mean()
human_median = np.median(participant_accuracies)

# AI detector performance
ai_accuracy = (valid_votes['detector_pred'] == valid_votes['true_label']).mean()

# Sample size
n_participants = len(participant_accuracies)
n_votes = len(valid_votes)




# ==============================================
# LEADERBOARD TABLE
# ==============================================


# st.markdown("---")
# st.subheader(" Detection Performance Leaderboard")

# Calculate metrics for table
total_human_votes = len(valid_votes)
total_ai_votes = len(valid_votes)  # AI detector processes same votes

human_correct = valid_votes['correct'].sum()
human_failures = total_human_votes - human_correct

ai_correct = (valid_votes['detector_pred'] == valid_votes['true_label']).sum()
ai_failures = total_ai_votes - ai_correct

# Determine ranking
if human_median > ai_accuracy:
    human_rank = "1st place"
    ai_rank = "2nd place"
elif ai_accuracy > human_median:
    human_rank = "2nd place"
    ai_rank = "1st place"
else:
    human_rank = "it's a tie"  # Tie - both get rank 1
    ai_rank = "it's a tie"   

# Create leaderboard table
leaderboard_data = {
    "Rank": [human_rank, ai_rank],
    "Detector Type": ["Humans", "AI Detector"],
    "Detection Accuracy": [f"{human_median:.1%}", f"{ai_accuracy:.1%}"],
    "No. of Detection Failures": [human_failures, ai_failures],
    "No. of Successful Detections": [human_correct, ai_correct]
}

leaderboard_df = pd.DataFrame(leaderboard_data)

# Sort by rank
leaderboard_df = leaderboard_df.sort_values("Rank")

# Display table with custom styling
st.dataframe(
    leaderboard_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Rank": st.column_config.TextColumn("Rank"),
        "Detector Type": st.column_config.TextColumn("Detector Type"),
        "Detection Accuracy": st.column_config.TextColumn("Detection Accuracy"),
        "No. of Detection Failures": st.column_config.NumberColumn("Detection Failures", format="%d"),
        "No. of Successful Detections": st.column_config.NumberColumn("Successful Detections", format="%d")
    }
)


# ==============================================
# BARCHART VISUALIZATION
# ==============================================

st.markdown("---")

# st.subheader("Human vs AI Performance:")
# st.subheader("Who's Better at Detecting AI Art?")

# Create a simple bar chart
fig = go.Figure()

fig.add_trace(go.Bar(
    x=['Humans (Median)', 'AI Detector'],
    y=[human_median, ai_accuracy],
    marker_color=['#1f77b4', '#ff7f0e'],
    text=[f'{human_median:.0%}', f'{ai_accuracy:.0%}'],
    textposition='auto',
    textfont_size=16
))

fig.update_layout(
    title="Who's Better at Detecting AI Art?",
    yaxis_title="Accuracy",
    yaxis=dict(tickformat='.0%', range=[0, 1]),
    height=400,
    showlegend=False,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)'
)

st.plotly_chart(fig, use_container_width=True)



# ==============================================
# PARTICIPATION STATS
# ==============================================

st.markdown("---")
st.subheader("Participation Stats")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Participants", n_participants)

with col2:
    st.metric("Images Classified", n_votes)

with col3:
    unique_images = valid_votes['image_id'].nunique()
    st.metric("Unique Images", unique_images)


# ==============================================
# CALL TO ACTION
# ==============================================

# if n_participants < 100:
#     st.markdown("---")
#     st.info(f" Help us reach 10 participants! We currently have {n_participants}.")

#     col1, col2, col3 = st.columns([1, 2, 1])
#     with col2:
#         if st.button(" Take the Test Now", type="primary", use_container_width=True):
#             st.switch_page("pages/1_Take_the_Test.py")



