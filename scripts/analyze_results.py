# Database analysis script
"""
Quick database analysis script for The Detection Game
"""

import sys
import os
sys.path.append('src')

from utils import get_engine
import pandas as pd
import numpy as np

def main():
    print("THE DETECTION GAME - DATABASE ANALYSIS")
    print("=" * 50)

    engine = get_engine()

    # Load all votes (excluding skipped ones)
    votes_df = pd.read_sql_query("""
        SELECT * FROM votes
        WHERE human_choice != 'seen'
        ORDER BY timestamp_utc DESC
    """, engine)

    if len(votes_df) == 0:
        print("No classification votes found in database yet.")
        print("Run some tests first, then come back to analyze!")
        return

    print(f"Total classification votes: {len(votes_df)}")
    print(f"Unique participants: {votes_df['participant_id'].nunique()}")

    # Calculate accuracy
    votes_df['correct'] = votes_df['true_label'] == votes_df['human_choice']
    overall_accuracy = votes_df['correct'].mean()

    print(f"Overall human accuracy: {overall_accuracy:.2%}")

    # Human vs AI detector comparison
    votes_df['detector_correct'] = votes_df['true_label'] == votes_df['detector_pred']
    detector_accuracy = votes_df['detector_correct'].mean()

    print(f"AI detector accuracy: {detector_accuracy:.2%}")

    if detector_accuracy > overall_accuracy:
        diff = detector_accuracy - overall_accuracy
        print(f"AI detector wins by {diff:.1%}")
    elif overall_accuracy > detector_accuracy:
        diff = overall_accuracy - detector_accuracy
        print(f"Humans win by {diff:.1%}")
    else:
        print("It's a tie!")

    # Label-specific accuracy
    print("\n📈 ACCURACY BY LABEL:")
    label_accuracy = votes_df.groupby('true_label')['correct'].agg(['count', 'mean'])
    for label, data in label_accuracy.iterrows():
        print(f"  {label.title()}: {data['mean']:.2%} ({data['count']} votes)")

    # Generator model analysis (for new data)
    if 'generator_model' in votes_df.columns and votes_df['generator_model'].notna().any():
        print("\n🎨 ACCURACY BY GENERATOR MODEL:")
        gen_accuracy = votes_df[votes_df['generator_model'].notna()].groupby('generator_model')['correct'].agg(['count', 'mean'])
        gen_accuracy_sorted = gen_accuracy.sort_values('mean')

        for generator, data in gen_accuracy_sorted.iterrows():
            if data['count'] >= 3:  # Only show if enough samples
                print(f"  {generator}: {data['mean']:.2%} ({data['count']} votes)")

    # Confidence analysis
    print(f"\nCONFIDENCE ANALYSIS:")
    avg_confidence = votes_df['confidence'].mean()
    print(f"  Average confidence: {avg_confidence:.2%}")

    # Confidence vs accuracy correlation
    high_conf = votes_df[votes_df['confidence'] >= 0.8]['correct'].mean()
    low_conf = votes_df[votes_df['confidence'] <= 0.5]['correct'].mean()
    print(f"  High confidence (≥80%): {high_conf:.2%} accuracy")
    print(f"  Low confidence (≤50%): {low_conf:.2%} accuracy")

    # Response time analysis
    print(f"\nRESPONSE TIME ANALYSIS:")
    avg_time = votes_df['response_time_ms'].mean() / 1000
    print(f"  Average response time: {avg_time:.1f} seconds")

    # Participants analysis
    participants_df = pd.read_sql_query("SELECT * FROM participants", engine)
    print(f"\nPARTICIPANTS OVERVIEW:")
    print(f"  Total registered: {len(participants_df)}")
    print(f"  Completed tests: {votes_df['participant_id'].nunique()}")

    # Recent activity
    if len(votes_df) > 0:
        latest_vote = pd.to_datetime(votes_df['timestamp_utc'].iloc[0])
        print(f"  Most recent test: {latest_vote.strftime('%Y-%m-%d %H:%M')}")

    print("\nAnalysis complete!")
    print("\nTip: Use the Research Dashboard in your web app for interactive analysis!")

if __name__ == "__main__":
    main()