"""Check rollout data availability for Gnosis training."""
import pandas as pd
import sys

try:
    df = pd.read_parquet("/scratch/metacognition/rollouts/rollouts_final.parquet")
    print(f"Rollouts: {len(df)}")
    print(f"Columns: {list(df.columns)[:10]}")
    correct = df["is_correct"].sum() if "is_correct" in df.columns else -1
    print(f"Correct: {correct}/{len(df)} ({correct/len(df)*100:.1f}%)")
    print(f"Problems: {df['problem_id'].nunique()}")
    print(f"Sample completion length: {len(str(df.iloc[0].get('completion', '')))}")
except Exception as e:
    print(f"Error: {e}")
