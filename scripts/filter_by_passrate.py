"""Filter training data by pass rate for effective GRPO training.

Problems with 0% or 100% pass rate produce zero reward variance.
Keep only problems with 10-90% pass rate (model sometimes correct).
"""
import pandas as pd
import json

df = pd.read_parquet("/scratch/metacognition/rollouts/rollouts_final.parquet")
print(f"Total rollouts: {len(df)}")

# Compute per-problem pass rate
pass_rates = df.groupby("problem_id")["is_correct"].mean().reset_index()
pass_rates.columns = ["problem_id", "pass_rate"]
print(f"Total problems: {len(pass_rates)}")
print(f"Pass rate distribution:")
print(f"  0% (always wrong):  {(pass_rates['pass_rate'] == 0).sum()}")
print(f"  1-9%:               {((pass_rates['pass_rate'] > 0) & (pass_rates['pass_rate'] < 0.1)).sum()}")
print(f"  10-50%:             {((pass_rates['pass_rate'] >= 0.1) & (pass_rates['pass_rate'] <= 0.5)).sum()}")
print(f"  51-90%:             {((pass_rates['pass_rate'] > 0.5) & (pass_rates['pass_rate'] <= 0.9)).sum()}")
print(f"  91-99%:             {((pass_rates['pass_rate'] > 0.9) & (pass_rates['pass_rate'] < 1.0)).sum()}")
print(f"  100% (always right): {(pass_rates['pass_rate'] == 1.0).sum()}")

# Filter: keep 10-90% pass rate
good = pass_rates[(pass_rates["pass_rate"] >= 0.1) & (pass_rates["pass_rate"] <= 0.9)]
print(f"\nFiltered: {len(good)} problems (pass rate 10-90%)")

# Build GRPO training data (same format as verl_train.parquet)
problems = df.drop_duplicates("problem_id")[["problem_id", "question", "gold_answer"]]
filtered = problems[problems["problem_id"].isin(good["problem_id"])]

verl_data = []
for _, row in filtered.iterrows():
    verl_data.append({
        "data_source": "metacot_math",
        "prompt": [{"role": "user", "content": row["question"]}],
        "reward_model": {"ground_truth": row["gold_answer"]},
    })

out_df = pd.DataFrame(verl_data)
out_df.to_parquet("/scratch/metacognition/verl_train_filtered.parquet", index=False)
print(f"Saved: verl_train_filtered.parquet ({len(out_df)} problems)")
