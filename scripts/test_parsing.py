"""Test math-verify parsing on actual training data."""
import pandas as pd
import json
import sys
sys.path.insert(0, "/scratch/metacognition")
from src.training.rewards import _check_correctness, HAS_MATH_VERIFY

print(f"math-verify available: {HAS_MATH_VERIFY}")

df = pd.read_parquet("/scratch/metacognition/verl_train_filtered.parquet")
print(f"Data: {len(df)} rows")

# Test first 10 problems
correct = 0
fail_examples = []
for i in range(min(10, len(df))):
    rm = json.loads(df.iloc[i]["reward_model"])
    gt = rm["ground_truth"]

    # Simulate model output: extract answer from GT, wrap in boxed
    import re
    boxed = re.findall(r'\\boxed\{[^}]+\}', gt)
    if boxed:
        fake_pred = f"Solution... {boxed[-1]}"
    else:
        nums = re.findall(r'(-?\d+(?:\.\d+)?)', gt)
        fake_pred = f"\\boxed{{{nums[-1]}}}" if nums else "no answer"

    result = _check_correctness(fake_pred, gt)
    if result:
        correct += 1
    else:
        fail_examples.append(f"Row {i}: pred={fake_pred[:50]} gt_end={gt[-60:]}")

print(f"\nParsing: {correct}/10 correct")
for ex in fail_examples[:3]:
    print(f"  FAIL: {ex}")
