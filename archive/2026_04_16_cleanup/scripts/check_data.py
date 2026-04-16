"""Check training data ground_truth format and test reward parsing."""
import pandas as pd
import json
import re
import sys
sys.path.insert(0, "/scratch/metacognition")

df = pd.read_parquet("/scratch/metacognition/verl_train_filtered.parquet")
print(f"Total: {len(df)} rows")

from src.training.rewards import _extract_answer, _check_correctness

# Analyze ground_truth formats
boxed = hash4 = ans_is = num_only = none_found = 0
for i in range(len(df)):
    rm = json.loads(df.iloc[i]["reward_model"]) if isinstance(df.iloc[i].get("reward_model"), str) else df.iloc[i].get("reward_model", {})
    gt = str(rm.get("ground_truth", ""))
    extracted = _extract_answer(gt)
    if extracted:
        boxed += 1
    elif "####" in gt:
        hash4 += 1
    else:
        # Try our new fallback
        m = re.search(r'(?:answer|result)\s+(?:is|=)\s+[\\$]*(-?[\d,.]+)', gt, re.I)
        nums = re.findall(r'(-?\d+(?:\.\d+)?)', gt)
        if m:
            ans_is += 1
        elif nums:
            num_only += 1
        else:
            none_found += 1

print(f"boxed: {boxed}")
print(f"####: {hash4}")
print(f"answer_is: {ans_is}")
print(f"number_fallback: {num_only}")
print(f"no_answer: {none_found}")

# Test actual parsing on 10 samples
print("\n=== Sample parsing test ===")
correct_count = 0
for i in range(min(20, len(df))):
    rm = json.loads(df.iloc[i]["reward_model"]) if isinstance(df.iloc[i].get("reward_model"), str) else df.iloc[i].get("reward_model", {})
    gt = str(rm.get("ground_truth", ""))
    extracted = _extract_answer(gt)
    # Simulate model output
    if extracted:
        fake_pred = f"\\boxed{{{extracted}}}"
        result = _check_correctness(fake_pred, gt)
        correct_count += 1 if result else 0
        if not result:
            print(f"  FAIL row {i}: extracted={extracted[:30]} gt_end={gt[-50:]}")
    else:
        # Try fallback
        m = re.search(r'(?:answer|result)\s+(?:is|=)\s+[\\$]*(-?[\d,.]+)', gt, re.I)
        nums = re.findall(r'(-?\d+(?:\.\d+)?)', gt)
        fallback = m.group(1) if m else (nums[-1] if nums else "?")
        fake_pred = f"\\boxed{{{fallback}}}"
        result = _check_correctness(fake_pred, gt)
        correct_count += 1 if result else 0
        if not result:
            print(f"  FAIL row {i}: fallback={fallback} gt_end={gt[-50:]}")

print(f"\nParsing success: {correct_count}/20")
