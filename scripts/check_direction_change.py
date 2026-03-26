"""Check if Meta-CoT data has direction changes (confidence decrease)."""
import pandas as pd
import json
import re

df = pd.read_parquet("/scratch/metacognition/sft_data/metacot_sft.parquet")
print(f"Total: {len(df)} chains")

n_decrease = 0
n_multi_meta = 0
examples = []

for idx, row in df.iterrows():
    msgs = json.loads(row["messages"])
    text = msgs[-1]["content"]

    blocks = re.findall(r"<\|meta\|>(.*?)<\|/meta\|>", text, re.DOTALL)
    confs = []
    for b in blocks:
        ms = re.findall(r"(?:probability|confidence)[:\s]*(\d+(?:\.\d+)?)", b, re.IGNORECASE)
        for m in ms:
            c = float(m)
            if c > 1:
                c /= 100
            confs.append(c)

    if len(confs) >= 2:
        n_multi_meta += 1
        for i in range(1, len(confs)):
            if confs[i] < confs[i - 1] - 0.1:
                n_decrease += 1
                if len(examples) < 3:
                    examples.append({"confs": confs, "text": text[:400]})
                break

print(f"Multi-meta chains: {n_multi_meta}")
print(f"Confidence DECREASE: {n_decrease} ({n_decrease / max(n_multi_meta, 1) * 100:.1f}%)")
print()
for i, ex in enumerate(examples):
    print(f"Example {i+1}: confs={ex['confs']}")
    print(f"  {ex['text'][:300]}...")
    print()
