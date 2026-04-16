"""Generate V3 Meta-CoT data via TRAPI (difficulty-adaptive meta)."""
import json, os, sys, time, random, re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.metacot.prompt_v3 import META_COT_V3_SYSTEM_PROMPT, build_v3_prompt, META_START, META_END

from openai import AzureOpenAI
from azure.identity import AzureCliCredential, get_bearer_token_provider

provider = get_bearer_token_provider(AzureCliCredential(), "api://trapi/.default")

def get_client():
    return AzureOpenAI(
        azure_endpoint="https://trapi.research.microsoft.com/gcr/shared",
        api_key=provider(), api_version="2025-04-01-preview"
    )

# ─── Load questions with difficulty labels ───
from datasets import load_dataset

questions = []

# GSM8K train → easy (pass_rate 0.85)
print("Loading GSM8K...")
gsm = load_dataset("openai/gsm8k", "main", split="train")
for row in gsm:
    ans = row["answer"].split("####")[-1].strip() if "####" in row["answer"] else row["answer"]
    questions.append({"q": row["question"], "gt": ans, "pr": 0.85, "src": "gsm8k"})
random.seed(42)
random.shuffle(questions)
questions_gsm = questions[:2500]
print(f"  GSM8K: {len(questions_gsm)} easy questions")

# MATH train → medium/hard
print("Loading MATH...")
questions_math = []
try:
    math_ds = load_dataset("hendrycks/competition_mathematics", split="train")
    for row in math_ds:
        level = row.get("level", "Level 3")
        level_num = int(re.search(r'\d', level).group()) if re.search(r'\d', level) else 3
        if level_num <= 2:
            pr = 0.7  # medium-easy
        elif level_num <= 3:
            pr = 0.5  # medium
        elif level_num <= 4:
            pr = 0.3  # hard
        else:
            pr = 0.15  # very hard
        questions_math.append({"q": row["problem"], "gt": row.get("answer", row.get("solution", "")), "pr": pr, "src": "math"})
    random.shuffle(questions_math)
    questions_math = questions_math[:2500]
    print(f"  MATH: {len(questions_math)} medium-hard questions")
except Exception as e:
    print(f"  MATH load failed: {e}")
    # Fallback: MATH-500
    math_ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    for row in math_ds:
        questions_math.append({"q": row["problem"], "gt": row.get("answer", ""), "pr": 0.4, "src": "math500"})
    print(f"  MATH-500 fallback: {len(questions_math)} questions")

all_questions = questions_gsm + questions_math
random.shuffle(all_questions)
N = len(all_questions)
print(f"\nTotal: {N} questions")

# ─── Generate ───
results = []
failed = 0
client = get_client()
client_refresh = time.time()

def gen_one(idx):
    global client, client_refresh
    if time.time() - client_refresh > 1800:
        try:
            client = get_client()
            client_refresh = time.time()
        except:
            pass

    q = all_questions[idx]
    prompt = build_v3_prompt(q["q"], q["pr"])

    for attempt in range(5):
        try:
            resp = client.responses.create(
                model="gpt-5.4-mini_2026-03-17",
                input=[
                    {"role": "system", "content": META_COT_V3_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
            )
            text = resp.output_text
            if META_START in text and "boxed" in text:
                meta_count = text.count(META_START)
                return {
                    "q": q["q"], "text": text, "gt": q["gt"],
                    "pr": q["pr"], "src": q["src"], "meta_count": meta_count,
                }
        except Exception as e:
            if "429" in str(e):
                time.sleep(5 * (2**attempt) + random.random()*3)
            else:
                time.sleep(3)
    return None

print(f"Generating {N} V3 chains with gpt-5.4-mini...")
with ThreadPoolExecutor(max_workers=10) as ex:
    futures = {ex.submit(gen_one, i): i for i in range(N)}
    for i, f in enumerate(as_completed(futures)):
        r = f.result()
        if r:
            results.append(r)
        else:
            failed += 1
        if (i+1) % 100 == 0:
            print(f"  {i+1}/{N}: {len(results)} valid, {failed} failed", flush=True)

# ─── Save V3 SFT data ───
import pandas as pd

records_meta = []
records_base = []
for r in results:
    messages_meta = json.dumps([
        {"role": "user", "content": r["q"]},
        {"role": "assistant", "content": r["text"]},
    ])
    records_meta.append({"messages": messages_meta, "source": f"metacot_v3_{r['src']}"})

    # Base version: strip meta blocks
    base_text = re.sub(r'<\|meta\|>.*?<\|/meta\|>\s*', '', r["text"], flags=re.DOTALL).strip()
    messages_base = json.dumps([
        {"role": "user", "content": r["q"]},
        {"role": "assistant", "content": base_text},
    ])
    records_base.append({"messages": messages_base, "source": f"base_v3_{r['src']}"})

pd.DataFrame(records_meta).to_parquet("/tmp/metacot_v3_trapi.parquet")
pd.DataFrame(records_base).to_parquet("/tmp/base_sft_v3.parquet")

# ─── Stats ───
all_confs = []
error_fix = 0
meta_by_diff = {"easy": [], "medium": [], "hard": []}
token_ratios = []

for r in results:
    confs = re.findall(r'(?:probability|confidence)[\s\w:]*?(\d+\.\d+)', r["text"], re.I)
    for c in confs:
        v = float(c)
        if 0 < v <= 1:
            all_confs.append(v)
    if re.search(r'\b(wait|wrong|fix|actually|mistake|forgot|oops)\b', r["text"], re.I):
        error_fix += 1

    # Meta count by difficulty
    if r["pr"] > 0.8:
        meta_by_diff["easy"].append(r["meta_count"])
    elif r["pr"] > 0.4:
        meta_by_diff["medium"].append(r["meta_count"])
    else:
        meta_by_diff["hard"].append(r["meta_count"])

    # Token ratio
    total_words = len(r["text"].split())
    meta_blocks = re.findall(r'<\|meta\|>(.*?)<\|/meta\|>', r["text"], re.DOTALL)
    meta_words = sum(len(b.split()) for b in meta_blocks)
    token_ratios.append(meta_words / max(total_words, 1))

import numpy as np
print(f"\n{'='*60}")
print(f"  V3 DATA GENERATION COMPLETE")
print(f"{'='*60}")
print(f"Valid: {len(results)}/{N} ({len(results)/N:.1%})")
print(f"Failed: {failed}")
print(f"\nConfidence: mean={np.mean(all_confs):.3f}, >0.95={sum(1 for c in all_confs if c>0.95)/len(all_confs):.1%}")
print(f"Error-fix patterns: {error_fix}/{len(results)} ({error_fix/len(results):.1%})")
print(f"\nMeta blocks by difficulty:")
for diff, counts in meta_by_diff.items():
    if counts:
        print(f"  {diff}: mean={np.mean(counts):.1f}, 1-block={sum(1 for c in counts if c==1)/len(counts):.0%}")
print(f"\nMeta token ratio: mean={np.mean(token_ratios):.1%} (V2 was 31.1%)")
print(f"\nSaved: /tmp/metacot_v3_trapi.parquet ({len(records_meta)} rows)")
print(f"Saved: /tmp/base_sft_v3.parquet ({len(records_base)} rows)")
