"""Generate V3 Meta-CoT data for HARD math problems (Level 4-5 + Omni-MATH)."""
import json, os, sys, time, random, re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.metacot.prompt_v3 import META_COT_V3_SYSTEM_PROMPT, build_v3_prompt, META_START, META_END

from openai import AzureOpenAI
from azure.identity import ChainedTokenCredential, AzureCliCredential, ManagedIdentityCredential, get_bearer_token_provider

scope = "api://trapi/.default"
credential = get_bearer_token_provider(ChainedTokenCredential(
    AzureCliCredential(),
    ManagedIdentityCredential(),
), scope)

def get_client():
    return AzureOpenAI(
        azure_endpoint="https://trapi.research.microsoft.com/gcr/shared",
        azure_ad_token_provider=credential,
        api_version="2025-04-01-preview",
    )

# ─── Load MATH Level 4-5 (hard) ───
from datasets import load_dataset, concatenate_datasets
import pandas as pd
import numpy as np

questions = []

print("=" * 60)
print("  Loading HARD math datasets")
print("=" * 60)

# 1. MATH Level 4-5 from EleutherAI/hendrycks_math
print("\nLoading MATH (EleutherAI/hendrycks_math)...")
try:
    subjects = ['algebra', 'counting_and_probability', 'geometry',
                'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']
    all_math = []
    for s in subjects:
        ds = load_dataset('EleutherAI/hendrycks_math', s, split='train')
        all_math.append(ds)
    math_ds = concatenate_datasets(all_math)
    print(f"  Total MATH: {len(math_ds)}")

    # Filter Level 4-5 only
    hard_math = [r for r in math_ds if r.get("level", "") in ["Level 4", "Level 5"]]
    print(f"  Level 4-5 (hard): {len(hard_math)}")

    # Extract answer from solution (last \boxed{})
    def extract_boxed(solution):
        """Extract the last \\boxed{...} answer from solution text."""
        matches = re.findall(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', solution)
        return matches[-1] if matches else ""

    for r in hard_math:
        level = r["level"]
        pr = 0.25 if level == "Level 5" else 0.35  # hard
        gt = extract_boxed(r.get("solution", ""))
        questions.append({
            "q": r["problem"], "gt": gt, "pr": pr,
            "src": "math_hard", "level": level, "type": r.get("type", "")
        })

except Exception as e:
    print(f"  MATH load failed: {e}")
    # Fallback: hendrycks/competition_math
    try:
        math_ds = load_dataset("hendrycks/competition_math", split="train")
        hard_math = [r for r in math_ds if r.get("level", "") in ["Level 4", "Level 5"]]
        for r in hard_math:
            pr = 0.25 if r["level"] == "Level 5" else 0.35
            questions.append({
                "q": r["problem"], "gt": r.get("answer", ""), "pr": pr,
                "src": "math_hard", "level": r["level"], "type": r.get("type", "")
            })
        print(f"  Fallback loaded: {len(questions)} hard MATH problems")
    except Exception as e2:
        print(f"  Fallback also failed: {e2}")

# Shuffle and take 1500
random.seed(42)
random.shuffle(questions)
math_questions = questions[:1500]
print(f"  Selected: {len(math_questions)} MATH L4-5 problems")
level_dist = {}
for q in math_questions:
    l = q.get("level", "?")
    level_dist[l] = level_dist.get(l, 0) + 1
print(f"  Level distribution: {level_dist}")

# 2. Omni-MATH (Olympiad level)
print("\nLoading Omni-MATH (KbsdJames/Omni-MATH)...")
omni_questions = []
try:
    omni = load_dataset("KbsdJames/Omni-MATH", split="test")
    print(f"  Total Omni-MATH: {len(omni)}")

    for r in omni:
        diff = r.get("difficulty", 5)
        # Omni-MATH difficulty is 1-10, all are hard
        pr = max(0.05, min(0.3, 0.35 - diff * 0.03))
        omni_questions.append({
            "q": r["problem"], "gt": r.get("answer", ""), "pr": pr,
            "src": "omni_math", "level": f"Olympiad-{diff}", "type": r.get("domain", [""])[0] if isinstance(r.get("domain"), list) else str(r.get("domain", ""))
        })

    random.shuffle(omni_questions)
    omni_questions = omni_questions[:500]
    print(f"  Selected: {len(omni_questions)} Omni-MATH problems")

except Exception as e:
    print(f"  Omni-MATH load failed: {e}")

# Combine
all_questions = math_questions + omni_questions
random.shuffle(all_questions)
N = len(all_questions)
print(f"\n{'='*60}")
print(f"  Total HARD questions to generate: {N}")
print(f"  MATH L4-5: {len(math_questions)}, Omni-MATH: {len(omni_questions)}")
print(f"{'='*60}\n")

# ─── Generate V3 chains ───
results = []
failed = 0
lock_results = __import__('threading').Lock()

client = get_client()
client_lock = __import__('threading').Lock()
client_refresh_time = time.time()

def gen_one(idx):
    global client, client_refresh_time

    # Refresh client every 30 min
    with client_lock:
        if time.time() - client_refresh_time > 1800:
            try:
                client = get_client()
                client_refresh_time = time.time()
                print("  [Client refreshed]", flush=True)
            except:
                pass

    q = all_questions[idx]
    prompt = build_v3_prompt(q["q"], q["pr"])

    for attempt in range(6):
        try:
            resp = client.responses.create(
                model="gpt-5.4-mini_2026-03-17",
                input=[
                    {"role": "system", "content": META_COT_V3_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_output_tokens=4096,
            )
            text = resp.output_text

            # Validate: has meta blocks, has boxed answer, meta count >= 2 for hard
            if META_START not in text:
                continue
            if "boxed" not in text:
                continue
            meta_count = text.count(META_START)
            if meta_count < 2:
                # Hard problems should have 2+ meta blocks, retry
                if attempt < 3:
                    continue

            return {
                "q": q["q"], "text": text, "gt": q["gt"],
                "pr": q["pr"], "src": q["src"], "meta_count": meta_count,
                "level": q.get("level", ""), "type": q.get("type", ""),
            }

        except Exception as e:
            err = str(e)
            if "429" in err or "rate" in err.lower():
                wait = min(120, 5 * (2 ** attempt) + random.random() * 5)
                time.sleep(wait)
            elif "timeout" in err.lower() or "connection" in err.lower():
                time.sleep(10 * (attempt + 1))
            else:
                print(f"  [Error idx={idx}] {err[:100]}", flush=True)
                time.sleep(5)
    return None

print(f"Generating {N} V3 HARD chains with gpt-5.4-mini...")
start_time = time.time()

with ThreadPoolExecutor(max_workers=10) as ex:
    futures = {ex.submit(gen_one, i): i for i in range(N)}
    for i, f in enumerate(as_completed(futures)):
        r = f.result()
        if r:
            with lock_results:
                results.append(r)
        else:
            failed += 1
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed * 60
            eta = (N - i - 1) / rate if rate > 0 else 0
            print(f"  {i+1}/{N}: {len(results)} valid, {failed} failed | "
                  f"{rate:.0f}/min, ETA {eta:.1f}min", flush=True)

elapsed_total = time.time() - start_time
print(f"\nGeneration complete in {elapsed_total/60:.1f} min")

# ─── Save V3 HARD SFT data ───
records_meta = []
records_base = []
for r in results:
    messages_meta = json.dumps([
        {"role": "user", "content": r["q"]},
        {"role": "assistant", "content": r["text"]},
    ])
    records_meta.append({
        "messages": messages_meta,
        "source": f"metacot_v3_{r['src']}",
    })

    # Base version: strip meta blocks
    base_text = re.sub(r'<\|meta\|>.*?<\|/meta\|>\s*', '', r["text"], flags=re.DOTALL).strip()
    messages_base = json.dumps([
        {"role": "user", "content": r["q"]},
        {"role": "assistant", "content": base_text},
    ])
    records_base.append({
        "messages": messages_base,
        "source": f"base_v3_{r['src']}",
    })

df_hard_meta = pd.DataFrame(records_meta)
df_hard_base = pd.DataFrame(records_base)

df_hard_meta.to_parquet("/tmp/metacot_v3_hard.parquet")
df_hard_base.to_parquet("/tmp/base_sft_v3_hard.parquet")
print(f"\nSaved: /tmp/metacot_v3_hard.parquet ({len(df_hard_meta)} rows)")
print(f"Saved: /tmp/base_sft_v3_hard.parquet ({len(df_hard_base)} rows)")

# ─── Merge with existing V3 data ───
print("\n" + "=" * 60)
print("  Merging with existing V3 data")
print("=" * 60)

try:
    v3_base_file = "/tmp/metacot_v3_trapi_fixed.parquet"
    v3_base = pd.read_parquet(v3_base_file)
    print(f"  Existing V3: {len(v3_base)} rows")
    print(f"  Sources: {v3_base['source'].value_counts().to_dict()}")

    # Merge meta versions
    v3_all = pd.concat([v3_base, df_hard_meta], ignore_index=True)
    v3_all.to_parquet("/tmp/metacot_v3_all.parquet")
    print(f"  Merged V3 (all): {len(v3_all)} rows")
    print(f"  Sources: {v3_all['source'].value_counts().to_dict()}")

    # Also merge base versions
    base_existing_file = "/tmp/base_sft_v3.parquet"
    if os.path.exists(base_existing_file):
        base_existing = pd.read_parquet(base_existing_file)
        base_all = pd.concat([base_existing, df_hard_base], ignore_index=True)
    else:
        base_all = df_hard_base
    base_all.to_parquet("/tmp/base_sft_v3_all.parquet")
    print(f"  Merged Base (all): {len(base_all)} rows")

except Exception as e:
    print(f"  Merge failed: {e}")
    print(f"  Hard-only data saved to /tmp/metacot_v3_hard.parquet")

# ─── Upload to HuggingFace ───
print("\n" + "=" * 60)
print("  Uploading to HuggingFace")
print("=" * 60)

try:
    from huggingface_hub import HfApi
    api = HfApi(token="${HF_TOKEN}")

    # Upload merged V3 all
    if os.path.exists("/tmp/metacot_v3_all.parquet"):
        api.upload_file(
            path_or_fileobj="/tmp/metacot_v3_all.parquet",
            path_in_repo="metacot_v3_all.parquet",
            repo_id="iamseungpil/metacot",
            repo_type="dataset",
            commit_message=f"Upload V3 all (easy+hard, {len(v3_all)} chains)"
        )
        print(f"  Uploaded metacot_v3_all.parquet ({len(v3_all)} rows)")

    # Upload hard-only
    api.upload_file(
        path_or_fileobj="/tmp/metacot_v3_hard.parquet",
        path_in_repo="metacot_v3_hard.parquet",
        repo_id="iamseungpil/metacot",
        repo_type="dataset",
        commit_message=f"Upload V3 hard (MATH L4-5 + Omni-MATH, {len(df_hard_meta)} chains)"
    )
    print(f"  Uploaded metacot_v3_hard.parquet ({len(df_hard_meta)} rows)")

    # Upload base version
    if os.path.exists("/tmp/base_sft_v3_all.parquet"):
        api.upload_file(
            path_or_fileobj="/tmp/base_sft_v3_all.parquet",
            path_in_repo="base_sft_v3_all.parquet",
            repo_id="iamseungpil/metacot",
            repo_type="dataset",
            commit_message=f"Upload Base V3 all (meta-stripped, {len(base_all)} chains)"
        )
        print(f"  Uploaded base_sft_v3_all.parquet ({len(base_all)} rows)")

except Exception as e:
    print(f"  HF upload failed: {e}")

# ─── Stats ───
print("\n" + "=" * 60)
print("  V3 HARD DATA GENERATION STATS")
print("=" * 60)

print(f"\nGeneration:")
print(f"  Total attempted: {N}")
print(f"  Valid: {len(results)}/{N} ({len(results)/N:.1%})")
print(f"  Failed: {failed}")
print(f"  Time: {elapsed_total/60:.1f} min ({elapsed_total/N:.1f}s/problem)")

# Meta block analysis
meta_by_src = {"math_hard": [], "omni_math": []}
all_confs = []
error_fix = 0
token_ratios = []

for r in results:
    src = r["src"]
    if src in meta_by_src:
        meta_by_src[src].append(r["meta_count"])

    # Extract confidence values
    confs = re.findall(r'(?:probability|confidence)[\s\w:]*?(\d+\.\d+)', r["text"], re.I)
    for c in confs:
        v = float(c)
        if 0 < v <= 1:
            all_confs.append(v)

    # Error-fix patterns
    if re.search(r'\b(wait|wrong|fix|actually|mistake|forgot|oops)\b', r["text"], re.I):
        error_fix += 1

    # Token ratio
    total_words = len(r["text"].split())
    meta_blocks = re.findall(r'<\|meta\|>(.*?)<\|/meta\|>', r["text"], re.DOTALL)
    meta_words = sum(len(b.split()) for b in meta_blocks)
    token_ratios.append(meta_words / max(total_words, 1))

print(f"\nMeta blocks by source:")
for src, counts in meta_by_src.items():
    if counts:
        print(f"  {src}: n={len(counts)}, mean={np.mean(counts):.1f}, "
              f">=2: {sum(1 for c in counts if c>=2)/len(counts):.0%}, "
              f">=3: {sum(1 for c in counts if c>=3)/len(counts):.0%}")

if all_confs:
    print(f"\nConfidence distribution (hard problems):")
    print(f"  mean={np.mean(all_confs):.3f}, median={np.median(all_confs):.3f}")
    print(f"  <0.3: {sum(1 for c in all_confs if c<0.3)/len(all_confs):.0%}")
    print(f"  0.3-0.5: {sum(1 for c in all_confs if 0.3<=c<0.5)/len(all_confs):.0%}")
    print(f"  0.5-0.7: {sum(1 for c in all_confs if 0.5<=c<0.7)/len(all_confs):.0%}")
    print(f"  >0.7: {sum(1 for c in all_confs if c>=0.7)/len(all_confs):.0%}")

print(f"\nError-fix patterns: {error_fix}/{len(results)} ({error_fix/len(results):.1%})")
print(f"Meta token ratio: mean={np.mean(token_ratios):.1%}")

# Final merged stats
if os.path.exists("/tmp/metacot_v3_all.parquet"):
    merged = pd.read_parquet("/tmp/metacot_v3_all.parquet")
    print(f"\n{'='*60}")
    print(f"  FINAL MERGED V3 DATASET")
    print(f"{'='*60}")
    print(f"  Total rows: {len(merged)}")
    print(f"  Source distribution:")
    for src, cnt in merged['source'].value_counts().items():
        print(f"    {src}: {cnt}")

print(f"\nFiles:")
print(f"  /tmp/metacot_v3_hard.parquet (hard only)")
print(f"  /tmp/metacot_v3_all.parquet (merged easy+hard)")
print(f"  /tmp/base_sft_v3_all.parquet (base, meta-stripped)")
print(f"\nHuggingFace: https://huggingface.co/datasets/iamseungpil/metacot")
