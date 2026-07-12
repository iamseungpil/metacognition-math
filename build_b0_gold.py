import os, json, re, statistics
import pandas as pd
from datasets import load_dataset
from math_verify import parse, verify

DATA = "/home/v-seungplee/metacognition-math/data"

# ---------- helpers ----------
def norm(s):
    # normalize whitespace for matching
    return re.sub(r"\s+", " ", s.strip())

def last_boxed(text):
    # find the content of the LAST \boxed{...} with balanced braces
    idxs = [m.start() for m in re.finditer(r"\\boxed", text)]
    if not idxs:
        return None
    start = idxs[-1]
    i = text.find("{", start)
    if i < 0:
        return None
    depth = 0
    j = i
    while j < len(text):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i+1:j]
        j += 1
    return None

def safe_verify(a, b):
    # returns True if a and b are math-equal
    try:
        pa = parse(a)
        pb = parse(b)
        return verify(pa, pb) or verify(pb, pa)
    except Exception:
        return False

# ---------- 1. load meta arm problems + gold ----------
df = pd.read_parquet(f"{DATA}/b23_rv_unmasked_sft.parquet")
prob2gold = {}
for _, row in df.iterrows():
    msgs = row["messages"]
    if isinstance(msgs, str):
        msgs = json.loads(msgs)
    user = next(m["content"] for m in msgs if m["role"] == "user")
    asst = next(m["content"] for m in msgs if m["role"] == "assistant")
    gold = last_boxed(asst)
    key = user.strip()
    if key not in prob2gold and gold is not None:
        prob2gold[key] = gold

uniq_problems = list(prob2gold.keys())
N = len(uniq_problems)
print(f"[meta] rows={len(df)} unique_problems_with_gold={N}")

# normalized lookup: norm(problem) -> original key
norm2key = {}
for k in uniq_problems:
    norm2key[norm(k)] = k

# ---------- 2. load benchmarks ----------
bench = {}  # norm(problem) -> (source, solution, ans)

# gsm8k
gsm = load_dataset("openai/gsm8k", "main", split="train")
for ex in gsm:
    q = ex["question"]
    a = ex["answer"]
    # ans = number after ####
    m = re.search(r"####\s*(.+)\s*$", a)
    ans = m.group(1).strip().replace(",", "") if m else None
    sol = re.sub(r"<<[^>]*>>", "", a)  # strip calc annotations
    sol = re.sub(r"\n?####.*$", "", sol).strip()  # remove #### line from solution body
    nq = norm(q)
    if nq not in bench:
        bench[nq] = ("gsm8k", sol, ans)

# MATH
MATH_SUBSETS = ["algebra", "prealgebra", "intermediate_algebra", "number_theory",
                "precalculus", "counting_and_probability", "geometry"]
for sub in MATH_SUBSETS:
    ds = load_dataset("EleutherAI/hendrycks_math", sub, split="train")
    for ex in ds:
        q = ex["problem"]
        sol = ex["solution"]
        ans = last_boxed(sol)
        nq = norm(q)
        if nq not in bench:
            bench[nq] = (f"MATH/{sub}", sol, ans)

# omni-math (try)
omni_note = ""
try:
    omni = load_dataset("KbsdJames/Omni-MATH", split="test")
    added = 0
    for ex in omni:
        q = ex.get("problem")
        sol = ex.get("solution")
        if not q or not sol:
            continue
        ans = last_boxed(sol)
        nq = norm(q)
        if nq not in bench:
            bench[nq] = ("omni-math", sol, ans if ans else "")
            added += 1
    omni_note = f"omni-math loaded, added {added} records"
except Exception as e:
    omni_note = f"omni-math SKIPPED: {type(e).__name__}: {e}"
print("[omni]", omni_note)

print(f"[bench] total benchmark records={len(bench)}")

# ---------- 3/4/5. match, format, correctness gate ----------
rows = []
dropped_mismatch = 0
unmatched = []
matched_by_source = {}
mismatch_by_source = {}

for k in uniq_problems:
    nk = norm(k)
    rec = bench.get(nk)
    if rec is None:
        unmatched.append(k)
        continue
    source, sol, ans = rec
    src_root = source.split("/")[0]
    matched_by_source[src_root] = matched_by_source.get(src_root, 0) + 1
    our_gold = prob2gold[k]
    if ans is None or ans == "":
        dropped_mismatch += 1
        mismatch_by_source[src_root] = mismatch_by_source.get(src_root, 0) + 1
        continue
    # correctness gate
    if not safe_verify(ans, our_gold):
        dropped_mismatch += 1
        mismatch_by_source[src_root] = mismatch_by_source.get(src_root, 0) + 1
        continue
    # format assistant turn
    formatted = f"<think>\n{sol.strip()}\n</think>\n\nThe answer is $\\boxed{{{ans}}}$."
    messages = [
        {"role": "user", "content": k},
        {"role": "assistant", "content": formatted},
    ]
    rows.append({
        "messages": json.dumps(messages, ensure_ascii=False),
        "source": source,
        "split_tags": json.dumps({"difficulty": "unknown"}),
    })

out = pd.DataFrame(rows, columns=["messages", "source", "split_tags"])
out.to_parquet(f"{DATA}/b0_gold_sft.parquet", index=False)
print(f"[save] wrote {len(out)} rows to b0_gold_sft.parquet")

# unmatched by trying to guess source is impossible; report count + samples
print(f"[match] matched_by_source={matched_by_source}")
print(f"[match] mismatch_dropped_by_source={mismatch_by_source}")
print(f"[match] unmatched_count={len(unmatched)}")

# ---------- 6/7. SELF-CHECK on reload ----------
rl = pd.read_parquet(f"{DATA}/b0_gold_sft.parquet")
total = len(rl)
# per source coverage
src_counts = {}
boxed_ok = 0
meta_ct = 0
lens = []
corr_ok = 0
for _, r in rl.iterrows():
    msgs = json.loads(r["messages"])
    asst = next(m["content"] for m in msgs if m["role"] == "assistant")
    src_root = r["source"].split("/")[0]
    src_counts[src_root] = src_counts.get(src_root, 0) + 1
    if "\\boxed" in asst:
        boxed_ok += 1
    if "<|meta|>" in asst:
        meta_ct += 1
    lens.append(len(asst))
    # re-verify correctness vs our RV gold
    user = next(m["content"] for m in msgs if m["role"] == "user")
    ans = last_boxed(asst)
    if ans is not None and safe_verify(ans, prob2gold[user.strip()]):
        corr_ok += 1

stats = {
    "total_rows": total,
    "coverage_overall": round(total / N, 4),
    "coverage_denominator_unique_problems": N,
    "matched_by_source": src_counts,
    "pct_boxed": round(100 * boxed_ok / total, 2) if total else 0,
    "pct_meta": round(100 * meta_ct / total, 2) if total else 0,
    "correctness_pass_rate_of_kept": round(100 * corr_ok / total, 2) if total else 0,
    "median_assistant_len": int(statistics.median(lens)) if lens else 0,
    "dropped_answer_mismatch": dropped_mismatch,
    "dropped_mismatch_by_source": mismatch_by_source,
    "unmatched_no_benchmark_record": len(unmatched),
    "omni_note": omni_note,
}
print("STATS_JSON_BEGIN")
print(json.dumps(stats, indent=2))
print("STATS_JSON_END")
