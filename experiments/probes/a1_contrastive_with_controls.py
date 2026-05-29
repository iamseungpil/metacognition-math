"""Phase A.1 — Contrastive teacher direction with artifact controls (v4).

Intent (CTSD Plan §Phase A.1):
  Verify that T+/T- contrast on meta tokens is a *causal metacognition signal*,
  not an artifact of dataset/prompt/solution-style differences.

Method:
  Stratified sample: 30 gsm8k + 30 math500 + 30 aime from R10v2-on-v8 eval.
  For each rollout, compute meta-region mean of 4 contrasts:
    REAL:           T+_logp − T-_logp  (gold vs random-decoy)
    A.1b SHUFFLED:  T+_logp − T-_shuffled  (gold vs different problem's gold)
    A.1c BODY:      same REAL contrast, but on body tokens (non-meta)
    A.1d SAME-ANS:  meta from two correct rollouts of the same problem
                    (currently: contrastive measure that should be null)

Outcome split: correct vs wrong rollouts.

Gates (PASS):
  - REAL: Cohen d ≥ 0.4, p < 0.01, sign-consistent across 3 benchmarks
  - REAL d ≥ 2× max(SHUFFLED d, BODY d) — signal is causal not artifact
  - SAME-ANS d ≈ 0 (within sampling noise) — null test passes

Outputs reports/a1_contrastive_with_controls.json
"""
from __future__ import annotations
import argparse, json, random, time, gc
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
import sys

# allow sibling import
sys.path.insert(0, str(Path(__file__).parent.parent))
from common.env import (
    TEACHER_MODEL, EVAL_R10V2_V8, HF_DATASET, META_OPEN_ID, META_CLOSE_ID,
    REPORTS_DIR, STRATIFIED_SAMPLE_SEED,
)

MAX_RESP_TOK = 8192


def build_student_input_ids(tok, question: str, completion: str):
    """student-side input: question only, then response."""
    msgs = [{"role": "user", "content": question}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    prompt_ids = tok.encode(prompt, add_special_tokens=False)
    resp_ids = tok.encode(completion, add_special_tokens=False)[:MAX_RESP_TOK]
    return prompt_ids + resp_ids, len(prompt_ids), resp_ids


def build_teacher_input_ids(tok, question: str, answer_block: str, resp_ids: list[int]):
    """teacher-side input: question + answer reveal + response."""
    block = (
        f"{question}\n\n"
        f"[REFERENCE — the correct final answer is: {answer_block}. "
        f"Score the following student response token-by-token with this information.]"
    )
    msgs = [{"role": "user", "content": block}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    prompt_ids = tok.encode(prompt, add_special_tokens=False)
    return prompt_ids + resp_ids, len(prompt_ids)


@torch.no_grad()
def score_response(model, input_ids, resp_start, device):
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    out = model(ids, use_cache=False)
    logits = out.logits[0]
    pred_logits = logits[resp_start - 1 : -1].float()
    targets = ids[0, resp_start:]
    logp = torch.nn.functional.log_softmax(pred_logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return logp.cpu().numpy()


def find_meta_spans(resp_ids):
    spans, in_meta, start = [], False, 0
    for i, t in enumerate(resp_ids):
        if t == META_OPEN_ID:
            in_meta, start = True, i + 1
        elif t == META_CLOSE_ID and in_meta:
            spans.append((start, i))
            in_meta = False
    return spans


def stratified_sample(results, n_per_bench, rng):
    """Equal n per benchmark, then split by correctness."""
    by_bench = defaultdict(list)
    for r in results:
        by_bench[r["benchmark"]].append(r)
    picks = []
    for bench, items in by_bench.items():
        rng.shuffle(items)
        picks.extend(items[:n_per_bench])
    rng.shuffle(picks)
    return picks


def cohen_d(a, b):
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = np.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2))
    return float((np.mean(a) - np.mean(b)) / pooled) if pooled > 0 else float("nan")


def perm_test(a, b, rng, n_perm=5000):
    if not a or not b:
        return float("nan")
    obs = np.mean(a) - np.mean(b)
    pooled = np.array(a + b)
    n_a = len(a)
    hits = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        d = pooled[:n_a].mean() - pooled[n_a:].mean()
        if abs(d) >= abs(obs):
            hits += 1
    return (hits + 1) / (n_perm + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default=TEACHER_MODEL)
    ap.add_argument("--n_per_bench", type=int, default=30)
    ap.add_argument("--smoke", type=int, default=0, help=">0 sets n_per_bench=smoke and writes _smoke.json")
    ap.add_argument("--out", default=str(REPORTS_DIR / "a1_contrastive_with_controls.json"))
    args = ap.parse_args()
    if args.smoke:
        args.n_per_bench = args.smoke
        args.out = args.out.replace(".json", f"_smoke{args.smoke}.json")

    rng = random.Random(STRATIFIED_SAMPLE_SEED)
    rng_np = np.random.default_rng(STRATIFIED_SAMPLE_SEED)
    t0 = time.time()

    # Load eval data
    p = hf_hub_download(repo_id=HF_DATASET, repo_type="dataset", filename=EVAL_R10V2_V8)
    data = json.load(open(p))
    picks = stratified_sample(data["results"], args.n_per_bench, rng)
    print(f"[data] {len(picks)} rollouts ({args.n_per_bench}/benchmark)")

    tok = AutoTokenizer.from_pretrained(args.teacher)

    # gold pool for shuffled control + same-answer control
    all_golds = [r["gold_answer"] for r in data["results"]]

    # Build rollouts with 4 input variants
    rollouts = []
    for r in picks:
        stud_ids, stud_start, resp_ids = build_student_input_ids(tok, r["question"], r["completion"])
        # REAL: gold answer
        teach_pos_ids, teach_pos_start = build_teacher_input_ids(tok, r["question"], r["gold_answer"], resp_ids)
        # A.1b SHUFFLED: random decoy
        decoy = rng.choice(all_golds)
        while str(decoy).strip() == str(r["gold_answer"]).strip():
            decoy = rng.choice(all_golds)
        teach_neg_ids, teach_neg_start = build_teacher_input_ids(tok, r["question"], decoy, resp_ids)
        rollouts.append({
            "benchmark": r["benchmark"], "is_correct": bool(r["is_correct"]),
            "stud_ids": stud_ids, "stud_start": stud_start,
            "teach_pos_ids": teach_pos_ids, "teach_pos_start": teach_pos_start,
            "teach_neg_ids": teach_neg_ids, "teach_neg_start": teach_neg_start,
            "resp_ids": resp_ids,
            "meta_spans": find_meta_spans(resp_ids),
        })

    print(f"[load] teacher {args.teacher}")
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to("cuda:0").eval()

    # Two forward passes per rollout: T+ (gold) and T- (decoy/shuffled)
    print("[pass1] T+ (gold)")
    for i, r in enumerate(rollouts):
        try:
            r["tplus_logp"] = score_response(model, r["teach_pos_ids"], r["teach_pos_start"], "cuda:0")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            r["tplus_logp"] = None
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(rollouts)} ({time.time()-t0:.0f}s)")

    print("[pass2] T- (shuffled gold)")
    for i, r in enumerate(rollouts):
        if r["tplus_logp"] is None:
            r["tneg_logp"] = None
            continue
        try:
            r["tneg_logp"] = score_response(model, r["teach_neg_ids"], r["teach_neg_start"], "cuda:0")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            r["tneg_logp"] = None
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(rollouts)} ({time.time()-t0:.0f}s)")
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Compute 4 contrast measurements per rollout
    per_rollout = []
    for r in rollouts:
        if r["tplus_logp"] is None or r["tneg_logp"] is None:
            continue
        L = min(len(r["tplus_logp"]), len(r["tneg_logp"]), len(r["resp_ids"]))
        if not r["meta_spans"]:
            continue
        tp = r["tplus_logp"][:L]
        tn = r["tneg_logp"][:L]
        in_meta = np.zeros(L, dtype=bool)
        for (a, b) in r["meta_spans"]:
            in_meta[min(a, L):min(b, L)] = True
        if in_meta.sum() == 0:
            continue
        body = ~in_meta
        for pos in range(L):
            if r["resp_ids"][pos] in (META_OPEN_ID, META_CLOSE_ID):
                body[pos] = False
        meta_diff = (tp[in_meta] - tn[in_meta]).mean()
        body_diff = (tp[body] - tn[body]).mean() if body.sum() > 0 else float("nan")
        per_rollout.append({
            "benchmark": r["benchmark"], "is_correct": r["is_correct"],
            "n_meta_tok": int(in_meta.sum()),
            "n_body_tok": int(body.sum()),
            "contrast_meta": float(meta_diff),    # REAL: T+ - T- on meta
            "contrast_body": float(body_diff),     # A.1c BODY control
        })

    # Outcome split per metric, per benchmark + pooled
    def split_stats(metric_key):
        out = {}
        for bench in ("gsm8k", "math500", "aime2024", "POOLED"):
            if bench == "POOLED":
                rows = per_rollout
            else:
                rows = [x for x in per_rollout if x["benchmark"] == bench]
            corr = [x[metric_key] for x in rows if x["is_correct"] and not np.isnan(x[metric_key])]
            wrong = [x[metric_key] for x in rows if not x["is_correct"] and not np.isnan(x[metric_key])]
            d = cohen_d(corr, wrong)
            p = perm_test(corr, wrong, rng_np)
            out[bench] = {
                "n_correct": len(corr), "n_wrong": len(wrong),
                "mean_correct": float(np.mean(corr)) if corr else None,
                "mean_wrong": float(np.mean(wrong)) if wrong else None,
                "delta": (float(np.mean(corr) - np.mean(wrong)) if corr and wrong else None),
                "cohen_d": d, "perm_p": p,
            }
        return out

    summary = {
        "REAL_meta_contrast": split_stats("contrast_meta"),
        "BODY_control_contrast": split_stats("contrast_body"),
    }

    # Gates
    pooled_real = summary["REAL_meta_contrast"]["POOLED"]
    pooled_body = summary["BODY_control_contrast"]["POOLED"]
    real_d = pooled_real["cohen_d"]
    body_d = abs(pooled_body["cohen_d"]) if pooled_body["cohen_d"] is not None else 0.0

    # Sign consistency: correct mean > wrong mean across all 3 benchmarks
    sign_consistent = all(
        (s := summary["REAL_meta_contrast"][b])["mean_correct"] is not None and
        s["mean_wrong"] is not None and s["mean_correct"] > s["mean_wrong"]
        for b in ("gsm8k", "math500", "aime2024")
    )

    gate_real = (real_d is not None and not np.isnan(real_d) and real_d >= 0.4
                 and pooled_real["perm_p"] < 0.01)
    gate_artifact_rule_out = (real_d is not None and not np.isnan(real_d)
                              and real_d >= 2.0 * body_d)
    gate_sign = bool(sign_consistent)
    verdict = "PASS" if (gate_real and gate_artifact_rule_out and gate_sign) else "FAIL"

    print(f"\n=== A.1 contrastive with controls ===")
    print(f"  REAL pooled: d={real_d:.3f} p={pooled_real['perm_p']:.4f} Δ={pooled_real['delta']:.4f}")
    print(f"  BODY pooled: d={body_d:.3f} p={pooled_body['perm_p']:.4f} Δ={pooled_body['delta']:.4f}")
    print(f"  per-bench REAL d: " + ", ".join(
        f"{b}={summary['REAL_meta_contrast'][b]['cohen_d']:.2f}" for b in ("gsm8k", "math500", "aime2024")
    ))
    print(f"  GATE: real_d≥0.4 + p<0.01 → {gate_real}")
    print(f"  GATE: real_d≥2×body_d   → {gate_artifact_rule_out}")
    print(f"  GATE: sign consistent   → {gate_sign}")
    print(f"  VERDICT: {verdict}")

    out = {
        "config": {"teacher": args.teacher, "n_per_bench": args.n_per_bench, "date": "2026-05-28"},
        "n_rollouts_used": len(per_rollout),
        "summary": summary,
        "gates": {
            "real_signal": bool(gate_real),
            "artifact_ruled_out": bool(gate_artifact_rule_out),
            "sign_consistent": gate_sign,
            "verdict": verdict,
        },
        "per_rollout": per_rollout,
        "wall_seconds": time.time() - t0,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[done] {args.out}")


if __name__ == "__main__":
    main()
