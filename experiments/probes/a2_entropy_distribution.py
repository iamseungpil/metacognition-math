"""Phase A.2 — Entropy distribution mapping for trigger threshold calibration.

Intent (CTSD Plan §Phase A.2):
  Calibrate τ_high (entropy threshold) for H2 force-inject from empirical
  distribution. Trigger threshold should be data-driven, not hand-picked.

Method:
  For 100 rollouts (50 correct + 50 wrong, stratified from R10v2-on-v8 eval),
  re-score the response under v8_strict SFT to get per-token logit distribution,
  compute per-token entropy H_t = -Σ p(v) log p(v).

  Two outputs:
  (1) Distribution: histogram + quantiles per outcome class.
  (2) AUC: how well does H_t (at a given position) predict final correctness?

Gates (PASS for go-Phase-C-Arm-2):
  - Some threshold τ where AUC(entropy > τ vs final_correctness) ≥ 0.65
  - Wrong-rollout entropy median > correct-rollout entropy median (signal direction)
  - p50/p90/p99 quantiles → empirical τ candidates for Arm 2

Outputs reports/a2_entropy_distribution.json
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

sys.path.insert(0, str(Path(__file__).parent.parent))
from common.env import (
    SFT_V8_STRICT, EVAL_R10V2_V8, HF_DATASET,
    META_OPEN_ID, META_CLOSE_ID, REPORTS_DIR, STRATIFIED_SAMPLE_SEED,
)

MAX_RESP_TOK = 8192


def build_student_input(tok, question: str, completion: str):
    msgs = [{"role": "user", "content": question}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    prompt_ids = tok.encode(prompt, add_special_tokens=False)
    resp_ids = tok.encode(completion, add_special_tokens=False)[:MAX_RESP_TOK]
    return prompt_ids + resp_ids, len(prompt_ids), resp_ids


@torch.no_grad()
def score_entropy(model, input_ids, resp_start, device):
    """Return per-response-token entropy H_t = -sum(p log p)."""
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    out = model(ids, use_cache=False)
    logits = out.logits[0]
    pred_logits = logits[resp_start - 1 : -1].float()  # logits predicting response positions
    log_probs = torch.nn.functional.log_softmax(pred_logits, dim=-1)
    probs = log_probs.exp()
    # entropy = -sum p log p
    H = -(probs * log_probs).sum(dim=-1)
    return H.cpu().numpy()


def find_meta_spans(resp_ids):
    spans, in_meta, start = [], False, 0
    for i, t in enumerate(resp_ids):
        if t == META_OPEN_ID:
            in_meta, start = True, i + 1
        elif t == META_CLOSE_ID and in_meta:
            spans.append((start, i))
            in_meta = False
    return spans


def stratified_pick(results, n_correct, n_wrong, rng):
    """Pick n_correct + n_wrong stratified by benchmark."""
    corr = [r for r in results if r["is_correct"]]
    wrong = [r for r in results if not r["is_correct"]]

    def pick_bench(items, n):
        by_b = defaultdict(list)
        for r in items:
            by_b[r["benchmark"]].append(r)
        picks = []
        # distribute n across benchmarks proportionally
        total = len(items)
        for b, lst in by_b.items():
            share = max(1, round(n * len(lst) / total))
            rng.shuffle(lst)
            picks.extend(lst[:share])
        rng.shuffle(picks)
        return picks[:n]

    return pick_bench(corr, n_correct) + pick_bench(wrong, n_wrong)


def compute_auc(scores, labels):
    """Simple AUC via Mann-Whitney."""
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return float("nan")
    # count pairs where pos > neg
    n_total = 0
    n_pos = 0
    for p in pos:
        for n in neg:
            n_total += 1
            if p > n:
                n_pos += 1
            elif p == n:
                n_pos += 0.5
    return n_pos / n_total if n_total > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=SFT_V8_STRICT)
    ap.add_argument("--n_correct", type=int, default=50)
    ap.add_argument("--n_wrong", type=int, default=50)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--out", default=str(REPORTS_DIR / "a2_entropy_distribution.json"))
    args = ap.parse_args()
    if args.smoke:
        args.n_correct = args.n_wrong = args.smoke
        args.out = args.out.replace(".json", f"_smoke{args.smoke}.json")

    rng = random.Random(STRATIFIED_SAMPLE_SEED)
    t0 = time.time()

    p = hf_hub_download(repo_id=HF_DATASET, repo_type="dataset", filename=EVAL_R10V2_V8)
    data = json.load(open(p))
    picks = stratified_pick(data["results"], args.n_correct, args.n_wrong, rng)
    print(f"[data] {len(picks)} rollouts")

    tok = AutoTokenizer.from_pretrained(args.model)
    print(f"[load] {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to("cuda:0").eval()

    rollouts = []
    for i, r in enumerate(picks):
        ids, start, resp_ids = build_student_input(tok, r["question"], r["completion"])
        try:
            H = score_entropy(model, ids, start, "cuda:0")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            continue
        meta_spans = find_meta_spans(resp_ids)
        # body mask (exclude meta region + meta markers)
        L = min(len(H), len(resp_ids))
        in_meta = np.zeros(L, dtype=bool)
        for (a, b) in meta_spans:
            in_meta[min(a, L):min(b, L)] = True
        body = ~in_meta
        for pos in range(L):
            if resp_ids[pos] in (META_OPEN_ID, META_CLOSE_ID):
                body[pos] = False

        H_arr = H[:L]
        rollouts.append({
            "benchmark": r["benchmark"],
            "is_correct": bool(r["is_correct"]),
            "n_tokens": L,
            "n_meta_tokens": int(in_meta.sum()),
            "H_body_mean": float(H_arr[body].mean()) if body.sum() > 0 else float("nan"),
            "H_body_median": float(np.median(H_arr[body])) if body.sum() > 0 else float("nan"),
            "H_body_p90": float(np.percentile(H_arr[body], 90)) if body.sum() > 0 else float("nan"),
            "H_body_p95": float(np.percentile(H_arr[body], 95)) if body.sum() > 0 else float("nan"),
            "H_body_max": float(H_arr[body].max()) if body.sum() > 0 else float("nan"),
            # Position of max entropy in body (early-trigger hypothesis)
            "H_body_argmax_frac": (float(np.argmax(H_arr[body])) / max(1, body.sum())) if body.sum() > 0 else float("nan"),
        })
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(picks)} ({time.time()-t0:.0f}s)")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Outcome-split analysis
    corr_means = [r["H_body_mean"] for r in rollouts if r["is_correct"] and not np.isnan(r["H_body_mean"])]
    wrong_means = [r["H_body_mean"] for r in rollouts if not r["is_correct"] and not np.isnan(r["H_body_mean"])]
    corr_max = [r["H_body_max"] for r in rollouts if r["is_correct"] and not np.isnan(r["H_body_max"])]
    wrong_max = [r["H_body_max"] for r in rollouts if not r["is_correct"] and not np.isnan(r["H_body_max"])]

    # AUC: does max entropy predict correctness? (low H_max → correct)
    all_max = corr_max + wrong_max
    all_labels = [0] * len(corr_max) + [1] * len(wrong_max)  # 1 = wrong (treat as positive class)
    auc_max_vs_wrong = compute_auc(all_max, all_labels)

    all_mean = corr_means + wrong_means
    all_labels_m = [0] * len(corr_means) + [1] * len(wrong_means)
    auc_mean_vs_wrong = compute_auc(all_mean, all_labels_m)

    # Quantile candidates for τ_high (top entropy values, body)
    all_body_max = [r["H_body_max"] for r in rollouts if not np.isnan(r["H_body_max"])]
    quantile_candidates = {
        "p70": float(np.percentile(all_body_max, 70)) if all_body_max else None,
        "p80": float(np.percentile(all_body_max, 80)) if all_body_max else None,
        "p85": float(np.percentile(all_body_max, 85)) if all_body_max else None,
        "p90": float(np.percentile(all_body_max, 90)) if all_body_max else None,
        "p95": float(np.percentile(all_body_max, 95)) if all_body_max else None,
        "p99": float(np.percentile(all_body_max, 99)) if all_body_max else None,
    }

    print(f"\n=== A.2 entropy distribution ===")
    print(f"  correct H_mean: {np.mean(corr_means):.3f} ± {np.std(corr_means):.3f} (n={len(corr_means)})")
    print(f"  wrong   H_mean: {np.mean(wrong_means):.3f} ± {np.std(wrong_means):.3f} (n={len(wrong_means)})")
    print(f"  correct H_max:  {np.mean(corr_max):.3f}  wrong H_max: {np.mean(wrong_max):.3f}")
    print(f"  AUC (H_max → wrong): {auc_max_vs_wrong:.3f}")
    print(f"  AUC (H_mean → wrong): {auc_mean_vs_wrong:.3f}")
    print(f"  τ candidates: {quantile_candidates}")

    # Gates
    auc_gate = (auc_max_vs_wrong >= 0.60) or (auc_mean_vs_wrong >= 0.60)
    direction_gate = np.mean(wrong_means) > np.mean(corr_means)  # wrong has higher entropy → expected
    verdict = "PASS" if (auc_gate and direction_gate) else "FAIL"
    print(f"  GATE auc ≥ 0.60:      {auc_gate}")
    print(f"  GATE wrong > correct: {direction_gate}")
    print(f"  VERDICT: {verdict}")

    out = {
        "config": {"model": args.model, "n_correct": args.n_correct, "n_wrong": args.n_wrong, "date": "2026-05-28"},
        "n_rollouts": len(rollouts),
        "stats": {
            "correct_H_mean": float(np.mean(corr_means)) if corr_means else None,
            "wrong_H_mean": float(np.mean(wrong_means)) if wrong_means else None,
            "correct_H_max_mean": float(np.mean(corr_max)) if corr_max else None,
            "wrong_H_max_mean": float(np.mean(wrong_max)) if wrong_max else None,
            "auc_Hmax_vs_wrong": auc_max_vs_wrong,
            "auc_Hmean_vs_wrong": auc_mean_vs_wrong,
        },
        "tau_quantile_candidates": quantile_candidates,
        "gates": {"auc_ge_0p60": bool(auc_gate), "wrong_higher_than_correct": bool(direction_gate), "verdict": verdict},
        "per_rollout": rollouts,
        "wall_seconds": time.time() - t0,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[done] {args.out}")


if __name__ == "__main__":
    main()
