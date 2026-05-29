"""A.6 — 6-cell teacher×context contrastive discrimination for redirect meta.

Intent (user's 2026-05-28 design):
  Test whether meta CONTENT direction (toward-gold vs toward-decoy) is
  discriminable by teacher logp, across (model × context) cells:
    Axis 1: context = T+ (gold-aware) OR T- (decoy-aware)
    Axis 2: model = v8_strict SFT, E20a SFT, Qwen3-8B-Base
  Total cells = 2 × 3 = 6.

Method (template-based, Option 1):
  For each wrong-rollout sample from R10v2-on-v8:
    - Truncate body to position before first meta block (pre-meta context)
    - Generate good_redirect: template referring to gold answer's path
    - Generate bad_redirect: template referring to decoy answer's path
    - Length/structure matched (same template skeleton, different routed answer)

  For each (model, context):
    - Score logp on good_meta tokens
    - Score logp on bad_meta tokens
    - Cell signal = mean_logp(good) - mean_logp(bad)

  Pre-registered gate per cell:
    AUC(good vs bad logp) ≥ 0.65 → cell discriminates redirect direction
    sign(good_logp > bad_logp) consistent across problems → not noise

Headline question: which cell (if any) gives the highest AUC?
  - Best cell becomes the candidate teacher for CTSD Phase C.
  - If no cell ≥ 0.65: meta-content discrimination is fundamentally limited
    in logp space (paper §8 robust null conclusion).
"""
from __future__ import annotations
import argparse, json, random, time, gc, re
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from common.env import (
    SFT_V8_STRICT, TEACHER_MODEL, EVAL_R10V2_V8, HF_DATASET,
    META_OPEN_ID, META_CLOSE_ID, REPORTS_DIR, STRATIFIED_SAMPLE_SEED,
)

QWEN3_8B_BASE = "/home/v-seungplee/Qwen3-8B-Base"
MAX_RESP_TOK = 4096


# --- Template-based meta generation -----------------------------------------

GOOD_REDIRECT_TEMPLATE = (
    "\n\n<|meta|>\n"
    "confidence: 0.25\n"
    "The current route is weak because it does not align with the structure of the problem. "
    "What is missing is the direct path: the answer should be {gold}, "
    "so I should switch to working backwards from {gold} to verify each step.\n"
    "study_need: align intermediate steps with the target value {gold}\n"
    "I should stop the current approach and switch to a backward-verification path.\n"
    "<|/meta|>\n\n"
)

BAD_REDIRECT_TEMPLATE = (
    "\n\n<|meta|>\n"
    "confidence: 0.25\n"
    "The current route is weak because it does not align with the structure of the problem. "
    "What is missing is the direct path: the answer should be {decoy}, "
    "so I should switch to working backwards from {decoy} to verify each step.\n"
    "study_need: align intermediate steps with the target value {decoy}\n"
    "I should stop the current approach and switch to a backward-verification path.\n"
    "<|/meta|>\n\n"
)


# --- Prompt builders --------------------------------------------------------

def build_prompt_with_meta(tokenizer, question: str, answer_for_context: str,
                            pre_meta_body: str, meta_block: str):
    """Build full input: chat-templated [question + answer_reveal] then forced
    response = pre_meta_body + meta_block. Returns input_ids and meta_start_idx
    (response token index where <|meta|> token begins)."""
    block = (
        f"{question}\n\n"
        f"[REFERENCE — the correct final answer is: {answer_for_context}. "
        f"Score the following student response token-by-token with this information.]"
    )
    msgs = [{"role": "user", "content": block}]
    prompt_str = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
    if len(prompt_ids) > 1024:
        prompt_ids = prompt_ids[:1024]

    pre_meta_ids = tokenizer.encode(pre_meta_body, add_special_tokens=False)
    meta_ids = tokenizer.encode(meta_block, add_special_tokens=False)

    input_ids = prompt_ids + pre_meta_ids + meta_ids
    response_start = len(prompt_ids)
    meta_start = len(prompt_ids) + len(pre_meta_ids)
    return input_ids, response_start, meta_start, len(meta_ids)


@torch.no_grad()
def score_meta_logp(model, input_ids, meta_start, meta_len, device):
    """Return per-meta-token logp (length = meta_len)."""
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    out = model(ids, use_cache=False)
    logits = out.logits[0]
    # token at position t predicted by logits[t-1]
    pred_logits = logits[meta_start - 1 : meta_start - 1 + meta_len].float()
    targets = ids[0, meta_start : meta_start + meta_len]
    log_probs = torch.nn.functional.log_softmax(pred_logits, dim=-1)
    logp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return logp.cpu().numpy()


def find_answer_token_mask(tokenizer, meta_block: str, answer: str, meta_token_ids: list[int]) -> np.ndarray:
    """Return boolean mask (length = len(meta_token_ids)) where True = answer-string token.

    Used to exclude answer-alignment confound (codex Q1 mitigation):
    score meta logp WITHOUT counting tokens that contain the routed answer.
    """
    answer_str = str(answer).strip()
    if not answer_str:
        return np.zeros(len(meta_token_ids), dtype=bool)
    # Find char positions of answer in meta_block
    char_pos = 0
    answer_char_spans = []
    while True:
        idx = meta_block.find(answer_str, char_pos)
        if idx < 0:
            break
        answer_char_spans.append((idx, idx + len(answer_str)))
        char_pos = idx + 1
    if not answer_char_spans:
        return np.zeros(len(meta_token_ids), dtype=bool)
    # Map meta_token_ids back to char positions by incremental decode
    mask = np.zeros(len(meta_token_ids), dtype=bool)
    decoded = ""
    for i, tid in enumerate(meta_token_ids):
        piece = tokenizer.decode([tid])
        start_char = len(decoded)
        decoded += piece
        end_char = len(decoded)
        # Token i covers chars [start_char, end_char)
        for (cs, ce) in answer_char_spans:
            if start_char < ce and end_char > cs:
                mask[i] = True
                break
    return mask


def get_pre_meta_body(completion: str) -> str:
    """Extract content before first <|meta|> in completion, or whole completion if none."""
    idx = completion.find("<|meta|>")
    if idx < 0:
        # Use first 300 chars if no meta marker (rare)
        return completion[:300]
    return completion[:idx]


def stratified_wrong_sample(results, n_per_bench, rng):
    """Pick n_per_bench wrong rollouts per benchmark."""
    wrong_by_bench = defaultdict(list)
    for r in results:
        if not r["is_correct"]:
            wrong_by_bench[r["benchmark"]].append(r)
    picks = []
    for bench, items in wrong_by_bench.items():
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


def compute_auc(scores_good, scores_bad):
    """AUC via Mann-Whitney: P(good > bad)."""
    if not scores_good or not scores_bad:
        return float("nan")
    hits = 0
    total = 0
    for g in scores_good:
        for b in scores_bad:
            total += 1
            if g > b:
                hits += 1
            elif g == b:
                hits += 0.5
    return hits / total if total > 0 else float("nan")


def perm_test_paired(diffs, rng, n_perm=5000):
    """Paired test: each problem has diff = good_logp - bad_logp. Test if mean diff ≠ 0."""
    if not diffs:
        return float("nan")
    diffs = np.array(diffs)
    obs = abs(diffs.mean())
    hits = 0
    for _ in range(n_perm):
        signs = rng.choice([-1, 1], size=len(diffs))
        perm = (diffs * signs).mean()
        if abs(perm) >= obs:
            hits += 1
    return (hits + 1) / (n_perm + 1)


# --- Main loop --------------------------------------------------------------

MODELS = {
    "v8_strict": SFT_V8_STRICT,
    "E20a": TEACHER_MODEL,
    "Qwen3_Base": QWEN3_8B_BASE,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_per_bench", type=int, default=7, help="wrong rollouts per benchmark (default 7→21 total)")
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--out", default=str(REPORTS_DIR / "a6_six_cell_teacher_swap.json"))
    args = ap.parse_args()
    if args.smoke:
        args.n_per_bench = args.smoke
        args.out = args.out.replace(".json", f"_smoke{args.smoke}.json")

    rng = random.Random(STRATIFIED_SAMPLE_SEED)
    rng_np = np.random.default_rng(STRATIFIED_SAMPLE_SEED)
    t0 = time.time()

    # Load eval data and prepare per-problem inputs
    p = hf_hub_download(repo_id=HF_DATASET, repo_type="dataset", filename=EVAL_R10V2_V8)
    data = json.load(open(p))
    all_golds = [r["gold_answer"] for r in data["results"]]

    picks = stratified_wrong_sample(data["results"], args.n_per_bench, rng)
    print(f"[data] {len(picks)} wrong rollouts ({args.n_per_bench}/bench)")

    # Pre-build problem inputs (problem + pre_meta_body + 2 meta variants)
    # using E20a tokenizer (all 3 models share Qwen3 tokenizer)
    tok = AutoTokenizer.from_pretrained(TEACHER_MODEL)
    problems = []
    for r in picks:
        gold = str(r["gold_answer"]).strip()
        decoy = str(rng.choice(all_golds)).strip()
        while decoy == gold:
            decoy = str(rng.choice(all_golds)).strip()
        pre_meta_body = get_pre_meta_body(r["completion"])
        # Truncate pre_meta_body to keep context reasonable
        if len(pre_meta_body) > 800:
            pre_meta_body = pre_meta_body[:800]
        good_meta = GOOD_REDIRECT_TEMPLATE.format(gold=gold)
        bad_meta = BAD_REDIRECT_TEMPLATE.format(decoy=decoy)
        problems.append({
            "benchmark": r["benchmark"],
            "question": r["question"],
            "gold": gold,
            "decoy": decoy,
            "pre_meta_body": pre_meta_body,
            "good_meta": good_meta,
            "bad_meta": bad_meta,
        })

    # Score under each (model, context, meta_variant) cell
    # Total: 3 models × 2 contexts × 2 variants × N problems = 12N forward passes
    # We minimize model load by iterating models in outer loop.
    results = {}  # results[(model_name, context, variant)] = list of mean meta logp per problem
    for model_name, model_path in MODELS.items():
        print(f"\n[load] {model_name} ← {model_path}")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
            ).to("cuda:0").eval()
        except Exception as e:
            print(f"  [error] {model_name}: {e}")
            continue

        for ctx_label, ctx_answer_key in (("Tplus", "gold"), ("Tminus", "decoy")):
            for variant_label, meta_key, routed_ans_key in (
                ("good", "good_meta", "gold"),
                ("bad", "bad_meta", "decoy"),
            ):
                key_full = (model_name, ctx_label, variant_label, "full")
                key_excl = (model_name, ctx_label, variant_label, "excl_answer")
                full_logps, excl_logps = [], []
                for prob in problems:
                    answer_for_context = prob[ctx_answer_key]
                    meta_block = prob[meta_key]
                    routed_answer = prob[routed_ans_key]
                    input_ids, resp_start, meta_start, meta_len = build_prompt_with_meta(
                        tok, prob["question"], answer_for_context,
                        prob["pre_meta_body"], meta_block,
                    )
                    # extract just the meta-region token ids for masking
                    meta_token_ids = input_ids[meta_start:meta_start + meta_len]
                    ans_mask = find_answer_token_mask(tok, meta_block, routed_answer, meta_token_ids)
                    try:
                        logp = score_meta_logp(model, input_ids, meta_start, meta_len, "cuda:0")
                        full_mean = float(np.mean(logp))
                        # Excluding answer-span tokens (codex Q1 mitigation)
                        non_ans = logp[~ans_mask]
                        excl_mean = float(np.mean(non_ans)) if len(non_ans) > 0 else float("nan")
                        full_logps.append(full_mean)
                        excl_logps.append(excl_mean)
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        full_logps.append(None)
                        excl_logps.append(None)
                results[key_full] = full_logps
                results[key_excl] = excl_logps
                valid_full = [x for x in full_logps if x is not None]
                valid_excl = [x for x in excl_logps if x is not None and not np.isnan(x)]
                print(f"  {ctx_label} {variant_label}: "
                      f"full_mean={np.mean(valid_full):.4f} (n={len(valid_full)}), "
                      f"excl_ans_mean={np.mean(valid_excl):.4f} (n={len(valid_excl)})")
        del model
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  [elapsed] {time.time()-t0:.0f}s")

    # Analyze: per (model, context, score_kind) cell. score_kind = full or excl_answer
    print(f"\n=== Cell analysis (3 models × 2 contexts × 2 score kinds) ===")
    cell_summary = {}
    for model_name in MODELS:
        for ctx_label in ("Tplus", "Tminus"):
            for score_kind in ("full", "excl_answer"):
                good_logps = results.get((model_name, ctx_label, "good", score_kind), [])
                bad_logps = results.get((model_name, ctx_label, "bad", score_kind), [])
                paired = [(g, b) for g, b in zip(good_logps, bad_logps)
                          if g is not None and b is not None and not np.isnan(g) and not np.isnan(b)]
                if not paired:
                    continue
                gs, bs = zip(*paired)
                gs, bs = list(gs), list(bs)
                diffs = [g - b for g, b in zip(gs, bs)]
                auc = compute_auc(gs, bs)
                d = cohen_d(gs, bs)
                paired_p = perm_test_paired(diffs, rng_np)
                cell_summary[(model_name, ctx_label, score_kind)] = {
                    "n_paired": len(paired),
                    "good_mean_logp": float(np.mean(gs)),
                    "bad_mean_logp": float(np.mean(bs)),
                    "delta": float(np.mean(diffs)),
                    "auc": auc, "cohen_d": d, "paired_perm_p": paired_p,
                    "gate_pass": (auc is not None and not np.isnan(auc) and auc >= 0.65
                                  and paired_p < 0.05 and float(np.mean(diffs)) > 0),
                }
                tag = "[FULL]" if score_kind == "full" else "[EXCL_ANS]"
                print(f"  {model_name:12s} {ctx_label:7s} {tag:11s}: "
                      f"good={np.mean(gs):.4f} bad={np.mean(bs):.4f} Δ={np.mean(diffs):+.4f} "
                      f"AUC={auc:.3f} d={d:.3f} p={paired_p:.3f} "
                      f"{'✓PASS' if cell_summary[(model_name, ctx_label, score_kind)]['gate_pass'] else 'fail'}")

    # Contrastive direction (T+ − T- on same meta variant) per model — paired diff-in-diff
    # Codex: T+ - T- should be paired per problem, not aggregate means.
    print(f"\n=== Contrastive direction (T+ − T-, paired per-problem) ===")
    contrast_summary = {}
    for model_name in MODELS:
        for score_kind in ("full", "excl_answer"):
            tp_g = results.get((model_name, "Tplus", "good", score_kind), [])
            tn_g = results.get((model_name, "Tminus", "good", score_kind), [])
            tp_b = results.get((model_name, "Tplus", "bad", score_kind), [])
            tn_b = results.get((model_name, "Tminus", "bad", score_kind), [])
            if not (tp_g and tn_g and tp_b and tn_b):
                continue
            # paired per-problem contrast diff-in-diff:
            #   per problem: did_p = (Tplus_good - Tminus_good) - (Tplus_bad - Tminus_bad)
            dids = []
            for i in range(len(tp_g)):
                vals = [tp_g[i], tn_g[i], tp_b[i], tn_b[i]]
                if any(v is None or np.isnan(v) for v in vals):
                    continue
                did = (vals[0] - vals[1]) - (vals[2] - vals[3])
                dids.append(did)
            if not dids:
                continue
            # Question: is did > 0 on average? (gold-aware advantage of good_meta is
            # larger than gold-aware advantage of bad_meta)
            mean_did = float(np.mean(dids))
            paired_p = perm_test_paired(dids, rng_np)
            # AUC: per-problem contrast(good) vs contrast(bad)
            cg = [tp_g[i] - tn_g[i] for i in range(len(tp_g))
                  if tp_g[i] is not None and tn_g[i] is not None
                  and not np.isnan(tp_g[i]) and not np.isnan(tn_g[i])]
            cb = [tp_b[i] - tn_b[i] for i in range(len(tp_b))
                  if tp_b[i] is not None and tn_b[i] is not None
                  and not np.isnan(tp_b[i]) and not np.isnan(tn_b[i])]
            auc = compute_auc(cg, cb)
            contrast_summary[(model_name, score_kind)] = {
                "n_paired": len(dids),
                "mean_did": mean_did,
                "paired_perm_p": paired_p,
                "auc_good_vs_bad_contrast": auc,
                "gate_pass": (mean_did > 0 and paired_p < 0.05 and auc >= 0.65),
            }
            tag = "[FULL]" if score_kind == "full" else "[EXCL_ANS]"
            print(f"  {model_name:12s} {tag:11s}: "
                  f"mean_did={mean_did:+.4f} (n={len(dids)}) "
                  f"p={paired_p:.3f} AUC={auc:.3f} "
                  f"{'✓PASS' if contrast_summary[(model_name, score_kind)]['gate_pass'] else 'fail'}")

    # Verdict
    print(f"\n=== Verdict ===")
    best_cell = None
    best_cell_auc = 0
    for key, s in cell_summary.items():
        if s["auc"] is not None and not np.isnan(s["auc"]) and s["auc"] > best_cell_auc:
            best_cell_auc = s["auc"]; best_cell = key
    best_contrast = None
    best_contrast_auc = 0
    for key, s in contrast_summary.items():
        if s.get("auc_good_vs_bad_contrast") is not None and not np.isnan(s["auc_good_vs_bad_contrast"]) and s["auc_good_vs_bad_contrast"] > best_contrast_auc:
            best_contrast_auc = s["auc_good_vs_bad_contrast"]; best_contrast = key
    # Critical: only EXCL_ANSWER cells count toward "real signal" (codex Q1 mitigation)
    excl_cells = {k: v for k, v in cell_summary.items() if k[2] == "excl_answer"}
    best_excl_cell = None
    best_excl_auc = 0
    for key, s in excl_cells.items():
        if s["auc"] is not None and not np.isnan(s["auc"]) and s["auc"] > best_excl_auc:
            best_excl_auc = s["auc"]; best_excl_cell = key
    print(f"  Best single cell (any):        {best_cell} (AUC={best_cell_auc:.3f})")
    print(f"  Best EXCL_ANSWER cell:         {best_excl_cell} (AUC={best_excl_auc:.3f})")
    print(f"  Best contrastive:              {best_contrast} (AUC={best_contrast_auc:.3f})")
    overall_pass = (best_excl_auc >= 0.65)  # primary criterion: excl_answer (no answer alignment)
    print(f"  Overall verdict:               {'PASS — viable meta-quality signal in EXCL_ANSWER' if overall_pass else 'FAIL — no meta-specific signal'}")

    # Convert tuple-keyed dicts to JSON-friendly
    cell_summary_json = {f"{m}|{c}|{k}": v for (m, c, k), v in cell_summary.items()}
    contrast_summary_json = {f"{m}|{k}": v for (m, k), v in contrast_summary.items()}
    raw_results_json = {f"{m}|{c}|{v}|{k}": logps for (m, c, v, k), logps in results.items()}

    out = {
        "config": {"n_per_bench": args.n_per_bench, "n_problems": len(problems), "date": "2026-05-28",
                   "models": list(MODELS.keys())},
        "cell_summary": cell_summary_json,
        "contrast_summary": contrast_summary_json,
        "verdict": {
            "best_cell_any": str(best_cell) if best_cell else None, "best_cell_auc": best_cell_auc,
            "best_excl_cell": str(best_excl_cell) if best_excl_cell else None, "best_excl_auc": best_excl_auc,
            "best_contrast": str(best_contrast) if best_contrast else None, "best_contrast_auc": best_contrast_auc,
            "overall_pass": bool(overall_pass),
            "primary_criterion": "best EXCL_ANSWER cell AUC ≥ 0.65 (codex Q1 mitigation against answer-alignment confound)",
        },
        "raw_results": raw_results_json,
        "problems": [{"benchmark": p["benchmark"], "gold": p["gold"], "decoy": p["decoy"]} for p in problems],
        "wall_seconds": time.time() - t0,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[done] {args.out}")


if __name__ == "__main__":
    main()
