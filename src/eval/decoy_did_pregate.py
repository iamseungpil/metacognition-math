"""Decoupled decoy-DiD pre-gate (PG0-real): does the meta-region make the GOLD
answer more reachable than a DECOY answer, beyond what the body alone does?

This is the make-or-break test the cheap log-based proxy could NOT answer (that
proxy used R_meta = PMI-toward-the-rollout's-OWN-answer, whose sign is
mechanically the correctness label = A.6 answer-leak family). Here we compute the
DIRECTION prescribed by the 2026-06-01 contrastive-teacher-confound autopsy:

    score(ctx)  = logp(gold_cont | ctx) - logp(decoy_cont | ctx)
    DiD         = score(body+META) - score(body+PLACEBO)

gold_cont and decoy_cont are IDENTICAL in structure and differ ONLY in the answer
value (decoy = _rule_based_decoy(gold)). The placebo arm neutralises the meta
content (tags kept, inner replaced), so DiD isolates the META's contribution to
the gold-over-decoy log-odds — NOT "is the produced answer correct". An
answer-free boilerplate meta scores DiD ~= 0 (it lifts gold and decoy equally).

GATES:
  - mean DiD > 0 with a significant sign/t test  -> meta favours gold-reachability
  - within-problem AUC of DiD discriminating correct-rollout metas from
    incorrect-rollout metas >= 0.60 (must beat the answer-coupled 0.56 / random
    0.50) -> the meta CONTENT discriminates -> decoy-DiD has a live substrate.

Pure helpers are CPU-unit-tested; main() needs a GPU (HF logp scoring).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from typing import Optional

META_OPEN = "<|meta|>"
META_CLOSE = "<|/meta|>"
# neutral placebo meta: tags kept by the caller, this is the inner content. It
# carries no problem-specific guidance, so score(placebo) reflects body-only.
PLACEBO_INNER = "\nconfidence: 0.50\n(no analysis)\ndecision: none\n"


def parse_body_meta(text: str) -> Optional[tuple[str, str]]:
    """Return (body_through_meta_close, meta_inner) or None if no complete meta.

    body_through_meta_close = everything up to AND INCLUDING the first
    <|/meta|>; the model's post-meta continuation is discarded (we substitute our
    own gold/decoy continuations). meta_inner = text strictly between the tags.
    """
    i = text.find(META_OPEN)
    if i < 0:
        return None
    j = text.find(META_CLOSE, i + len(META_OPEN))
    if j < 0:
        return None
    inner = text[i + len(META_OPEN):j]
    body_through = text[: j + len(META_CLOSE)]
    return body_through, inner


def make_placebo(body_through: str, meta_inner: str) -> str:
    """Replace the meta inner content with the neutral filler (tags preserved)."""
    needle = META_OPEN + meta_inner + META_CLOSE
    repl = META_OPEN + PLACEBO_INNER + META_CLOSE
    if needle not in body_through:
        # defensive: rebuild from the close tag if exact slice drifted
        k = body_through.rfind(META_OPEN)
        return body_through[:k] + repl
    return body_through.replace(needle, repl, 1)


def build_continuation(answer: str) -> str:
    """Fixed gold/decoy continuation differing only in the answer value."""
    a = str(answer).strip()
    return f"\nTherefore the final answer is {a}.\n\\boxed{{{a}}}{ '' }"


def did(lp_gold_meta: float, lp_decoy_meta: float,
        lp_gold_plac: float, lp_decoy_plac: float) -> float:
    """Difference-in-differences: meta's contribution to gold-over-decoy log-odds."""
    return (lp_gold_meta - lp_decoy_meta) - (lp_gold_plac - lp_decoy_plac)


def rank_auc(scores: list[float], labels: list[int]) -> Optional[float]:
    """AUC via Mann-Whitney U with tie-averaged ranks. labels: 1=positive."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return None
    order = sorted(range(len(scores)), key=lambda k: scores[k])
    ranks = [0.0] * len(scores)
    k = 0
    while k < len(order):
        m = k
        while m + 1 < len(order) and scores[order[m + 1]] == scores[order[k]]:
            m += 1
        avg = (k + m) / 2.0 + 1.0
        for t in range(k, m + 1):
            ranks[order[t]] = avg
        k = m + 1
    sum_pos = sum(ranks[i] for i, y in enumerate(labels) if y == 1)
    u = sum_pos - len(pos) * (len(pos) + 1) / 2.0
    return u / (len(pos) * len(neg))


# ----------------------------------------------------------------------------
# GPU path
# ----------------------------------------------------------------------------
def _score_logp(model, tok, prompt: str, continuation: str, device) -> float:
    """Sum log p(continuation | prompt) over the continuation tokens only."""
    import torch
    p_ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
    full_ids = tok(prompt + continuation, return_tensors="pt",
                   add_special_tokens=False).input_ids
    full_ids = full_ids.to(device)
    n_prompt = p_ids.shape[1]
    with torch.no_grad():
        logits = model(full_ids).logits  # [1, L, V]
    logp = torch.log_softmax(logits[0, :-1].float(), dim=-1)
    tgt = full_ids[0, 1:]
    tok_lp = logp[range(tgt.shape[0]), tgt]
    cont_lp = tok_lp[n_prompt - 1:]  # tokens predicting the continuation
    return float(cont_lp.sum().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--rollouts_csv", required=True,
                    help="CSV with columns main_tail, gt, c_with, group")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_rows", type=int, default=0)
    ap.add_argument("--decoy_seed", type=int, default=42)
    args = ap.parse_args()

    sys.path.insert(0, ".")
    from src.training._decoy_utils import _rule_based_decoy
    try:
        from src.training.rewards import _check_correctness as _checker
    except Exception:
        _checker = None
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    device = next(model.parameters()).device

    rows = list(csv.DictReader(open(args.rollouts_csv)))
    if args.max_rows:
        rows = rows[: args.max_rows]

    results = []
    for r in rows:
        text = r.get("main_tail", "")
        gt = str(r.get("gt", "")).strip()
        if not gt:
            continue
        pm = parse_body_meta(text)
        if pm is None:
            continue  # no meta -> not a decoy-DiD candidate
        body_meta, inner = pm
        body_plac = make_placebo(body_meta, inner)
        checker = (lambda d, g: _checker(g, d)) if _checker else None
        decoy = _rule_based_decoy(gt, seed=args.decoy_seed, checker=checker)
        gold_c = build_continuation(gt)
        decoy_c = build_continuation(decoy)
        lp_gm = _score_logp(model, tok, body_meta, gold_c, device)
        lp_dm = _score_logp(model, tok, body_meta, decoy_c, device)
        lp_gp = _score_logp(model, tok, body_plac, gold_c, device)
        lp_dp = _score_logp(model, tok, body_plac, decoy_c, device)
        d = did(lp_gm, lp_dm, lp_gp, lp_dp)
        results.append({
            "group": r.get("group", ""),
            "correct": int(float(r.get("c_with", 0) or 0) >= 0.5),
            "did": d,
            "score_meta": lp_gm - lp_dm,
            "score_plac": lp_gp - lp_dp,
            "decoy": decoy, "gt": gt,
        })

    # aggregate ----------------------------------------------------------------
    dids = [x["did"] for x in results]
    n = len(dids)
    mean_did = sum(dids) / n if n else float("nan")
    pos_frac = sum(1 for d in dids if d > 0) / n if n else float("nan")
    sd = (sum((d - mean_did) ** 2 for d in dids) / (n - 1)) ** 0.5 if n > 1 else float("nan")
    t_stat = mean_did / (sd / math.sqrt(n)) if n > 1 and sd > 0 else float("nan")

    pooled_auc = rank_auc(dids, [x["correct"] for x in results])
    # within-problem AUC: per group with both classes, then average
    from collections import defaultdict
    g = defaultdict(list)
    for x in results:
        g[x["group"]].append(x)
    wp_aucs = []
    for k, xs in g.items():
        labs = [x["correct"] for x in xs]
        if 0 < sum(labs) < len(labs):
            a = rank_auc([x["did"] for x in xs], labs)
            if a is not None:
                wp_aucs.append(a)
    within_auc = sum(wp_aucs) / len(wp_aucs) if wp_aucs else None

    summary = {
        "n_metas_scored": n,
        "mean_did": mean_did, "did_sd": sd, "did_t": t_stat,
        "did_pos_frac": pos_frac,
        "pooled_auc_did_vs_correct": pooled_auc,
        "within_problem_auc": within_auc,
        "n_mixed_groups": len(wp_aucs),
        "gate_mean_did_gt0": bool(n > 1 and t_stat == t_stat and t_stat > 2.0),
        "gate_within_auc_ge_060": bool(within_auc is not None and within_auc >= 0.60),
        "verdict": None,
    }
    sig = summary["gate_mean_did_gt0"]
    disc = summary["gate_within_auc_ge_060"]
    summary["verdict"] = "PASS" if (sig and disc) else ("MARGINAL" if (sig or disc) else "FAIL")

    json.dump({"summary": summary, "rows": results}, open(args.out, "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
