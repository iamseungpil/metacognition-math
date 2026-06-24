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
    """MINIMAL gold/decoy continuation: just the boxed answer, NO shared prose.

    A verbose continuation ("Therefore the final answer is X. \\boxed{X}") buries
    the gold-vs-decoy signal — only the answer token differs, the ~10 shared
    prose tokens contribute DiD~=0 yet add noise, diluting the sum (the user's
    point). The minimal form concentrates the contrast on the answer tokens so
    the per-token MAX (worst/biggest-difference token) is read cleanly.
    """
    return f"\\boxed{{{str(answer).strip()}}}"


def did(lp_gold_meta: float, lp_decoy_meta: float,
        lp_gold_plac: float, lp_decoy_plac: float) -> float:
    """Difference-in-differences: meta's contribution to gold-over-decoy log-odds."""
    return (lp_gold_meta - lp_decoy_meta) - (lp_gold_plac - lp_decoy_plac)


# ----------------------------------------------------------------------------
# mg (meta|gold) direction — pure helpers
# ----------------------------------------------------------------------------
# The mg direction scores the META tokens themselves, conditioned on a gold vs a
# decoy ANSWER HINT injected into the context. A meta whose CONTENT is made more
# probable by knowing the gold answer (vs a decoy) is one that genuinely "points
# at" the gold reasoning. The hint is injected ONLY at reward/scoring time — the
# model never sees it at inference, so this is leak-free w.r.t. generation.
#
#   δ_tok = logp(meta_tok | prompt + GOLD_HINT  + body_before_meta)
#         − logp(meta_tok | prompt + DECOY_HINT + body_before_meta)
#
# (gm scores the ANSWER given the meta; mg scores the META given the answer.)
GOLD_HINT_TMPL = "[Reference: the answer is {ans}]\n"


def build_hint(answer: str) -> str:
    """Construct the answer HINT injected into the mg scoring context.

    GOLD_HINT injects the gold answer; DECOY_HINT injects _rule_based_decoy(gold)
    via the SAME template (the caller picks which answer to pass). Identical in
    structure apart from the answer value, so the only thing distinguishing the
    gold and decoy arms is the answer the meta is conditioned on."""
    return GOLD_HINT_TMPL.format(ans=str(answer).strip())


def mg_token_deltas(meta_gold_lp: list, meta_decoy_lp: list) -> list:
    """Per-META-token δ = logp(meta_tok | gold_hint) − logp(meta_tok | decoy_hint),
    aligned position-wise to the min length. Returns the list of per-token δ."""
    n = min(len(meta_gold_lp), len(meta_decoy_lp))
    return [meta_gold_lp[t] - meta_decoy_lp[t] for t in range(n)]


def max_token_mg(meta_gold_lp: list, meta_decoy_lp: list) -> float:
    """mg row score = MAX over META tokens of the per-token δ (the EVAL/detection
    metric). Max isolates the single meta token most lifted by the gold hint; the
    REWARD variant (out of scope here) would use mean_min instead."""
    deltas = mg_token_deltas(meta_gold_lp, meta_decoy_lp)
    if not deltas:
        return 0.0
    return max(deltas)


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
def _score_logp(model, tok, prompt: str, continuation: str, device):
    """Return (sum_logp, per_token_logp_list) over the continuation tokens only."""
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
    return float(cont_lp.sum().item()), [float(x) for x in cont_lp.tolist()]


def max_token_did(gm: list, dm: list, gp: list, dp: list) -> float:
    """Per-token DiD = (gold_meta - gold_plac) - (decoy_meta - decoy_plac),
    aligned position-wise to the min length; return the MAX (the biggest-
    difference token = typically the answer token). Isolates the answer signal
    from shared structural tokens (the user's worst/max-token point)."""
    n = min(len(gm), len(dm), len(gp), len(dp))
    if n == 0:
        return 0.0
    return max(((gm[t] - gp[t]) - (dm[t] - dp[t])) for t in range(n))


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
        lp_gm, pt_gm = _score_logp(model, tok, body_meta, gold_c, device)
        lp_dm, pt_dm = _score_logp(model, tok, body_meta, decoy_c, device)
        lp_gp, pt_gp = _score_logp(model, tok, body_plac, gold_c, device)
        lp_dp, pt_dp = _score_logp(model, tok, body_plac, decoy_c, device)
        d = did(lp_gm, lp_dm, lp_gp, lp_dp)              # sequence-sum DiD
        d_max = max_token_did(pt_gm, pt_dm, pt_gp, pt_dp)  # worst/max-token DiD

        # mg (meta|gold): score the META tokens under a gold vs decoy answer hint
        # injected into the context (leak-free — hint only at scoring time). The
        # prompt is [hint]+body-before-<|meta|>+<|meta|>; the scored continuation
        # is the meta inner + close tag (the META tokens).
        k_open = body_meta.rfind(META_OPEN)
        body_before_meta = body_meta[:k_open]          # everything before <|meta|>
        meta_cont = META_OPEN + inner + META_CLOSE      # the META tokens to score
        gold_hint = build_hint(gt)
        decoy_hint = build_hint(decoy)
        _, pt_meta_gold = _score_logp(
            model, tok, gold_hint + body_before_meta, meta_cont, device)
        _, pt_meta_decoy = _score_logp(
            model, tok, decoy_hint + body_before_meta, meta_cont, device)
        mg_max = max_token_mg(pt_meta_gold, pt_meta_decoy)  # EVAL/detection metric

        results.append({
            "group": r.get("group", ""),
            "correct": int(float(r.get("c_with", 0) or 0) >= 0.5),
            "did": d, "did_max": d_max, "mg_max": mg_max,
            "decoy": decoy, "gt": gt,
        })

    # aggregate ----------------------------------------------------------------
    from collections import defaultdict
    labels = [x["correct"] for x in results]
    n = len(results)
    groups = defaultdict(list)
    for x in results:
        groups[x["group"]].append(x)

    def _stats(key):
        vals = [x[key] for x in results]
        m = sum(vals) / n if n else float("nan")
        sd = (sum((d - m) ** 2 for d in vals) / (n - 1)) ** 0.5 if n > 1 else float("nan")
        t = m / (sd / math.sqrt(n)) if n > 1 and sd > 0 else float("nan")
        pooled = rank_auc(vals, labels)
        wp = []
        for _, xs in groups.items():
            labs = [x["correct"] for x in xs]
            if 0 < sum(labs) < len(labs):
                a = rank_auc([x[key] for x in xs], labs)
                if a is not None:
                    wp.append(a)
        within = sum(wp) / len(wp) if wp else None
        return {"mean": m, "sd": sd, "t": t,
                "pos_frac": sum(1 for d in vals if d > 0) / n if n else float("nan"),
                "pooled_auc": pooled, "within_auc": within, "n_mixed": len(wp)}

    s_sum = _stats("did")        # sequence-sum DiD (diluted)
    s_max = _stats("did_max")    # worst/max-token DiD (user's point — answer token)
    s_mg = _stats("mg_max")      # mg (meta|gold) max-token — parallel direction

    # verdict on the TOKEN-LEVEL (max-token) signal — the un-diluted one.
    sig = bool(s_max["t"] == s_max["t"] and s_max["t"] > 2.0)
    disc = bool(s_max["within_auc"] is not None and s_max["within_auc"] >= 0.60)
    # mg gate (parallel to gm): within-problem AUC of mg-score predicting correct.
    mg_sig = bool(s_mg["t"] == s_mg["t"] and s_mg["t"] > 2.0)
    mg_disc = bool(s_mg["within_auc"] is not None and s_mg["within_auc"] >= 0.60)
    summary = {
        "n_metas_scored": n,
        "TOKEN_LEVEL_max_did": s_max,
        "sequence_sum_did": s_sum,
        "MG_direction": s_mg,
        "gate_maxtok_mean_gt0": sig,
        "gate_maxtok_within_auc_ge_060": disc,
        "gate_mg_mean_gt0": mg_sig,
        "gate_mg_within_auc_ge_060": mg_disc,
        "verdict": "PASS" if (sig and disc) else ("MARGINAL" if (sig or disc) else "FAIL"),
        "mg_verdict": "PASS" if (mg_sig and mg_disc) else ("MARGINAL" if (mg_sig or mg_disc) else "FAIL"),
    }

    json.dump({"summary": summary, "rows": results}, open(args.out, "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
