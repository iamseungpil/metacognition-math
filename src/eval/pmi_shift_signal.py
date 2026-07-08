r"""PMI-SHIFT-ACROSS-META signal test (cheap GPU forward on a frozen model).

Make-or-break evidence for the PMI-SHIFT reward (design 2026-06-25): does the
META block actually SHIFT the model's gold-vs-decoy belief across its own two
positions (open vs close), and does that shift DISCRIMINATE correct from wrong
rollouts — beyond an A.6 answer-identity confound?

Per rollout (parsed meta block), we teacher-force the GOLD and DECOY `\boxed{...}`
answer strings at TWO contexts of the SAME rollout and read per-token logprobs:

    OPEN  = body BEFORE <|meta|>             (belief before reading its meta)
    CLOSE = body + <|meta|>...<|/meta|>      (belief after reading its meta)
    PMI_open  = Σ_div (logp(gold|OPEN)  − logp(decoy|OPEN))
    PMI_close = Σ_div (logp(gold|CLOSE) − logp(decoy|CLOSE))
    shift     = PMI_close − PMI_open

Σ_div sums only the DIVERGENT answer tokens (where gold and decoy actually differ;
shared `\boxed{`/`}` structure is excluded, mirroring the gm DiD locus).

Reports:
  1. shift discrimination — AUC of `shift` separating correct vs wrong rollouts
     (and of PMI_close / PMI_open for reference).
  2. sign-reversal vs SAVE — correlation/contingency of decoy→gold reversals with
     correct outcomes (and gold→decoy reversals with wrong outcomes).
  3. ★ CONFOUND CHECK (A.6 answer-identity) — is `shift` just the model favoring
     its OWN final answer, not a genuine gold-vs-decoy belief update? We stratify
     on whether the model's parsed final answer EQUALS gold:
       - own==gold subset: shift could be answer-identity OR genuine.
       - own!=gold subset: if shift STILL points toward gold (positive close-PMI /
         positive reversal rate) here, it is a GENUINE gold-belief update, not mere
         own-answer favoring. A confounded signal would collapse to ~0 / flip on
         this subset (shift would track the OWN wrong answer, i.e. toward decoy or
         away from gold).

CLI:
  python -m src.eval.pmi_shift_signal --model_path PATH --n N --out OUT.json \
      [--rollouts PARQUET_OR_CSV] [--decoy_seed 42]

If --rollouts is given it reads logged rollouts (columns: text/main_tail, gt/answer,
c_with optional); otherwise it errors (generation is out of scope for the cheap
signal test — supply parsed meta rollouts, e.g. data/redirect_verify_build/*.parquet).
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

import numpy as np

META_OPEN = "<|meta|>"
META_CLOSE = "<|/meta|>"


# ── pure helpers (CPU-testable) ──────────────────────────────────────────────
def parse_open_close(text: str) -> Optional[tuple[str, str]]:
    """Return (body_before_open, body_through_close) for the FIRST closed meta.

    body_before_open  = everything strictly before <|meta|>  (the OPEN context).
    body_through_close = everything up to AND INCLUDING <|/meta|> (the CLOSE
    context). None if there is no complete meta block."""
    i = text.find(META_OPEN)
    if i < 0:
        return None
    j = text.find(META_CLOSE, i + len(META_OPEN))
    if j < 0:
        return None
    return text[:i], text[: j + len(META_CLOSE)]


def rank_auc(scores: list, labels: list) -> Optional[float]:
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


def reversal_label(pmi_open: float, pmi_close: float) -> int:
    """+1 = decoy→gold (SAVE), −1 = gold→decoy (DERAIL), 0 = no reversal."""
    if pmi_open < 0.0 and pmi_close > 0.0:
        return 1
    if pmi_open > 0.0 and pmi_close < 0.0:
        return -1
    return 0


def corrupt_meta_text(meta_block: str, seed: int = 0) -> str:
    """Token-shuffle the INNER content of a meta block, preserving the tags.

    Placebo for the meta-CONTENT confound test (review 2026-06-25): a corrupted meta
    keeps the same length / token PRESENCE but destroys the reasoning CONTENT. If the
    shift on real meta is NOT >> the shift on corrupted meta (on the own≠gold subset),
    the signal is driven by meta-presence-as-confidence, not by reading the meta.
    `meta_block` is the tag-inclusive <|meta|>...<|/meta|> span (the body_close tail);
    returns a same-length block with the inner tokens randomly permuted."""
    import random
    i = meta_block.find(META_OPEN)
    j = meta_block.find(META_CLOSE, i + len(META_OPEN)) if i >= 0 else -1
    if i < 0 or j < 0:
        return meta_block
    inner = meta_block[i + len(META_OPEN):j]
    toks = inner.split()
    if len(toks) <= 1:
        return meta_block  # nothing to scramble
    rng = random.Random(seed)
    rng.shuffle(toks)
    scrambled = (" " if inner[:1].isspace() else "") + " ".join(toks)
    return meta_block[: i + len(META_OPEN)] + scrambled + meta_block[j:]


def gold_is_default(pmi_open: float) -> int:
    """1 if the model already LEANS gold before reading its meta (PMI_open>0).

    Operationalizes the 'safe-default' bias: when gold is the model's prior-favored
    (default) answer at OPEN, a positive shift could be a default-to-correct
    correlation rather than a genuine meta-driven update. The confound report splits
    own≠gold on this flag so shift>0 ONLY in the gold-is-default sub-case is flagged
    as confounded, while shift>0 equally in BOTH sub-cases is more likely genuine."""
    return 1 if pmi_open > 0.0 else 0


# ── GPU path ─────────────────────────────────────────────────────────────────
def _score_per_token(model, tok, prompt: str, continuation: str, device):
    """Return per-token logp list over the continuation tokens only."""
    import torch
    p_ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
    full_ids = tok(prompt + continuation, return_tensors="pt",
                   add_special_tokens=False).input_ids.to(device)
    n_prompt = p_ids.shape[1]
    with torch.no_grad():
        logits = model(full_ids).logits
    logp = torch.log_softmax(logits[0, :-1].float(), dim=-1)
    tgt = full_ids[0, 1:]
    tok_lp = logp[range(tgt.shape[0]), tgt]
    cont_lp = tok_lp[n_prompt - 1:]
    return [float(x) for x in cont_lp.tolist()]


def _pmi_position(gold_lp: list, decoy_lp: list, dmask) -> float:
    """Σ over divergent positions of (gold_lp − decoy_lp), gold-span aligned.

    LENGTH-MISMATCH (review 2026-06-25): gold and decoy are both `\\boxed{...}`
    tokenizations and should share length on the divergent span. The previous code
    zero-PADDED a shorter decoy span, which fabricates logp=0 (P=1.0) for the
    missing decoy tokens and inflates the gold-vs-decoy contrast by orders of
    magnitude — corrupting auc_shift / save_correct_rate / confound metrics with
    tokenization accidents rather than genuine belief updates. We now FAIL CLOSED
    (return NaN) on any mismatch so the caller drops the row. diff.size==0 (all
    tokens masked as identical) likewise returns NaN (no divergent positions)."""
    import numpy as np
    g = np.asarray(gold_lp, dtype=np.float64)
    d = np.asarray(decoy_lp, dtype=np.float64)
    n = g.size
    if n == 0:
        return float("nan")
    if d.size != n:
        # Misaligned spans: do NOT pad with 0 (fabricates logp=0 / P=1). Fail closed.
        return float("nan")
    mask = np.asarray(dmask, dtype=bool)
    if mask.size != n:
        mask = np.ones(n, dtype=bool)
    diff = (g - d)[mask]
    if diff.size == 0:
        return float("nan")
    return float(diff.sum())


def _load_rows(path: str, n: int) -> list:
    """Load logged rollout rows from a parquet/csv. Returns list of dicts with
    keys text, gt, c_with (best-effort over common column names)."""
    rows = []
    if path.endswith(".parquet"):
        import pandas as pd
        df = pd.read_parquet(path)
        recs = df.to_dict("records")
    else:
        import csv
        recs = list(csv.DictReader(open(path)))
    for r in recs:
        text = r.get("text") or r.get("main_tail") or r.get("response") or ""
        gt = r.get("gt") or r.get("answer") or r.get("ground_truth") or ""
        c_with = r.get("c_with", None)
        rows.append({"text": str(text), "gt": str(gt).strip(),
                     "c_with": c_with})
        if n and len(rows) >= n:
            break
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rollouts", required=True,
                    help="parquet/csv of parsed meta rollouts "
                         "(e.g. data/redirect_verify_build/rv_redirect_verify_b600.parquet)")
    ap.add_argument("--decoy_seed", type=int, default=42)
    args = ap.parse_args()

    sys.path.insert(0, ".")
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.training._decoy_utils import _rule_based_decoy
    from src.training.dcpo_directional import (
        boxed_answer_string,
        divergent_token_mask,
    )
    from src.training.dcpo_pmi_shift import pmi_shift_reward
    try:
        from src.training.rewards import _check_correctness as _checker
        from src.training.rewards import _extract_answer_fallback as _extract_ans
    except Exception:
        _checker = None
        _extract_ans = None

    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    device = next(model.parameters()).device

    rows = _load_rows(args.rollouts, args.n)
    results = []
    n_len_mismatch = 0     # gold/decoy answer-token length mismatch (dropped rows)
    n_zero_divergent = 0   # gold==decoy on the divergent span (no signal, dropped)
    for r in rows:
        text, gt = r["text"], r["gt"]
        if not gt:
            continue
        oc = parse_open_close(text)
        if oc is None:
            continue
        body_open, body_close = oc
        checker = (lambda d, g: _checker(g, d)) if _checker else None
        decoy = _rule_based_decoy(gt, seed=args.decoy_seed, checker=checker)
        gold_str = boxed_answer_string(gt)
        decoy_str = boxed_answer_string(decoy)
        gold_ids = tok.encode(gold_str, add_special_tokens=False)
        decoy_ids = tok.encode(decoy_str, add_special_tokens=False)
        if not gold_ids:
            continue
        # Length-match guard (review 2026-06-25): gold/decoy are both \boxed{...}
        # tokenizations; a mismatch would (post-fix) fail _pmi_position closed and
        # silently drop the row. Skip + count so the report exposes the rate.
        if len(decoy_ids) != len(gold_ids):
            n_len_mismatch += 1
            continue
        dmask = divergent_token_mask(gold_ids, decoy_ids)
        if int(np.asarray(dmask, dtype=bool).sum()) == 0:
            n_zero_divergent += 1   # gold==decoy on the divergent span -> no signal
            continue
        g_open = _score_per_token(model, tok, body_open, gold_str, device)
        d_open = _score_per_token(model, tok, body_open, decoy_str, device)
        g_close = _score_per_token(model, tok, body_close, gold_str, device)
        d_close = _score_per_token(model, tok, body_close, decoy_str, device)
        pmi_open = _pmi_position(g_open, d_open, dmask)
        pmi_close = _pmi_position(g_close, d_close, dmask)
        if not (np.isfinite(pmi_open) and np.isfinite(pmi_close)):
            continue
        shift = pmi_close - pmi_open
        r_shift = pmi_shift_reward(pmi_open, pmi_close)

        # ★ PLACEBO (meta-CONTENT vs meta-PRESENCE confound): re-score CLOSE with a
        # token-SHUFFLED meta (same length/presence, destroyed content). A genuine
        # content-driven update has shift_real >> shift_placebo; a presence-as-
        # confidence confound has shift_real ≈ shift_placebo.
        placebo_close = corrupt_meta_text(body_close, seed=args.decoy_seed)
        gp_close = _score_per_token(model, tok, placebo_close, gold_str, device)
        dp_close = _score_per_token(model, tok, placebo_close, decoy_str, device)
        pmi_close_placebo = _pmi_position(gp_close, dp_close, dmask)
        shift_placebo = (pmi_close_placebo - pmi_open
                         if np.isfinite(pmi_close_placebo) else float("nan"))

        # correctness label: from c_with if logged, else grade the rollout text.
        if r["c_with"] is not None:
            correct = int(float(r["c_with"]) >= 0.5)
        elif _checker is not None:
            correct = int(bool(_checker(text, gt)))
        else:
            correct = -1
        # model's OWN final answer (for the confound stratification).
        own = _extract_ans(text) if _extract_ans else None
        own_eq_gold = None
        if own is not None and _checker is not None:
            own_eq_gold = int(bool(_checker(boxed_answer_string(own), gt)))
        results.append({
            "pmi_open": pmi_open, "pmi_close": pmi_close,
            "pmi_close_placebo": pmi_close_placebo,
            "shift": shift, "shift_placebo": shift_placebo, "r_shift": r_shift,
            "reversal": reversal_label(pmi_open, pmi_close),
            "correct": correct, "own_eq_gold": own_eq_gold,
            "gold_is_default": gold_is_default(pmi_open),
            "gt": gt, "decoy": decoy,
        })

    _diag_counts = {"n_len_mismatch": n_len_mismatch,
                    "n_zero_divergent": n_zero_divergent}

    report = _summarize(results, diag_counts=_diag_counts)
    with open(args.out, "w") as f:
        json.dump({"report": report, "rows": results}, f, indent=2)
    print(json.dumps(report, indent=2))


def _summarize(results: list, diag_counts: Optional[dict] = None) -> dict:
    """Build the discrimination / reversal / CONFOUND report from scored rows."""
    n = len(results)
    rep = {"n_scored": n}
    if diag_counts:
        rep["dropped"] = dict(diag_counts)
    if n == 0:
        return rep
    correct = [r["correct"] for r in results]
    labeled = [r for r in results if r["correct"] in (0, 1)]
    if labeled:
        lab = [r["correct"] for r in labeled]
        rep["auc_shift"] = rank_auc([r["shift"] for r in labeled], lab)
        rep["auc_pmi_close"] = rank_auc([r["pmi_close"] for r in labeled], lab)
        rep["auc_pmi_open"] = rank_auc([r["pmi_open"] for r in labeled], lab)
    # sign-reversal vs SAVE (correct) contingency.
    n_save = sum(1 for r in results if r["reversal"] == 1)
    n_derail = sum(1 for r in results if r["reversal"] == -1)
    rep["n_save_reversal"] = n_save
    rep["n_derail_reversal"] = n_derail
    save_rows = [r for r in results if r["reversal"] == 1 and r["correct"] in (0, 1)]
    derail_rows = [r for r in results if r["reversal"] == -1 and r["correct"] in (0, 1)]
    rep["save_correct_rate"] = (
        sum(r["correct"] for r in save_rows) / len(save_rows) if save_rows else None)
    rep["derail_correct_rate"] = (
        sum(r["correct"] for r in derail_rows) / len(derail_rows) if derail_rows else None)
    # ★ CONFOUND CHECK: stratify on own==gold vs own!=gold.
    own_ne = [r for r in results if r["own_eq_gold"] == 0]
    own_eq = [r for r in results if r["own_eq_gold"] == 1]
    # Diagnostic (review): own_eq_gold is None when answer extraction / checker is
    # unavailable; report the count so a small own≠gold n is not mistaken for a clean
    # confound result when it is really an extraction-failure dropout.
    n_own_missing = sum(1 for r in results if r["own_eq_gold"] is None)

    def _mean(rows, key):
        return float(sum(r[key] for r in rows) / len(rows)) if rows else None

    rep["confound"] = {
        "n_own_ne_gold": len(own_ne),
        "n_own_eq_gold": len(own_eq),
        "n_own_eq_gold_missing": n_own_missing,
        # mean close-PMI on the own!=gold subset: a GENUINE gold-belief update stays
        # POSITIVE (toward gold) even when the model's own answer is NOT gold; an
        # A.6 answer-identity confound would go NEGATIVE (toward the own/decoy-ish).
        "mean_pmi_close_own_ne_gold": _mean(own_ne, "pmi_close"),
        "mean_shift_own_ne_gold": _mean(own_ne, "shift"),
        "save_rate_own_ne_gold": (
            sum(1 for r in own_ne if r["reversal"] == 1) / len(own_ne) if own_ne else None),
        "mean_pmi_close_own_eq_gold": _mean(own_eq, "pmi_close"),
        "mean_shift_own_eq_gold": _mean(own_eq, "shift"),
        # discrimination AUC restricted to own!=gold (genuine-update test): if shift
        # still separates correct from wrong HERE, it is not mere own-answer identity.
        "auc_shift_own_ne_gold": (
            rank_auc([r["shift"] for r in own_ne],
                     [r["correct"] for r in own_ne])
            if own_ne and len({r["correct"] for r in own_ne}) > 1 else None),
    }

    # ★ PLACEBO CHECK (meta-CONTENT vs meta-PRESENCE-as-confidence): on the own≠gold
    # subset, compare the real-meta shift to the shuffled-meta (placebo) shift. A
    # GENUINE content-driven update has mean_shift_real >> mean_shift_placebo; a
    # presence-as-confidence confound has them ≈ equal (the shift survives content
    # destruction). `placebo_gap` = real − placebo; near 0 ⇒ CONFOUNDED.
    plac_ne = [r for r in own_ne if np.isfinite(r.get("shift_placebo", float("nan")))]
    mean_real = _mean(plac_ne, "shift")
    mean_plac = _mean(plac_ne, "shift_placebo")
    rep["confound"]["placebo"] = {
        "n_own_ne_gold_with_placebo": len(plac_ne),
        "mean_shift_real_own_ne_gold": mean_real,
        "mean_shift_placebo_own_ne_gold": mean_plac,
        "placebo_gap_own_ne_gold": (
            (mean_real - mean_plac) if (mean_real is not None and mean_plac is not None)
            else None),
    }

    # ★ SAFE-DEFAULT CHECK: split own≠gold into gold-IS-default vs gold-is-NOT-default
    # (gold_is_default = model already leans gold at OPEN, PMI_open>0). If shift>0
    # ONLY where gold is the default, the signal is a default-to-correct correlation,
    # not a meta-driven update. shift>0 in BOTH ⇒ more likely genuine.
    ne_def = [r for r in own_ne if r.get("gold_is_default") == 1]
    ne_nondef = [r for r in own_ne if r.get("gold_is_default") == 0]
    rep["confound"]["safe_default"] = {
        "n_own_ne_gold_is_default": len(ne_def),
        "n_own_ne_gold_not_default": len(ne_nondef),
        "mean_shift_own_ne_gold_is_default": _mean(ne_def, "shift"),
        "mean_shift_own_ne_gold_not_default": _mean(ne_nondef, "shift"),
        "save_rate_own_ne_gold_not_default": (
            sum(1 for r in ne_nondef if r["reversal"] == 1) / len(ne_nondef)
            if ne_nondef else None),
    }
    return rep


if __name__ == "__main__":
    main()
