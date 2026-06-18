"""Redirect-priming v2 CONTINUOUS counterfactual R_meta (spec 2026-06-18 §3).

TORCH-FREE (intent-check w4udybnbv C6) so it is testable in any env and importable
by the Stage-C CF producer without pulling torch. Re-exported from dcpo_region.

★NOT SAFE TO WIRE into the live Stage-C loop until the C2 gap-gaming tripwire
(halt if frozen-hard-band acc_without FALLS while rmeta_pos rises) is implemented
in verl_sdc and gated to halt BEFORE reward accumulates (intent-check w4udybnbv C7).
This function only credits a redirect that did not degrade the suppressed arm by
construction of the gates below; the running-degradation case is the tripwire's job.

Composes the verified primitives (does NOT reimplement): detect_redirect
(prose redirect, judge primary), degeneracy_flags (garbled/looped/no-answer).
"""


def redirect_cf_rmeta(
    c_with: int,
    c_without_draws: list,
    emit_switch: bool,
    cf_texts: list,
    in_hard_band: bool,
    with_text: str = "",
    lam: float = 0.25,
    min_len: int = 20,
    llm_judge=None,
    regex_only: bool = False,   # C4: live path REQUIRES a judge (fail-closed via detect_redirect)
    min_valid: int = 1,
) -> float:
    """Continuous CF R_meta. Positive credit ONLY when the row actually emitted a
    redirect AND is in the frozen hard band (intent-check C1); otherwise penalty-only.
    Returns 0.0 (ABSTAIN) when the counterfactual was never validly established —
    a hollow WITH arm (C3) or zero valid suppressed draws (C2) — so a non-causal /
    unmeasured redirect can never be scored as 'redirect saved it'."""
    from src.eval.redirect_behavior_detector import detect_redirect
    from src.eval.cf_stats import degeneracy_flags

    def _degenerate(text):
        f = degeneracy_flags(text, min_len=min_len)
        return bool(f.get("repetition") or f.get("too_short") or f.get("no_answer"))

    # C3 — hollow WITH arm: a garbled-but-parses WITH rollout must not be a win.
    if with_text and _degenerate(with_text):
        return 0.0

    draws = list(c_without_draws or [])
    texts = list(cf_texts or [])
    n = min(len(draws), len(texts))

    # C2 — a draw is a VALID meta-free counterfactual only if it is clean AND does
    # NOT still redirect in prose. Invalid draws are DISCARDED (abstain), never
    # counted as a meta-free failure (which would inflate the redirect's credit).
    n_valid = 0
    valid_correct = 0
    for i in range(n):
        if _degenerate(texts[i]) or detect_redirect(texts[i], llm_judge=llm_judge, regex_only=regex_only):
            continue
        n_valid += 1
        if bool(draws[i]):
            valid_correct += 1
    if n_valid < min_valid:
        return 0.0  # CF never validly established → cannot attribute anything to redirect

    c_without = valid_correct / n_valid
    base = float(c_with) - c_without

    # NEGATIVE term: redirect emitted but the suppressed arm was also (mostly) correct
    # → unnecessary redirect.
    r = base - lam if (emit_switch and c_without >= 0.5) else base

    # C1 + C5 — positive redirect credit requires emit_switch AND in_hard_band;
    # every other case is penalty-only (and the −lam penalty survives, C5).
    if not (emit_switch and in_hard_band):
        return min(0.0, r)
    return r


def rmeta_pos(r: float, thr: float = 0.25) -> bool:
    """Continuous-regime POSITIVE classifier (spec §3, replaces old >0.5)."""
    return r >= thr


def rmeta_neg(r: float, thr: float = 0.25) -> bool:
    """Continuous-regime NEGATIVE classifier."""
    return r <= -thr
