"""R-B' causal-measurement core for Redirect-Priming v2 (spec 2026-06-18 REV-6 §5).

The load-bearing estimand is **R - B'** = redirect *content* given a matched
second attempt, NOT meta-on/off. This module is pure (GPU-free): generation
happens elsewhere and is passed in as per-problem records. Each record `row`
carries arms:
  row["R"]      = {"correct": bool, "text": str}   redirect allowed
  row["Bprime"] = {"correct": bool, "text": str}   <|switch|> -inf-masked, may
                                                    restart in plain prose
  row["P"]      = {"correct": bool}                placebo (a matched NON-redirect
                                                    meta token banned)

It COMPOSES the already-tested primitives in cf_stats.py and
redirect_behavior_detector.py — it does not reimplement them.
"""
from src.eval.cf_stats import is_parsed, degeneracy_flags, mcnemar_status
from src.eval.redirect_behavior_detector import detect_redirect


def row_usable(row, min_len: int = 20, llm_judge=None, regex_only: bool = False) -> bool:
    """A per-problem record is usable for the R-B' contrast iff:
    - BOTH arms parse a final answer (cf_stats.is_parsed);
    - arm-B' is NOT off-policy degenerate while arm-R is clean (a B' that loops /
      truncates / loses its answer when R didn't is an artifact, not a redirect
      effect — drop it); and
    - arm-B' does NOT still exhibit redirect BEHAVIOR in prose (it routed around
      the masked <|switch|>; counting it would fake separability).
    Any failure -> False.
    """
    r = row["R"]
    b = row["Bprime"]
    r_text = r.get("text", "")
    b_text = b.get("text", "")

    # 1. parse gate (both arms)
    if not is_parsed(r_text) or not is_parsed(b_text):
        return False

    # 2. off-policy degeneracy: drop if B' is degenerate while R is clean
    r_flags = degeneracy_flags(r_text, min_len=min_len)
    b_flags = degeneracy_flags(b_text, min_len=min_len)
    r_clean = not any(r_flags.values())
    b_degenerate = any(b_flags.values())
    if b_degenerate and r_clean:
        return False

    # 3. B' must not still redirect in prose
    if detect_redirect(b_text, llm_judge=llm_judge, regex_only=regex_only):
        return False

    return True


def saved_broke(rows, min_len: int = 20, llm_judge=None, regex_only: bool = False):
    """Over USABLE rows return (b, c):
      b = redirect SAVED   = count(R correct AND B' wrong)
      c = redirect BROKE   = count(B' correct AND R wrong)
    These are the discordant-pair counts fed to McNemar.
    """
    b = c = 0
    for row in rows:
        if not row_usable(row, min_len=min_len, llm_judge=llm_judge, regex_only=regex_only):
            continue
        r_ok = bool(row["R"]["correct"])
        b_ok = bool(row["Bprime"]["correct"])
        if r_ok and not b_ok:
            b += 1
        elif b_ok and not r_ok:
            c += 1
    return b, c


def _usable_rows(rows, min_len, llm_judge, regex_only):
    return [r for r in rows
            if row_usable(r, min_len=min_len, llm_judge=llm_judge, regex_only=regex_only)]


def beats_placebo(rows, min_len: int = 20, llm_judge=None, regex_only: bool = False) -> bool:
    """The redirect arm's advantage over B' must EXCEED its advantage over the
    placebo P (a matched non-redirect meta token banned). I.e. require
    (acc_R - acc_Bprime) > (acc_R - acc_P), equivalently acc_P > acc_Bprime —
    so a B' that merely loses accuracy for non-redirect reasons (which P also
    captures) is not scored as a redirect win. Computed over usable rows.
    """
    usable = _usable_rows(rows, min_len, llm_judge, regex_only)
    n = len(usable)
    if n == 0:
        return False
    acc_r = sum(bool(r["R"]["correct"]) for r in usable) / n
    acc_b = sum(bool(r["Bprime"]["correct"]) for r in usable) / n
    acc_p = sum(bool(r["P"]["correct"]) for r in usable) / n
    return (acc_r - acc_b) > (acc_r - acc_p)


def is_monotone_saturating(effects_by_bias) -> bool:
    """`effects_by_bias` = list of (bias_level, effect) sorted by |bias|. A TRUE
    causal effect is monotone non-decreasing in suppression strength AND
    saturates (plateaus); a decoding-break artifact keeps growing.

    Return True iff effects are monotone non-decreasing AND the last step's
    increment is <= 0.25x the first step's increment (saturating). Pre-register:
    non-monotone OR non-saturating ⇒ artifact.
    """
    if effects_by_bias is None or len(effects_by_bias) < 2:
        return False
    effs = [e for _, e in effects_by_bias]
    incs = [effs[i] - effs[i - 1] for i in range(1, len(effs))]

    # monotone non-decreasing
    if any(inc < 0 for inc in incs):
        return False

    if len(incs) < 2:
        # one step can't show a plateau
        return False
    first, last = incs[0], incs[-1]
    if first <= 0:
        # no initial rise to saturate from; treat flat-then-flat as non-saturating
        return False
    return last <= 0.25 * first


def rml_verdict(rows, effects_by_bias=None, alpha: float = 0.05,
                min_discordant: int = 10, llm_judge=None, regex_only: bool = False) -> dict:
    """Composed final verdict for the R-B' redirect-content measurement.

    Returns {"status", "b", "c", "n_usable", "beats_placebo", "monotone_saturating"}
    where status ∈ {SIGNIFICANT, NOT_SIGNIFICANT, UNDERPOWERED, INSUFFICIENT}.

    status = INSUFFICIENT if too few usable rows, or (bias sweep given AND not
    monotone-saturating), or placebo not beaten. Otherwise the cf_stats McNemar
    status. Significance therefore REQUIRES beats_placebo AND (if a bias sweep is
    given) monotone_saturating — these are pre-registered artifact guards.
    """
    usable = _usable_rows(rows, 20, llm_judge, regex_only)
    n_usable = len(usable)
    b, c = saved_broke(rows, llm_judge=llm_judge, regex_only=regex_only)
    placebo_ok = beats_placebo(rows, llm_judge=llm_judge, regex_only=regex_only)
    mono = None if effects_by_bias is None else is_monotone_saturating(effects_by_bias)

    out = {
        "b": b,
        "c": c,
        "n_usable": n_usable,
        "beats_placebo": placebo_ok,
        "monotone_saturating": mono,
    }

    if n_usable < min_discordant:
        out["status"] = "INSUFFICIENT"
    elif effects_by_bias is not None and not mono:
        out["status"] = "INSUFFICIENT"
    elif not placebo_ok:
        out["status"] = "INSUFFICIENT"
    else:
        out["status"] = mcnemar_status(b, c, alpha=alpha, min_discordant=min_discordant)

    return out
