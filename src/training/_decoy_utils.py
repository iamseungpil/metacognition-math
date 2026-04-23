"""Decoy generators for N3 contrastive RLSD (plan §2.1 / §H3).

Extracted from ``contrastive_meta_rlsd_trainer`` so that CPU-only unit tests
can import the decoy helpers without pulling in ``trl``/``transformers``.

Public API (re-exported by the trainer):
    * ``_numerically_equal(a, b) -> bool``
    * ``_rule_based_decoy(gold, seed) -> str``    (§2.1)
    * ``_random_noise_decoy(gold, seed) -> str``  (§H3 ablation)

All functions are deterministic in ``(gold, seed)`` — a precondition of the
leakage-free proof in plan §2.5.
"""
from __future__ import annotations

import hashlib
import random
import re
from typing import Callable, List, Optional


def _numerically_equal(a: str, b: str) -> bool:
    """Float equality with ValueError fallback (§2.1 numerical filter).

    LaTeX strings like ``\\frac{2}{1}`` are NOT parsed — the filter is
    intentionally conservative: only pure-numeric candidates are compared.
    Strategy 1-6 output classes are disjoint by construction so a
    float-level check suffices (see plan §2.1 invariant proof).
    """
    try:
        return abs(float(a) - float(b)) < 1e-9
    except (ValueError, TypeError):
        return False


def _rule_based_decoy(
    gold: str,
    seed: int = 42,
    checker: Optional[Callable[[str, str], bool]] = None,
) -> str:
    """Deterministic rule-based decoy generator (plan §2.1).

    Guarantees:
        (A) ``decoy != gold`` strictly (string compare).
        (B) Not numerically equivalent when both parse as float.
        (C) ``f(gold, seed)`` is deterministic — required for §2.5 leakage
            isolation proof.
        (D) ``decoy`` is syntactically a parse-valid answer string.
        (E) (when ``checker`` is passed) ``checker(decoy, gold) == False`` —
            aligns decoy equivalence with the training-time correctness
            checker (e.g. ``math_verify``-based ``_check_correctness``).
            Without this, symbolic equivalences like ``\\sqrt5`` ≡ ``-\\sqrt5``
            or ``\\frac{1}{2}`` ≡ ``-\\frac{1}{2}`` (via sympy) slip past the
            float-only filter and make decoy context equivalent to gold in
            the training loop, destroying the T+ vs T- contrastive signal.

    Strategies (§2.1):
        1. integer perturbation (±1, ±2, ±5, ±10)
        2. float perturbation (±0.1, ±0.5, ±1.0)
        3. LaTeX constants (drop ``\\pi``, strip ``\\sqrt{}``)
        4. fraction manipulation (swap num/denom, perturb numerator;
           skip palindromes ``\\frac{a}{a}``)
        5. sign flip (excluding ``0`` / ``-0``)
        6. fallback — append ``"+1"`` (symbolic, cannot equal gold)
    """
    # C2 fix: use hashlib.md5 for cross-process determinism (§2.5 Proposition).
    # Python's built-in hash() randomizes PYTHONHASHSEED per-process, breaking
    # leakage-free proof precondition.
    _seed_bytes = hashlib.md5(f"{gold}|{seed}".encode("utf-8")).hexdigest()[:8]
    rng = random.Random(int(_seed_bytes, 16))
    s = str(gold).strip()
    candidates: List[str] = []

    # Strategy 1 — integer perturbation.
    if re.fullmatch(r"-?\d+", s):
        n = int(s)
        for delta in (1, -1, 2, -2, 10, -10, 5, -5):
            c = str(n + delta)
            if c != s:
                candidates.append(c)
    # Strategy 2 — float perturbation (not an integer match above).
    elif re.fullmatch(r"-?\d+\.\d*|-?\.\d+|-?\d+\.?", s):
        try:
            v = float(s)
            for delta in (0.1, -0.1, 1.0, -1.0, 0.5, -0.5):
                c = str(round(v + delta, 2))
                if c != s:
                    candidates.append(c)
        except ValueError:
            pass

    # Strategy 3 — LaTeX constants.
    if "\\pi" in s:
        candidates.append(s.replace("\\pi", ""))
        candidates.append(s.replace("\\pi", "\\pi/2"))
    if "\\sqrt" in s:
        candidates.append(re.sub(r"\\sqrt\{(\d+)\}", r"\1", s))

    # Strategy 4 — fraction manipulation (palindrome guarded).
    m = re.match(r"\\?frac\{(-?\d+)\}\{(-?\d+)\}", s)
    if m and m.group(1) != m.group(2):
        swapped = f"\\frac{{{m.group(2)}}}{{{m.group(1)}}}"
        if swapped != s:
            candidates.append(swapped)
        # Perturb numerator by +1.
        try:
            candidates.append(f"\\frac{{{int(m.group(1)) + 1}}}{{{m.group(2)}}}")
        except ValueError:
            pass

    # Strategy 5 — sign flip (guard 0 and -0 explicitly).
    if s.startswith("-") and s != "-0":
        candidates.append(s[1:])
    elif s not in {"0", "0.0", "-0", "-0.0"}:
        candidates.append("-" + s)

    # Filter 1: decoy must differ both as string and numerically (when numeric),
    # and must not be empty (empty decoy would silently break teacher context).
    valid = [
        c for c in candidates
        if c != s and c.strip() != "" and not _numerically_equal(c, s)
    ]

    # Filter 2 (new — aligns with training-time checker): if caller passes a
    # ``checker`` (e.g. math_verify-based), reject candidates that the checker
    # grades as equivalent to gold. This catches symbolic equivalences the
    # float-only filter misses (e.g., ``\sqrt5`` vs ``-\sqrt5``).
    if checker is not None and valid:
        valid = [c for c in valid if not checker(c, s)]

    if valid:
        return rng.choice(valid)

    # Absolute fallback — symbolic sum, guaranteed distinct from gold by both
    # string and math_verify (sympy parses "\sqrt5 + 1" ≠ "\sqrt5"). Still
    # re-check if a checker is provided; if even this fails (unlikely) use a
    # large integer offset as last resort.
    fb1 = s + " + 1"
    if checker is None or not checker(fb1, s):
        return fb1
    fb2 = s + " + 1000000"
    return fb2


def _random_noise_decoy(gold: str, seed: int = 42) -> str:
    """Operational random-noise decoy baseline (plan §H3).

    Used for the H3 ablation (``variant=n3-random``). Mechanism:
        * integer gold → uniform random int in [-100, 100] ≠ gold
        * float gold   → uniform random float in [gold−10, gold+10]
                         with |c − gold| ≥ 0.1
        * LaTeX gold   → random char sequence of matching length drawn from
                         ``[0-9a-zA-Z\\pi\\sqrt\\frac]``
        * fallback     → ``random.randint(1, 1000)``

    All paths are deterministic via ``hashlib.md5((gold, "random", seed))`` —
    avoiding Python's PYTHONHASHSEED randomization (§2.5 precondition).
    """
    _seed_bytes = hashlib.md5(f"{gold}|random|{seed}".encode("utf-8")).hexdigest()[:8]
    rng = random.Random(int(_seed_bytes, 16))
    s = str(gold).strip()

    if re.fullmatch(r"-?\d+", s):
        target = int(s)
        # Rejection sample to guarantee ≠ gold.
        for _ in range(32):
            c = rng.randint(-100, 100)
            if c != target:
                return str(c)
        return str(target + 1)

    if re.fullmatch(r"-?\d+\.\d*|-?\.\d+|-?\d+\.?", s):
        try:
            v = float(s)
            for _ in range(32):
                c = round(v + rng.uniform(-10.0, 10.0), 2)
                if abs(c - v) >= 0.1:
                    return str(c)
            return str(round(v + 1.0, 2))
        except ValueError:
            pass

    # LaTeX / generic symbolic fallback — random char sequence, same length.
    charset = list("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    specials = ["\\pi", "\\sqrt", "\\frac"]
    L = max(len(s), 4)
    out_chars: List[str] = []
    while len("".join(out_chars)) < L:
        if rng.random() < 0.15 and specials:
            out_chars.append(rng.choice(specials))
        else:
            out_chars.append(rng.choice(charset))
    cand = "".join(out_chars)[: max(L, 4)]
    if cand == s:
        cand = cand + str(rng.randint(1, 1000))
    return cand


__all__ = ["_numerically_equal", "_rule_based_decoy", "_random_noise_decoy"]
