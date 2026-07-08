"""Shared helpers for the experiments/analysis CLI scripts (not a CLI itself).

Provides:
  - load_eval_frame():   normalize eval outputs from scripts/eval_vllm_1030.py
                         (parquet / {"summary","results"} json) and
                         src/eval/eval_hf.py ({"model","run_metadata","results"}
                         json / parquet), plus plain jsonl, into one schema.
  - robust grading:      reuses experiments/common/grading.robust_grade
                         (validated boxed-wrap math_verify grader). The stored
                         `is_correct` column comes from the legacy
                         src.training.rewards._check_correctness path, which is
                         DOCUMENTED to mis-grade (raw-text math_verify.parse
                         drops ~21% of math500 golds; string fallback mis-cut
                         base by 26%). Final-judgment numbers must therefore be
                         re-graded here. If math_verify is not installed we fall
                         back to a numeric/string boxed compare and every script
                         prints a loud warning.
  - closed <|meta|> block parsing (metric rule: meta emission is counted by
    CLOSED <|meta|>...<|/meta|> blocks only, never the free-text fallback).
  - stated-confidence extraction from closed meta blocks (gold is NEVER used
    to fabricate confidence; it only grades correctness).
  - calibration math (15-bin ECE, Brier, overconfidence rate) shared by
    calibration.py and aggregate_tables.py.

Import pattern from sibling scripts (works when run as plain files):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiments.analysis.analysis_common import ...
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

META_START = "<|meta|>"
META_END = "<|/meta|>"
# Metric rule 5: meta emission is counted by CLOSED blocks only.
CLOSED_META_RE = re.compile(
    re.escape(META_START) + r"(.*?)" + re.escape(META_END), re.DOTALL
)

BENCH_ORDER = ["gsm8k", "math500", "aime2024"]
# OOD split for T3: AIME is the OOD (hard-domain) benchmark; GSM8K/MATH-500 are ID.
OOD_BENCHMARKS = {"aime2024"}


# ── grading ──────────────────────────────────────────────────────────────────

try:
    # Validated grader: wraps BOTH pred's last \boxed{...} and the gold in
    # \boxed{} before math_verify parse+verify (see experiments/common/grading.py
    # for the validation history). Import fails cleanly when math_verify is absent.
    from experiments.common.grading import extract_last_boxed, robust_grade

    HAS_MATH_VERIFY = True
except Exception:  # math_verify not installed in this environment
    HAS_MATH_VERIFY = False

    def extract_last_boxed(t: str):
        r"""Brace-balanced content of the LAST \boxed{...}, or None (local copy)."""
        i = t.rfind(r"\boxed{")
        if i < 0:
            return None
        j = i + len(r"\boxed{")
        depth, out = 1, []
        while j < len(t) and depth > 0:
            c = t[j]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            if depth > 0:
                out.append(c)
            j += 1
        return "".join(out)

    def robust_grade(pred_text: str, gold) -> bool:
        """DEGRADED fallback when math_verify is missing: numeric/string compare
        of the last boxed answer. Only use for smoke tests, never for the paper."""
        box = extract_last_boxed(str(pred_text))
        if box is None:
            return False
        p, g = _normalize_answer(box), _normalize_answer(str(gold))
        if p == g:
            return True
        try:
            return abs(float(p) - float(g)) < 1e-6
        except (TypeError, ValueError):
            return False


def _normalize_answer(s: str) -> str:
    """Light normalization for the degraded string-compare fallback only."""
    s = s.strip().strip("$").strip()
    s = re.sub(r"(?<=\d),(?=\d{3})", "", s)  # 70,000 -> 70000
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def warn_if_no_math_verify(script_name: str) -> None:
    """Print a loud warning when grading falls back to string compare."""
    if not HAS_MATH_VERIFY:
        print(
            f"[{script_name}] WARNING: math_verify is NOT installed — grading "
            "falls back to a numeric/string boxed compare. Numbers produced in "
            "this mode are for smoke tests only, NOT for the paper.",
            file=sys.stderr,
        )


class Grader:
    """Memoized robust grader. Key = (last-boxed answer, gold) so identical
    answers across samples/arms are verified only once (sympy verify is slow)."""

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str], bool] = {}

    def grade(self, pred_text: str, gold: str, fallback_answer=None) -> bool:
        box = extract_last_boxed(str(pred_text))
        if box is None:
            # Format-fair fallback: some arms state the final answer in prose
            # ("the verified answer is 540.") WITHOUT a \boxed{} wrapper, which
            # extract_last_boxed cannot see. Rather than score those legitimate
            # answers as wrong (a per-arm format bias — see the gandhi arm, which
            # boxes only ~85% of GSM8K vs ~100% for base/pmishift), grade the
            # runtime-extracted answer through the SAME validated math_verify path
            # by wrapping it in \boxed{}. Arms that always emit \boxed{} never
            # reach this branch, so this is uniform and cannot inflate them.
            fb = None if fallback_answer is None else str(fallback_answer).strip()
            if not fb or fb.lower() in ("none", "nan"):
                return False
            key = ("FB:" + fb, str(gold))
            if key not in self._cache:
                self._cache[key] = bool(robust_grade(r"\boxed{" + fb + "}", gold))
            return self._cache[key]
        key = (box, str(gold))
        if key not in self._cache:
            self._cache[key] = bool(robust_grade(str(pred_text), gold))
        return self._cache[key]


def regrade_frame(df: pd.DataFrame, grader: Grader | None = None) -> pd.Series:
    """Re-grade every completion against its gold with the robust grader.

    When a completion carries no \\boxed{} answer, fall back to the stored
    runtime `answer_extracted` (graded through the same math_verify path) so a
    prose-only final answer is not scored wrong purely for lacking the wrapper.
    """
    grader = grader or Grader()
    has_fb = "answer_extracted" in df.columns
    return df.apply(
        lambda r: grader.grade(
            r["completion"], r["gold_answer"],
            r["answer_extracted"] if has_fb else None,
        ),
        axis=1,
    )


def resolve_correct(
    df: pd.DataFrame, no_regrade: bool, script_name: str, grader: Grader | None = None
) -> pd.Series:
    """Return the correctness series per metric rule 2 (math_verify regrade by
    default); `--no-regrade` trusts the stored is_correct (fast, NOT for paper)."""
    if no_regrade:
        print(
            f"[{script_name}] using STORED is_correct (--no-regrade); stored "
            "grades come from the legacy check_correctness path and are known "
            "to mis-grade — do not use for final numbers.",
            file=sys.stderr,
        )
        return df["is_correct"].astype(bool)
    warn_if_no_math_verify(script_name)
    return regrade_frame(df, grader)


# ── eval-output loading ──────────────────────────────────────────────────────

def _records_from_json(payload) -> list[dict]:
    """Accept both eval_vllm_1030.py json ({"summary","results"}) and
    eval_hf.py json ({"model","run_metadata","results"}), or a bare list."""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and "results" in payload:
        return payload["results"]
    raise ValueError("Unrecognized eval json layout (no 'results' key).")


def load_eval_frame(path: str | Path) -> pd.DataFrame:
    """Load one eval output file into the normalized schema.

    Normalized columns: benchmark, question, gold_answer, completion,
    is_correct (stored grade), sample_idx, num_meta_blocks_closed, qid,
    completion_length_tokens (NaN if absent).
    """
    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.DataFrame(pd.read_parquet(path))
    elif path.suffix == ".jsonl":
        df = pd.read_json(path, lines=True)
    elif path.suffix == ".json":
        with open(path) as f:
            df = pd.DataFrame(_records_from_json(json.load(f)))
    else:
        raise ValueError(f"Unsupported eval file type: {path}")

    # eval_hf.py truncates `question`/`gold_answer` and keeps the full text in
    # full_question / full_gold_answer — prefer the full columns when present.
    if "full_question" in df.columns:
        df["question"] = df["full_question"]
    if "full_gold_answer" in df.columns:
        df["gold_answer"] = df["full_gold_answer"]

    required = ["benchmark", "question", "gold_answer", "completion", "is_correct"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing required columns {missing}")

    if "sample_idx" not in df.columns:
        df["sample_idx"] = 0
    if "completion_length_tokens" not in df.columns:
        df["completion_length_tokens"] = np.nan

    df["is_correct"] = df["is_correct"].astype(bool)
    # Recount meta blocks from raw text: the stored num_meta_blocks column may
    # include the free-text fallback (parse_meta_blocks synthesizes a block when
    # a bare confidence phrase appears), which violates metric rule 5.
    df["num_meta_blocks_closed"] = df["completion"].map(count_closed_meta_blocks)
    df["qid"] = [
        make_qid(b, q) for b, q in zip(df["benchmark"], df["question"])
    ]
    df["source_file"] = str(path)
    return df


def make_qid(benchmark: str, question: str) -> str:
    """Stable question id for cross-arm joins (same 1030 set in every arm)."""
    h = hashlib.sha1(str(question).strip().encode("utf-8")).hexdigest()[:12]
    return f"{benchmark}:{h}"


def parse_arm_specs(specs: list[str]) -> list[tuple[str, str]]:
    """Parse repeated NAME=PATH CLI args into (name, path) pairs."""
    out = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Expected NAME=PATH, got: {spec}")
        name, path = spec.split("=", 1)
        out.append((name.strip(), path.strip()))
    return out


# ── closed meta blocks ───────────────────────────────────────────────────────

def count_closed_meta_blocks(text: str) -> int:
    return len(CLOSED_META_RE.findall(str(text)))


def meta_block_spans(text: str) -> list[tuple[int, int, str]]:
    """(start, end, inner_content) for every CLOSED meta block, in order.
    `start`/`end` are offsets of the full block including the markers."""
    return [
        (m.start(), m.end(), m.group(1)) for m in CLOSED_META_RE.finditer(str(text))
    ]


# ── stated confidence ────────────────────────────────────────────────────────

# Confidence statements inside meta blocks look like "Confidence: 0.90",
# "probability of solving it correctly is about 0.40", "confidence is 85%".
# The value must be STATED BY THE MODEL — gold answers are never consulted here.
CONF_RE = re.compile(
    r"(?:probability|confidence)[^0-9]{0,60}?(\d+(?:\.\d+)?)\s*(%?)",
    re.IGNORECASE,
)


def stated_confidences(text: str) -> list[float]:
    """All confidences stated inside CLOSED meta blocks, normalized to [0, 1]."""
    vals: list[float] = []
    for _, _, inner in meta_block_spans(text):
        for num, pct in CONF_RE.findall(inner):
            v = float(num)
            if pct == "%" or v > 1.0:
                v /= 100.0
            v = min(1.0, max(0.0, v))
            if v > 0.001:  # near-zero values are parsing artifacts (see prompt.py)
                vals.append(v)
    return vals


# ── calibration math (shared by calibration.py and aggregate_tables.py) ─────

def ece_15bin(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error over `n_bins` equal-width bins on [0, 1]."""
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    if len(conf) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # digitize: conf == 1.0 must land in the last bin, not overflow it.
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / len(conf)) * abs(
            correct[mask].mean() - conf[mask].mean()
        )
    return float(ece)


def brier(conf: np.ndarray, correct: np.ndarray) -> float:
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    if len(conf) == 0:
        return float("nan")
    return float(np.mean((conf - correct) ** 2))


def overconfidence_rate(
    conf: np.ndarray, correct: np.ndarray, high_conf: float = 0.8
) -> float:
    """P(wrong | stated confidence >= high_conf); NaN when nothing qualifies."""
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    mask = conf >= high_conf
    if mask.sum() == 0:
        return float("nan")
    return float((~correct[mask]).mean())


# ── markdown helpers ─────────────────────────────────────────────────────────

def md_table(header: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def fmt_pct(x: float, digits: int = 1) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{100.0 * x:.{digits}f}%"
