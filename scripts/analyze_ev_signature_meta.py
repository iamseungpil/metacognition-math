#!/usr/bin/env python3
"""Epistemic Verbalization (EV) signature analysis for Meta-CoT completions.

Computes the four EV signature metrics defined in plan_EAD_unified Section 6:

  1. delta_H +/- 5     : mean full-vocab Shannon entropy difference between 5
                         tokens AFTER and 5 BEFORE an EV marker.
  2. d_M                : Mahalanobis distance in (H_t, top1_prob, top1-top2) space
                         between the EV pair (marker-token, next-token) and a
                         neutral pair drawn from a random position, with
                         bootstrap 95% CI.
  3. I(M_c ; Y | D)     : mutual information between meta-content indicator and
                         correctness Y, conditioned on difficulty tercile D
                         (plug-in histogram estimator).
  4. C_t                : cumulative confidence gain over 5 post-marker tokens,
                         C_t = sum_s (1 - H_s / log2(V)). Cohen's d reported
                         across splits.

Adapted from:
  - metacognition-behavior-uncertainty/scripts/analyze_deep_epistemic.py
    (local before/on/after windows with correctness split)
  - metacognition-behavior-uncertainty/scripts/analyze_epistemic_trajectories.py
    (aggregate summaries)
  - metacognition/scripts/analyze_entropy_meta.py (HF forward-pass pattern)

Input schema: the JSON produced by src.eval.eval_hf save_results_bundle -- i.e.
an object with key 'results', each element having 'completion', 'is_correct',
'num_meta_blocks', and 'benchmark' (at minimum; 'full_question' preferred for
prompt reconstruction, 'question' used as fallback).

Usage:
    python scripts/analyze_ev_signature_meta.py \\
        --model_path checkpoints/v6_clean_10k_E19 \\
        --eval_json   results/eval_v6_E19/eval_v6_clean_10k_E19.json \\
        --output_dir  results/ev_signature/ \\
        --max_samples 200 \\
        --marker_mode meta
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WINDOW = 5  # plan §6 specifies +/- 5 tokens
N_BOOTSTRAP = 1000
DIFFICULTY_TERCILE_LABELS = ("easy", "medium", "hard")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MarkerSpan:
    """Start/end (inclusive) token indices of an EV marker content region."""
    start: int
    end: int


@dataclass
class SampleTrace:
    """Per-sample forward-pass output plus marker metadata.

    Used as the canonical interchange between the GPU-backed extractor and
    the pure-numerical metric computers. The test harness fabricates these
    directly so no model is required.
    """
    sample_id: int
    is_correct: bool
    benchmark: str
    entropy: np.ndarray           # shape (T,) nats
    top1_prob: np.ndarray         # shape (T,)
    top2_prob: np.ndarray         # shape (T,)
    difficulty_proxy: float       # e.g. -log(accuracy_rate) per benchmark
    spans: list[MarkerSpan]
    vocab_size: int


# ---------------------------------------------------------------------------
# Marker detection (reuses pattern from analyze_entropy_meta.py)
# ---------------------------------------------------------------------------

def find_meta_spans_in_text(text: str) -> list[tuple[int, int]]:
    """Locate `<|meta|>...<|/meta|>` character spans in completion text."""
    pat = re.compile(r"<\|meta\|>(.*?)<\|/meta\|>", re.DOTALL)
    return [(m.start(1), m.end(1)) for m in pat.finditer(text)]


def find_confidence_spans_in_text(text: str) -> list[tuple[int, int]]:
    """Locate `confidence: 0.XX` character spans in completion text."""
    pat = re.compile(r"confidence\s*:\s*\d+(?:\.\d+)?", re.IGNORECASE)
    return [(m.start(), m.end()) for m in pat.finditer(text)]


# ---------------------------------------------------------------------------
# Metric 1: delta_H +/- 5
# ---------------------------------------------------------------------------

def compute_delta_h_window(
    entropy: np.ndarray,
    span: MarkerSpan,
    window: int = WINDOW,
) -> Optional[float]:
    """Mean entropy after - mean entropy before, with symmetric window.

    Returns None if either side of the window lies entirely outside the
    sequence.
    """
    T = len(entropy)
    before_start = max(0, span.start - window)
    before_end = span.start
    after_start = min(T, span.end + 1)
    after_end = min(T, after_start + window)

    if before_end <= before_start or after_end <= after_start:
        return None
    before_mean = float(np.mean(entropy[before_start:before_end]))
    after_mean = float(np.mean(entropy[after_start:after_end]))
    return after_mean - before_mean


def aggregate_delta_h(
    samples: Iterable[SampleTrace],
    window: int = WINDOW,
) -> dict:
    """Aggregate delta_H stats across samples, split by correctness."""
    correct_vals: list[float] = []
    incorrect_vals: list[float] = []
    for trace in samples:
        for span in trace.spans:
            d = compute_delta_h_window(trace.entropy, span, window=window)
            if d is None or not math.isfinite(d):
                continue
            if trace.is_correct:
                correct_vals.append(d)
            else:
                incorrect_vals.append(d)
    all_vals = correct_vals + incorrect_vals
    return {
        "n_all": len(all_vals),
        "mean_all": float(np.mean(all_vals)) if all_vals else 0.0,
        "std_all": float(np.std(all_vals)) if all_vals else 0.0,
        "n_correct": len(correct_vals),
        "mean_correct": float(np.mean(correct_vals)) if correct_vals else 0.0,
        "n_incorrect": len(incorrect_vals),
        "mean_incorrect": float(np.mean(incorrect_vals)) if incorrect_vals else 0.0,
    }


# ---------------------------------------------------------------------------
# Metric 2: Mahalanobis distance in 3-axis space
# ---------------------------------------------------------------------------

def _sample_ev_pair(trace: SampleTrace, span: MarkerSpan) -> Optional[np.ndarray]:
    """Return concatenated [H_m, p1_m, margin_m, H_n, p1_n, margin_n] (6-dim).

    Marker-token = first content token of the span; next-token = token
    immediately after the span. Returns None if indices fall outside bounds.
    """
    m = span.start
    n = span.end + 1
    T = len(trace.entropy)
    if m < 0 or m >= T or n >= T:
        return None
    margin_m = float(trace.top1_prob[m] - trace.top2_prob[m])
    margin_n = float(trace.top1_prob[n] - trace.top2_prob[n])
    return np.array([
        float(trace.entropy[m]), float(trace.top1_prob[m]), margin_m,
        float(trace.entropy[n]), float(trace.top1_prob[n]), margin_n,
    ], dtype=np.float64)


def _sample_neutral_pair(trace: SampleTrace, rng: random.Random) -> Optional[np.ndarray]:
    """Random neutral (t, t+1) pair drawn from outside any marker span."""
    T = len(trace.entropy)
    if T < 2:
        return None
    excluded = set()
    for span in trace.spans:
        for i in range(max(0, span.start - 1), min(T, span.end + 2)):
            excluded.add(i)
    candidates = [i for i in range(T - 1) if i not in excluded and (i + 1) not in excluded]
    if not candidates:
        return None
    m = rng.choice(candidates)
    n = m + 1
    margin_m = float(trace.top1_prob[m] - trace.top2_prob[m])
    margin_n = float(trace.top1_prob[n] - trace.top2_prob[n])
    return np.array([
        float(trace.entropy[m]), float(trace.top1_prob[m]), margin_m,
        float(trace.entropy[n]), float(trace.top1_prob[n]), margin_n,
    ], dtype=np.float64)


def _mahalanobis_between_sets(A: np.ndarray, B: np.ndarray, ridge: float = 1e-6) -> float:
    """Mahalanobis distance between set means using pooled covariance."""
    if A.shape[0] == 0 or B.shape[0] == 0:
        return 0.0
    mean_a = A.mean(axis=0)
    mean_b = B.mean(axis=0)
    pooled = np.cov(np.vstack([A, B]).T)
    pooled = np.atleast_2d(pooled)
    pooled = pooled + ridge * np.eye(pooled.shape[0])
    try:
        inv = np.linalg.pinv(pooled)
    except np.linalg.LinAlgError:
        return 0.0
    diff = mean_a - mean_b
    val = float(diff @ inv @ diff.T)
    return math.sqrt(max(val, 0.0))


def compute_mahalanobis_d(
    samples: list[SampleTrace],
    rng_seed: int = 0,
    n_bootstrap: int = N_BOOTSTRAP,
) -> dict:
    """Metric 2: d_M between EV pairs and neutral pairs, with bootstrap CI."""
    rng = random.Random(rng_seed)
    ev_vecs: list[np.ndarray] = []
    neutral_vecs: list[np.ndarray] = []
    for trace in samples:
        for span in trace.spans:
            ev = _sample_ev_pair(trace, span)
            if ev is not None and np.all(np.isfinite(ev)):
                ev_vecs.append(ev)
        neu = _sample_neutral_pair(trace, rng)
        if neu is not None and np.all(np.isfinite(neu)):
            neutral_vecs.append(neu)

    if len(ev_vecs) == 0 or len(neutral_vecs) == 0:
        return {
            "d_m": 0.0, "ci_low": 0.0, "ci_high": 0.0,
            "n_ev": len(ev_vecs), "n_neutral": len(neutral_vecs),
        }

    A = np.stack(ev_vecs)
    B = np.stack(neutral_vecs)
    d_point = _mahalanobis_between_sets(A, B)

    # Bootstrap: resample each group with replacement.
    boots: list[float] = []
    for _ in range(n_bootstrap):
        a_idx = np.array([rng.randrange(A.shape[0]) for _ in range(A.shape[0])])
        b_idx = np.array([rng.randrange(B.shape[0]) for _ in range(B.shape[0])])
        boots.append(_mahalanobis_between_sets(A[a_idx], B[b_idx]))
    lo = float(np.percentile(boots, 2.5)) if boots else 0.0
    hi = float(np.percentile(boots, 97.5)) if boots else 0.0

    return {
        "d_m": d_point,
        "ci_low": lo,
        "ci_high": hi,
        "n_ev": int(A.shape[0]),
        "n_neutral": int(B.shape[0]),
    }


# ---------------------------------------------------------------------------
# Metric 3: I(M_c ; Y | D)
# ---------------------------------------------------------------------------

def _difficulty_terciles(values: list[float]) -> list[int]:
    """Assign each value to tercile 0/1/2 based on rank."""
    if not values:
        return []
    arr = np.asarray(values, dtype=float)
    if np.all(arr == arr[0]):
        return [0] * len(values)
    q1, q2 = np.percentile(arr, [100 / 3, 200 / 3])
    bins = []
    for v in arr:
        if v <= q1:
            bins.append(0)
        elif v <= q2:
            bins.append(1)
        else:
            bins.append(2)
    return bins


def _mutual_information_discrete(m: list[int], y: list[int]) -> float:
    """Plug-in MI estimate I(M ; Y) for discrete variables (nats)."""
    if not m or not y or len(m) != len(y):
        return 0.0
    n = len(m)
    joint: dict[tuple[int, int], int] = {}
    pm: dict[int, int] = {}
    py: dict[int, int] = {}
    for a, b in zip(m, y):
        joint[(a, b)] = joint.get((a, b), 0) + 1
        pm[a] = pm.get(a, 0) + 1
        py[b] = py.get(b, 0) + 1
    mi = 0.0
    for (a, b), c in joint.items():
        p_ab = c / n
        p_a = pm[a] / n
        p_b = py[b] / n
        if p_ab > 0 and p_a > 0 and p_b > 0:
            mi += p_ab * math.log((p_ab + 1e-12) / (p_a * p_b + 1e-12))
    return max(mi, 0.0)


def compute_conditional_mi(samples: list[SampleTrace]) -> dict:
    """I(M_c ; Y | D) with M = has-marker indicator, Y = correctness, D = tercile."""
    if not samples:
        return {"i_mi": 0.0, "n_samples": 0, "per_tercile": {}}

    diffs = [s.difficulty_proxy for s in samples]
    terciles = _difficulty_terciles(diffs)
    n = len(samples)

    groups: dict[int, list[tuple[int, int]]] = {0: [], 1: [], 2: []}
    for idx, trace in enumerate(samples):
        m = 1 if trace.spans else 0
        y = 1 if trace.is_correct else 0
        groups[terciles[idx]].append((m, y))

    total_mi = 0.0
    per_t: dict[str, dict] = {}
    for t_label, t_idx in zip(DIFFICULTY_TERCILE_LABELS, (0, 1, 2)):
        pairs = groups[t_idx]
        if not pairs:
            per_t[t_label] = {"n": 0, "mi": 0.0}
            continue
        m_vals = [p[0] for p in pairs]
        y_vals = [p[1] for p in pairs]
        mi_t = _mutual_information_discrete(m_vals, y_vals)
        weight = len(pairs) / n
        total_mi += weight * mi_t
        per_t[t_label] = {"n": len(pairs), "mi": mi_t}

    return {"i_mi": total_mi, "n_samples": n, "per_tercile": per_t}


# ---------------------------------------------------------------------------
# Metric 4: C_t cumulative confidence gain
# ---------------------------------------------------------------------------

def compute_c_t_single(
    entropy: np.ndarray,
    span: MarkerSpan,
    vocab_size: int,
    window: int = WINDOW,
) -> Optional[float]:
    """C_t = sum_{s=1..W} (1 - H_s / log2(V)) over post-marker tokens."""
    if vocab_size <= 1:
        return None
    T = len(entropy)
    post_start = min(T, span.end + 1)
    post_end = min(T, post_start + window)
    if post_end <= post_start:
        return None
    log2v = math.log2(vocab_size)
    # entropy is in nats; convert to bits via /ln(2)
    window_ent = entropy[post_start:post_end]
    window_ent_bits = window_ent / math.log(2.0)
    gain = 1.0 - (window_ent_bits / log2v)
    return float(np.sum(gain))


def aggregate_c_t(
    samples: Iterable[SampleTrace],
    window: int = WINDOW,
) -> dict:
    """Aggregate C_t per sample (one value per marker span)."""
    correct_vals: list[float] = []
    incorrect_vals: list[float] = []
    for trace in samples:
        for span in trace.spans:
            c = compute_c_t_single(trace.entropy, span, trace.vocab_size, window=window)
            if c is None or not math.isfinite(c):
                continue
            if trace.is_correct:
                correct_vals.append(c)
            else:
                incorrect_vals.append(c)
    all_vals = correct_vals + incorrect_vals
    return {
        "n_all": len(all_vals),
        "mean_all": float(np.mean(all_vals)) if all_vals else 0.0,
        "std_all": float(np.std(all_vals)) if all_vals else 0.0,
        "mean_correct": float(np.mean(correct_vals)) if correct_vals else 0.0,
        "mean_incorrect": float(np.mean(incorrect_vals)) if incorrect_vals else 0.0,
        "correct_values": correct_vals,
        "incorrect_values": incorrect_vals,
    }


def cohen_d(a: list[float], b: list[float]) -> float:
    """Standard Cohen's d between two independent samples."""
    if len(a) < 2 or len(b) < 2:
        return 0.0
    ma, mb = float(np.mean(a)), float(np.mean(b))
    va, vb = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    s = math.sqrt((va + vb) / 2.0)
    if s <= 0:
        return 0.0
    return (ma - mb) / s


# ---------------------------------------------------------------------------
# Forward-pass extraction (GPU path)
# ---------------------------------------------------------------------------

class HFForwardExtractor:
    """Wraps a HuggingFace CausalLM and produces SampleTrace objects.

    Lazy-imports torch / transformers so that metric-only code paths (and the
    smoke test) remain importable without a GPU.
    """

    def __init__(self, model_path: str, dtype: str = "bfloat16", max_seq_len: int = 8192):
        import torch  # noqa: F401  # side-effect import check
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype_map = {
            "bfloat16": __import__("torch").bfloat16,
            "float16": __import__("torch").float16,
            "float32": __import__("torch").float32,
        }
        self._torch = __import__("torch")
        self._F = __import__("torch.nn.functional", fromlist=["softmax"])
        torch_dtype = dtype_map[dtype]

        print(f"Loading tokenizer from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print(f"Loading model from {model_path} (dtype={dtype}, device_map=auto)")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        self.max_seq_len = max_seq_len
        self.vocab_size = int(self.model.config.vocab_size)

    def _build_prompt_text(self, question: str) -> str:
        messages = [{"role": "user", "content": question}]
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            return f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"

    def run(
        self,
        question: str,
        completion: str,
    ) -> Optional[dict]:
        """Returns dict with entropy, top1, top2, prompt_token_len, completion_text.

        entropy/top1/top2 are aligned with target positions (length seq_len-1).
        """
        torch = self._torch
        F = self._F
        prompt_text = self._build_prompt_text(question)
        prompt_ids = self.tokenizer(
            prompt_text, return_tensors="pt", add_special_tokens=False,
        )["input_ids"]
        prompt_token_len = int(prompt_ids.shape[1])

        full_text = prompt_text + completion
        enc = self.tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"]
        if input_ids.shape[1] > self.max_seq_len:
            return None

        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)
        logits = outputs.logits.float()
        probs = F.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-12)
        entropy = -(probs * log_probs).sum(dim=-1).squeeze(0)  # nats

        top2 = torch.topk(probs, k=2, dim=-1).values.squeeze(0)  # (seq, 2)
        top1 = top2[:, 0]
        top2_second = top2[:, 1]

        # Align with "predicting next token": drop last position.
        entropy = entropy[:-1]
        top1 = top1[:-1]
        top2_second = top2_second[:-1]
        token_ids = input_ids.squeeze(0).cpu().tolist()

        return {
            "entropy": entropy.cpu().numpy().astype(np.float64),
            "top1": top1.cpu().numpy().astype(np.float64),
            "top2": top2_second.cpu().numpy().astype(np.float64),
            "prompt_token_len": prompt_token_len,
            "token_ids": token_ids,
        }

    def text_spans_to_token_spans(
        self,
        completion_text: str,
        prompt_text: str,
        token_ids: list[int],
        prompt_token_len: int,
        char_spans: list[tuple[int, int]],
    ) -> list[MarkerSpan]:
        """Map (char_start, char_end) spans within completion to token indices.

        Uses the running-decode approach from analyze_entropy_meta.py.
        Returns spans in the post-shift (predict-next-token) index space.
        """
        answer_start = max(prompt_token_len - 1, 0)
        if answer_start >= len(token_ids):
            return []
        completion_ids = token_ids[answer_start:]

        char_offsets = [0]
        acc = ""
        for t in completion_ids:
            acc += self.tokenizer.decode([t], skip_special_tokens=False)
            char_offsets.append(len(acc))

        def char_to_token(char_pos: int) -> int:
            for i, off in enumerate(char_offsets):
                if off >= char_pos:
                    return i
            return len(char_offsets) - 1

        out: list[MarkerSpan] = []
        for c_start, c_end in char_spans:
            tok_start_rel = char_to_token(c_start)
            tok_end_rel = max(char_to_token(c_end) - 1, tok_start_rel)
            tok_start = answer_start + tok_start_rel
            tok_end = answer_start + tok_end_rel
            if tok_end >= tok_start:
                out.append(MarkerSpan(start=tok_start, end=tok_end))
        return out


# ---------------------------------------------------------------------------
# Driver utilities
# ---------------------------------------------------------------------------

def load_eval_results(eval_json: str) -> tuple[list[dict], dict]:
    """Load eval JSON; handle both bundle ({results: [...]}) and flat list."""
    with open(eval_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "results" in data:
        return data["results"], data.get("run_metadata", {})
    if isinstance(data, list):
        return data, {}
    raise ValueError(f"Unrecognized eval JSON format: {eval_json}")


def build_difficulty_proxy(results: list[dict]) -> dict[str, float]:
    """Per-benchmark difficulty = 1 - accuracy; higher means harder."""
    per_bench: dict[str, list[float]] = {}
    for r in results:
        b = r.get("benchmark", "unknown")
        per_bench.setdefault(b, []).append(1.0 if r.get("is_correct") else 0.0)
    return {b: 1.0 - (sum(v) / len(v)) for b, v in per_bench.items()}


def build_samples_from_forward(
    extractor: HFForwardExtractor,
    results: list[dict],
    marker_mode: str,
    max_samples: int,
    difficulty_by_bench: dict[str, float],
) -> list[SampleTrace]:
    """Run forward pass on each result row and assemble SampleTrace list."""
    from tqdm import tqdm

    if marker_mode == "meta":
        find_char_spans = find_meta_spans_in_text
        trig = "<|meta|>"
    elif marker_mode == "confidence":
        find_char_spans = find_confidence_spans_in_text
        trig = "confidence:"
    else:
        raise ValueError(f"Unknown marker_mode: {marker_mode}")

    filtered = [r for r in results if trig.lower() in str(r.get("completion", "")).lower()]
    if max_samples > 0 and len(filtered) > max_samples:
        filtered = filtered[:max_samples]

    traces: list[SampleTrace] = []
    for idx, row in enumerate(tqdm(filtered, desc="Forward pass")):
        question = row.get("full_question") or row.get("question") or ""
        completion = row.get("completion", "")
        is_correct = bool(row.get("is_correct", False))
        benchmark = row.get("benchmark", "unknown")

        prompt_text = extractor._build_prompt_text(question)
        fwd = extractor.run(question, completion)
        if fwd is None:
            continue

        char_spans = find_char_spans(completion)
        if not char_spans:
            continue

        # Offset completion-level char positions by prompt text length, since
        # our token mapping works on `prompt + completion` string.
        # However text_spans_to_token_spans expects char positions within the
        # completion portion (not prepended with prompt). Adjust accordingly.
        token_spans = extractor.text_spans_to_token_spans(
            completion_text=completion,
            prompt_text=prompt_text,
            token_ids=fwd["token_ids"],
            prompt_token_len=fwd["prompt_token_len"],
            char_spans=char_spans,
        )
        if not token_spans:
            continue

        traces.append(SampleTrace(
            sample_id=idx,
            is_correct=is_correct,
            benchmark=benchmark,
            entropy=fwd["entropy"],
            top1_prob=fwd["top1"],
            top2_prob=fwd["top2"],
            difficulty_proxy=float(difficulty_by_bench.get(benchmark, 0.5)),
            spans=token_spans,
            vocab_size=extractor.vocab_size,
        ))
    return traces


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def compute_all_metrics(samples: list[SampleTrace]) -> dict:
    """Run the four EV metrics and return a JSON-serializable dict."""
    correct_samples = [s for s in samples if s.is_correct]
    incorrect_samples = [s for s in samples if not s.is_correct]

    delta_h = aggregate_delta_h(samples)
    d_m = compute_mahalanobis_d(samples)
    mi = compute_conditional_mi(samples)
    c_t = aggregate_c_t(samples)
    c_d = cohen_d(c_t["correct_values"], c_t["incorrect_values"])

    c_t_out = {k: v for k, v in c_t.items() if k not in {"correct_values", "incorrect_values"}}

    return {
        "n_samples_total": len(samples),
        "n_samples_correct": len(correct_samples),
        "n_samples_incorrect": len(incorrect_samples),
        "metric_1_delta_h": delta_h,
        "metric_2_mahalanobis_d": d_m,
        "metric_3_mi_m_y_given_d": mi,
        "metric_4_c_t": c_t_out,
        "metric_4_cohen_d_correct_vs_incorrect": c_d,
    }


def save_per_sample_csv(samples: list[SampleTrace], path: str) -> None:
    fieldnames = [
        "sample_id", "is_correct", "benchmark", "num_spans",
        "delta_h_first_span", "c_t_first_span", "difficulty_proxy",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for s in samples:
            delta_h_val = None
            c_t_val = None
            if s.spans:
                delta_h_val = compute_delta_h_window(s.entropy, s.spans[0])
                c_t_val = compute_c_t_single(s.entropy, s.spans[0], s.vocab_size)
            w.writerow({
                "sample_id": s.sample_id,
                "is_correct": s.is_correct,
                "benchmark": s.benchmark,
                "num_spans": len(s.spans),
                "delta_h_first_span": f"{delta_h_val:.6f}" if delta_h_val is not None else "",
                "c_t_first_span": f"{c_t_val:.6f}" if c_t_val is not None else "",
                "difficulty_proxy": f"{s.difficulty_proxy:.6f}",
            })


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_path", required=True, help="HF checkpoint path")
    p.add_argument("--eval_json", required=True, help="eval_hf.py output JSON (with 'results' key)")
    p.add_argument("--output_dir", default="results/ev_signature/")
    p.add_argument("--max_samples", type=int, default=200)
    p.add_argument("--max_seq_len", type=int, default=8192)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--marker_mode", default="meta", choices=["meta", "confidence"])
    p.add_argument("--window", type=int, default=WINDOW)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results, run_metadata = load_eval_results(args.eval_json)
    if not results:
        print("ERROR: empty results list")
        sys.exit(1)
    difficulty_by_bench = build_difficulty_proxy(results)
    print(f"Difficulty-by-benchmark: {difficulty_by_bench}")

    extractor = HFForwardExtractor(
        model_path=args.model_path,
        dtype=args.dtype,
        max_seq_len=args.max_seq_len,
    )

    samples = build_samples_from_forward(
        extractor=extractor,
        results=results,
        marker_mode=args.marker_mode,
        max_samples=args.max_samples,
        difficulty_by_bench=difficulty_by_bench,
    )
    print(f"Built {len(samples)} SampleTraces")

    if not samples:
        print("ERROR: zero samples with valid markers. Nothing to analyze.")
        sys.exit(1)

    report = compute_all_metrics(samples)
    report["config"] = {
        "model_path": args.model_path,
        "eval_json": args.eval_json,
        "max_samples": args.max_samples,
        "window": args.window,
        "dtype": args.dtype,
        "marker_mode": args.marker_mode,
    }

    out_json = output_dir / "ev_signature_stats.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Saved {out_json}")

    out_csv = output_dir / "ev_signature_per_sample.csv"
    save_per_sample_csv(samples, str(out_csv))
    print(f"Saved {out_csv}")

    # Print a terse summary.
    print("\n=== EV SIGNATURE SUMMARY ===")
    dh = report["metric_1_delta_h"]
    dm = report["metric_2_mahalanobis_d"]
    mi = report["metric_3_mi_m_y_given_d"]
    ct = report["metric_4_c_t"]
    print(f"  (1) delta_H       mean={dh['mean_all']:+.4f} nats  "
          f"(correct={dh['mean_correct']:+.4f}, incorrect={dh['mean_incorrect']:+.4f})")
    print(f"  (2) d_M           {dm['d_m']:.4f}  [95% CI {dm['ci_low']:.4f}..{dm['ci_high']:.4f}]")
    print(f"  (3) I(M;Y|D)      {mi['i_mi']:.4f} nats  (N={mi['n_samples']})")
    print(f"  (4) C_t           mean={ct['mean_all']:+.4f} bits  "
          f"Cohen_d(correct,incorrect)={report['metric_4_cohen_d_correct_vs_incorrect']:+.3f}")


if __name__ == "__main__":
    main()
