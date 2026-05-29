"""Shared primitives for Phase A inference probes.

Canonical home for helpers that were duplicated across a1/a2/a6.
Keep ONLY stable, probe-agnostic primitives here:
  - meta-region tokenization (find_meta_spans)
  - forward-pass scoring (score_response_logp, score_token_entropy)
  - statistics (cohen_d, mann_whitney_auc, paired_perm_test, perm_test_two_sample)
  - prompt builders (build_student_input, build_teacher_input)

Experiment-specific sampling / prompt-templates / reward formulas stay local
to each probe (they drift; do not over-abstract — Karpathy).
"""
from __future__ import annotations
import numpy as np
import torch

from .env import META_OPEN_ID, META_CLOSE_ID

MAX_RESP_TOK_DEFAULT = 8192
MAX_PROMPT_TOK_DEFAULT = 1024


# ── meta-region tokenization ────────────────────────────────────────────────

def find_meta_spans(resp_ids: list[int]) -> list[tuple[int, int]]:
    """Return (open_exclusive, close_exclusive) spans for <|meta|>...<|/meta|>.

    Span covers the *content* tokens (excludes the open/close marker tokens).
    """
    spans, in_meta, start = [], False, 0
    for i, t in enumerate(resp_ids):
        if t == META_OPEN_ID:
            in_meta, start = True, i + 1
        elif t == META_CLOSE_ID and in_meta:
            spans.append((start, i))
            in_meta = False
    return spans


def meta_region_mask(resp_ids: list[int], length: int) -> np.ndarray:
    """Boolean mask over resp_ids[:length]; True = inside a meta span (content only)."""
    mask = np.zeros(length, dtype=bool)
    for a, b in find_meta_spans(resp_ids):
        mask[min(a, length):min(b, length)] = True
    return mask


# ── prompt builders ─────────────────────────────────────────────────────────

def build_student_input(tokenizer, question: str, completion: str,
                        max_resp: int = MAX_RESP_TOK_DEFAULT,
                        max_prompt: int = MAX_PROMPT_TOK_DEFAULT):
    """Student-side input (question only) + response. Returns (ids, resp_start, resp_ids)."""
    messages = [{"role": "user", "content": question}]
    prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)[:max_prompt]
    resp_ids = tokenizer.encode(completion, add_special_tokens=False)[:max_resp]
    return prompt_ids + resp_ids, len(prompt_ids), resp_ids


def build_teacher_input(tokenizer, question: str, answer: str, resp_ids: list[int],
                        max_prompt: int = MAX_PROMPT_TOK_DEFAULT):
    """Teacher-side input (question + answer reveal) + response. Returns (ids, resp_start).

    Used for both T+ (gold answer) and T- (decoy answer) by passing different `answer`.
    """
    block = (
        f"{question}\n\n"
        f"[REFERENCE — the correct final answer is: {answer}. "
        f"Score the following student response token-by-token with this information.]"
    )
    messages = [{"role": "user", "content": block}]
    prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)[:max_prompt]
    return prompt_ids + resp_ids, len(prompt_ids)


# ── forward-pass scoring ────────────────────────────────────────────────────

@torch.no_grad()
def score_response_logp(model, input_ids: list[int], resp_start: int, device: str) -> np.ndarray:
    """Per-response-token logp under model. Length = len(input_ids) - resp_start."""
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    out = model(ids, use_cache=False)
    logits = out.logits[0]
    pred_logits = logits[resp_start - 1: -1].float()
    targets = ids[0, resp_start:]
    logp = torch.nn.functional.log_softmax(pred_logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return logp.cpu().numpy()


@torch.no_grad()
def score_token_entropy(model, input_ids: list[int], resp_start: int, device: str) -> np.ndarray:
    """Per-response-token predictive entropy H_t = -sum p log p."""
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    out = model(ids, use_cache=False)
    logits = out.logits[0]
    pred_logits = logits[resp_start - 1: -1].float()
    log_probs = torch.nn.functional.log_softmax(pred_logits, dim=-1)
    H = -(log_probs.exp() * log_probs).sum(dim=-1)
    return H.cpu().numpy()


# ── statistics ──────────────────────────────────────────────────────────────

def cohen_d(a, b) -> float:
    a, b = list(a), list(b)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = np.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2))
    return float((np.mean(a) - np.mean(b)) / pooled) if pooled > 0 else float("nan")


def mann_whitney_auc(scores_pos, scores_neg) -> float:
    """AUC = P(pos > neg) via Mann-Whitney U."""
    pos, neg = list(scores_pos), list(scores_neg)
    if not pos or not neg:
        return float("nan")
    hits, total = 0.0, 0
    for p in pos:
        for n in neg:
            total += 1
            if p > n:
                hits += 1
            elif p == n:
                hits += 0.5
    return hits / total if total else float("nan")


def perm_test_two_sample(a, b, rng, n_perm: int = 5000) -> float:
    """Two-sided permutation test on mean(a) - mean(b)."""
    a, b = list(a), list(b)
    if not a or not b:
        return float("nan")
    obs = abs(np.mean(a) - np.mean(b))
    pooled = np.array(a + b)
    n_a = len(a)
    hits = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        if abs(pooled[:n_a].mean() - pooled[n_a:].mean()) >= obs:
            hits += 1
    return (hits + 1) / (n_perm + 1)


def paired_perm_test(diffs, rng, n_perm: int = 5000) -> float:
    """Paired sign-flip permutation test on mean of per-item diffs ≠ 0."""
    diffs = np.asarray([d for d in diffs if d is not None and not np.isnan(d)])
    if len(diffs) == 0:
        return float("nan")
    obs = abs(diffs.mean())
    hits = 0
    for _ in range(n_perm):
        signs = rng.choice([-1, 1], size=len(diffs))
        if abs((diffs * signs).mean()) >= obs:
            hits += 1
    return (hits + 1) / (n_perm + 1)
