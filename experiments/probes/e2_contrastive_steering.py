"""E.3 — Contrastive-Direction A/B for Meta-Content Steering (CTSD Phase E,
plan_ctsd_E3_contrastive_direction_2026_06_03, LOCKED pre-registration).

This file EXTENDS the E.2 probe (e2-steering-probe) — Karpathy minimal-change: the working
machinery (v8_strict self-meta harvest, the META-ONLY ContrastiveMetaSteerProcessor, pass@k
capability split, headroom, paired Δacc + MDE power gate + paired_perm_test, the leakage guard)
is PRESERVED. Three changes implement E.3:

  CHANGE 1 — PERFORMANCE HANDOFF (HF-meta → vLLM-continuation). E.2 ran the WHOLE steered
    continuation on slow HF (~10-20x vLLM) → infeasible at 6 arms x N. E.3 restructures so HF
    steering covers ONLY the meta span (max_new = META_CAP ~256, stop at <|/meta|>), then the
    long UNSTEERED continuation runs on vLLM resumed from the EXACT handoff token ids (steering is
    already baked into the meta). HF and vLLM are NEVER co-resident (b1 16k OOM lesson): P1-HF frees
    the HF model before P1-vLLM re-creates VllmGen.

  CHANGE 2 — 6 CONTRAST MODES (the grounding factorial arms). A CONTRASTS registry maps each
    steered mode -> (ctx_A_suffix, ctx_B_suffix); build_reveal_ids is generalized to take an
    arbitrary reveal SUFFIX (not only the answer-reveal). steer = alpha*(logit_A - logit_B), reusing
    ContrastiveMetaSteerProcessor unchanged. gold_decoy is kept BYTE-IDENTICAL to E.2 (REVEAL
    constant + REVEAL_SUFFIX reproduce the exact old string). Single moderate alpha (default 0.6,
    NOT 1.0 — E.2 saw 1.0 destabilize off-distribution); --alpha flag. Arms per problem: self
    (baseline) + the 5 steered modes.

  CHANGE 3 — OBJECTIVE UNCERTAINTY METRICS (the E.2 gap). Per (problem, arm), over k continuations:
    accuracy, self_consistency (modal-equivalence-class fraction via the GRADER's equivalence),
    agree_with_gold, verbalized_conf (parsed "confidence: X" from the steered meta), calibration_gap,
    and a best-effort answer_entropy (one HF forward over the final-answer span of one chosen
    continuation, taken WHILE the HF model is still resident; null + noted if unclean — never blocks).

Steering mechanism (FIXED, unchanged from E.2): LOGIT-level contrastive decoding, META-CONTENT-ONLY:
  at each self-emitted meta token scores += alpha*(logit|ctx_A - logit|ctx_B), ctx_A/ctx_B the two
  reveal contexts of the SAME frozen v8_strict advanced in KV-lockstep, applied ONLY between 151669
  and 151670. Position = the model's OWN self-emitted <|meta|> (self-trigger). E20a self-emits 0% so
  the self-trigger harvest is empty for it (opt-in via --model e20a). v8_strict is the model.

Two phases (DISCOVERY -> CONFIRMATION; winner's-curse-safe): Phase 1 (n~40-50, all 6 arms, fresh
  seed A) ranks directions by paired Δacc + the 4 decomposition contrasts + objective metrics; output
  = best 1-2 directions, NO PASS claim. Phase 2 (n~100-120, --arms subset, FRESH seed B → a DIFFERENT
  problem pool) is the pre-registered confirmation. Discovery/confirmation separation is enforced by
  the fresh problem set (different seed), NOT by in-run alpha selection — the E.2 cross-fit-alpha block
  is dropped (alpha is fixed per-phase).

Tokenizer: EVERY v8_strict tokenizer use (HF encode/decode, the LogitsProcessor reveal-stream, the
  vLLM handoff continuation, grading) goes through the SAME safe_tokenizer_path-substituted tokenizer
  (the v8_strict checkpoint tokenizer is broken under transformers 4.57; the E20a tokenizer has an
  identical Qwen3 vocab). The handoff is therefore token-id consistent between HF and vLLM.

Karpathy minimal-change: IMPORTS only from a3 (raw_entropy/first_boxed_token_idx — find_meta_spans
  NOT used), b2 (extract_first_meta_block/make_decoy), b4 (representative_pool), a6
  (find_answer_token_mask), _decoy_utils (_rule_based_decoy/_numerically_equal), rewards (signal
  predicates), common.grading (robust_grade/is_gradeable/extract_last_boxed) / vllm_gen
  (VllmGen/safe_tokenizer_path) / probe_utils (paired_perm_test). a3/a3b/a6/probe_utils/env/grading/
  vllm_gen/rewards are NOT modified — all new logic lives in this file.

Outputs reports/e2_steering_<model>_<tag>.jsonl (turn-granular, one row per (problem, arm)) +
  reports/e2_steering_<model>_<tag>.json (summary).
"""
from __future__ import annotations
import argparse, json, re, time, gc
import random as _random
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LogitsProcessor, LogitsProcessorList
from huggingface_hub import hf_hub_download
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))         # experiments/
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # repo root (src/)
from common.env import (
    TEACHER_MODEL, SFT_V8_STRICT, EVAL_R10V2_E20A, EVAL_R10V2_V8, HF_DATASET,
    META_OPEN_ID, META_CLOSE_ID, REPORTS_DIR, STRATIFIED_SAMPLE_SEED,
)
from common.vllm_gen import VllmGen, safe_tokenizer_path
from common.probe_utils import paired_perm_test
from common.grading import robust_grade, is_gradeable, extract_last_boxed
# IMPORT ONLY from a3 (HARD constraint: never modify a3). find_meta_spans is NOT used for content
# extraction (it requires the close token 151670 -> EMPTY metas on open-only self-emits); the entropy
# helpers are kept for failure-tagging + best-effort answer_entropy.
from probes.a3_inject_causal import (
    raw_entropy, first_boxed_token_idx,
)
from probes.b4_teacher_steering import representative_pool
from probes.b2_e20a_content_variance import extract_first_meta_block, make_decoy
from probes.a6_six_cell_teacher_swap import find_answer_token_mask
from src.training._decoy_utils import _rule_based_decoy, _numerically_equal
from src.training.rewards import (
    _has_effective_verification_signal, _has_verification_signal,
    _has_redirection_signal, _has_strategy_switch_signal,
    _has_uncertainty_signal, _has_overconfidence_signal,
)

# E20a tokenizer fallback (identical Qwen3 vocab) — same as B.1/B.2/B.4/C.1.
E20A_TOKENIZER_PATH = "/home/v-seungplee/sft_e20a_local"
MODELS = {"e20a": TEACHER_MODEL, "v8_strict": SFT_V8_STRICT}
EVALS = {"e20a": EVAL_R10V2_E20A, "v8_strict": EVAL_R10V2_V8}

# ── E.3 arms (Change 2) ────────────────────────────────────────────────────────
# Arms run per problem: `self` (baseline) + the 5 steered contrast modes. A single moderate alpha
# (default 0.6, NOT 1.0 — E.2 saw 1.0 destabilize into off-distribution tokens) for Phase 1.
STEERED_MODES = ("gold_decoy", "cautious", "gold_stance", "conf_down", "gold_conf_down", "conf_adaptive")
ARMS = ("self",) + STEERED_MODES
GOLD_BEARING_MODES = ("gold_decoy", "gold_stance", "gold_conf_down")   # leakage-guarded arms
DEFAULT_ALPHA = 0.6
META_CAP = 256              # HF steers ONLY the meta span (Change 1); this caps it.
DELTA_THR = 0.05            # Δacc PASS gate (Phase 2)
POWER_MDE_THR = 0.05        # realized_MDE hard-gate

# Stance instructions (Change 2). cautious steers TOWARD verification; confident is ctx_B.
CAUTIOUS_INSTR = ("Reason cautiously: question whether your current approach is right, verify each "
                  "step with an alternative method, avoid premature confidence.")
CONFIDENT_INSTR = ("Reason decisively: commit to your current approach with confidence and proceed.")

# gold_decoy reveal — BYTE-IDENTICAL to E.2. The E.2 REVEAL was a full-question template; here we
# express it as a SUFFIX appended to the bare question so the generalized build_reveal_ids reproduces
# the exact old string (question + REVEAL_SUFFIX(ans) == old REVEAL.format(question, ans)).
REVEAL = ("{question}\n\n[REFERENCE — the correct final answer is: {ans}. "
          "Continue the solution from where it stops.]")


def REVEAL_SUFFIX(ans: str) -> str:
    """The answer-reveal SUFFIX such that question + REVEAL_SUFFIX(ans) == the E.2 REVEAL string."""
    return ("\n\n[REFERENCE — the correct final answer is: {ans}. "
            "Continue the solution from where it stops.]").format(ans=ans)


# CONTRASTS[mode](gold, decoy) -> (A_suffix, B_suffix). A = ctx_A (steer TOWARD), B = ctx_B. All modes
# append their suffix to the SAME bare question + the SAME pre_meta_body, so the contrast isolates
# only the intended axis (answer / stance / confidence-number), gold cancelling on the gold-grounded
# arms. gold_decoy reuses REVEAL_SUFFIX verbatim -> byte-identical reveal stream to E.2.
CONTRASTS = {
    "gold_decoy":     lambda g, d: (REVEAL_SUFFIX(g), REVEAL_SUFFIX(d)),
    "cautious":       lambda g, d: (" " + CAUTIOUS_INSTR, " " + CONFIDENT_INSTR),
    "gold_stance":    lambda g, d: (f" (answer is {g}) " + CAUTIOUS_INSTR,
                                    f" (answer is {g}) " + CONFIDENT_INSTR),
    "conf_down":      lambda g, d: (" confidence: 0.15", " confidence: 0.95"),
    "gold_conf_down": lambda g, d: (f" (answer is {g}) confidence: 0.15",
                                    f" (answer is {g}) confidence: 0.95"),
}


# ── per-file helpers (copied from b4/c1 — they are per-probe, not shared) ──────

def load_tokenizer(path: str):
    """Robust tokenizer load. The v8_strict checkpoint tokenizer FAILS under transformers 4.57
    (extra_special_tokens is a list); we route through common.vllm_gen.safe_tokenizer_path, which
    returns the E20a-substituted tokenizer path (identical Qwen3 vocab, <|meta|>=151669,
    <|/meta|>=151670) in exactly that case. This is the ONE tokenizer object used everywhere for this
    model: HF encode/decode, grading, build_reveal_ids, the LogitsProcessor reveal-stream prefill, AND
    the vLLM handoff continuation — so HF and vLLM share a vocab and the token-id handoff is exact.
    Always assert the meta-token IDs."""
    safe_path = safe_tokenizer_path(path)
    tok = AutoTokenizer.from_pretrained(safe_path)
    assert tok.convert_tokens_to_ids("<|meta|>") == META_OPEN_ID, "META_OPEN_ID mismatch"
    assert tok.convert_tokens_to_ids("<|/meta|>") == META_CLOSE_ID, "META_CLOSE_ID mismatch"
    return tok


def build_prompt_ids(tok, question: str):
    msgs = [{"role": "user", "content": question}]
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok.encode(s, add_special_tokens=False)[:1024]


def build_reveal_ids(tok, question: str, reveal_suffix: str, pre_meta_body: str):
    """Generalized reveal-context prefix ids (Change 2). The reveal SUFFIX is appended to the bare
    user question (suffix = REVEAL_SUFFIX(ans) for gold_decoy, or a stance/confidence string for the
    other modes), then the SAME pre_meta_body the base stream reached at p_self (decode(base[:p_self]),
    the body BEFORE the self-emitted <|meta|>; the marker token itself is NOT included — the reveal
    streams are advanced only by the generated meta-CONTENT tokens). Because BOTH the A and B streams
    of a mode share an identical pre_meta_body and differ only in the suffix, logit_A - logit_B
    isolates exactly that mode's axis at every meta-content token. gold_decoy reproduces the E.2 byte
    string (question + REVEAL_SUFFIX(ans) == old REVEAL.format(...))."""
    user = question + reveal_suffix
    msgs = [{"role": "user", "content": user}]
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    prompt = tok.encode(s, add_special_tokens=False)[:1024]
    body = tok.encode(pre_meta_body, add_special_tokens=False)
    return prompt + body


def _json_safe(o):
    """Recursively cast numpy bool/float/int to python and NaN/Inf -> None so output is STRICT
    JSON. Same helper as b1/b2/b3/b4/c1."""
    import math
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    return o


# ── the core: META-ONLY contrastive steering LogitsProcessor (UNCHANGED from E.2) ──

class ContrastiveMetaSteerProcessor(LogitsProcessor):
    """Apply, INSIDE a meta span only, the contrastive shift
        scores += alpha * (logit_A - logit_B)
    where logit_A/logit_B are this model's next-token logits under reveal contexts A and B (E.2 called
    these gold/decoy; E.3 generalizes A/B to any contrast mode — the class body is UNCHANGED, only the
    A/B id CONSTRUCTION differs per mode, Change 2). Outside the span alpha is treated as 0 => identity
    (the reasoning body is NEVER touched). Span state is tracked by the meta token ids 151669/151670:
    in_meta turns True the step AFTER <|meta|> is appended and False once <|/meta|> is appended, so
    steering covers exactly the meta CONTENT tokens.

    Lockstep three-stream decode: the processor seeds the A/B streams once with a full prefill forward
    to get their KV caches, then advances each by exactly the newly-sampled base token — but ONLY while
    in_meta — keeping all three contexts byte-aligned on the META-CONTENT suffix. Outside the meta span
    the reveal streams are NOT advanced at all (the shift is zero there; skipping the 2 forwards is the
    perf optimization). Here in E.3 the HF generation ALSO stops at <|/meta|>/META_CAP (Change 1), so
    the body is never generated on HF at all — vLLM resumes it unsteered.

    Resume-at-self-meta: steered generation RESUMES from the baseline prefix ending at AND INCLUDING
    the self-emitted <|meta|> (151669), so start_in_meta=True and the first generated token is meta
    content. The A/B reveal streams are prefilled with [reveal_prompt + pre_meta_body] (the base body
    BEFORE <|meta|>) then primed through one <|meta|> token; thereafter each in-meta __call__ advances
    them by the just-generated meta-content token.

    `record` (optional) receives, for the smoke trace, a per-step {step, last_token, in_meta,
    delta_norm, would_steer_norm, applied} dict for the staging assertions."""

    def __init__(self, model, gold_ids: list[int], decoy_ids: list[int], alpha: float,
                 prompt_len: int, device, record: list | None = None,
                 start_in_meta: bool = True):
        self.model = model
        self.alpha = float(alpha)
        self.prompt_len = prompt_len            # number of base-prompt tokens (suffix begins here)
        self.device = device
        self.in_meta = bool(start_in_meta)
        self.meta_done = False                  # set once <|/meta|> closes: freeze reveal streams
        self.g_past = None                      # ctx_A reveal KV cache
        self.d_past = None                      # ctx_B reveal KV cache
        self.record = record
        self._step = 0
        self.g_past = self._prefill(gold_ids)
        self.d_past = self._prefill(decoy_ids)
        # Resume-path priming: prime both reveal streams through one <|meta|> token at init and stash
        # the resulting next-token logits as _pending, applied on the first __call__ (where no base
        # token has been generated yet). Thereafter each in-meta __call__ advances by the just-
        # generated meta-content token, keeping all three contexts byte-aligned.
        self._pending = None
        if self.in_meta and self.alpha != 0.0:
            g0, self.g_past = self._advance(self.g_past, META_OPEN_ID)
            d0, self.d_past = self._advance(self.d_past, META_OPEN_ID)
            self._pending = (g0, d0)

    @torch.no_grad()
    def _prefill(self, ids: list[int]):
        t = torch.tensor([ids], dtype=torch.long, device=self.device)
        out = self.model(t, use_cache=True)
        return out.past_key_values

    @torch.no_grad()
    def _advance(self, past, token_id: int):
        """Feed one token into a reveal stream, return (next_token_logits, new_past)."""
        t = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
        out = self.model(t, past_key_values=past, use_cache=True)
        return out.logits[0, -1].float(), out.past_key_values

    @torch.no_grad()
    def __call__(self, input_ids, scores):
        last = int(input_ids[0, -1].item())
        if last == META_OPEN_ID:
            self.in_meta = True
        elif last == META_CLOSE_ID:
            self.in_meta = False
            self.meta_done = True
        generating = input_ids.shape[1] > self.prompt_len
        in_span = self.in_meta and not self.meta_done and self.alpha != 0.0
        if in_span and self._pending is not None and not generating:
            g_logits, d_logits = self._pending      # first meta-content token (post-<|meta|> logits)
            self._pending = None
        elif in_span and generating:
            g_logits, self.g_past = self._advance(self.g_past, last)
            d_logits, self.d_past = self._advance(self.d_past, last)
        else:
            g_logits = d_logits = None
        would_steer_norm = 0.0
        if g_logits is not None:
            cand_delta = self.alpha * (g_logits - d_logits)
            would_steer_norm = float(cand_delta.norm().item())
        applied = bool(self.in_meta and self.alpha != 0.0 and g_logits is not None)
        delta_norm = 0.0
        if applied:
            delta = self.alpha * (g_logits - d_logits)
            scores = scores + delta.to(scores.dtype).unsqueeze(0)
            delta_norm = float(delta.norm().item())
        if self.record is not None:
            self.record.append({"step": self._step, "last_token": last,
                                 "in_meta": bool(self.in_meta), "delta_norm": delta_norm,
                                 "would_steer_norm": would_steer_norm, "applied": applied})
        self._step += 1
        return scores


class BanMetaOpenProcessor(LogitsProcessor):
    """no_meta arm under the HF decode path: forbid the model from opening a meta span by setting the
    <|meta|> logit to -inf at every step. Kept available (E.2 carryover); E.3's `self` arm uses the
    harvested baseline directly."""

    def __call__(self, input_ids, scores):
        scores[:, META_OPEN_ID] = float("-inf")
        return scores


class StopAtMetaClose(LogitsProcessor):
    """Force HF meta-only generation (Change 1) to terminate at <|/meta|>: once the close token has
    been emitted, every subsequent step is pinned to the close token so model.generate stops cleanly
    even without eos wiring. (Belt-and-suspenders alongside eos_token_id; the caller post-slices at
    the FIRST close regardless.)"""

    def __init__(self):
        self.closed = False

    def __call__(self, input_ids, scores):
        if int(input_ids[0, -1].item()) == META_CLOSE_ID:
            self.closed = True
        if self.closed:
            scores[:] = float("-inf")
            scores[:, META_CLOSE_ID] = 0.0
        return scores


# ── leakage control (H3) ──────────────────────────────────────────────────────

def meta_contains_answer(meta_text: str, gold: str, decoy: str, tok) -> bool:
    """True if the GENERATED steered meta CONTENT contains the gold or decoy answer (string, numeric,
    or token-level). Used for the leakage guard on the gold-bearing arms: a gold-bearing arm's gain
    must survive restriction to answer-free metas. Computed on the generated meta, NOT the reveal
    context."""
    if meta_text is None:
        return False
    g, d = str(gold).strip(), str(decoy).strip()
    if (g and g in meta_text) or (d and d in meta_text):
        return True
    for tok_num in re.findall(r"-?\d+\.?\d*", meta_text):
        if _numerically_equal(tok_num, g):
            return True
    meta_ids = tok.encode(meta_text, add_special_tokens=False)
    if find_answer_token_mask(tok, meta_text, g, meta_ids).any():
        return True
    if find_answer_token_mask(tok, meta_text, d, meta_ids).any():
        return True
    return False


# ── stance classification (qualitative co-gate) ───────────────────────────────

def classify_stance(meta_text: str) -> str:
    """Map a meta block to {verify, redirect, commit, generic} using the rewards signal vocabulary
    (import-only)."""
    if not meta_text:
        return "generic"
    if _has_effective_verification_signal(meta_text) or _has_verification_signal(meta_text):
        return "verify"
    if _has_redirection_signal(meta_text) or _has_strategy_switch_signal(meta_text):
        return "redirect"
    if _has_overconfidence_signal(meta_text):
        return "commit"
    if _has_uncertainty_signal(meta_text):
        return "verify"
    return "generic"


# ── small utilities ────────────────────────────────────────────────────────────

def pass_at_k(correct_list) -> int:
    """1 if ANY of the k baseline rollouts is correct (capability), else 0."""
    return int(any(bool(c) for c in correct_list))


def first_meta_open_pos(resp_ids: list[int]) -> int | None:
    """Token index of the FIRST self-emitted <|meta|> (151669) in a response token list, or None.
    Captures OPEN-ONLY metas (no close needed), unlike a3.find_meta_spans."""
    try:
        return resp_ids.index(META_OPEN_ID)
    except ValueError:
        return None


def split_around_meta(tok, resp_ids: list[int]):
    """Split a RESPONSE token sequence into (pre_meta_text, meta_content_text, meta_block, post_text)
    via b2.extract_first_meta_block on the DECODED response (open-only safe)."""
    text = tok.decode(resp_ids, skip_special_tokens=False)
    block = extract_first_meta_block(text)
    if block is None:
        return text, None, None, ""
    o = text.find("<|meta|>")
    pre = text[:o] if o >= 0 else text
    inner = block
    if inner.startswith("<|meta|>"):
        inner = inner[len("<|meta|>"):]
    if inner.endswith("<|/meta|>"):
        inner = inner[:-len("<|/meta|>")]
    inner = inner.strip("\n")
    idx = text.find(block)
    if idx >= 0:
        post = text[idx + len(block):]
    else:
        anchor = text.find(inner, o if o >= 0 else 0)
        post = text[anchor + len(inner):] if (anchor >= 0 and inner) else ""
    return pre, inner, block, post


def _unpack_split(tok, prompt_ids, full_ids):
    """Split the RESPONSE region (full_ids beyond prompt) into (pre, meta_content, meta_block, post)
    via b2.extract_first_meta_block (open-only safe)."""
    resp_ids = full_ids[len(prompt_ids):] if len(full_ids) > len(prompt_ids) else full_ids
    return split_around_meta(tok, resp_ids)


def contrast_stats(diffs, rng_np):
    """(mean, sd, n, paired-perm p, realized_MDE) for a list of paired diffs. MDE = 1.96 sd sqrt(2/n)."""
    d = np.asarray([x for x in diffs if x is not None and not np.isnan(x)])
    nd = len(d)
    if nd == 0:
        return None, None, 0, None, None
    mean = float(d.mean())
    sd = float(np.std(d, ddof=1)) if nd > 1 else None
    pval = float(paired_perm_test(d.tolist(), rng_np))
    mde = (1.96 * sd * np.sqrt(2.0 / nd)) if (sd is not None and nd > 0) else None
    return mean, sd, nd, pval, (float(mde) if mde is not None else None)


def headroom_band(bacc: float) -> str:
    """Stratification band from the baseline accuracy: floor (==0), ceiling (==1), else headroom."""
    if bacc <= 0.0:
        return "floor"
    if bacc >= 1.0:
        return "ceiling"
    return "headroom"


# ── Change 3: objective uncertainty metrics over k continuations ──────────────

def parse_verbalized_conf(meta_text: str):
    """Parse a verbalized 'confidence: X' from the steered meta text (subjective; reported only).
    Returns a float in [0,1] (clamped) or None."""
    if not meta_text:
        return None
    m = re.search(r"confidence:\s*([0-9]*\.?[0-9]+)", meta_text, re.IGNORECASE)
    if not m:
        return None
    try:
        v = float(m.group(1))
    except ValueError:
        return None
    return max(0.0, min(1.0, v))


def _grade_equiv(a: str, b: str) -> bool:
    """Grader-equivalence of two final answers (wrap each in \\boxed and use robust_grade). Avoids
    string-equality inflation of disagreement (½ vs 0.5). a/b are raw boxed-content strings."""
    if a is None or b is None:
        return a == b
    try:
        return bool(robust_grade(r"\boxed{" + str(a) + "}", str(b)))
    except Exception:
        return str(a).strip() == str(b).strip()


def self_consistency(final_answers: list) -> float | None:
    """Fraction of continuations whose final answer equals the MODAL answer under GRADER equivalence
    (empirical confidence). final_answers = list of boxed-content strings (None = unparseable)."""
    parsed = [a for a in final_answers if a is not None]
    if not parsed:
        return None
    # build equivalence classes by grader equivalence; modal class size / total.
    classes = []                                  # list of (representative, count)
    for a in parsed:
        placed = False
        for i, (rep, _cnt) in enumerate(classes):
            if _grade_equiv(a, rep):
                classes[i] = (rep, classes[i][1] + 1)
                placed = True
                break
        if not placed:
            classes.append((a, 1))
    modal = max(c for _, c in classes)
    return modal / len(parsed)


def compute_arm_metrics(continuations_text: list, gold: str, steered_meta_text: str):
    """Per-arm objective metrics (Change 3) over the k continuations' decoded RESPONSE text.
    Returns the metrics dict (answer_entropy filled later, possibly null)."""
    k = len(continuations_text)
    corrects = [bool(robust_grade(c, gold)) for c in continuations_text]
    accuracy = float(np.mean([int(x) for x in corrects])) if k else None
    final_answers = [extract_last_boxed(c) for c in continuations_text]
    sc = self_consistency(final_answers)
    agree_with_gold = (float(np.mean([int(_grade_equiv(a, gold)) for a in final_answers]))
                       if final_answers else None)
    vconf = parse_verbalized_conf(steered_meta_text)
    calib = (abs(vconf - agree_with_gold) if (vconf is not None and agree_with_gold is not None)
             else None)
    pass_at_k_cont = int(any(corrects)) if k else 0
    return {
        "accuracy": accuracy,
        "self_consistency": sc,
        "agree_with_gold": agree_with_gold,
        "verbalized_conf": vconf,
        "calibration_gap": calib,
        "answer_entropy": None,
        "answer_entropy_method": "null",
    }, pass_at_k_cont, corrects, final_answers


@torch.no_grad()
def answer_span_entropy(model, tok, full_ids: list[int], resp_start: int, dev):
    """Best-effort answer-token entropy (Change 3): mean a3.raw_entropy over the final-answer span
    (first_boxed_token_idx -> end) of ONE continuation. full_ids = handoff_ids + cont_ids (the COMPLETE
    sequence); resp_start = len(prompt_ids). Run WHILE the HF model is still resident (before the free),
    never gates the run. Returns (mean_entropy, method) — (None, "null") if the span is empty/unclean."""
    try:
        resp_ids = full_ids[resp_start:]
        box_idx = first_boxed_token_idx(tok, resp_ids)         # response-relative index of first \boxed
        if box_idx >= len(resp_ids):
            return None, "null"                                # no boxed answer in this continuation
        H = raw_entropy(model, full_ids, resp_start, dev)      # per-response-token entropy
        span = H[box_idx:]
        span = span[np.isfinite(span)]
        if span.size == 0:
            return None, "null"
        return float(span.mean()), "hf_raw_entropy"
    except Exception as e:
        print(f"  [answer_entropy] skipped ({type(e).__name__}: {str(e)[:60]})")
        return None, "null"


# ── HF meta-only generation (Change 1) ─────────────────────────────────────────

@torch.no_grad()
def hf_generate(model, tok, prefix_ids, max_new, device, processors=None,
                stop_token: int | None = None):
    """One sampled HF generation from prefix_ids. Returns full_ids = prefix_ids + generated. When
    `stop_token` is given (META_CLOSE_ID for the meta-only phase), it is added to eos_token_id so
    generation HALTS at the meta close (Change 1: HF covers ONLY the meta span). The caller still
    post-slices at the first stop token and appends a close if absent."""
    ids = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    lp = LogitsProcessorList(processors) if processors else None
    eos = tok.eos_token_id
    eos_list = [eos] if eos is not None else []
    if stop_token is not None:
        eos_list = eos_list + [stop_token]
    out = model.generate(
        ids, max_new_tokens=max_new, do_sample=True, temperature=0.7, top_p=0.95,
        num_return_sequences=1, logits_processor=lp,
        eos_token_id=(eos_list or None),
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
    )
    return out[0].tolist()


def steer_meta_span(model, tok, c, mode, alpha, dev, record=None):
    """P1-HF for ONE (problem, steered mode): generate ONLY the steered meta span (max_new=META_CAP),
    stop at <|/meta|>, and return the EXACT token-id handoff + steered meta text (Change 1 + 2).

    handoff_ids = prompt_ids + base[:p_self] + [<|meta|>] + steered_meta_content + [<|/meta|>]
    built by PURE INT CONCATENATION from out[0].tolist() slicing — NO decode->re-encode (correctness
    rule 1). The meta is CLOSED before handoff: if META_CLOSE_ID is absent after META_CAP, we append
    it (correctness rule 2); the content is sliced at the FIRST close so no stray 151670 sits mid-span.
    """
    prompt_ids = c["prompt_ids"]
    base = c["base_roll"]
    p_self = c["p_self"]
    q = c["r"]["question"]
    gold, decoy = c["gold"], c["decoy"]
    # meta_prefix = prompt + base[:p_self+1]  (ends AT and INCLUDING the self-emitted <|meta|>)
    meta_prefix_ids = prompt_ids + list(base[:p_self + 1])
    # CORRECTNESS rule 1: assert the boundary so a single off-by-one on <|meta|> cannot slip through.
    assert meta_prefix_ids == prompt_ids + list(base[:p_self + 1])
    assert base[p_self] == META_OPEN_ID, "p_self does not point at <|meta|>"
    pre_meta_body = tok.decode(base[:p_self], skip_special_tokens=False)

    if mode == "conf_adaptive":
        # E.7 adaptive self-referential steering: teacher = the student's OWN verbalized confidence
        # INVERTED (1-c) vs its actual c, so steer = alpha*(logit|conf:1-c - logit|conf:c) moves the
        # meta FROM where the student is TOWARD 1-c. Both contexts differ only in the confidence value
        # (same prefix) -> the confidence axis is isolated, gold-free. c is parsed from the harvested
        # baseline meta (c["_self_meta_text"], set just above in the PHASE 1-HF loop). If the student
        # stated no confidence, c defaults to 0.5 -> (0.5,0.5) -> zero steer (honest no-op).
        sc = parse_verbalized_conf(c.get("_self_meta_text", "") or "")
        sc = 0.5 if sc is None else min(max(float(sc), 0.0), 1.0)
        sfx_A, sfx_B = (f" confidence: {1.0 - sc:.2f}", f" confidence: {sc:.2f}")
    else:
        sfx_A, sfx_B = CONTRASTS[mode](gold, decoy)
    A_ids = build_reveal_ids(tok, q, sfx_A, pre_meta_body)
    B_ids = build_reveal_ids(tok, q, sfx_B, pre_meta_body)
    proc = ContrastiveMetaSteerProcessor(
        model, A_ids, B_ids, alpha, prompt_len=len(meta_prefix_ids),
        device=dev, record=record, start_in_meta=True)
    full_ids = hf_generate(model, tok, meta_prefix_ids, META_CAP, dev,
                           processors=[proc, StopAtMetaClose()], stop_token=META_CLOSE_ID)
    # generated meta tail = everything beyond meta_prefix (these are int ids from out[0].tolist()).
    gen_tail = full_ids[len(meta_prefix_ids):]
    # slice at the FIRST close (correctness rule 2); strip any trailing eos.
    if META_CLOSE_ID in gen_tail:
        content_ids = gen_tail[:gen_tail.index(META_CLOSE_ID)]
    else:
        content_ids = list(gen_tail)
        if content_ids and tok.eos_token_id is not None and content_ids[-1] == tok.eos_token_id:
            content_ids = content_ids[:-1]
    # well-formed CLOSED meta: prompt + base[:p_self] + <|meta|> + content + <|/meta|>
    handoff_ids = (list(prompt_ids) + list(base[:p_self]) + [META_OPEN_ID]
                   + list(content_ids) + [META_CLOSE_ID])
    assert handoff_ids[-1] == META_CLOSE_ID, "handoff meta not closed"
    assert META_CLOSE_ID not in content_ids, "stray close mid-span"
    steered_meta_text = tok.decode(content_ids, skip_special_tokens=False).strip("\n")
    return handoff_ids, steered_meta_text


# ── main (one model per invocation) ────────────────────────────────────────────

def run_model(model_key: str, args, rng, rng_np, t0):
    model_path = MODELS[model_key]
    eval_path = EVALS[model_key]
    n = args.smoke or args.n
    k = args.k if not args.smoke else 2
    max_new = args.max_new if not args.smoke else min(args.max_new, 256)
    meta_cap = META_CAP if not args.smoke else 64
    alpha = args.alpha
    arms = args.arms_list                          # set in main() per --arms / phase default
    shard_sfx = f"_shard{args.shard}" if getattr(args, "n_shards", 1) > 1 else ""
    out_jsonl = Path(args.out_dir) / f"e2_steering_{model_key}_{args.tag}{shard_sfx}.jsonl"
    out_json = Path(args.out_dir) / f"e2_steering_{model_key}_{args.tag}{shard_sfx}.json"
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if out_jsonl.exists():
        out_jsonl.unlink()                         # fresh append-log per run
    record_smoke = [] if args.smoke else None      # per-step trace for the first smoke (problem,mode)
    steered_arms = [a for a in arms if a in STEERED_MODES]

    ev = hf_hub_download(repo_id=HF_DATASET, repo_type="dataset", filename=eval_path)
    results = json.load(open(ev))
    results = results if isinstance(results, list) else results.get("results") or list(results.values())[0]
    all_golds = [r.get("gold_answer") for r in results if r.get("gold_answer") is not None]
    if args.n_pool is not None:
        n_pool = args.n_pool
    elif args.smoke:
        n_pool = max(40, 20 * n)
    else:
        n_pool = max(8 * n, n + 200)
    # Phase-derived problem-pool seed: Phase-1 = seed A, Phase-2 = seed B (a DIFFERENT pool ⇒
    # discovery/confirmation separation, NOT in-run alpha selection). --phase {1,2} picks the seed.
    pool_seed = args.pool_seed
    pool = representative_pool(results, n_pool, rng)
    if getattr(args, "n_shards", 1) > 1:
        # data-parallel sharding across GPUs: each rank takes a disjoint stride of the
        # deterministically-built pool (same pool_seed ⇒ identical pool in every process),
        # so the n_shards ranks union to the full pool with zero overlap. The launcher pins
        # each rank to one GPU via CUDA_VISIBLE_DEVICES, so vllm_gen/HF (single-device) are
        # untouched; merge the per-shard JSONLs afterward.
        pool = pool[args.shard :: args.n_shards]
    print(f"[e3:{model_key}] model={model_path} phase={args.phase} n_target={n} pool={len(pool)} "
          f"shard={getattr(args,'shard',0)}/{getattr(args,'n_shards',1)} "
          f"k={k} alpha={alpha} arms={arms} max_new={max_new} meta_cap={meta_cap} "
          f"pool_seed={pool_seed} max_model_len={args.max_model_len}")

    tok = load_tokenizer(model_path)
    dev = "cuda"

    def make_rule_decoy(gold: str) -> str:
        d = _rule_based_decoy(gold, seed=STRATIFIED_SAMPLE_SEED,
                              checker=lambda c, g: bool(robust_grade(c, g)))
        if d == gold or _numerically_equal(d, gold) or bool(robust_grade(d, gold)):
            d = make_decoy(gold, all_golds, rng)
        assert d != gold and not _numerically_equal(d, gold), f"degenerate decoy {d!r} for {gold!r}"
        return d

    # ── PHASE 0 (vLLM): k baseline rollouts = pass@k + SELF-META HARVEST + the `self` arm ──
    vgen = VllmGen(model_path, tokenizer_path=safe_tokenizer_path(model_path),
                   gpu_memory_utilization=0.45, max_model_len=args.max_model_len,
                   seed=pool_seed)
    pool_prompt_ids = [build_prompt_ids(tok, r["question"]) for r in pool]
    pool_rollouts = vgen.generate(pool_prompt_ids, n=k, max_tokens=max_new, seed=pool_seed)
    grade_n, grade_d, box_n = 0, 0, 0
    cand, drop_ceil, drop_floor, drop_noemit = [], 0, 0, 0
    pool_seen = 0
    for r, pids, rolls in zip(pool, pool_prompt_ids, pool_rollouts):
        if len(cand) >= n:
            break
        pool_seen += 1
        gold = str(r["gold_answer"]).strip()
        # decode each baseline rollout into RESPONSE text (for grading + the self-arm continuations).
        roll_texts = [tok.decode(roll, skip_special_tokens=False) for roll in rolls]
        correct = [bool(robust_grade(rt, gold)) for rt in roll_texts]
        for rt in roll_texts:
            grade_d += 1; grade_n += int(is_gradeable(rt)); box_n += int(r"\boxed" in rt)
        emit_roll, p_self = None, None
        for roll, rt in zip(rolls, roll_texts):
            if emit_roll is None and extract_first_meta_block(rt) is not None:
                pos = first_meta_open_pos(roll)
                if pos is not None:
                    emit_roll, p_self = roll, pos
        bacc = float(np.mean([int(c) for c in correct])) if correct else 0.0
        if emit_roll is None:
            drop_noemit += 1
            continue
        if bacc <= 0.0:
            drop_floor += 1
        elif bacc >= 1.0:
            drop_ceil += 1
        cand.append({
            "r": r, "prompt_ids": pids, "gold": gold, "decoy": make_rule_decoy(gold),
            "benchmark": r.get("benchmark"), "baseline_acc": bacc,
            "pass_at_k": pass_at_k(correct), "headroom_band": headroom_band(bacc),
            "base_roll": emit_roll, "p_self": int(p_self),
            "per_k_correct": [int(b) for b in correct],
            # `self` arm continuations = the k baseline ROLLS themselves (own meta + continuation),
            # graded over the SAME k with the SAME path as steered arms (correctness rule 3).
            "self_cont_text": roll_texts,
            "headroom": (0.0 < bacc < 1.0),
        })
    self_emit_harvest_rate = (len(cand) / pool_seen) if pool_seen else None
    print(f"[e3:{model_key}][P0] pool_seen={pool_seen} kept={len(cand)} no_self_emit={drop_noemit} "
          f"ceiling={drop_ceil} floor={drop_floor} harvest_rate={self_emit_harvest_rate} "
          f"({time.time()-t0:.0f}s)")
    if not cand:
        raise RuntimeError(
            f"[e3:{model_key}] NO self-emitting problems harvested from pool_seen={pool_seen} "
            f"(self-emit is sparse ~9-32% for v8_strict; raise --n_pool). 0 for E20a is expected.")
    vgen.free(); gc.collect(); torch.cuda.empty_cache()

    # ── PHASE 1-HF (Change 1): ONE HF load, steer ONLY the meta span over ALL (problem × steered arm) ──
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    handoffs = {}                                  # (pi, mode) -> handoff_ids (exact int sequence)
    steered_meta = {}                              # (pi, mode) -> steered meta text
    failure_tags, stance_self_map = {}, {}
    for pi, c in enumerate(cand):
        prompt_ids, base, p_self = c["prompt_ids"], c["base_roll"], c["p_self"]
        # failure_tag from pre-meta entropy at the self-trigger position (a3.raw_entropy, import-only).
        H = raw_entropy(model, prompt_ids + base, len(prompt_ids), dev)
        cap = first_boxed_token_idx(tok, base)
        H_at = float(H[p_self]) if p_self < len(H) else float("nan")
        wrong = c["baseline_acc"] < 1.0
        body_med = float(np.median(H[:cap])) if cap > 1 else H_at
        if wrong and not np.isnan(H_at) and H_at < body_med:
            failure_tags[pi] = "overconfident-wrong"
        elif wrong:
            failure_tags[pi] = "wrong-direction"
        else:
            failure_tags[pi] = "none"
        # self-arm stance (from the harvested baseline meta).
        _, s_meta_content, _, _ = _unpack_split(tok, prompt_ids, prompt_ids + list(base))
        stance_self_map[pi] = classify_stance(s_meta_content)
        c["_self_meta_text"] = s_meta_content
        for mode in steered_arms:
            rec = record_smoke if (pi == 0 and mode == steered_arms[0]) else None
            hid, mtext = steer_meta_span(model, tok, c, mode, alpha, dev, record=rec)
            handoffs[(pi, mode)] = hid
            steered_meta[(pi, mode)] = mtext
        print(f"  [P1-HF {pi+1}/{len(cand)}] {c['benchmark']:8s} tag={failure_tags[pi]} "
              f"bacc={c['baseline_acc']:.2f} p@k={c['pass_at_k']} p_self={p_self} "
              f"steered_arms={len(steered_arms)} ({time.time()-t0:.0f}s)")

    # ── best-effort answer_entropy (Change 3): one HF forward per (problem × steered arm) on the
    # chosen continuation. Done HERE while the HF model is still resident (rule 5: never co-resident
    # with vLLM). We need a continuation first, so we DEFER the actual forward to after P1-vLLM by
    # RE-LOADING? No — instead we hold the model, run vLLM AFTER freeing, then reload only if entropy
    # requested. To keep HF/vLLM non-co-resident AND avoid a second HF load, we compute entropy on a
    # SHORT greedy meta-free continuation generated HERE on HF (one per (pi,mode)) purely for the
    # entropy probe; the GRADED continuations come from vLLM. This keeps entropy best-effort and
    # non-blocking without a second model load.
    answer_entropy = {}                            # (pi, mode) -> (val, method)
    if args.answer_entropy:
        for pi, c in enumerate(cand):
            for mode in steered_arms:
                hid = handoffs[(pi, mode)]
                probe_full = hf_generate(model, tok, hid, min(meta_cap, 256), dev)
                val, method = answer_span_entropy(
                    model, tok, probe_full, len(c["prompt_ids"]), dev)
                answer_entropy[(pi, mode)] = (val, method)

    del model; gc.collect(); torch.cuda.empty_cache()       # HF freed BEFORE vLLM (rule 5)

    # ── PHASE 1-vLLM (Change 1): re-create VllmGen, ONE batched generate from EVERY handoff (UNSTEERED) ──
    cont_text = {}                                  # (pi, mode) -> list[k] of decoded RESPONSE text
    if steered_arms:
        vgen = VllmGen(model_path, tokenizer_path=safe_tokenizer_path(model_path),
                       gpu_memory_utilization=0.45, max_model_len=args.max_model_len,
                       seed=pool_seed)
        keys = [(pi, mode) for pi in range(len(cand)) for mode in steered_arms]
        batch = [handoffs[key] for key in keys]
        conts = vgen.generate(batch, n=k, max_tokens=max_new, temperature=0.7, seed=pool_seed)
        for key, cont_list in zip(keys, conts):
            pi, _mode = key
            plen = len(cand[pi]["prompt_ids"])
            hid = handoffs[key]
            # full continuation = handoff_ids + cont_ids; GRADE the RESPONSE region (rule: decode the
            # response, not the prompt). handoff_ids[plen:] is the resumed body+steered meta.
            texts = []
            for cids in cont_list:
                full_resp_ids = list(hid[plen:]) + list(cids)
                texts.append(tok.decode(full_resp_ids, skip_special_tokens=False))
            cont_text[key] = texts
            for tx in texts:
                grade_d += 1; grade_n += int(is_gradeable(tx)); box_n += int(r"\boxed" in tx)
        vgen.free(); gc.collect(); torch.cuda.empty_cache()

    # ── PHASE 2: grade + metrics per (problem, arm) + JSONL + stats + verdict ──────────────
    rows = []
    metrics_grid = {}                              # (pi, arm) -> metrics dict (for stats)
    for pi, c in enumerate(cand):
        r, gold, decoy = c["r"], c["gold"], c["decoy"]
        pid = r.get("id", r.get("problem_id", f"{model_key}_{pi}"))
        base_fields = {
            "model": model_key, "problem_id": pid, "benchmark": c["benchmark"],
            "gold": gold, "decoy": decoy, "baseline_acc": c["baseline_acc"],
            "pass_at_k": c["pass_at_k"], "headroom_band": c["headroom_band"],
            "failure_tag": failure_tags[pi], "p_self": int(c["p_self"]), "alpha": alpha,
        }
        # ── self arm: baseline's OWN meta + continuation (the harvested k rolls), same k + grader. ──
        if "self" in arms:
            self_meta_text = c.get("_self_meta_text")
            m_self, pak_self, _corr, _fa = compute_arm_metrics(
                c["self_cont_text"], gold, self_meta_text)
            metrics_grid[(pi, "self")] = m_self
            base_text = c["self_cont_text"][0] if c["self_cont_text"] else ""
            pre0, _mc0, _blk0, post0 = split_around_meta(
                tok, c["base_roll"])  # for reasoning excerpt
            rows.append(_json_safe({**base_fields, "arm": "self", "contrast_mode": None,
                                    "meta_text": self_meta_text, "metrics": m_self,
                                    "pass_at_k_cont": pak_self, "meta_contains_answer": False,
                                    "reasoning": {"pre_meta": pre0[-3000:], "post_meta": post0[:3000]}}))
        # ── steered arms ──
        for mode in steered_arms:
            key = (pi, mode)
            mtext = steered_meta[key]
            texts = cont_text.get(key, [])
            m, pak, _corr, _fa = compute_arm_metrics(texts, gold, mtext)
            if key in answer_entropy:
                val, method = answer_entropy[key]
                m["answer_entropy"], m["answer_entropy_method"] = val, method
            metrics_grid[key] = m
            mca = (meta_contains_answer(mtext, gold, decoy, tok)
                   if mode in GOLD_BEARING_MODES else False)
            stance_steered = classify_stance(mtext)
            ex = texts[0] if texts else ""
            rows.append(_json_safe({**base_fields, "arm": mode, "contrast_mode": mode,
                                    "meta_text": mtext, "metrics": m, "pass_at_k_cont": pak,
                                    "meta_contains_answer": bool(mca),
                                    "stance": {"self": stance_self_map[pi], "steered": stance_steered,
                                               "shift": f"{stance_self_map[pi]}->{stance_steered}"},
                                    "reasoning": {"pre_meta": tok.decode(
                                        c["base_roll"][:c["p_self"]],
                                        skip_special_tokens=False)[-3000:],
                                        "post_meta": ex[:3000]}}))

    with open(out_jsonl, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    # ── stats: paired Δacc(arm − self) over pass@k>0 problems, pooled + per-benchmark ──
    primary = [pi for pi, c in enumerate(cand) if c["pass_at_k"] > 0]
    wall = [pi for pi, c in enumerate(cand) if c["pass_at_k"] == 0]

    def acc_of(pi, arm):
        m = metrics_grid.get((pi, arm))
        return None if m is None else m["accuracy"]

    def delta_list(pi_list, arm, answer_free_only=False):
        ds = []
        for pi in pi_list:
            a_arm, a_self = acc_of(pi, arm), acc_of(pi, "self")
            if a_arm is None or a_self is None:
                continue
            if answer_free_only and arm in GOLD_BEARING_MODES:
                row = next((x for x in rows if x["arm"] == arm
                            and x["problem_id"] == cand[pi]["r"].get(
                                "id", cand[pi]["r"].get("problem_id", f"{model_key}_{pi}"))), None)
                if row is not None and row.get("meta_contains_answer"):
                    continue
            ds.append(a_arm - a_self)
        return ds

    benches = sorted({c["benchmark"] for c in cand if c["benchmark"] is not None})
    per_arm = {}
    for arm in steered_arms:
        m, sd, nd, p, mde = contrast_stats(delta_list(primary, arm), rng_np)
        pb = {}
        for b in benches:
            idx = [pi for pi in primary if cand[pi]["benchmark"] == b]
            bm, bsd, bn, bp, bmde = contrast_stats(delta_list(idx, arm), rng_np)
            pb[b] = {"delta": bm, "sd": bsd, "n": bn, "p": bp, "realized_mde": bmde}
        lg = None
        if arm in GOLD_BEARING_MODES:
            lm, lsd, ln, lp_, lmde = contrast_stats(
                delta_list(primary, arm, answer_free_only=True), rng_np)
            lg = {"delta": lm, "sd": lsd, "n": ln, "p": lp_, "realized_mde": lmde,
                  "survives": bool(lm is not None and lm >= DELTA_THR and lp_ is not None
                                   and lp_ < 0.05 and lmde is not None and lmde <= POWER_MDE_THR)}
        per_arm[arm] = {"delta": m, "sd": sd, "n": nd, "p": p, "realized_mde": mde,
                        "per_benchmark": pb, "leakage_guard": lg}

    # decomposition contrasts (descriptive; plan §"What the decomposition answers"). Each = paired
    # diff of (acc[arm_X] − acc[arm_Y]) over primary problems where both arms exist.
    def pair_contrast(arm_x, arm_y):
        if arm_x not in steered_arms or arm_y not in steered_arms:
            return None
        ds = []
        for pi in primary:
            ax, ay = acc_of(pi, arm_x), acc_of(pi, arm_y)
            if ax is None or ay is None:
                continue
            ds.append(ax - ay)
        m, sd, nd, p, mde = contrast_stats(ds, rng_np)
        return {"delta": m, "sd": sd, "n": nd, "p": p, "realized_mde": mde}

    decomposition = {
        "answer_vs_stance__gold_decoy_minus_gold_stance": pair_contrast("gold_decoy", "gold_stance"),
        "grounding_stance__gold_stance_minus_cautious": pair_contrast("gold_stance", "cautious"),
        "grounding_conf__gold_conf_down_minus_conf_down": pair_contrast("gold_conf_down", "conf_down"),
        "stance_vs_confnum__cautious_minus_conf_down": pair_contrast("cautious", "conf_down"),
    }

    # power gate.
    gradeable_rate = (grade_n / grade_d) if grade_d else None
    boxed_str_rate = (box_n / grade_d) if grade_d else None
    power_ok = (gradeable_rate is not None and gradeable_rate >= 0.5)

    # ranking + verdict. Phase 1 = DISCOVERY (rank, NO pass claim). Phase 2 = CONFIRMATION.
    ranked = sorted(
        [(arm, per_arm[arm]["delta"]) for arm in steered_arms if per_arm[arm]["delta"] is not None],
        key=lambda kv: kv[1], reverse=True)
    best_arm = ranked[0][0] if ranked else None

    if args.phase == 1:
        status = "DISCOVERY"
        verdict = (f"DISCOVERY (Phase 1, fresh seed={pool_seed}, n_primary={len(primary)}) — ranked "
                   f"directions by paired Δacc: {ranked}. Best = {best_arm}. NO pass claim; feed the "
                   f"top 1-2 directions to Phase 2 (--phase 2 --arms self,<winner> on a FRESH seed).")
    else:
        # Phase 2 CONFIRMATION: pre-registered PASS on the winner(s) tested here.
        passing = []
        for arm in steered_arms:
            pa = per_arm[arm]
            pooled = (pa["delta"] is not None and pa["delta"] >= DELTA_THR and pa["p"] is not None
                      and pa["p"] < 0.05 and pa["realized_mde"] is not None
                      and pa["realized_mde"] <= POWER_MDE_THR)
            leak_ok = (arm not in GOLD_BEARING_MODES) or (pa["leakage_guard"] is not None
                                                          and pa["leakage_guard"]["survives"])
            if pooled and leak_ok:
                passing.append(arm)
        mde_ok = all(per_arm[a]["realized_mde"] is not None and per_arm[a]["realized_mde"] <= POWER_MDE_THR
                     for a in steered_arms) if steered_arms else False
        if (not power_ok) or (not mde_ok):
            status = "INCONCLUSIVE"
            verdict = (f"INCONCLUSIVE — power gate (gradeable_rate={gradeable_rate}, "
                       f"realized_MDE>thr for some arm); scale k/N, NEVER a substantive null")
        elif passing:
            status = "PASS"
            verdict = (f"PASS — direction(s) {passing} beat self (Δacc>=+0.05, p<0.05, MDE<=thr) on "
                       f"pooled pass@k>0 AND survive the answer-free leakage guard -> validated "
                       f"steering/RL teacher direction(s)")
        else:
            status = "FAIL"
            verdict = ("FAIL — no direction powered-beats self by +0.05 surviving the leakage guard "
                       "-> reconsider (activation-steering axis, or RL-only)")

    summary = {
        "status": status, "phase": args.phase, "model": model_key, "model_path": model_path,
        "pool_seed": pool_seed, "arms": list(arms), "alpha": alpha,
        "n_kept": len(cand), "n_primary_passk_gt0": len(primary), "n_capability_wall_passk0": len(wall),
        "k": k, "max_new": max_new, "meta_cap": meta_cap, "max_model_len": args.max_model_len,
        "self_emit_harvest_rate": self_emit_harvest_rate, "pool_seen": pool_seen,
        "n_no_self_emit": drop_noemit,
        "per_arm": per_arm, "decomposition": decomposition, "ranked_by_delta": ranked,
        "best_arm": best_arm,
        "gradeable_rate": gradeable_rate, "boxed_str_rate": boxed_str_rate,
        "power_ok": bool(power_ok), "n_graded": grade_d,
        "verdict": verdict, "wall_seconds": time.time() - t0,
    }

    if args.smoke and record_smoke is not None and record_smoke:
        # STAGING assertions for the RESUME-AT-SELF-META steered gen + the handoff well-formedness.
        in_meta_from_start = bool(record_smoke[0]["in_meta"])
        nonzero_inside = any(rec["in_meta"] and rec["delta_norm"] > 0 for rec in record_smoke)
        applied_outside = any((not rec["in_meta"]) and rec["applied"] for rec in record_smoke)
        nonzero_would_outside = any((not rec["in_meta"]) and rec["would_steer_norm"] > 0
                                    for rec in record_smoke)
        closed = any((not rec["in_meta"]) for rec in record_smoke)
        # handoff well-formedness: every handoff ends in 151670, and token-id round-trip identity.
        all_closed = all(handoffs[k_][-1] == META_CLOSE_ID for k_ in handoffs)
        # token-id identity: handoff prefix == prompt + base[:p_self] + <|meta|> (exact, no re-encode).
        identity_ok = True
        for pi, c in enumerate(cand):
            for mode in steered_arms:
                hid = handoffs[(pi, mode)]
                want = list(c["prompt_ids"]) + list(c["base_roll"][:c["p_self"]]) + [META_OPEN_ID]
                if hid[:len(want)] != want:
                    identity_ok = False
        summary["smoke_assert"] = {
            "in_meta_from_resumed_open": bool(in_meta_from_start),
            "delta_nonzero_inside_span": bool(nonzero_inside),
            "steering_applied_only_inside_span": bool(not applied_outside),
            "reveal_streams_frozen_outside_span": bool(not nonzero_would_outside),
            "span_closed_in_trace": bool(closed),
            "all_handoffs_closed": bool(all_closed),
            "handoff_token_id_identity": bool(identity_ok),
            "n_steps_traced": len(record_smoke),
        }
        assert in_meta_from_start, "SMOKE FAIL: steered gen did not start in_meta at the resumed <|meta|>"
        assert nonzero_inside, "SMOKE FAIL: steering delta was never nonzero inside the meta span"
        assert not applied_outside, "SMOKE FAIL: steering delta APPLIED OUTSIDE the meta span"
        assert not nonzero_would_outside, (
            "SMOKE FAIL: a nonzero would-steer candidate appeared OUTSIDE the span — reveal streams "
            "were NOT frozen post-close")
        assert all_closed, "SMOKE FAIL: a handoff did not end in <|/meta|> (151670)"
        assert identity_ok, "SMOKE FAIL: handoff token-id identity broke (decode->re-encode drift?)"
        print(f"[smoke] in_meta_from_start={in_meta_from_start} nonzero_inside={nonzero_inside} "
              f"applied_only_inside={not applied_outside} frozen_outside={not nonzero_would_outside} "
              f"span_closed={closed} all_handoffs_closed={all_closed} identity={identity_ok} "
              f"steps={len(record_smoke)}")

    payload = _json_safe({"summary": summary})
    json.dump(payload, open(out_json, "w"), indent=2)
    print(json.dumps(payload["summary"], indent=2)[:4000])
    print(f"[done:{model_key}] {out_jsonl}  +  {out_json}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), default=None,
                    help="single model key; default = v8_strict (the PRIMARY self-emitting model; "
                         "e20a self-emits 0 meta so its harvest is empty, opt-in only)")
    ap.add_argument("--models", default=None, help="comma-sep model keys (overrides --model)")
    ap.add_argument("--phase", type=int, choices=(1, 2), default=1,
                    help="1=DISCOVERY (n~40-50, all 6 arms, seed A); 2=CONFIRMATION (n~100-120, arm "
                         "subset via --arms, FRESH seed B). Separation is via the fresh problem pool.")
    ap.add_argument("--arms", default=None,
                    help="comma-sep arms; default Phase1=all 6, Phase2=self only (set the winner(s)). "
                         f"choices: {','.join(ARMS)}")
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                    help="single steering alpha (default 0.6 — NOT 1.0; E.2 saw 1.0 destabilize)")
    ap.add_argument("--n", type=int, default=None, help="target primary N (default: phase 1=45, 2=110)")
    ap.add_argument("--n_pool", type=int, default=None, help="override pool size")
    ap.add_argument("--k", type=int, default=8, help="continuations per (problem,arm) (self-consistency)")
    ap.add_argument("--max_new", type=int, default=16384, help="continuation budget (vLLM)")
    ap.add_argument("--max_model_len", type=int, default=20480, help="vLLM context window")
    ap.add_argument("--shard", type=int, default=0, help="data-parallel shard rank (0..n_shards-1)")
    ap.add_argument("--n_shards", type=int, default=1,
                    help="number of GPU shards (1 = no sharding; launcher pins each rank to one GPU)")
    ap.add_argument("--answer_entropy", action="store_true",
                    help="best-effort HF answer-span entropy (Change 3; non-blocking, off by default "
                         "for speed)")
    ap.add_argument("--smoke", type=int, default=0,
                    help="N kept self-emitting problems, all arms, k=2, small caps + assertions")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out_dir", default=str(REPORTS_DIR))
    args = ap.parse_args()

    # phase defaults.
    if args.n is None:
        args.n = (args.smoke or (45 if args.phase == 1 else 110))
    # fresh-seed-per-phase (discovery/confirmation separation): seed A vs seed B.
    args.pool_seed = STRATIFIED_SAMPLE_SEED + (0 if args.phase == 1 else 7919)
    if args.tag is None:
        args.tag = f"p{args.phase}"
    # arms resolution.
    if args.smoke and args.arms:
        # honor --arms in smoke so the node pre-flight exercises exactly the arms the
        # real run uses (e.g. self,cautious,conf_down) — keeps the smoke faithful and
        # routes the nonzero_inside steer-fired assert onto a GOLD-FREE arm.
        sel = [a for a in args.arms.split(",") if a in ARMS]
        args.arms_list = (["self"] + [a for a in sel if a != "self"]) if sel else list(ARMS)
    elif args.smoke:
        args.arms_list = list(ARMS)                 # smoke = all arms
    elif args.arms:
        sel = [a for a in args.arms.split(",") if a in ARMS]
        args.arms_list = (["self"] + [a for a in sel if a != "self"]) if sel else list(ARMS)
    elif args.phase == 1:
        args.arms_list = list(ARMS)                 # Phase 1 = all 6 arms
    else:
        args.arms_list = list(ARMS)                 # Phase 2 default = all (caller should pass winner)

    if args.smoke:
        model_keys = ["v8_strict"]
    elif args.models:
        model_keys = [m for m in args.models.split(",") if m in MODELS]
    elif args.model:
        model_keys = [args.model]
    else:
        model_keys = ["v8_strict"]

    rng = _random.Random(args.pool_seed)
    rng_np = np.random.default_rng(args.pool_seed)
    t0 = time.time()
    summaries = {}
    for mk in model_keys:
        summaries[mk] = run_model(mk, args, rng, rng_np, t0)
    print("\n=== E.3 per-model status ===")
    for mk, s in summaries.items():
        print(f"  {mk:10s} {s['status']:12s} phase={s['phase']} best={s['best_arm']} "
              f"ranked={s['ranked_by_delta']} self_emit={s['self_emit_harvest_rate']}")


if __name__ == "__main__":
    main()
