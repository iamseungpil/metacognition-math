"""Reward functions for Meta-CoT GRPO.

Four independent reward functions, each usable with vanilla GRPOTrainer.
GDPO (per-reward normalization) prevents strong rewards from drowning weak ones.

R1: correctness_reward   — sympy-based math verification (Open-R1 style)
R2: meta_quality_reward  — meta block presence, length, Q&A structure
R3: calibration_reward   — Rewarding Doubt log scoring rule, summation across blocks
R4: uncertainty_meta_reward — (1-conf) weighted meta quality, summation

Reference: Open-R1 (https://github.com/huggingface/open-r1)
"""
import math
from pathlib import Path
import re
from src.training.meta_quality import score_meta_commit_quality

# ── thread-safe signal guard for math_verify ────────────────────────────────
# math_verify wraps every comparison in a SIGALRM-based timeout
# (math_verify/utils.py: signal.signal(signal.SIGALRM, ...); signal.alarm(t)).
# SIGALRM only works in the main thread, so inside Ray RewardLoopWorker threads
# EVERY comparison raises ValueError("signal only works in main thread of the
# main interpreter"): the real symbolic comparison never runs (it falls back to
# string-match, mis-grading 1/2==0.5, fractions, radicals), and the per-call
# exception + traceback flood also massively slows the run. Passing
# timeout_seconds=None does NOT help — the installed math_verify calls
# signal.signal unconditionally. Fix: make signal.signal / signal.alarm no-ops
# when called off the main thread (where they cannot work anyway), so the
# comparison runs without a timeout in worker threads. Main-thread behaviour
# (a real timeout) is fully preserved. Idempotent.
import signal as _signal
import threading as _threading

if not getattr(_signal, "_metacot_threadsafe_patch", False):
    _orig_signal_signal = _signal.signal
    _orig_signal_alarm = _signal.alarm

    def _threadsafe_signal(signalnum, handler):
        if _threading.current_thread() is not _threading.main_thread():
            return None  # SIGALRM unavailable off the main thread; no-op
        return _orig_signal_signal(signalnum, handler)

    def _threadsafe_alarm(seconds):
        if _threading.current_thread() is not _threading.main_thread():
            return 0
        # Newer math_verify calls signal.alarm(None) to DISABLE its timeout, but
        # the real signal.alarm requires an int -> TypeError that aborts EVERY
        # comparison (even "2"=="2") in main-thread grading (e.g. pg0 pilot, eval
        # scripts). Coerce None/float -> int; None -> 0 cancels any pending alarm.
        return _orig_signal_alarm(int(seconds or 0))

    _signal.signal = _threadsafe_signal
    _signal.alarm = _threadsafe_alarm
    _signal._metacot_threadsafe_patch = True

# Math verification via sympy (same as Open-R1)
try:
    from math_verify import parse, verify
    from latex2sympy2_extended import NormalizationConfig
    HAS_MATH_VERIFY = True
except ImportError:
    HAS_MATH_VERIFY = False


def _check_correctness(pred_text, gold):
    """Verify math answer using sympy (Open-R1 style).

    IMPORTANT: math_verify silently fails in Ray RewardLoopWorker threads —
    its grader uses signal.SIGALRM which only works in the main thread, raising
    ValueError("signal only works in main thread") that gets swallowed inside
    the math_verify wrapper, and verify() returns False even for correct
    answers. Therefore: always also run the string-match fallback when
    math_verify says False, so correct answers don't get misclassified in
    async-rollout worker threads (verl_sdc RewardLoopManager). In the main
    thread verify() works, so the fallback is a no-op rescue (never flips a
    genuinely-wrong answer to correct — it only matches exact/numeric equals).

    Handles: fractions, decimals, LaTeX expressions, boxed answers, etc.
    """
    if HAS_MATH_VERIFY:
        try:
            # parsing_timeout=None / timeout_seconds=None disable math_verify's
            # signal.SIGALRM timeout, which only works in the main thread and
            # raises ValueError("signal only works in main thread") in Ray
            # RewardLoopWorker threads. The installed math_verify RE-RAISES that
            # ValueError (its own docstring says any timeout > 0 raises in a
            # threaded environment and recommends parsing_timeout=None), so a
            # positive timeout silently degrades correct symbolic answers
            # (1/2==0.5, radicals, fractions) to string-match → mis-grades them.
            # With the signal disabled the real symbolic comparison runs
            # correctly in worker threads. Main-thread behaviour is preserved.
            gold_parsed = parse(str(gold), extraction_mode="first_match", parsing_timeout=None)
            pred_parsed = parse(str(pred_text), extraction_mode="first_match", parsing_timeout=None)
            if bool(verify(gold_parsed, pred_parsed, timeout_seconds=None)):
                return True
            # fall through to string-match (math_verify may have silent-failed)
        except Exception:
            pass  # fallback to string matching

    # Always run string-match fallback (covers worker-thread silent fail)
    p = _extract_answer_fallback(pred_text)
    g = _extract_answer_fallback(str(gold))
    if not p or not g:
        return False
    if p == g:
        return True
    try:
        return abs(float(p) - float(g)) < 1e-6
    except (ValueError, TypeError):
        return p.lower().strip() == g.lower().strip()


def _extract_answer_fallback(text):
    """Fallback answer extraction (string-based)."""
    # Normalize currency: \$70,000 or $70,000 → 70000
    # Only remove $ when followed by digits (currency), not LaTeX $x^2$
    text_norm = re.sub(r'\\?\$\s*(?=\d)', '', text)  # remove \$ or $ before digits only
    text_norm = re.sub(r'(?<=\d),(?=\d{3})', '', text_norm)  # remove commas in numbers: 70,000 → 70000

    pattern = r'\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}'
    matches = re.findall(pattern, text_norm)
    if matches:
        return matches[-1].strip()
    m = re.search(r'####\s*(.+?)(?:\n|$)', text_norm)
    if m:
        return m.group(1).strip()
    nums = re.findall(r'(-?\d+(?:\.\d+)?)', text_norm)
    return nums[-1] if nums else ""


def _last_confidence(text):
    """Return the last stated confidence in the completion, if any."""
    blocks = _parse_meta_blocks(text)
    confs = [b["confidence"] for b in blocks if b["confidence"] is not None]
    return confs[-1] if confs else None


def _meta_block_texts(text):
    """Return raw meta-block texts only."""
    return [block["text"] for block in _parse_meta_blocks(text)]


def _meta_joined_text(text):
    """Join meta blocks into one string for meta-only pattern checks."""
    return "\n".join(_meta_block_texts(text))


def _text_after_last_meta(text):
    """Return the non-meta tail, where verification and redirected solving should appear."""
    end_token = "<|/meta|>"
    idx = text.rfind(end_token)
    if idx == -1:
        return text
    return text[idx + len(end_token):]


def _has_verification_signal(text):
    """Detect explicit verification/checking language."""
    return bool(re.search(
        r'\b(verify|verified|verification|double-check|check again|re-check|'
        r'check my answer|confirm|confirmation|cross-?check|validate|validation|'
        r'consistency check|sanity check)\b',
        text,
        re.IGNORECASE,
    ))


def _has_uncertainty_signal(text):
    """Detect uncertainty/stuck language before a correction attempt."""
    return bool(re.search(
        r'\b(wait|hmm|not sure|uncertain|stuck|hold on|let me think)\b',
        text,
        re.IGNORECASE,
    ))


def _has_redirection_signal(text):
    """Detect a change of strategy rather than decorative meta text."""
    return bool(re.search(
        r'\b(instead|another way|different way|alternative|let me try|reframe|case split|fix|mistake|forgot)\b',
        text,
        re.IGNORECASE,
    ))


def _has_effective_verification_signal(text):
    """Detect verification that re-checks the answer, not just decorative wording."""
    keyword = bool(re.search(
        r'\b(substitut\w*(?:\s+back)?|plug(?:ging)?(?:\s+\w+){0,3}\s+(?:back|in|into)|back-?substitut|'
        r'recomput|recalculat|independent check|check by|check that|check:|sanity check|'
        r'verify by|verify that|verify:|test the result|cross-?check|confirm by|working backward|'
        r'work backwards|small case|boundary case|compare both sides|numerically)\b',
        text,
        re.IGNORECASE,
    ))
    if keyword:
        return True
    # Numerical verification: chained equalities like "(-1)^2 + 6(-1) + 5 = 1 - 6 + 5 = 0"
    has_chained = bool(re.search(
        r'=\s*-?[\d.]+\s*[-+*/]\s*-?[\d.]+\s*[-+*/=]', text,
    )) or bool(re.search(r'=\s*-?[\d.]+\s*=\s*-?[\d.]', text))
    # f(number) pattern
    has_eval = bool(re.search(r'(?:\w+)?\s*\(\s*-?\d+(?:\.\d+)?\s*\)\s*=', text))
    return has_chained or has_eval


def _has_anomaly_notice_signal(text):
    """Detect an explicit notice that the current route feels unreliable."""
    return bool(re.search(
        r'\b(something feels off|this feels off|that seems off|this seems off|'
        r'I should not trust this|I do not trust this|I don\'t trust this|'
        r'anomaly|inconsistent|not consistent|contradiction|'
        r'doesn\'t satisfy|does not satisfy|fails|mismatch)\b',
        text,
        re.IGNORECASE,
    ))


def _has_conflict_trigger(text):
    """Detect explicit evidence that the current path may be wrong."""
    return _has_anomaly_notice_signal(text) or bool(re.search(
        r'\b(too large|too small|cannot be|can\'t be|impossible|unit mismatch|unsupported|'
        r'I may be forcing|I am forcing|I committed too early|I overcommitted)\b',
        text,
        re.IGNORECASE,
    ))


def _has_strategy_switch_signal(text):
    """Detect a real switch in solving method."""
    return bool(re.search(
        r'\b(switch(?:ing)? to|different method|alternative approach|instead use|instead I\'ll|instead I will|'
        r'reframe|case split|solve via|let me use|another method|better to use|'
        r'use a parity|use an invariant|use a direct check|step back and)\b',
        text,
        re.IGNORECASE,
    ))


def _has_overconfidence_signal(text):
    """Detect explicit notice that confidence is running ahead of support."""
    return bool(re.search(
        r'\b(overconfiden|overcommit|committing too quickly|too quickly|too certain|too sure|'
        r'confidence is outrunning the support|support is thinner than the confidence|'
        r'about to commit without an independent check|answer came too quickly|'
        r'might be committing too quickly|risk of overcommitting|single route|'
        r'single familiar route|recognition alone|over-trusting|committing without checking)\b',
        text,
        re.IGNORECASE,
    ))


def _has_failure_diagnosis(text):
    """Detect explicit explanation of why the current route is failing."""
    return bool(
        re.search(
            r'(the issue is|the problem is|the current route fails because|this approach fails because|'
            r'this route is weak because|current route is weak because|route is weak because|approach is weak because|'
            r'I may be forcing|I am forcing|I committed too early|'
            r'I overcommitted|I am missing|I\'m missing|this only checks|this does not control|'
            r'this does not explain|the real task is|not needed here|unnecessary|'
            r'would be unnecessary|would be weak|too indirect|too complicated|'
            r'solve the wrong problem|not the game structure|can hide a mismatch|'
            r'does not match the structure|wrong formula|wrong assumption|'
            r'ignored a constraint|ignore(?:d)? the constraint|conflated two cases|'
            r'this is going nowhere|too brittle|unjustified)',
            text,
            re.IGNORECASE,
        )
    )


def _has_missing_skill_or_blocker(text):
    """Detect an explicit blocker statement."""
    return bool(
        re.search(
            r'(weakness tag:|blocker:|I am over-committing|I am missing|I\'m missing|'
            r'missing piece|missing ingredient|not controlling the key constraint)',
            text,
            re.IGNORECASE,
        )
    )


def _has_decomposition_plan(text):
    """Detect decomposition of failure or a missing requirement, not solve-time CoT."""
    return bool(re.search(
        r'(missing skill|missing perspective|missing structure|missing ingredient|'
        r'the bottleneck is|the blocker is|the failure is|'
        r'this is not a calculation problem|this is not an algebra problem|'
        r'need a structural view|need a constraint-based view|need an invariant|'
        r'missing the real invariant|missing the invariant|'
        r'need a different object of study|subgoal is to recover the missing constraint)',
        text,
        re.IGNORECASE,
    ))


def _has_next_strategy(text):
    """Detect an explicit next-strategy declaration."""
    return bool(
        re.search(
            r'(switch_method|switch to|different method|different approach|'
            r'alternative approach|try a different|try another|try different|case split|'
            r'reframe|instead I\'ll|instead I will|better to use|use a parity|'
            r'use an invariant|use a direct check|constraint-based analysis|'
            r'parity-based case split|switch to a parity|'
            r'study before retrying|redirect to|start over|restart|work backwards?|'
            r'factor(?:ing)? instead|use symmetry|use the discriminant|'
            r'count directly|direct enumeration|try a cleaner approach|'
            r'try an alternative|alternative algebraic manipulation|'
            r'identity-based approach|exact fraction-based method|'
            r'base-height analysis|sample-space view|enumerat(?:e|ing) the joint outcomes|'
            r'boundary[- ]tracing|trace the shaded boundary|'
            r'use a substitution|switch to multiplying|year-by-year factors|'
            r'tangent-center configuration|structural identity)',
            text,
            re.IGNORECASE,
        )
    )


def _strategy_terms(text: str) -> set[str]:
    s = text.lower()
    mapping = {
        "invariant": [r"\binvariant\b"],
        "parity": [r"\bparity\b"],
        "factor": [r"\bfactor(?:ing)?\b"],
        "identity": [r"\bidentity\b", r"\bproduct-to-sum\b"],
        "fraction": [r"\bfraction\b", r"\brepeating decimal\b"],
        "substitution": [r"\bsubstitution\b", r"\buse a substitution\b"],
        "sample_space": [r"\bsample[- ]space\b", r"\bjoint outcomes\b", r"\bequally likely pairs\b", r"\benumerat"],
        "boundary_trace": [r"\bboundary\b", r"\btrace the shaded boundary\b"],
        "base_height": [r"\bbase-height\b", r"\bparallel sides\b", r"\btrapezoid area\b"],
        "direct_check": [r"\bdirect check\b", r"\bplug(?:ging)? .* (?:into|in|back)\b"],
        "symmetry": [r"\bsymmetry\b"],
        "discriminant": [r"\bdiscriminant\b"],
        "scale_factors": [r"\bscale factors\b", r"\byear-by-year factors\b", r"\bmultiplying\b"],
        "tangent_center": [r"\btangent-center configuration\b", r"\btangent\b", r"\bradius\b", r"\bincenter\b"],
        "backward": [r"\bwork backwards?\b", r"\bworking backward\b"],
    }
    terms = set()
    for label, patterns in mapping.items():
        if any(re.search(p, s, re.IGNORECASE) for p in patterns):
            terms.add(label)
    return terms


def _has_confidence_drop(text, margin=0.08):
    """Return True if later confidence drops meaningfully from an earlier one."""
    confs = []
    explicit = re.findall(
        r'confidence[:\s]+(\d+\.\d+|\d+)',
        text,
        re.IGNORECASE,
    )
    for conf in explicit:
        value = float(conf)
        if value > 1:
            value /= 100
        confs.append(value)

    if len(confs) < 2:
        blocks = _parse_meta_blocks(text)
        confs = [b["confidence"] for b in blocks if b["confidence"] is not None]
    if len(confs) < 2:
        return False
    best_seen = confs[0]
    for conf in confs[1:]:
        if conf <= best_seen - margin:
            return True
        best_seen = max(best_seen, conf)
    return False


def _has_low_confidence(text, threshold=0.55):
    """Return True if any stated confidence is already low enough to justify redirect."""
    confs = re.findall(r'confidence[:\s]+(\d+\.\d+|\d+)', text, re.IGNORECASE)
    vals = []
    for conf in confs:
        value = float(conf)
        if value > 1:
            value /= 100
        vals.append(value)
    if not vals:
        blocks = _parse_meta_blocks(text)
        vals = [b["confidence"] for b in blocks if b["confidence"] is not None]
    return any(v <= threshold for v in vals)


def _first_and_last_confidence(text):
    """Return first and last parsed confidence values, if any."""
    blocks = _parse_meta_blocks(text)
    confs = [b["confidence"] for b in blocks if b["confidence"] is not None]
    if not confs:
        return None, None
    return confs[0], confs[-1]


def _get_text(completion):
    """Extract text from TRL completion format."""
    if isinstance(completion, list):
        return completion[0]["content"] if completion else ""
    return str(completion)


def _parse_meta_blocks(text, allow_free_text_fallback=True):
    """Parse meta blocks from text.

    Returns list of dicts: [{text, confidence, length}, ...]
    """
    return [
        {
            "text": block["text"],
            "confidence": block["confidence"],
            "length": block["length"],
        }
        for block in _parse_meta_blocks_with_spans(
            text,
            allow_free_text_fallback=allow_free_text_fallback,
        )
    ]


def _parse_meta_blocks_with_spans(text, allow_free_text_fallback=True):
    """Parse meta blocks and retain end offsets for prefix-based probe scoring."""
    blocks = []

    token_pattern = re.compile(r'<\|meta\|>(.*?)<\|/meta\|>', re.IGNORECASE | re.DOTALL)
    for match in token_pattern.finditer(text):
        block_text = match.group(1).strip()
        conf = _parse_confidence(block_text)
        blocks.append({
            "text": block_text,
            "confidence": conf,
            "length": len(block_text.split()),
            "start": match.start(),
            "end": match.end(),
        })

    # Fallback: try [META] / [/META] text markers (when special tokens are stripped)
    if not blocks:
        text_pattern = re.compile(r'\[META\](.*?)\[/META\]', re.IGNORECASE | re.DOTALL)
        for match in text_pattern.finditer(text):
            block_text = match.group(1).strip()
            conf = _parse_confidence(block_text)
            blocks.append({
                "text": block_text,
                "confidence": conf,
                "length": len(block_text.split()),
                "start": match.start(),
                "end": match.end(),
            })

    # Fallback: detect meta-like patterns in stripped text
    if allow_free_text_fallback and not blocks:
        conf_matches = re.findall(
            r'(?:probability|confidence|확률|확신)(?:\s*\([^)]*\))?[:\s\w]*?(\d+\.\d+|\d+)\s*%?',
            text, re.IGNORECASE
        )
        for m in conf_matches:
            v = float(m)
            if v > 1:
                v /= 100
            v = max(0.0, min(1.0, v))
            blocks.append({"text": "", "confidence": v, "length": 0, "start": 0, "end": len(text)})

    return blocks


def _has_structured_meta(text):
    return bool(_parse_meta_blocks(text, allow_free_text_fallback=False))


def _has_free_text_confidence_without_structure(text):
    return (
        not _has_structured_meta(text)
        and bool(_parse_meta_blocks(text, allow_free_text_fallback=True))
    )


def _meta_block_prefixes(text):
    """Return prefixes ending at each meta block for per-block probe scoring."""
    prefixes = []
    for block in _parse_meta_blocks_with_spans(text):
        end = block.get("end")
        if end is None:
            continue
        prefixes.append(text[:end])
    return prefixes


def _render_prompt_text(prompt, tokenizer=None):
    if prompt is None:
        return ""
    if isinstance(prompt, str):
        return prompt
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    if isinstance(prompt, list):
        rendered = []
        for message in prompt:
            if isinstance(message, dict):
                role = str(message.get("role", "user")).strip().capitalize()
                content = str(message.get("content", "")).strip()
                rendered.append(f"{role}: {content}")
        if rendered:
            rendered.append("Assistant:")
            return "\n\n".join(rendered)
    return str(prompt)


def _prefix_payloads_for_probe(text, *, prompt=None, tokenizer=None):
    completion_prefixes = _meta_block_prefixes(text)
    if not completion_prefixes:
        return []
    prompt_text = _render_prompt_text(prompt, tokenizer=tokenizer).rstrip("\n")
    if not prompt_text:
        return completion_prefixes
    return [prompt_text + prefix for prefix in completion_prefixes]


def _safe_model_device(model):
    if hasattr(model, "device"):
        return model.device
    try:
        return next(model.parameters()).device
    except Exception:
        return "cpu"


def _load_probe_head(hidden_dim, device, probe_path):
    """Load a SimpleCorrectnessProbe checkpoint if available."""
    try:
        import torch
        from src.probes.simple_probe import SimpleCorrectnessProbe
    except Exception:
        return None

    if probe_path is None:
        return None
    probe_path = Path(probe_path)
    if not probe_path.exists():
        return None

    probe = SimpleCorrectnessProbe(hidden_dim=hidden_dim)
    state = torch.load(probe_path, map_location="cpu", weights_only=False)
    temperature = None
    if isinstance(state, dict) and "state_dict" in state:
        temperature = state.get("temperature")
        state = state["state_dict"]
    probe.load_state_dict(state)
    if temperature is not None:
        try:
            probe.temperature.fill_(float(temperature))
        except Exception:
            pass
    probe.to(device)
    probe.eval()
    return probe


def _predict_probe_probabilities(
    prefix_texts,
    *,
    model=None,
    tokenizer=None,
    probe_head=None,
    probe_path=None,
    probe_predictor=None,
    max_length=2048,
):
    """Predict correctness probabilities for a batch of text prefixes."""
    if not prefix_texts:
        return []
    if probe_predictor is not None:
        return [float(x) for x in probe_predictor(prefix_texts)]
    if model is None or tokenizer is None:
        return [None] * len(prefix_texts)

    try:
        import torch
    except Exception:
        return [None] * len(prefix_texts)

    device = _safe_model_device(model)
    local_probe = probe_head
    if local_probe is None:
        hidden_dim = getattr(getattr(model, "config", None), "hidden_size", None)
        if hidden_dim is None:
            return [None] * len(prefix_texts)
        local_probe = _load_probe_head(hidden_dim, device, probe_path)
    if local_probe is None:
        return [None] * len(prefix_texts)
    # W1 fix: ensure probe is on same device as model (multi-GPU DDP)
    local_probe = local_probe.to(device)

    orig_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "right"
    try:
        encoded = tokenizer(
            prefix_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
    finally:
        tokenizer.padding_side = orig_padding_side
    encoded = {k: v.to(device) for k, v in encoded.items()}

    was_training = getattr(model, "training", False)
    try:
        model.eval()
        with torch.no_grad():
            outputs = model(
                **encoded,
                output_hidden_states=True,
                use_cache=False,
            )
            hidden_states = outputs.hidden_states[-1].float()
            probs = local_probe(hidden_states, encoded.get("attention_mask"))
            return probs.detach().cpu().tolist()
    except Exception as e:
        import warnings
        warnings.warn(
            f"_predict_probe_probabilities failed: {type(e).__name__}: {e}. "
            f"Returning [None]*{len(prefix_texts)} — probe reward will be 0.",
            stacklevel=2,
        )
        return [None] * len(prefix_texts)
    finally:
        if was_training:
            model.train()


def _score_stepwise_blocks(
    text,
    *,
    is_correct,
    probe_scores=None,
):
    """Score each meta block individually and aggregate.

    The intended object of learning is each meta intervention, not just the
    final confidence. Each block gets local credit for:
      - expressing confidence aligned with local belief / uncertainty
      - revising confidence downward after anomaly/conflict
      - avoiding unjustified early overconfidence
      - ending with final confidence aligned to correctness
    """
    blocks = _parse_meta_blocks(text)
    conf_blocks = [b for b in blocks if b["confidence"] is not None]
    if not conf_blocks:
        return 0.0

    scores = []
    prev_conf = None
    error_correction_re = re.compile(
        r'\b(wait|wrong|fix|actually|mistake|no,|let me re|hold on|incorrect|error)\b',
        re.IGNORECASE,
    )
    for idx, block in enumerate(conf_blocks):
        conf = block["confidence"]
        block_text = block["text"]
        local_probe = None
        if probe_scores is not None and idx < len(probe_scores):
            local_probe = probe_scores[idx]

        is_last = idx == len(conf_blocks) - 1
        has_conflict = _has_conflict_trigger(block_text) or _has_uncertainty_signal(block_text)
        has_diag = _has_failure_diagnosis(block_text) or _has_decomposition_plan(block_text)
        has_overconf = _has_overconfidence_signal(block_text)
        has_verify = _has_verification_signal(block_text)
        has_error_correction = bool(error_correction_re.search(block_text))

        score = 0.0

        if local_probe is not None:
            score += -(conf - local_probe) ** 2
        elif is_last:
            target = 1.0 if is_correct else 0.0
            score += -(conf - target) ** 2
        elif has_conflict:
            # When the model says something is wrong, local confidence should drop.
            target = 0.35 if has_diag else 0.45
            score += -(conf - target) ** 2
        elif prev_conf is None:
            # Early meta should not start with unjustified certainty.
            if conf < 0.55:
                score += 0.25
            elif conf > 0.9:
                score -= 0.3
        else:
            # Neutral intermediate block: moderate confidence is safer.
            score += -(conf - 0.6) ** 2

        if has_conflict:
            if prev_conf is not None and conf <= prev_conf - 0.08:
                score += 0.35
            elif prev_conf is not None and conf >= prev_conf and has_error_correction:
                score += 0.15
            elif prev_conf is not None and conf >= prev_conf:
                score -= 0.25
            if has_diag:
                score += 0.15

        if has_error_correction:
            score += 0.2

        if has_overconf or has_verify:
            # Verify-oriented meta should appear when confidence is high enough to commit.
            if conf >= 0.75:
                score += 0.15
            else:
                score -= 0.05

        if is_last:
            if prev_conf is not None:
                recovered = conf >= prev_conf + 0.15
                if is_correct and recovered:
                    # Successful redirect/verify should be allowed to regain confidence.
                    score += 0.35
                elif not is_correct and recovered:
                    # Confidence rebound on a wrong final trajectory is bad calibration.
                    score -= 0.25

            conf_clamped = max(0.01, min(0.99, conf))
            if is_correct:
                score += 0.5 * math.log(conf_clamped)
            else:
                score += 0.5 * math.log(1.0 - conf_clamped)

        scores.append(score)
        prev_conf = conf

    return max(sum(scores) / len(scores), -3.0)


def _parse_confidence(text):
    """Extract confidence value from a meta block."""
    matches = re.findall(
        r'(?:probability|confidence|확률|확신)(?:\s*\([^)]*\))?[:\s\w]*?(\d+\.\d+|\d+)\s*%?',
        text, re.IGNORECASE
    )
    for m in matches:
        v = float(m)
        if v > 1:
            v /= 100
        return max(0.01, min(0.99, v))
    return None


def _parse_confidence_charspan(text):
    """Char-span variant of _parse_confidence (ADDITIVE, TRIOBJ_DCPO_V2).

    Returns the (start, end) character offsets of the FIRST confidence NUMBER
    literal in `text` using the SAME regex as `_parse_confidence`, or None if no
    confidence is present. Used by build_dcpo_region_masks to map the confidence
    char-span to a token span (spec §4 Pass B). The matched float itself is
    identical to what _parse_confidence returns — only the offset is added.
    """
    m = re.search(
        r'(?:probability|confidence|확률|확신)(?:\s*\([^)]*\))?[:\s\w]*?(\d+\.\d+|\d+)\s*%?',
        text, re.IGNORECASE
    )
    if m is None:
        return None
    return m.span(1)


def meta_count_bonus(completions, ground_truth=None, max_rewarded_blocks=3, per_block_bonus=0.1, **kwargs):
    """Correctness-conditioned reward for multiple meta checkpoints.

    Bonus is applied ONLY if the final answer is correct. This prevents the
    naive "add meta blocks to farm reward" hacking observed in earlier runs
    where easy problems lost -10pp accuracy when meta blocks were rewarded
    unconditionally.

    When is_correct:
      - 0 blocks -> 0.0
      - 1 block  -> 0.1
      - 2 blocks -> 0.2
      - 3+       -> 0.3 (cap)
    When not is_correct:
      - any      -> 0.0
    """
    rewards = []
    for i, c in enumerate(completions):
        text = _get_text(c)
        gt = ground_truth[i] if ground_truth is not None else ""
        if not _check_correctness(text, gt):
            rewards.append(0.0)
            continue
        n_blocks = len(_parse_meta_blocks(text, allow_free_text_fallback=False))
        rewards.append(float(min(n_blocks, max_rewarded_blocks)) * per_block_bonus)
    return rewards


# ─── R0: Outcome-Aware Calibration (E21R-v2) ───

def outcome_calibration_reward(completions, ground_truth=None, **kwargs):
    """Structured proper-scoring calibration with revision credit.

    This reward is intentionally strict about structure:
      - only confidence inside wrapped meta blocks counts
      - free-text confidence without meta wrapping gets no calibration credit

    Components:
      1. Endpoint proper score using Brier against binary correctness
      2. Revision credit when final confidence is closer to the true outcome
         than the first confidence

    Range: approximately [-0.25, +0.25].
    """
    rewards = []
    for i, c in enumerate(completions):
        text = _get_text(c)
        gt = ground_truth[i] if ground_truth is not None else ""
        is_correct = _check_correctness(text, gt)
        blocks = _parse_meta_blocks(text, allow_free_text_fallback=False)
        confs = [b["confidence"] for b in blocks if b["confidence"] is not None]

        if not confs:
            rewards.append(0.0)
            continue

        target = 1.0 if is_correct else 0.0
        last_conf = confs[-1]
        endpoint = 0.3 * (1.0 - (last_conf - target) ** 2) - 0.15

        trajectory = 0.0
        if len(confs) >= 2:
            first_conf = confs[0]
            first_err = (first_conf - target) ** 2
            last_err = (last_conf - target) ** 2
            trajectory = 0.1 * (first_err - last_err)

        rewards.append(endpoint + trajectory)
    return rewards


def meta_structure_reward(completions, ground_truth=None, **kwargs):
    """Reward preserving wrapped meta structure.

    The goal is not to reward meta content directly, but to prevent RL from
    collapsing wrapped controller state into reward-equivalent free text.

    Rewards:
      +0.10  at least one wrapped meta block exists
      -0.10  confidence/meta-like text exists only in free text
       0.00  no meta-like content at all
    """
    rewards = []
    for c in completions:
        text = _get_text(c)
        if _has_structured_meta(text):
            rewards.append(0.10)
        elif _has_free_text_confidence_without_structure(text):
            rewards.append(-0.10)
        else:
            rewards.append(0.0)
    return rewards


def meta_commit_shape_reward(completions, ground_truth=None, **kwargs):
    """Analysis-driven reward for good meta -> commit behavior.

    This is intentionally controller-focused rather than style-focused.
    It rewards:
      - wrapped meta with diagnosis/study_need
      - concise post-meta reasoning that reaches a boxed commit
    And it penalizes:
      - no-boxed/non-commit outputs
      - repeated meta loops
      - post-boxed drift
      - decoherence-like delimiter imbalance / repeated tail fragments

    Range is modest on purpose so correctness remains dominant.
    """
    rewards = []
    for c in completions:
        text = _get_text(c)
        quality = score_meta_commit_quality(text)
        shaped = 0.30 * float(quality["total"])
        if float(quality.get("no_boxed_penalty", 0.0)) > 0.0:
            shaped -= 0.15
        if float(quality.get("decoherence_penalty", 0.0)) > 0.0:
            shaped -= 0.10 * float(quality["decoherence_penalty"])
        rewards.append(max(-0.35, min(0.35, shaped)))
    return rewards


# ─── R1: Correctness ───

def correctness_reward(completions, ground_truth=None, **kwargs):
    """Binary: +1 correct, -1 incorrect."""
    rewards = []
    for i, c in enumerate(completions):
        text = _get_text(c)
        gt = ground_truth[i] if ground_truth is not None else ""
        rewards.append(1.0 if _check_correctness(text, gt) else -1.0)
    return rewards


# ─── R1b: Format Reward (Open-R1 style) ───

def format_reward(completions, **kwargs):
    """Reward for using \\boxed{} answer format.

    +1 if completion contains \\boxed{...}
    0 otherwise
    """
    rewards = []
    for c in completions:
        text = _get_text(c)
        has_boxed = bool(re.search(r'\\boxed\{', text))
        rewards.append(1.0 if has_boxed else 0.0)
    return rewards


# ─── R2: Meta Quality ───

def meta_quality_reward(completions, **kwargs):
    """Reward for meta reasoning presence and quality.

    -0.5 if no meta blocks
    +0.1 per meta block (up to 3)
    +0.1 bonus if block has Q&A structure
    +0.1 bonus if block has 10+ words
    Max per block: 0.3, max total: ~1.0
    """
    rewards = []
    for c in completions:
        text = _get_text(c)
        blocks = _parse_meta_blocks(text)

        if not blocks:
            rewards.append(-0.5)
            continue

        r = 0.0
        for block in blocks[:4]:  # cap at 4 blocks
            r += 0.1  # existence bonus
            if block["length"] >= 10:
                r += 0.1  # length bonus
            if re.search(r'[?？]', block["text"]):
                r += 0.1  # Q&A structure bonus
        rewards.append(min(r, 1.2))
    return rewards


# ─── R3: Calibration (Group-based Doubt, CoCA style) ───

def calibration_reward(completions, ground_truth=None, **kwargs):
    """Group-based Brier score calibration (CoCA/Rewarding Doubt hybrid).

    Uses group empirical accuracy as target (not binary per-sample):
      p̂ = fraction of group that got correct answer
      r = -(stated_confidence - p̂)² per meta block, summed

    This gives smoother calibration signal than binary correct/wrong.
    """
    # First pass: compute group accuracy
    correct_flags = []
    for i, c in enumerate(completions):
        text = _get_text(c)
        gt = ground_truth[i] if ground_truth is not None else ""
        correct_flags.append(1.0 if _check_correctness(text, gt) else 0.0)
    group_accuracy = sum(correct_flags) / max(len(correct_flags), 1)

    # Second pass: compute calibration reward per completion
    rewards = []
    for i, c in enumerate(completions):
        text = _get_text(c)
        blocks = _parse_meta_blocks(text)

        if not blocks or all(b["confidence"] is None for b in blocks):
            rewards.append(0.0)
            continue

        r = 0.0
        n = 0
        for block in blocks:
            conf = block["confidence"]
            if conf is None:
                continue
            # Brier score: -(conf - p̂)²
            r += -(conf - group_accuracy) ** 2
            n += 1

        # Also add log scoring (Rewarding Doubt) for individual correctness
        is_correct = bool(correct_flags[i])
        last_conf = None
        for block in reversed(blocks):
            if block["confidence"] is not None:
                last_conf = block["confidence"]
                break
        if last_conf is not None:
            if is_correct:
                r += math.log(max(last_conf, 0.01))
            else:
                r += math.log(max(1.0 - last_conf, 0.01))
            n += 1

        if n > 0:
            r = r / n  # average
        rewards.append(max(r, -5.0))
    return rewards


# ─── R4: Uncertainty-Aware Meta ───

def uncertainty_meta_reward(completions, ground_truth=None, **kwargs):
    """Higher reward when meta blocks appear at uncertain moments.

    (1 - confidence) × meta_quality → summation
    Uncertain + good meta = high reward
    Confident + any meta = low reward (unnecessary reflection)
    """
    rewards = []
    for i, c in enumerate(completions):
        text = _get_text(c)
        blocks = _parse_meta_blocks(text)

        if not blocks:
            rewards.append(0.0)
            continue

        r = 0.0
        for block in blocks[:4]:
            conf = block["confidence"]
            if conf is None:
                conf = 0.5  # unknown confidence treated as moderate

            uncertainty = 1.0 - conf  # high when unsure

            # Quality component
            quality = 0.0
            if block["length"] >= 10:
                quality += 0.5
            if re.search(r'[?？]', block["text"]):
                quality += 0.5
            quality = min(quality, 1.0)

            r += uncertainty * quality  # uncertain + quality meta = high

        rewards.append(r)
    return rewards


# ─── R5: Stepwise Confidence Trajectory ───

def stepwise_trajectory_reward(completions, ground_truth=None, **kwargs):
    """Reward stepwise meta control at the block level.

    This is no longer just a global monotonicity heuristic. The intended object
    is each meta intervention:
      - early blocks should avoid unjustified overconfidence
      - anomaly/diagnosis blocks should revise confidence downward
      - final blocks should calibrate confidence to outcome
    """
    rewards = []
    for i, c in enumerate(completions):
        text = _get_text(c)
        gt = ground_truth[i] if ground_truth is not None else ""
        is_correct = _check_correctness(text, gt)
        rewards.append(_score_stepwise_blocks(text, is_correct=is_correct))
    return rewards


# ─── R6: Probe Calibration Reward ───
def stepwise_probe_reward(completions, ground_truth=None, **kwargs):
    """Probe-based stepwise reward with per-meta-block credit assignment.

    For each meta block, score stated confidence against probe-estimated local
    correctness probability from the prefix ending at that block. When a probe
    is unavailable, fall back to heuristic block-wise targets so the reward
    function remains smoke-testable.
    """
    model = kwargs.get("model")
    tokenizer = kwargs.get("tokenizer")
    probe_head = kwargs.get("probe_head")
    probe_path = kwargs.get("probe_path")
    probe_predictor = kwargs.get("probe_predictor")
    max_length = kwargs.get("max_length", 2048)
    prompts = kwargs.get("prompts")

    rewards = []
    for i, c in enumerate(completions):
        text = _get_text(c)
        gt = ground_truth[i] if ground_truth is not None else ""
        is_correct = _check_correctness(text, gt)
        prompt = prompts[i] if prompts is not None and i < len(prompts) else None
        all_prefixes = _prefix_payloads_for_probe(text, prompt=prompt, tokenizer=tokenizer)
        # W5 fix: only send conf-block prefixes to probe (avoid wasted inference)
        all_blocks = _parse_meta_blocks(text)
        conf_indices = [j for j, b in enumerate(all_blocks) if b["confidence"] is not None]
        aligned_prefixes = [all_prefixes[j] for j in conf_indices if j < len(all_prefixes)]
        probe_scores = _predict_probe_probabilities(
            aligned_prefixes,
            model=model,
            tokenizer=tokenizer,
            probe_head=probe_head,
            probe_path=probe_path,
            probe_predictor=probe_predictor,
            max_length=max_length,
        ) if aligned_prefixes else []
        rewards.append(_score_stepwise_blocks(text, is_correct=is_correct, probe_scores=probe_scores))
    return rewards


def probe_calibration_reward(completions, ground_truth=None,
                              model=None, tokenizer=None, **kwargs):
    """Calibration reward using a hidden-state probe on meta-block prefixes.

    The reward aligns each stated confidence with the probe's local correctness
    estimate p_hat for the prefix ending at that meta block:

        R = mean_k [ -(conf_k - p_hat_k)^2 ]

    This is the verifiable bridge between verbal confidence and internal belief.
    """
    probe_head = kwargs.get("probe_head")
    probe_path = kwargs.get("probe_path")
    probe_predictor = kwargs.get("probe_predictor")
    max_length = kwargs.get("max_length", 2048)
    prompts = kwargs.get("prompts")

    rewards = []
    for i, c in enumerate(completions):
        text = _get_text(c)
        all_blocks = _parse_meta_blocks(text)
        prompt = prompts[i] if prompts is not None and i < len(prompts) else None
        all_prefixes = _prefix_payloads_for_probe(text, prompt=prompt, tokenizer=tokenizer)

        # Align: keep only (block, prefix) pairs where block has confidence
        conf_aligned = [
            (b, all_prefixes[idx])
            for idx, b in enumerate(all_blocks)
            if b["confidence"] is not None and idx < len(all_prefixes)
        ]

        if not conf_aligned:
            rewards.append(0.0)
            continue

        aligned_blocks, aligned_prefixes = zip(*conf_aligned)
        probe_scores = _predict_probe_probabilities(
            list(aligned_prefixes),
            model=model,
            tokenizer=tokenizer,
            probe_head=probe_head,
            probe_path=probe_path,
            probe_predictor=probe_predictor,
            max_length=max_length,
        )
        paired = [
            -(block["confidence"] - p_hat) ** 2
            for block, p_hat in zip(aligned_blocks, probe_scores)
            if p_hat is not None
        ]
        if not paired:
            rewards.append(0.0)
            continue
        rewards.append(max(sum(paired) / len(paired), -3.0))

    return rewards


# ─── R7: Length Penalty ───

def length_penalty_reward(completions, **kwargs):
    """Penalize verbose responses to prevent reward hacking via length.

    No penalty below 500 tokens. Linear penalty above 500.
    -0.001 per token over 500 (max penalty -1.5 at 2000 tokens).

    This prevents GRPO from gaming meta_quality/format rewards
    by generating long, low-quality responses.
    """
    rewards = []
    for c in completions:
        text = _get_text(c)
        n_tokens = len(text.split())  # word-level approximation
        if n_tokens <= 500:
            rewards.append(0.0)
        else:
            penalty = -0.001 * (n_tokens - 500)
            rewards.append(max(penalty, -1.5))
    return rewards


# ─── R8: Correctness-Conditional Meta ───

def correct_meta_reward(completions, ground_truth=None, **kwargs):
    """Only reward meta quality when the answer is correct.

    This prevents the model from generating verbose meta blocks
    on wrong answers just to collect meta_quality reward.

    Correct + good meta → +0.5
    Correct + no meta → 0.0
    Wrong + any meta → -0.3 (penalty for wasting tokens on wrong answer)
    Wrong + no meta → 0.0
    """
    rewards = []
    for i, c in enumerate(completions):
        text = _get_text(c)
        gt = ground_truth[i] if ground_truth is not None else ""
        is_correct = _check_correctness(text, gt)
        blocks = _parse_meta_blocks(text)

        if is_correct:
            if blocks:
                # Reward proportional to meta quality
                q = 0.0
                for b in blocks[:3]:
                    if b["confidence"] is not None:
                        q += 0.1
                    if b["length"] >= 10:
                        q += 0.05
                rewards.append(min(q, 0.5))
            else:
                rewards.append(0.0)
        else:
            if blocks:
                rewards.append(-0.3)  # Penalize meta on wrong answers
            else:
                rewards.append(0.0)
    return rewards


def self_correction_reward(completions, ground_truth=None, **kwargs):
    """Reward uncertainty followed by a genuine change in approach.

    This targets the intended metacognitive behavior:
    when the model realizes it may be wrong, it should redirect and recover.
    """
    rewards = []
    for i, c in enumerate(completions):
        text = _get_text(c)
        gt = ground_truth[i] if ground_truth is not None else ""
        is_correct = _check_correctness(text, gt)

        has_uncertainty = _has_uncertainty_signal(text)
        has_redirection = _has_redirection_signal(text)

        if has_uncertainty and has_redirection:
            rewards.append(0.8 if is_correct else 0.2)
        elif has_uncertainty:
            rewards.append(-0.2)
        else:
            rewards.append(0.0)
    return rewards


def verification_reward(completions, ground_truth=None, **kwargs):
    """Reward final verification, especially for high-confidence answers."""
    rewards = []
    for i, c in enumerate(completions):
        text = _get_text(c)
        meta_text = _meta_joined_text(text)
        solve_tail = _text_after_last_meta(text)
        gt = ground_truth[i] if ground_truth is not None else ""
        is_correct = _check_correctness(text, gt)
        conf = _last_confidence(text)
        has_verify_intent = _has_verification_signal(meta_text)
        has_verify = _has_effective_verification_signal(solve_tail)

        if conf is None:
            rewards.append(0.0)
            continue

        if conf >= 0.75 and has_verify_intent and has_verify:
            rewards.append(0.5 if is_correct else 0.05)
        elif conf >= 0.75 and has_verify_intent and not has_verify:
            rewards.append(-0.1 if is_correct else -0.6)
        else:
            rewards.append(0.1 if has_verify_intent and has_verify and is_correct else 0.0)
    return rewards


def overconfidence_penalty_reward(completions, ground_truth=None, **kwargs):
    """Strongly penalize wrong answers delivered with high confidence."""
    rewards = []
    for i, c in enumerate(completions):
        text = _get_text(c)
        gt = ground_truth[i] if ground_truth is not None else ""
        is_correct = _check_correctness(text, gt)
        conf = _last_confidence(text)

        if conf is None or is_correct:
            rewards.append(0.0)
            continue

        if conf >= 0.95:
            rewards.append(-1.0)
        elif conf >= 0.85:
            rewards.append(-0.6)
        elif conf >= 0.7:
            rewards.append(-0.25)
        else:
            rewards.append(0.0)
    return rewards


def confidence_revision_reward(completions, ground_truth=None, **kwargs):
    """Reward lowering confidence when evidence suggests the current path is weak."""
    rewards = []
    for c in completions:
        text = _get_text(c)
        meta_text = _meta_joined_text(text)
        has_conflict = _has_conflict_trigger(meta_text) or _has_uncertainty_signal(meta_text)
        has_drop = _has_confidence_drop(text) or _has_low_confidence(text)

        if has_conflict and has_drop:
            rewards.append(0.5)
        elif has_conflict and not has_drop:
            rewards.append(-0.35)
        elif has_drop and not has_conflict:
            rewards.append(-0.1)
        else:
            rewards.append(0.0)
    return rewards


def effective_verification_reward(completions, ground_truth=None, **kwargs):
    """Reward verification that uses an explicit checking mechanism."""
    rewards = []
    for i, c in enumerate(completions):
        text = _get_text(c)
        meta_text = _meta_joined_text(text)
        solve_tail = _text_after_last_meta(text)
        gt = ground_truth[i] if ground_truth is not None else ""
        is_correct = _check_correctness(text, gt)
        conf = _last_confidence(text)
        has_verify_intent = _has_verification_signal(meta_text) or _has_overconfidence_signal(meta_text)
        has_effective_verify = _has_effective_verification_signal(solve_tail)

        if has_verify_intent and has_effective_verify:
            rewards.append(0.8 if is_correct else 0.1)
        elif has_verify_intent and conf is not None and conf >= 0.8:
            rewards.append(-0.2 if is_correct else -0.8)
        else:
            rewards.append(0.0)
    return rewards


def effective_redirection_reward(completions, ground_truth=None, **kwargs):
    """Reward genuine redirection after detecting a conflict or being stuck."""
    rewards = []
    for i, c in enumerate(completions):
        text = _get_text(c)
        meta_text = _meta_joined_text(text)
        gt = ground_truth[i] if ground_truth is not None else ""
        is_correct = _check_correctness(text, gt)

        has_conflict = _has_conflict_trigger(meta_text) or _has_uncertainty_signal(meta_text)
        has_switch = _has_next_strategy(meta_text)
        has_drop = _has_confidence_drop(text) or _has_low_confidence(text)
        solve_tail = _text_after_last_meta(text)
        has_tail_recovery = bool(solve_tail.strip())

        if has_conflict and has_switch and has_drop and has_tail_recovery:
            rewards.append(1.0 if is_correct else 0.2)
        elif has_conflict and has_switch:
            rewards.append(0.5 if is_correct else 0.0)
        elif has_conflict and not has_switch:
            rewards.append(-0.4)
        elif has_switch and not has_conflict:
            rewards.append(-0.1)
        else:
            rewards.append(0.0)
    return rewards


def diagnosis_reward(completions, ground_truth=None, **kwargs):
    """Reward natural-language diagnosis after uncertainty or conflict."""
    rewards = []
    for c in completions:
        text = _get_text(c)
        meta_text = _meta_joined_text(text)
        has_conflict = _has_conflict_trigger(meta_text) or _has_uncertainty_signal(meta_text)
        has_diag = _has_failure_diagnosis(meta_text)
        has_blocker = _has_missing_skill_or_blocker(meta_text) or _has_decomposition_plan(meta_text)

        if has_conflict and has_diag and has_blocker:
            rewards.append(0.7)
        elif has_conflict and has_diag:
            rewards.append(0.35)
        elif has_conflict and not has_diag:
            rewards.append(-0.3)
        elif has_diag and not has_conflict:
            rewards.append(-0.05)
        else:
            rewards.append(0.0)
    return rewards


def decomposition_reward(completions, ground_truth=None, **kwargs):
    """Reward decomposition plans that accompany a real redirect."""
    rewards = []
    for i, c in enumerate(completions):
        text = _get_text(c)
        meta_text = _meta_joined_text(text)
        gt = ground_truth[i] if ground_truth is not None else ""
        is_correct = _check_correctness(text, gt)

        has_conflict = _has_conflict_trigger(meta_text) or _has_uncertainty_signal(meta_text)
        has_plan = _has_decomposition_plan(meta_text)
        has_strategy = _has_next_strategy(meta_text)
        has_drop = _has_confidence_drop(text) or _has_low_confidence(text)

        if has_conflict and has_plan and has_strategy and has_drop:
            rewards.append(0.8 if is_correct else 0.15)
        elif has_conflict and has_plan and has_strategy:
            rewards.append(0.35 if is_correct else 0.0)
        elif has_conflict and not has_plan:
            rewards.append(-0.25)
        else:
            rewards.append(0.0)
    return rewards


def anomaly_notice_reward(completions, ground_truth=None, **kwargs):
    """Reward noticing that the current route feels unreliable before redirect."""
    rewards = []
    for c in completions:
        text = _get_text(c)
        meta_text = _meta_joined_text(text)
        has_notice = _has_anomaly_notice_signal(meta_text)
        has_drop = _has_confidence_drop(text)
        if has_notice and has_drop:
            rewards.append(0.45)
        elif has_notice:
            rewards.append(0.15)
        else:
            rewards.append(0.0)
    return rewards


def repeated_intervention_reward(completions, ground_truth=None, **kwargs):
    """Reward multiple interventions only when they accompany real control signals."""
    rewards = []
    for c in completions:
        text = _get_text(c)
        meta_count = len(_parse_meta_blocks(text))
        meta_text = _meta_joined_text(text)
        has_control = (
            (_has_verification_signal(meta_text) and _has_effective_verification_signal(_text_after_last_meta(text)))
            or ((_has_conflict_trigger(meta_text) or _has_uncertainty_signal(meta_text)) and _has_next_strategy(meta_text))
        )
        if meta_count >= 2 and has_control:
            rewards.append(0.35)
        elif meta_count >= 2 and not has_control:
            rewards.append(-0.1)
        else:
            rewards.append(0.0)
    return rewards


def overconfidence_verify_reward(completions, ground_truth=None, **kwargs):
    """Reward verification specifically when confidence is high."""
    rewards = []
    for i, c in enumerate(completions):
        text = _get_text(c)
        meta_text = _meta_joined_text(text)
        gt = ground_truth[i] if ground_truth is not None else ""
        is_correct = _check_correctness(text, gt)
        conf = _last_confidence(text)
        has_verify = _has_effective_verification_signal(_text_after_last_meta(text))
        has_overconfidence = _has_overconfidence_signal(meta_text) or _has_verification_signal(meta_text)
        if conf is None or conf < 0.8:
            rewards.append(0.0)
        elif has_overconfidence and has_verify:
            rewards.append(0.45 if is_correct else 0.05)
        else:
            rewards.append(-0.35 if not is_correct else -0.1)
    return rewards


# ─── V2 Reward Functions (2026-04-04) ────────────────────────────────────────
# Fix 1: same-route repetition penalty
# Fix 2: route-switch evidence reward
# Fix 3: confidence omission floor


def _text_before_last_meta(text):
    """Return text up to (not including) the last <|meta|> block."""
    tag = "<|meta|>"
    idx = text.rfind(tag)
    if idx == -1:
        return text
    return text[:idx]


def _redirect_context(text):
    """Shared redirect context for confidence-centered control rewards."""
    meta_text = _meta_joined_text(text)
    solve_tail = _text_after_last_meta(text)
    prefix = _text_before_last_meta(text)
    first_conf, last_conf = _first_and_last_confidence(text)
    has_trigger = (
        _has_conflict_trigger(meta_text)
        or _has_uncertainty_signal(meta_text)
        or _has_low_confidence(text)
        or _has_confidence_drop(text)
    )
    has_diag = _has_failure_diagnosis(meta_text) or _has_decomposition_plan(meta_text)
    has_next = _has_next_strategy(meta_text)
    method_diff = _method_diff_score(prefix, solve_tail)
    strategy_overlap = len(_strategy_terms(meta_text) & _strategy_terms(solve_tail))
    tail_has_alt_method = bool(re.search(
        r'\b(factor|discriminant|symmetry|work backwards?|'
        r'count directly|enumerat|identity|product-to-sum|'
        r'fraction|parallel sides|trapezoid area)\b',
        solve_tail,
        re.IGNORECASE,
    ))
    followthrough_keywords = (
        ("invariant" in meta_text.lower() and "invariant" in solve_tail.lower())
        or ("parity" in meta_text.lower() and "parity" in solve_tail.lower())
        or ("case split" in meta_text.lower() and "case split" in solve_tail.lower())
        or ("constraint" in meta_text.lower() and "constraint" in solve_tail.lower())
    )
    nontrivial_tail = len(solve_tail.split()) >= 8
    has_verify_tail = _has_effective_verification_signal(solve_tail)
    has_route_replacement = nontrivial_tail and (
        method_diff >= 0.30
        or (has_diag and has_next and method_diff >= 0.18 and not has_verify_tail)
        or strategy_overlap > 0
        or (has_next and followthrough_keywords)
        or (has_next and tail_has_alt_method)
    )
    has_execution = has_route_replacement
    return {
        "meta_text": meta_text,
        "solve_tail": solve_tail,
        "prefix": prefix,
        "first_conf": first_conf,
        "last_conf": last_conf,
        "has_trigger": has_trigger,
        "has_diag": has_diag,
        "has_next": has_next,
        "method_diff": method_diff,
        "strategy_overlap": strategy_overlap,
        "has_verify_tail": has_verify_tail,
        "has_route_replacement": has_route_replacement,
        "has_execution": has_execution,
    }


_REPETITION_RE = re.compile(
    r'\b(repeat(?:ing)?|same (calculation|approach|route|steps|method|way|chain)|'
    r'again compute|re-?compute the same|re-?check the same|'
    r'verify again the same|confirm by repeating)\b',
    re.IGNORECASE,
)

_INDEPENDENT_METHOD_RE = re.compile(
    r'\b(different method|alternative approach|working backwards|'
    r'from the opposite end|direct check|boundary case|edge case|'
    r'special case|numerical check|dimensional analysis|'
    r'let me approach this differently|solve it another way|'
    r'independent(?:ly)? (?:verify|check|confirm|compute)|'
    r'cross-?check|plug .{0,20} into the original)\b',
    re.IGNORECASE,
)


def same_route_repetition_penalty(completions, ground_truth=None, **kwargs):
    """Penalize verification that repeats the same calculation instead of
    using an independent checking method.

    Heuristic: if the solve tail after the last meta block contains explicit
    repetition language OR lacks independent-method language while the meta
    block announced a verify intent, apply a penalty.

    Rewards:
      -0.5  verify intent + repetition detected in tail
      -0.3  verify intent + short tail (<20 words, likely no real check)
       0.0  no verify intent, or verify intent + independent method signal
    """
    rewards = []
    for c in completions:
        text = _get_text(c)
        meta_text = _meta_joined_text(text)
        solve_tail = _text_after_last_meta(text)

        has_verify_intent = (
            _has_verification_signal(meta_text)
            or _has_overconfidence_signal(meta_text)
        )
        if not has_verify_intent:
            rewards.append(0.0)
            continue

        is_repetition = bool(_REPETITION_RE.search(solve_tail))
        has_independent = bool(_INDEPENDENT_METHOD_RE.search(solve_tail))
        tail_words = len(solve_tail.split())

        if is_repetition and not has_independent:
            rewards.append(-0.5)
        elif tail_words < 20 and not has_independent:
            rewards.append(-0.3)
        else:
            rewards.append(0.0)
    return rewards


def route_switch_evidence_reward(completions, ground_truth=None, **kwargs):
    """Reward evidence that a redirect actually changed the solving method.

    Checks whether the solve tail after a redirect announcement uses
    structurally different keywords from the prefix before the meta block.

    Rewards:
      +0.9 / +0.25  methods differ + coherent tail (correct / wrong)
      +0.5 / +0.1   partial evidence of switch
      -0.3           switch announced but no structural difference in tail
       0.0           no redirect intent
    """
    rewards = []
    for i, c in enumerate(completions):
        text = _get_text(c)
        meta_text = _meta_joined_text(text)
        solve_tail = _text_after_last_meta(text)
        gt = ground_truth[i] if ground_truth is not None else ""
        is_correct = _check_correctness(text, gt)

        has_conflict = (
            _has_conflict_trigger(meta_text) or _has_uncertainty_signal(meta_text)
        )
        has_switch = _has_next_strategy(meta_text)
        has_drop = _has_confidence_drop(text) or _has_low_confidence(text)

        if not (has_conflict and has_switch and has_drop):
            rewards.append(0.0)
            continue

        prefix = _text_before_last_meta(text)
        methods_differ = _methods_structurally_differ(prefix, solve_tail)
        tail_coherent = (
            len(solve_tail.split()) >= 15
            and bool(re.search(r'\\boxed\{|####|\bans', solve_tail, re.IGNORECASE))
        )

        if methods_differ and tail_coherent:
            rewards.append(0.9 if is_correct else 0.25)
        elif methods_differ or tail_coherent:
            rewards.append(0.5 if is_correct else 0.1)
        else:
            rewards.append(-0.3)
    return rewards


def redirect_execution_reward(completions, ground_truth=None, **kwargs):
    """Reward confidence-triggered redirect that actually changes the route.

    This is intentionally simpler than the older full controller:
      - a redirect should be triggered by low confidence, anomaly, or diagnosis
      - the meta block should name a next strategy
      - the post-meta solve tail should differ structurally from the pre-meta route

    The reward stays mildly positive for incomplete-but-valid redirect attempts,
    so hard problems are not forced into all-or-nothing credit.
    """
    rewards = []
    for i, c in enumerate(completions):
        text = _get_text(c)
        meta_text = _meta_joined_text(text)
        solve_tail = _text_after_last_meta(text)
        prefix = _text_before_last_meta(text)
        gt = ground_truth[i] if ground_truth is not None else ""
        is_correct = _check_correctness(text, gt)

        has_trigger = (
            _has_low_confidence(text)
            or _has_conflict_trigger(meta_text)
            or _has_uncertainty_signal(meta_text)
            or _has_failure_diagnosis(meta_text)
            or _has_decomposition_plan(meta_text)
        )
        has_strategy = _has_next_strategy(meta_text)
        methods_differ = _methods_structurally_differ(prefix, solve_tail)
        tail_coherent = (
            len(solve_tail.split()) >= 10
            and bool(re.search(r'\\boxed\{|####|[=+\-*/^()]|\d', solve_tail))
        )

        if has_trigger and has_strategy and methods_differ and tail_coherent:
            rewards.append(0.9 if is_correct else 0.25)
        elif has_trigger and has_strategy and (methods_differ or tail_coherent):
            rewards.append(0.45 if is_correct else 0.1)
        elif has_trigger and has_strategy:
            rewards.append(0.1)
        elif has_trigger and not has_strategy:
            rewards.append(-0.2)
        elif has_strategy and not has_trigger:
            rewards.append(-0.05)
        else:
            rewards.append(0.0)
    return rewards


def verify_execution_reward(completions, ground_truth=None, **kwargs):
    """Reward verification only when it is actually executed.

    Combines:
      - `effective_verification_reward`: real check in the solve tail
      - `overconfidence_verify_reward`: high-confidence / overcommit gating

    This keeps verify as a conditional controller rather than a ubiquitous
    phrase-level reward.
    """
    r1 = effective_verification_reward(completions, ground_truth=ground_truth, **kwargs)
    r2 = overconfidence_verify_reward(completions, ground_truth=ground_truth, **kwargs)
    return [0.7 * a + 0.3 * b for a, b in zip(r1, r2)]

def _methods_structurally_differ(prefix_text, tail_text):
    """Heuristic: do prefix and tail use substantively different approaches?

    Checks for method-pair switches (e.g., algebra→invariant) and
    explicit switch declarations in the tail.
    """
    p = prefix_text.lower()
    t = tail_text.lower()

    method_pairs = [
        ("algebra", "invariant"), ("direct", "case split"), ("direct", "case analysis"),
        ("expand", "contract"), ("forward", "backward"),
        ("coordinate", "vector"), ("parity", "modular"),
        ("inclusion-exclusion", "recurrence"), ("brute", "structural"),
        ("substitut", "parity"), ("counting", "generating function"),
        ("direct", "recursion"), ("algebra", "geometric"),
        ("analytic", "numeric"), ("exact", "approximat"),
        ("induction", "direct"), ("constructive", "contradiction"),
        ("greedy", "dynamic programming"), ("brute force", "dynamic programming"),
        ("brute force", "clever"),
        ("trial", "systematic"), ("guess", "deriv"),
    ]
    for old_m, new_m in method_pairs:
        if (old_m in p and new_m in t) or (new_m in p and old_m in t):
            return True

    if re.search(
        r'\b(different method|alternative|instead of the previous|'
        r'another approach|let me try a different)\b',
        t, re.IGNORECASE,
    ):
        return True

    return False


def confidence_omission_floor(completions, ground_truth=None, **kwargs):
    """Penalize completions that emit no meta blocks at all.

    Without this floor, a model can escape calibration pressure by simply
    not emitting any <|meta|> blocks, receiving 0.0 from calibration_reward
    instead of being penalized for poor calibration.

    Rewards:
      -0.5  no meta blocks emitted (penalty for omission)
       0.0  at least one meta block present (passes the floor)
    """
    rewards = []
    for c in completions:
        text = _get_text(c)
        blocks = _parse_meta_blocks(text, allow_free_text_fallback=False)
        if not blocks:
            rewards.append(-0.5)
        else:
            rewards.append(0.0)
    return rewards


# ─── V6.1 Rewards (2026-04-05): structural switch + Brier calibration + verify outcome ───


def structural_switch_reward(completions, ground_truth=None, **kwargs):
    """R2: Reward structural method switching with partial credit.

    Decomposed (per Codex review):
      - Switch attempt alone: +0.3 (regardless of correctness)
      - Switch + correct answer: +1.0 (full bonus)
    This lets the model learn switching behavior even on hard problems
    where it cannot yet execute the new method correctly.
    """
    rewards = []
    for i, c in enumerate(completions):
        text = _get_text(c)
        gt = ground_truth[i] if ground_truth is not None else ""

        meta_start = text.find("<|meta|>")
        meta_end = text.rfind("<|/meta|>")

        if meta_start < 0 or meta_end < 0:
            rewards.append(0.0)
            continue

        pre = text[:meta_start]
        post = text[meta_end + len("<|/meta|>"):]

        if len(pre) < 30 or len(post) < 30:
            rewards.append(0.0)
            continue

        methods_differ = _methods_structurally_differ(pre, post)

        if not methods_differ:
            rewards.append(0.0)
            continue

        is_correct = _check_correctness(text, gt)
        if is_correct:
            rewards.append(1.0)
        else:
            rewards.append(0.3)  # partial credit for switching attempt

    return rewards


def brier_calibration_reward(completions, ground_truth=None, **kwargs):
    """R3 (legacy): Simple Brier score. Kept for backward compat."""
    rewards = []
    for i, c in enumerate(completions):
        text = _get_text(c)
        gt = ground_truth[i] if ground_truth is not None else ""
        conf = _last_confidence(text)
        if conf is None:
            rewards.append(0.0)
            continue
        is_correct = 1.0 if _check_correctness(text, gt) else 0.0
        rewards.append(1.0 - (conf - is_correct) ** 2)
    return rewards


def confidence_trajectory_reward(completions, ground_truth=None, **kwargs):
    """R3v2: Continuous confidence trajectory reward.

    Scores 3 axes:
      1. Calibration: last conf vs actual correctness (Brier)
      2. Gradual change: penalizes abrupt jumps (subgoal trap avoidance)
      3. Direction: correct→rise, wrong→drop is healthy metacognition

    Single confidence: discounted Brier (0.2x) to encourage multi-meta.
    Empty confidence: small penalty (-0.05).
    """
    rewards = []
    for i, c in enumerate(completions):
        text = _get_text(c)
        gt = ground_truth[i] if ground_truth is not None else ""
        is_correct = _check_correctness(text, gt)
        correct_float = 1.0 if is_correct else 0.0

        # Extract all confidences from meta blocks
        blocks = _parse_meta_blocks(text)
        confs = [b["confidence"] for b in blocks if b["confidence"] is not None]

        # Empty: small penalty
        if not confs:
            rewards.append(-0.05)
            continue

        # Single: discounted Brier (cap at 0.2 to discourage single-conf)
        if len(confs) == 1:
            brier = 1.0 - (confs[0] - correct_float) ** 2
            rewards.append(min(brier * 0.2, 0.2))
            continue

        # Multiple confidences: 3-axis continuous scoring
        first, last = confs[0], confs[-1]

        # Axis 1: Calibration (Brier on last confidence)
        cal_score = 1.0 - (last - correct_float) ** 2

        # Axis 2: Gradual change (penalize abrupt jumps)
        steps = [confs[j + 1] - confs[j] for j in range(len(confs) - 1)]
        max_jump = max(abs(s) for s in steps)
        gradual_score = max(0.0, 1.0 - max_jump * 2)

        # Axis 3: Direction (correct→rise, wrong→drop, floor at 0)
        direction = last - first
        if is_correct:
            dir_score = max(0.0, min(direction * 2, 1.0))
        else:
            dir_score = max(0.0, min(-direction * 2, 1.0))

        score = cal_score * 0.4 + gradual_score * 0.3 + dir_score * 0.3
        rewards.append(score)

    return rewards


def verify_outcome_reward(completions, ground_truth=None, **kwargs):
    """R4: Reward independent verification that leads to correct answers.

    Strengthened per Codex review:
      - Require numerical content in verification (not just phrase)
      - Correct + verified: +0.3
      - Wrong + verified: -0.2 (increased penalty to discourage spurious verification)
      - Phrase-only (no numbers): +0.05 (minimal, discourage text-only hack)
    """
    _verify_re = re.compile(
        r"\b(substitut\w*\s+back|plug\w*\s+(back|in)|"
        r"reverse|inverse|check\w*\s+by|boundary|special\s+case|"
        r"sanity\s+check|verify\w*\s+by)\b",
        re.IGNORECASE,
    )
    _has_numbers_re = re.compile(r"-?\d[\d,]*\.?\d*\s*[=<>≤≥≠!]|[a-zA-Z]\s*=\s*-?\d")
    rewards = []
    for i, c in enumerate(completions):
        text = _get_text(c)
        gt = ground_truth[i] if ground_truth is not None else ""

        meta_end = text.rfind("<|/meta|>")
        verify_region = text[meta_end:] if meta_end >= 0 else text[-500:]

        if not _verify_re.search(verify_region):
            rewards.append(0.0)
            continue

        has_numerical = bool(_has_numbers_re.search(verify_region))
        is_correct = _check_correctness(text, gt)

        if not has_numerical:
            rewards.append(0.05)  # phrase-only: minimal credit
        elif is_correct:
            rewards.append(0.3)
        else:
            rewards.append(-0.2)

    return rewards


def _method_diff_score(pre: str, post: str) -> float:
    """Soft method difference score (0-1) for switch_v2.

    3 components per Codex review:
      method_family_change (0.60): keyword pair matching (expanded)
      representation_change (0.25): variable/notation differences
      opener_change (0.15): first-line approach indicator
    """
    p = pre.lower()
    t = post.lower()

    # 1. Method family change (0.60)
    method_family = 0.0
    method_pairs = [
        ("algebra", "invariant"), ("direct", "case split"), ("direct", "case analysis"),
        ("expand", "contract"), ("forward", "backward"),
        ("coordinate", "vector"), ("parity", "modular"),
        ("inclusion-exclusion", "recurrence"), ("brute", "structural"),
        ("substitut", "parity"), ("counting", "generating function"),
        ("direct", "recursion"), ("algebra", "geometric"),
        ("analytic", "numeric"), ("exact", "approximat"),
        ("induction", "direct"), ("constructive", "contradiction"),
        ("greedy", "dynamic programming"), ("brute force", "dynamic programming"),
        ("brute force", "clever"), ("trial", "systematic"), ("guess", "deriv"),
        ("factoring", "completing the square"), ("synthetic division", "long division"),
        ("trigonometric", "algebraic"), ("polar", "cartesian"),
    ]
    for old_m, new_m in method_pairs:
        if (old_m in p and new_m in t) or (new_m in p and old_m in t):
            method_family = 1.0
            break

    # Explicit switch phrases (weaker signal)
    if method_family == 0.0 and re.search(
        r'\b(different method|alternative|instead of the previous|'
        r'another approach|let me try a different|let me reconsider)\b',
        t, re.IGNORECASE,
    ):
        method_family = 0.5

    # 2. Representation change (0.25): different variables/notation
    repr_change = 0.0
    # Check if variables change
    pre_vars = set(re.findall(r'\b([a-z])\s*=', p))
    post_vars = set(re.findall(r'\b([a-z])\s*=', t))
    if pre_vars and post_vars:
        overlap = len(pre_vars & post_vars) / max(len(pre_vars | post_vars), 1)
        repr_change = 1.0 - overlap  # less overlap = more change

    # Check if notation style changes (e.g., fraction vs decimal)
    pre_has_frac = '\\frac' in p or '/' in p
    post_has_frac = '\\frac' in t or '/' in t
    pre_has_decimal = bool(re.search(r'\d+\.\d+', p))
    post_has_decimal = bool(re.search(r'\d+\.\d+', t))
    if pre_has_frac and post_has_decimal and not pre_has_decimal:
        repr_change = max(repr_change, 0.7)
    elif pre_has_decimal and post_has_frac and not pre_has_frac:
        repr_change = max(repr_change, 0.7)

    # 3. Opener change (0.15): different first approach indicator
    opener_change = 0.0
    pre_first = p.strip().split('\n')[0][:80] if p.strip() else ''
    post_first = t.strip().split('\n')[0][:80] if t.strip() else ''
    if pre_first and post_first:
        # Simple word overlap for first line
        pre_words = set(pre_first.split())
        post_words = set(post_first.split())
        if pre_words and post_words:
            overlap = len(pre_words & post_words) / max(len(pre_words | post_words), 1)
            opener_change = 1.0 - overlap

    return method_family * 0.60 + repr_change * 0.25 + opener_change * 0.15


def structural_switch_reward_v2(completions, ground_truth=None, **kwargs):
    """R2v2: Soft structural switch reward with gating.

    Per Codex review:
      - Soft method_diff_score (0-1) instead of binary
      - Gating: meta required, verify is soft multiplier
      - Post-meta length: soft ramp instead of hard threshold
      - Partial credit for attempt even when wrong
    """
    _verify_re = re.compile(
        r"\b(substitut\w*\s+back|plug\w*\s+(back|in)|"
        r"reverse|inverse|check\w*\s+by|boundary|special\s+case|"
        r"sanity\s+check|verify\w*\s+by)\b",
        re.IGNORECASE,
    )
    rewards = []
    for i, c in enumerate(completions):
        text = _get_text(c)
        gt = ground_truth[i] if ground_truth is not None else ""

        # Gate: meta must be present
        meta_start = text.find("<|meta|>")
        meta_end = text.rfind("<|/meta|>")
        if meta_start < 0 or meta_end < 0:
            rewards.append(0.0)
            continue

        # Strip think tags for content analysis
        text_clean = re.sub(r'</?think>', '', text)
        ms = text_clean.find("<|meta|>")
        me = text_clean.rfind("<|/meta|>")
        pre = text_clean[:ms].strip()
        post = text_clean[me + len("<|/meta|>"):].strip()
        # Remove boxed answer from post for content analysis
        post_content = re.sub(r'The answer is.*$', '', post, flags=re.DOTALL).strip()

        if len(pre) < 30:
            rewards.append(0.0)
            continue

        # Soft post-meta length gate: ramp from 5 to 30 math-ish tokens
        math_tokens = len(re.findall(r'[=+\-*/^()]|\d+|\\[a-z]+', post_content))
        length_mult = min(max((math_tokens - 5) / 25.0, 0.0), 1.0)

        if length_mult < 0.01:
            rewards.append(0.0)
            continue

        # Soft verify multiplier
        has_verify = bool(_verify_re.search(post))
        verify_mult = 1.0 if has_verify else 0.6

        # Soft method diff score
        diff_score = _method_diff_score(pre, post_content)

        # Combine: diff_score × length_mult × verify_mult
        raw_score = diff_score * length_mult * verify_mult

        # Scale by correctness
        is_correct = _check_correctness(text, gt)
        if is_correct:
            final = raw_score * 1.0  # full credit
        else:
            final = raw_score * 0.5  # partial credit

        rewards.append(min(final, 1.0))

    return rewards


# Template phrases for verify_outcome_v2 penalty
_VERIFY_TEMPLATES = [
    "let me verify by estimating",
    "sanity check:",
    "reverse verification: starting from",
    "verification: working backwards from",
    "let me verify: substituting",
    "quick check: plugging",
    "cross-check: an alternative approach",
    "confirming:",
    "double-checking: recomputing",
    "testing boundary cases:",
]


def verify_outcome_v2(completions, ground_truth=None, **kwargs):
    """R4v2: Verify reward with template penalty.

    Per Codex review:
      - Penalize formulaic template phrases when no computation present
      - Bonus for actual computation (new equations with prior symbols)
      - Capped in [-0.3, 0.3] for GDPO stability
    """
    _verify_re = re.compile(
        r"\b(substitut\w*\s+back|plug\w*\s+(back|in)|"
        r"reverse|inverse|check\w*\s+by|boundary|special\s+case|"
        r"sanity\s+check|verify\w*\s+by)\b",
        re.IGNORECASE,
    )
    _has_numbers_re = re.compile(r"-?\d[\d,]*\.?\d*\s*[=<>≤≥≠!]|[a-zA-Z]\s*=\s*-?\d")

    rewards = []
    for i, c in enumerate(completions):
        text = _get_text(c)
        gt = ground_truth[i] if ground_truth is not None else ""

        meta_end = text.rfind("<|/meta|>")
        verify_region = text[meta_end:] if meta_end >= 0 else text[-500:]

        if not _verify_re.search(verify_region):
            rewards.append(0.0)
            continue

        has_numerical = bool(_has_numbers_re.search(verify_region))
        is_correct = _check_correctness(text, gt)

        # Template detection
        verify_lower = verify_region.lower()
        is_template = any(t in verify_lower for t in _VERIFY_TEMPLATES)

        if is_template and not has_numerical:
            # Template phrase with no actual computation → penalty
            rewards.append(-0.3)
        elif has_numerical and is_correct:
            rewards.append(0.3)
        elif has_numerical and not is_correct:
            rewards.append(-0.2)
        elif is_template:
            # Template but has some numbers → minimal
            rewards.append(0.05)
        else:
            rewards.append(0.1)  # non-template verify phrase

    return rewards


def confidence_revision_reward_v2(completions, ground_truth=None, **kwargs):
    """Confidence-centered revision reward.

    This is the main controller reward. It prefers:
      - confidence drops when the meta text signals conflict/low-confidence
      - lower final confidence on triggered trajectories
      - calibrated final confidence on non-triggered trajectories
    """
    rewards = []
    for i, c in enumerate(completions):
        text = _get_text(c)
        gt = ground_truth[i] if ground_truth is not None else ""
        is_correct = _check_correctness(text, gt)
        ctx = _redirect_context(text)
        first_conf = ctx["first_conf"]
        last_conf = ctx["last_conf"]
        has_trigger = ctx["has_trigger"]

        if first_conf is None and last_conf is None:
            rewards.append(-0.05)
            continue

        conf = last_conf if last_conf is not None else first_conf
        target = 1.0 if is_correct else 0.0
        base = 1.0 - (conf - target) ** 2

        if has_trigger:
            drop = (
                first_conf is not None
                and last_conf is not None
                and last_conf <= first_conf - 0.08
            )
            low_enough = conf <= 0.55
            has_action = ctx["has_execution"] or ctx["has_next"]
            if drop and has_action:
                rewards.append(min(0.15 + 0.45 * base, 0.6))
            elif drop or (low_enough and has_action):
                rewards.append(min(0.1 + 0.3 * base, 0.4))
            elif low_enough and not has_action:
                # Low confidence but no action: small credit for honesty
                rewards.append(min(0.05 * base, 0.1))
            else:
                rewards.append(max(-0.35 + 0.15 * base, -0.35))
        else:
            rewards.append(max(min(base * 0.2, 0.2), -0.2))
    return rewards


def redirect_execution_reward_v2(completions, ground_truth=None, **kwargs):
    """Reward true redirect execution after confidence-triggered diagnosis.

    A redirect gets credit only if:
      - a trigger exists,
      - the meta contains diagnosis or next-strategy evidence,
      - and the post-meta solve tail structurally changes.
    """
    rewards = []
    for i, c in enumerate(completions):
        text = _get_text(c)
        gt = ground_truth[i] if ground_truth is not None else ""
        is_correct = _check_correctness(text, gt)
        ctx = _redirect_context(text)

        if not ctx["has_trigger"]:
            rewards.append(0.0)
            continue

        if ctx["has_diag"] and ctx["has_next"] and ctx["has_execution"]:
            rewards.append(0.6 if is_correct else 0.2)
        elif ctx["has_diag"] and ctx["has_execution"]:
            rewards.append(0.35 if is_correct else 0.1)
        elif ctx["has_next"] and not ctx["has_execution"]:
            rewards.append(-0.2)
        elif ctx["has_diag"] and not ctx["has_execution"]:
            # Diagnosed but didn't act — mild penalty
            rewards.append(-0.1)
        elif ctx["has_trigger"] and not (ctx["has_diag"] or ctx["has_next"]):
            rewards.append(-0.25)
        else:
            rewards.append(0.0)
    return rewards


def verify_execution_reward_v2(completions, ground_truth=None, **kwargs):
    """Reward verification only for high-confidence overcommit cases.

    This avoids giving dense reward to generic verification on every sample.
    """
    rewards = []
    for i, c in enumerate(completions):
        text = _get_text(c)
        gt = ground_truth[i] if ground_truth is not None else ""
        is_correct = _check_correctness(text, gt)
        meta_text = _meta_joined_text(text)
        solve_tail = _text_after_last_meta(text)
        _, last_conf = _first_and_last_confidence(text)

        if last_conf is None:
            rewards.append(0.0)
            continue

        high_conf = last_conf >= 0.85
        overcommit = _has_overconfidence_signal(meta_text)
        gated = high_conf and overcommit
        has_intent = _has_verification_signal(meta_text) or overcommit
        has_exec = _has_effective_verification_signal(solve_tail)

        if not gated:
            rewards.append(0.0)
        elif has_intent and has_exec:
            rewards.append(0.25 if is_correct else 0.05)
        elif gated and has_intent and not has_exec:
            rewards.append(-0.15 if is_correct else -0.3)
        elif gated and not has_intent and not has_exec and not is_correct:
            rewards.append(-0.2)
        else:
            rewards.append(0.0)
    return rewards


def efficiency_bonus_reward(completions, **kwargs):
    """R_len: Reward efficient solutions (shorter = bonus)."""
    rewards = []
    for c in completions:
        text = _get_text(c)
        ratio = min(len(text) / 8000, 1.0)
        rewards.append(max(0.0, (1.0 - ratio) * 0.1))
    return rewards


# ─── Phase 6 [NEW 2026-04-16]: No-Boxed Commit Penalty ───
#
# Rationale (Meta-CoT V8 plan, Phase 6, H6):
#   `results/aime_failure_analysis_16k/aime_failure_modes.json` shows that at
#   16k max_tokens on AIME, the Meta GRPO (E21R-v2 step 300) model decoheres
#   on 13/26 wrong cases and runs out of tokens without emitting \boxed{} on
#   12/26 wrong cases — only 1/26 commits to a coherent wrong answer. Base
#   GRPO commits to a coherent \boxed{} on 18/19 wrong cases. Meta training
#   teaches verify/redirect/epistemic patterns, which under GRPO on OOD hard
#   problems loop infinitely → no commit → token exhaustion → decoherence.
#
#   H6 claims that the reward surface has no force pushing the model to stop
#   deliberating and write a boxed answer. This penalty is the minimal reward
#   intervention: subtract a fixed amount whenever a completion never emits
#   \boxed, independent of correctness. It is additive and does not modify
#   any existing reward head.
#
# Evidence class: side_evidence (Phase 6 smoke only). Do not mix into
# claim-bearing Phase 5 self-distill tables.


def compute_no_boxed_penalty(completion: str, penalty: float = -0.3) -> float:
    """Penalize completions that never emit ``\\boxed{...}``.

    Scalar variant. Used by Phase 6 ``E21R-v3-smoke`` reward composition to
    push GRPO away from epistemic-loop decoherence on OOD hard problems.

    Args:
        completion: The raw generated text of a single completion.
        penalty: Penalty applied when ``\\boxed`` is absent. Defaults to
            ``-0.3`` per the Phase 6 plan.

    Returns:
        ``penalty`` if ``\\boxed`` is not present in ``completion``,
        otherwise ``0.0``.

    Notes:
        - Detection uses a plain substring check on the literal ``\\boxed``
          prefix so that malformed tails (e.g., ``\\boxed{`` without a
          closing brace) still count as "committed": the goal is to reward
          any commit attempt, not only well-formed commits. Downstream
          correctness reward already handles whether the commit is valid.
        - A ``None`` or empty string is treated as no-commit and receives
          the full penalty.
        - This function is a scalar helper. For the batched reward-function
          signature expected by the reward manager, use
          :func:`no_boxed_penalty_reward` below.

    References:
        Plan section "Phase 6: Decoherence / no-commit fix [NEW 2026-04-16]",
        intervention I1. Evidence: ``results/aime_failure_analysis_16k/
        aime_failure_modes.json`` (2026-04-16).
    """
    if not completion:
        return penalty
    if r"\boxed" not in completion:
        return penalty
    return 0.0


def no_boxed_penalty_reward(completions, ground_truth=None, penalty: float = -0.3, **kwargs):
    """Batched reward wrapper around :func:`compute_no_boxed_penalty`.

    Matches the batched ``(completions, ground_truth=None, **kwargs)``
    signature used by the rest of this module so it can be composed into
    ``src/training/verl_reward.py::compute_score_e21r_v3`` without special
    casing.

    Args:
        completions: Iterable of completion objects (text, dict, or list
            form). Parsed via the module-private ``_get_text``.
        ground_truth: Unused; accepted for signature parity with other
            rewards in this file.
        penalty: Forwarded to :func:`compute_no_boxed_penalty`.

    Returns:
        List of floats, one per completion.
    """
    rewards = []
    for c in completions:
        text = _get_text(c)
        rewards.append(compute_no_boxed_penalty(text, penalty=penalty))
    return rewards


# ====================================================================
# ADDITIVE RESTORE (do not modify above): missing names required by
# src.training.verl_sdc import. Definitions copied byte-identical from
# release snapshot metacognition_v7_2_8 (asset 419914748) rewards.py.
# ====================================================================


def meta_penalty_reward(completions, ground_truth=None, **kwargs):
    """Asymmetric penalty for missing wrapped meta — no positive bonus.

    Unlike `meta_structure_reward` (symmetric ±0.10) which contributed to a
    +0.9 reward floor on the SFT base (saturated meta heads taught the policy
    to emit clean wrong answers), this is penalty-only:

      0.0    wrapped meta block exists  (no positive bias, no reward floor)
     -0.20   no wrapped meta block      (penalty to prevent meta collapse)

    Rationale:
      • Meta region is where the SDC contrastive teacher applies
        (lambda_meta ≠ 0). If the policy stops emitting meta tags, the
        contrastive signal can't apply and the meta scaffold breaks down.
      • Penalty-only avoids the saturation floor problem from prior 5-head
        config: ~95% of SFT-base rollouts emit meta, so penalty fires only
        on the 5% drop-outs and provides asymmetric correction without
        reward inflation.
    """
    rewards = []
    for c in completions:
        text = _get_text(c)
        rewards.append(0.0 if _has_structured_meta(text) else -0.20)
    return rewards
# ===================================================================
# R16: Degeneration penalty (codex-locked V4 spec, 2026-05-12)
#
# Goal: stabilize hard-regime generation so R17 (control-field RLSD +
# follow-through reward) can be measured cleanly. Targets three failure
# modes observed in ROD-PT R14 AIME traces:
#   α — pure LaTeX bracket/backslash collapse tail
#   β — answer-line repetition after \boxed
#   γ — single-character / unicode-fragment repetition tail
#
# Spec finalized via 4-round codex review + data-grounded validation
# on 1030 R14 traces (iamseungpil/metacot:eval/rod_pt_R10_step_100_16k/).
# Acceptance: C1 97.2%, C2 97.7%, C3 11.2%, C4 100% (PASS-PASS-borderline-PASS).
#
# Compose with R17: add r_followthrough on top with its own GDPO key;
# raise λ_repeat inside R17 to penalize "ritual meta" without follow-through.
# r_degen returns single scalar via dedicated GDPO key `degeneration`, weight 0.3.
# ===================================================================

_DEGEN_LAMBDA_REPEAT = 0.45
_DEGEN_LAMBDA_LATEX = 0.35
_DEGEN_LAMBDA_TAIL = 0.15
_DEGEN_LAMBDA_LEN = 0.20
_DEGEN_LAMBDA_REPEAT_HIGH = 0.15
_DEGEN_THETA_TAIL = 0.18
_DEGEN_THETA_REPEAT_HIGH = 0.60
_DEGEN_LMAX_TOKENS = 12500
_DEGEN_CAP = 0.70
_DEGEN_SHORT_PENALTY = 0.25
_DEGEN_SHORT_LEN = 200
_DEGEN_MIN_LEN_FOR_DEGEN = 400
_DEGEN_MATH_RATIO_LOW = 0.24
_DEGEN_MATH_RATIO_HIGH = 0.55
_DEGEN_MATH_TOKEN_TERMS = (
    "\\frac", "\\binom", "\\sqrt", "\\cdot", "\\times", "\\div",
    "\\sum", "\\prod", "\\int", "\\mod", "\\equiv", "\\pmod",
)
_DEGEN_ALLOWED_MATH_UNICODE = set("∀∃∈∉⊂⊃∑∏∫√≤≥≠≈±÷×αβγδεθλμπσφψω°")
_DEGEN_BOXED_RE = re.compile(r"\\boxed\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}")


def _degen_find_tail_window(text, total_chars):
    """Returns (tail_text, tail_start, tail_len). Empty tail signals clean ending."""
    m = _DEGEN_BOXED_RE.search(text)
    if m:
        tail_start = m.end()
        post = text[tail_start:]
        clean = post.strip().strip("$.,;:!?")
        if len(clean) < 80:
            return "", tail_start, 0
        cap = min(4096, max(512, len(post)))
        return post[:cap], tail_start, min(cap, len(post))
    if total_chars >= 800:
        return text[-800:], total_chars - 800, 800
    return text, 0, total_chars


def _degen_math_ratio(tail):
    if not tail:
        return 0.0
    n = len(tail)
    digits = sum(1 for c in tail if c.isdigit())
    ops = sum(tail.count(o) for o in ("=", "+", "-", "*", "^", "_"))
    math_terms = sum(tail.count(t) for t in _DEGEN_MATH_TOKEN_TERMS) * 4
    return (digits + ops + math_terms) / n


def _degen_math_gate(mr):
    if mr <= _DEGEN_MATH_RATIO_LOW:
        return 1.0
    if mr >= _DEGEN_MATH_RATIO_HIGH:
        return 0.0
    return 1.0 - (mr - _DEGEN_MATH_RATIO_LOW) / (_DEGEN_MATH_RATIO_HIGH - _DEGEN_MATH_RATIO_LOW)


def _degen_garbage_components(tail):
    n = max(len(tail), 1)
    backslash = tail.count("\\") / n
    brackets = sum(tail.count(c) for c in "[](){}") / n
    tokens = tail.split()
    short_tokens = [t for t in tokens if 1 <= len(t) <= 3]
    repeat_short = (len(short_tokens) - len(set(short_tokens))) / max(len(tokens), 1) if tokens else 0.0
    non_alnum = sum(1 for c in tail if not c.isalnum() and not c.isspace()) / n
    weird_unicode = 1.0 if any(
        ord(c) > 127 and not c.isspace() and c not in _DEGEN_ALLOWED_MATH_UNICODE for c in tail
    ) else 0.0
    return backslash, brackets, repeat_short, non_alnum, weird_unicode


def _degen_bigram_repeat(tail):
    tokens = tail.split()
    if len(tokens) < 2:
        return 0.0
    bigrams = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
    return (len(bigrams) - len(set(bigrams))) / len(bigrams)


def compute_degeneration_penalty(completion_text, completion_len_tokens, answer_extracted):
    """R16 degeneration penalty (single-sample, returns negative scalar in [-CAP, 0]).

    Returns
    -------
    (penalty: float, breakdown: dict)
        penalty <= 0; breakdown carries diagnostic fields for logging.

    Spec: post-boxed tail with length-scaled cap, math-gated structural sum,
    bigram repeat (continuous + binary bump), length penalty, short-truncation
    guard. See module docstring above for full provenance.
    """
    br = {
        "tail_garbage_score": 0.0, "math_ratio": 0.0, "math_gate": 1.0,
        "bigram_repeat": 0.0, "p_repeat": 0.0, "p_latex": 0.0,
        "p_tail": 0.0, "p_len": 0.0, "p_repeat_high": 0.0, "p_short": 0.0,
        "triggered": False,
    }

    # Short-truncation: model gave up before reaching an answer. Empty string
    # also counts as "no answer" (the fallback extractor returns '' rather
    # than None when no \boxed match is found).
    if completion_len_tokens < _DEGEN_SHORT_LEN and not answer_extracted:
        br["p_short"] = _DEGEN_SHORT_PENALTY
        br["triggered"] = True
        return -_DEGEN_SHORT_PENALTY, br

    if completion_len_tokens < _DEGEN_MIN_LEN_FOR_DEGEN:
        return 0.0, br

    total_chars = len(completion_text)
    tail, _, tlen = _degen_find_tail_window(completion_text, total_chars)

    p_len = _DEGEN_LAMBDA_LEN if completion_len_tokens > _DEGEN_LMAX_TOKENS else 0.0
    br["p_len"] = p_len

    if tlen == 0:
        total = min(p_len, _DEGEN_CAP)
        br["triggered"] = total > 0
        return -total, br

    mr = _degen_math_ratio(tail)
    gate = _degen_math_gate(mr)
    br["math_ratio"] = mr
    br["math_gate"] = gate

    bs, brc, rs, na, weird = _degen_garbage_components(tail)
    tgs = gate * (bs + brc + rs + na) + weird
    br["tail_garbage_score"] = tgs

    bgr = _degen_bigram_repeat(tail)
    br["bigram_repeat"] = bgr

    p_repeat = _DEGEN_LAMBDA_REPEAT * bgr
    p_latex = _DEGEN_LAMBDA_LATEX * tgs
    p_tail = _DEGEN_LAMBDA_TAIL if tgs > _DEGEN_THETA_TAIL else 0.0
    p_repeat_high = _DEGEN_LAMBDA_REPEAT_HIGH if bgr >= _DEGEN_THETA_REPEAT_HIGH else 0.0

    br["p_repeat"] = p_repeat
    br["p_latex"] = p_latex
    br["p_tail"] = p_tail
    br["p_repeat_high"] = p_repeat_high

    total = min(p_repeat + p_latex + p_tail + p_repeat_high + p_len, _DEGEN_CAP)
    br["triggered"] = total > 0
    return -total, br


def degeneration_penalty_reward(completions, ground_truth=None, **kwargs):
    """Batched reward wrapper for compute_degeneration_penalty.

    Matches the (completions, ground_truth=None, **kwargs) signature used by
    the rest of this module. ``kwargs`` may contain ``completion_lengths`` and
    ``answer_extracted`` lists; if absent, falls back to safe defaults
    (length = len(text.split()), answer = None).
    """
    completion_lengths = kwargs.get("completion_lengths")
    answers_extracted = kwargs.get("answer_extracted") or kwargs.get("answers_extracted")
    rewards = []
    for idx, c in enumerate(completions):
        text = _get_text(c)
        if completion_lengths and idx < len(completion_lengths):
            length = int(completion_lengths[idx])
        else:
            length = len(text.split())
        ans = None
        if answers_extracted and idx < len(answers_extracted):
            ans = answers_extracted[idx]
        penalty, _ = compute_degeneration_penalty(text, length, ans)
        rewards.append(penalty)
    return rewards


def meta_penalty_adaptive_reward(completions, ground_truth=None, **kwargs):
    """Adaptive variant of meta_penalty_reward.

    ADDITIVE RESTORE: this name is imported by src.training.verl_sdc but is
    absent from both the committed module and the release snapshot. It is
    defined here as a penalty-only meta head matching meta_penalty_reward's
    contract (0.0 when a structured meta block is present, -0.20 otherwise),
    so the import resolves and the reward semantics stay consistent with the
    non-adaptive head. Signature mirrors the rest of this module:
    (completions, ground_truth=None, **kwargs) -> list[float].
    """
    return meta_penalty_reward(completions, ground_truth=ground_truth, **kwargs)
