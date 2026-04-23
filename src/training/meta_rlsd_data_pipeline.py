"""Meta-RLSD data pipeline — preflight checks + dataset loader + meta-mask.

Implements §2.10 "Data pre-flight checks" and §2.1 meta-mask construction from
``results/plan_meta_rlsd_v2_2026_04_17.md``.

Design notes (modular plug-and-play):
    * No mutation of existing modules. Reuses:
        - ``src.metacot.prompt.META_START / META_END / parse_meta_blocks``
        - ``src.training.self_distill.kl._assistant_offsets``
        - ``src.training.tokenizer_utils.ensure_meta_tokens_not_special``
    * Exposes three public callables:
        - :func:`preflight_checks`
        - :func:`load_meta_rlsd_dataset`
        - :func:`build_meta_mask` (thin wrapper around private helper)

The meta-mask operates on the *completion* (assistant) token span only. The caller
is responsible for passing the decoded completion text that corresponds to the
same token ids so character offsets align with the mask tensor.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Sequence

import pandas as pd

try:  # torch is a hard runtime dep for the trainer but keep the import light
    import torch
except ImportError:  # pragma: no cover — torch absent == the file will never be used
    torch = None  # type: ignore[assignment]

from datasets import Dataset

from src.metacot.prompt import META_END, META_START, parse_meta_blocks

_META_BLOCK_RE = re.compile(
    rf"{re.escape(META_START)}(.*?){re.escape(META_END)}",
    re.DOTALL | re.IGNORECASE,
)


def _assistant_offsets(tokenizer, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    """Return token ids + char offsets without depending on self-distill package init.

    veRL SDC only needs a tokenizer-to-char alignment helper. Importing
    ``src.training.self_distill.kl`` triggers the full package ``__init__``,
    which in turn pulls optional research dependencies on cluster nodes.
    Keeping the narrow helper local avoids unrelated import-time failures.
    """
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        input_ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]
        return [int(x) for x in input_ids], [(int(s), int(e)) for s, e in offsets]
    except Exception:
        encoded = tokenizer(text, add_special_tokens=False)
        input_ids = encoded["input_ids"]
        offsets: list[tuple[int, int]] = []
        cursor = 0
        for token_id in input_ids:
            piece = tokenizer.decode([int(token_id)], skip_special_tokens=False)
            piece = piece.replace(" ", "")
            if not piece:
                offsets.append((cursor, cursor))
                continue
            start = text.find(piece, cursor)
            if start < 0:
                start = cursor
            end = min(len(text), start + len(piece))
            offsets.append((start, end))
            cursor = end
        return [int(x) for x in input_ids], offsets


# ─── Types ─────────────────────────────────────────────────────────────────

@dataclass
class PFReport:
    """Pre-flight inspection outcome — see §2.10."""

    passed: bool
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def format_report(self) -> str:
        lines = [f"PFReport(passed={self.passed})"]
        if self.violations:
            lines.append("  VIOLATIONS:")
            for v in self.violations:
                lines.append(f"    - {v}")
        if self.warnings:
            lines.append("  WARNINGS:")
            for w in self.warnings:
                lines.append(f"    - {w}")
        if self.stats:
            lines.append("  STATS:")
            for k, v in self.stats.items():
                lines.append(f"    {k}: {v}")
        return "\n".join(lines)


# ─── Dataset loader ────────────────────────────────────────────────────────

def _extract_prompt_messages(raw: Any) -> List[dict]:
    """Normalize parquet ``prompt`` column into chat-template list[dict]."""
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return [{"role": "user", "content": raw}]
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    if isinstance(raw, dict):
        return [raw]
    if isinstance(raw, (list, tuple)):
        normalized: List[dict] = []
        for msg in raw:
            if isinstance(msg, dict):
                normalized.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", ""),
                })
            else:
                normalized.append({"role": "user", "content": str(msg)})
        return normalized
    return [{"role": "user", "content": str(raw)}]


def _extract_ground_truth(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        # May be JSON-encoded dict or plain string
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            return raw
        if isinstance(obj, dict):
            return str(obj.get("ground_truth", ""))
        return str(obj)
    if hasattr(raw, "item") and not isinstance(raw, (list, tuple, dict)):
        try:
            raw = raw.item()
        except Exception:  # pragma: no cover
            pass
    if isinstance(raw, dict):
        return str(raw.get("ground_truth", ""))
    return str(raw)


def _user_content_of(prompt_messages: Sequence[dict]) -> str:
    """Return the last user-role content string; fallback to first message."""
    for msg in reversed(prompt_messages):
        if msg.get("role") == "user":
            return str(msg.get("content", ""))
    if prompt_messages:
        return str(prompt_messages[0].get("content", ""))
    return ""


def load_meta_rlsd_dataset(parquet_path: str) -> Dataset:
    """Load parquet into HF ``Dataset`` with normalized ``prompt`` / ``ground_truth``.

    Expected columns (verl-style): ``prompt`` (list[dict]), ``reward_model``
    (dict with ``ground_truth``). Extra columns are preserved verbatim.
    """
    df = pd.read_parquet(parquet_path)
    records: List[dict] = []
    for _, row in df.iterrows():
        prompt_messages = _extract_prompt_messages(row.get("prompt"))
        gt = _extract_ground_truth(row.get("reward_model"))
        records.append({
            "prompt": prompt_messages,
            "ground_truth": gt,
            "problem": _user_content_of(prompt_messages),
        })
    return Dataset.from_list(records)


# ─── Meta-mask construction — §2.1 m_t ─────────────────────────────────────

def _manual_offset_scan(
    tokenizer,
    completion_ids: Sequence[int],
    completion_text: str,
) -> List[tuple]:
    """Fallback offset scan for added-vocab tokens (C11).

    HF fast tokenizers can return ``(0, 0)`` offsets for tokens that were added
    after the tokenizer was trained (e.g., ``<|meta|>``/``<|/meta|>``). A (0,0)
    offset on a non-leading token would silently zero the mask for that token.
    This helper walks the completion text with a char cursor and matches each
    decoded piece to produce usable offsets.
    """
    # Normalize to a list of ints regardless of input container type.
    if hasattr(completion_ids, "tolist"):
        ids_iter = completion_ids.tolist()
    else:
        ids_iter = list(completion_ids)

    offsets: List[tuple] = []
    cursor = 0
    for tid in ids_iter:
        piece = tokenizer.decode([int(tid)], skip_special_tokens=False)
        if not piece:
            offsets.append((cursor, cursor))
            continue
        idx = completion_text.find(piece, cursor)
        if idx < 0:
            # Piece couldn't be located (e.g., whitespace normalization); keep cursor.
            offsets.append((cursor, cursor))
            continue
        end = idx + len(piece)
        offsets.append((idx, end))
        cursor = end
    return offsets


def _build_meta_mask(
    tokenizer,
    completion_ids: Sequence[int],
    completion_text: str,
):
    """Return a 1-D ``torch.Tensor`` of 0/1 marking tokens inside <|meta|> blocks.

    Args:
        tokenizer: HF tokenizer with ``return_offsets_mapping`` support (fast
            variant preferred).
        completion_ids: the exact token ids whose mask we want (length ``T``).
            Used only for its length — the offset map is derived from
            ``completion_text`` so it must be the decoded version of these ids.
        completion_text: ``tokenizer.decode(completion_ids, skip_special_tokens=False)``.

    Returns:
        ``torch.Tensor`` shape ``[T]`` dtype ``float32`` with values in {0., 1.}.

    Notes:
        * Re-uses ``kl._assistant_offsets`` to handle slow-tokenizer fallback.
        * If the re-tokenization yields a different number of tokens than
          ``completion_ids`` we align by index up to the shorter of the two and
          pad with zeros — this matches the "meta_only" convention used by
          ``kl.build_control_span_weights``.
        * C11 fix: HF fast tokenizers emit ``(0, 0)`` offsets for added-vocab
          tokens (e.g., ``<|meta|>``). If more than 5 % of non-leading tokens
          have ``(0, 0)`` offsets, we fall back to a manual piece-scan aligned
          against ``completion_ids`` so the mask covers the actual meta span.
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required for _build_meta_mask")

    expected_len = len(completion_ids)
    if expected_len == 0:
        return torch.zeros(0, dtype=torch.float32)

    _, offsets = _assistant_offsets(tokenizer, completion_text)

    # C11: detect suspicious (0,0) offset rate and fall back to manual scan.
    if offsets:
        suspicious = sum(
            1
            for i, (s, e) in enumerate(offsets)
            if i > 0 and s == 0 and e == 0
        )
        if suspicious > max(1, int(len(offsets) * 0.05)):
            offsets = _manual_offset_scan(tokenizer, completion_ids, completion_text)

    mask = [0.0] * len(offsets)

    for match in _META_BLOCK_RE.finditer(completion_text):
        char_start, char_end = match.start(), match.end()
        if char_end <= char_start:
            continue
        for token_idx, (tok_s, tok_e) in enumerate(offsets):
            if tok_e <= char_start or tok_s >= char_end:
                continue
            mask[token_idx] = 1.0

    if len(mask) < expected_len:
        mask.extend([0.0] * (expected_len - len(mask)))
    mask_tensor = torch.tensor(mask[:expected_len], dtype=torch.float32)

    # Sanity self-check: if completion text contains <|meta|> but mask is empty,
    # the offset alignment failed — warn (mask stays zero; reward's meta floor
    # would then penalize as if no meta were present, which masks the bug).
    if META_START in completion_text and float(mask_tensor.sum().item()) == 0.0:
        print(
            "[WARN] meta_mask alignment failed: completion text has "
            f"{META_START!r} but mask is all zeros (len={expected_len})"
        )

    return mask_tensor


def build_meta_mask(tokenizer, completion_ids, completion_text):  # public alias
    """Public wrapper — see :func:`_build_meta_mask`."""
    return _build_meta_mask(tokenizer, completion_ids, completion_text)


# ─── SDC post-meta mask — plan_SDC_v2 §3.2 ─────────────────────────────────

# Matches first ``\boxed{...}`` group; captures brace content for payload.
# Supports up to 2 levels of nested braces inside (common in LaTeX fractions),
# matching the same convention as ``_BOXED_RE`` below in the preflight section.
# Supports up to 2 levels of nested braces (e.g., \boxed{\frac{a}{b}}). Deeper
# nesting (3+ levels like \boxed{\frac{\sqrt{x}}{b}}) may miscapture closing
# brace; fallback to end-of-completion if this happens.
_BOXED_CAPTURE_RE = re.compile(
    r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"
)


def _build_postmeta_mask(
    tokenizer,
    completion_ids: Sequence[int],
    completion_text: str,
    meta_mask,
):
    """Return a 1-D tensor marking tokens in the post-meta region (SDC §3.2).

    Post-meta region = tokens after the **last** ``<|/meta|>`` through the
    closing ``}`` of the first subsequent ``\\boxed{...}`` (inclusive).

    **Fallback** (when ``\\boxed{}`` missing after final ``<|/meta|>``): region
    extends from end-of-``<|/meta|>`` to end-of-completion.

    The returned mask is disjoint from ``meta_mask`` by construction — meta
    tokens are removed from the postmeta span so the three-region partition
    ``meta + post + body == 1`` holds element-wise (plan §3.1).

    Args:
        tokenizer: HF tokenizer (used for offset map via ``_assistant_offsets``).
        completion_ids: token ids whose mask we want (length ``T``).
        completion_text: ``tokenizer.decode(completion_ids, skip_special_tokens=False)``.
        meta_mask: the already-computed meta mask (same shape as return value) —
            used to enforce disjointness.

    Returns:
        Tuple ``(mask_tensor, fallback_triggered)`` where ``mask_tensor`` is a
        float32 tensor of shape ``[T]`` in {0., 1.} and ``fallback_triggered``
        is a bool indicating whether the ``\\boxed{}`` boundary was missing
        (plan §3.2 logging contract).
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required for _build_postmeta_mask")

    expected_len = len(completion_ids)
    if expected_len == 0:
        return torch.zeros(0, dtype=torch.float32), False

    _, offsets = _assistant_offsets(tokenizer, completion_text)

    # Same (0,0) suspicion guard as _build_meta_mask.
    if offsets:
        suspicious = sum(
            1 for i, (s, e) in enumerate(offsets)
            if i > 0 and s == 0 and e == 0
        )
        if suspicious > max(1, int(len(offsets) * 0.05)):
            offsets = _manual_offset_scan(tokenizer, completion_ids, completion_text)

    mask = [0.0] * len(offsets)

    # Locate LAST <|/meta|> in the completion. SDC §3.2 uses the final closing
    # tag so nested/multiple meta blocks still yield a single post-meta region.
    meta_end_positions = [m.end() for m in re.finditer(
        re.escape(META_END), completion_text
    )]
    fallback_triggered = False
    if not meta_end_positions:
        # No meta end tag → no post-meta region by this definition.
        # Return all-zero mask; fallback is not "triggered" in the §3.2 sense
        # (that counter tracks "\boxed{} missing after <|/meta|>").
        if len(mask) < expected_len:
            mask.extend([0.0] * (expected_len - len(mask)))
        return torch.tensor(mask[:expected_len], dtype=torch.float32), False

    post_start = meta_end_positions[-1]

    # Find first \boxed{...} at or after post_start.
    boxed_match = _BOXED_CAPTURE_RE.search(completion_text, post_start)
    if boxed_match is not None:
        post_end = boxed_match.end()  # includes the closing '}'
    else:
        fallback_triggered = True
        post_end = len(completion_text)

    if post_end <= post_start:
        # Degenerate (shouldn't happen — len of completion >= meta_end).
        if len(mask) < expected_len:
            mask.extend([0.0] * (expected_len - len(mask)))
        return torch.tensor(mask[:expected_len], dtype=torch.float32), fallback_triggered

    for token_idx, (tok_s, tok_e) in enumerate(offsets):
        # Skip degenerate zero-width offsets.
        if tok_e <= tok_s:
            continue
        # Any overlap with [post_start, post_end) counts.
        if tok_e <= post_start or tok_s >= post_end:
            continue
        mask[token_idx] = 1.0

    if len(mask) < expected_len:
        mask.extend([0.0] * (expected_len - len(mask)))
    mask_tensor = torch.tensor(mask[:expected_len], dtype=torch.float32)

    # Enforce disjointness with meta_mask (plan §3.1 invariant).
    if meta_mask is not None:
        mm = meta_mask
        if hasattr(mm, "to"):
            mm = mm.to(mask_tensor.device).float()
        # Clip to same length in case callers passed a longer meta_mask.
        mm = mm[: mask_tensor.size(0)]
        mask_tensor = mask_tensor * (1.0 - mm)
        mask_tensor = torch.clamp(mask_tensor, 0.0, 1.0)

    return mask_tensor, fallback_triggered


def build_postmeta_mask(tokenizer, completion_ids, completion_text, meta_mask):
    """Public wrapper — see :func:`_build_postmeta_mask`."""
    return _build_postmeta_mask(tokenizer, completion_ids, completion_text, meta_mask)


# ─── Pre-flight checks — §2.10 ─────────────────────────────────────────────

_BOXED_RE = re.compile(r'\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}')


def _normalize_for_boxed(gold: str) -> str:
    """Strip leading/trailing whitespace and surrounding $ or \\boxed wrapper."""
    s = gold.strip()
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()
    m = _BOXED_RE.search(s)
    if m:
        s = m.group(1).strip()
    return s


def _boxed_extractable(gold: str) -> bool:
    """PF2: whether ``\\boxed{gold}`` is a valid LaTeX-balanced pattern."""
    if not gold:
        return False
    normalized = _normalize_for_boxed(gold)
    if not normalized:
        return False
    # Sanity: must not contain unbalanced braces that would break \boxed{...}
    depth = 0
    for ch in normalized:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _tokenize_len(tokenizer, text: str) -> int:
    try:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])
    except Exception:
        return len(tokenizer.encode(text, add_special_tokens=False))


def preflight_checks(
    parquet_path: str,
    tokenizer,
    *,
    prompt_length: int = 2048,
    meta_min_length_tokens: int = 20,
    sample_size: int | None = None,
) -> PFReport:
    """Run PF1–PF5 from §2.10. Abort training if ``report.passed`` is False.

    Args:
        parquet_path: verl-format parquet, columns ``prompt`` + ``reward_model``.
        tokenizer: loaded tokenizer (meta tokens already registered).
        prompt_length: hard limit for PF3 (§2.7 ``prompt_length``).
        meta_min_length_tokens: threshold used by the reward, surfaced here for
            the PF5 informational warning only.
        sample_size: if set, only check first ``n`` rows (debug). Default
            ``None`` → full scan.
    """
    report = PFReport(passed=True)

    try:
        df = pd.read_parquet(parquet_path)
    except Exception as exc:
        report.passed = False
        report.violations.append(f"Failed to read parquet {parquet_path!r}: {exc}")
        return report

    if sample_size is not None:
        df = df.head(sample_size)

    n = len(df)
    report.stats["rows"] = n

    # PF1 + PF2 + PF3 + PF5 (row-level)
    empty_gold = 0
    non_boxed = 0
    overflow = 0
    meta_rows = 0
    gold_samples: List[str] = []

    for _, row in df.iterrows():
        prompt_messages = _extract_prompt_messages(row.get("prompt"))
        gt = _extract_ground_truth(row.get("reward_model"))
        problem = _user_content_of(prompt_messages)

        # PF1
        if not gt or not str(gt).strip():
            empty_gold += 1
            continue

        # PF2 — accept \boxed{ground_truth} pattern
        if not _boxed_extractable(gt):
            non_boxed += 1

        # PF3 — tokenized length bound
        teacher_text = f"{problem} Answer: {gt}"
        if _tokenize_len(tokenizer, teacher_text) > prompt_length:
            overflow += 1

        # PF5 tally — meta rate in train data (informational)
        for msg in prompt_messages:
            content = msg.get("content", "") if isinstance(msg, dict) else ""
            if META_START in content and META_END in content:
                meta_rows += 1
                break
        else:
            parsed = parse_meta_blocks(problem, allow_free_text_fallback=False)
            if parsed.get("num_blocks", 0) > 0:
                meta_rows += 1

        if len(gold_samples) < 5:
            gold_samples.append(str(gt))

    report.stats["empty_gold"] = empty_gold
    report.stats["non_boxed"] = non_boxed
    report.stats["overflow"] = overflow
    report.stats["meta_rows"] = meta_rows
    report.stats["gold_samples"] = gold_samples

    # PF1 — strict
    if empty_gold > 0:
        report.passed = False
        report.violations.append(
            f"PF1 FAIL: {empty_gold}/{n} rows have empty gold_answer"
        )

    # PF2 — ≥99 % extractable
    boxed_rate = 1.0 - (non_boxed / max(n, 1))
    report.stats["boxed_rate"] = round(boxed_rate, 4)
    if boxed_rate < 0.99:
        report.passed = False
        report.violations.append(
            f"PF2 FAIL: \\boxed-extractable rate {boxed_rate:.3%} < 99%"
        )

    # PF3 — strict
    if overflow > 0:
        report.passed = False
        report.violations.append(
            f"PF3 FAIL: {overflow}/{n} rows exceed prompt_length={prompt_length}"
        )

    # PF4 — tokenizer vocab presence of meta tokens (include added_tokens)
    vocab = tokenizer.get_vocab()
    added = {}
    try:
        added = tokenizer.get_added_vocab()
    except Exception:
        pass
    combined_vocab = {**vocab, **added}
    missing = [t for t in (META_START, META_END) if t not in combined_vocab]
    if missing:
        # Also accept if tokenize(token) produces a single non-UNK token id
        accepted = []
        for t in missing:
            try:
                ids = tokenizer(t, add_special_tokens=False)["input_ids"]
                if len(ids) >= 1 and ids[0] != tokenizer.unk_token_id:
                    accepted.append(t)
            except Exception:
                pass
        missing = [t for t in missing if t not in accepted]
    if missing:
        report.passed = False
        report.violations.append(
            f"PF4 FAIL: meta tokens missing from vocab: {missing}"
        )

    # PF5 — informational only
    meta_rate = meta_rows / max(n, 1)
    report.stats["train_meta_rate"] = round(meta_rate, 4)
    if meta_rate == 0.0:
        report.warnings.append(
            "PF5 NOTE: 0% meta in train data. OK because Meta-RLSD uses fresh "
            "rollouts — prompts are math problems, not meta examples."
        )

    # Expose threshold for visibility
    report.stats["meta_min_length_tokens"] = meta_min_length_tokens
    return report


__all__ = [
    "PFReport",
    "preflight_checks",
    "load_meta_rlsd_dataset",
    "build_meta_mask",
    "_build_meta_mask",
    "build_postmeta_mask",
    "_build_postmeta_mask",
]
