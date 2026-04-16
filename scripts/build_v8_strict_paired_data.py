#!/usr/bin/env python3
"""Build strict paired Meta/Base SFT data from the current V8 corpora.

Goal:
  - keep the exact same problem slice for paired comparison
  - enforce strong verify vs redirect semantics on the meta corpus
  - rebuild the base corpus as direct one-pass reasoning targets

Outputs:
  - data/v8_meta_inside_strict.parquet
  - data/v8_base_matched_strict.parquet
  - optional JSON summary
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd


META_BLOCK_RE = re.compile(r"<\|meta\|>\s*(.*?)\s*<\|/meta\|>", re.DOTALL)
THINK_RE = re.compile(r"^\s*<think>\s*(.*?)\s*</think>\s*(.*)\s*$", re.DOTALL)
CONF_RE = re.compile(r"confidence:\s*([0-9]*\.?[0-9]+)")
BLANK_INLINE_MATH_RE = re.compile(r"\\\(\s*\\\)", re.DOTALL)

VERIFY_LINE_PATTERNS = [
    re.compile(
        r"^\s*(verification|verify|quick check|cross-check|double-checking|"
        r"confirming|reverse verification|sanity check|checking with|"
        r"testing boundary cases|numerical spot-check|dimensional analysis|"
        r"parity check|let me verify)\b",
        re.IGNORECASE,
    ),
]
REDIRECT_TRANSITION_PATTERNS = [
    re.compile(r"^\s*(now switch to|i should switch to|switch to|now use)\b", re.IGNORECASE),
    re.compile(r"^\s*(this means i should|instead,)\b", re.IGNORECASE),
]
BASE_ROUTE_PATTERNS = [
    re.compile(r"^\s*(a first thought is|a tempting first thought is|at first glance)\b", re.IGNORECASE),
    re.compile(r"^\s*(i might try|one might try|the initial route is weak)\b", re.IGNORECASE),
    re.compile(r"^\s*(i should switch|i should redirect|what is missing is)\b", re.IGNORECASE),
]
VERIFY_META_PATTERNS = [
    "action:",
    "verify",
    "double-check",
    "cross-check",
    "substitute",
]
BASE_FORBIDDEN_SUBSTRINGS = [
    "a first thought is",
    "a tempting first thought is",
    "at first glance",
    "i might try",
    "one might try",
    "i should switch",
    "i should redirect",
    "what is missing is",
    "study_need:",
]
REDIRECT_SWITCH_PATTERNS = [
    "i should switch",
    "switch to",
    "redirect",
]
REDIRECT_DIAGNOSIS_PATTERNS = [
    "what is missing",
    "study_need:",
    "the current route is weak",
]


def load_messages(raw_messages) -> list[dict]:
    if isinstance(raw_messages, list):
        return raw_messages
    if isinstance(raw_messages, str):
        try:
            return json.loads(raw_messages)
        except json.JSONDecodeError:
            return ast.literal_eval(raw_messages)
    raise TypeError(f"Unsupported messages type: {type(raw_messages)!r}")


def dump_messages(messages: list[dict]) -> str:
    return json.dumps(messages, ensure_ascii=False)


def extract_last_boxed(text: str) -> str | None:
    starts = [m.start() for m in re.finditer(r"\\boxed\{", text)]
    if not starts:
        return None
    start = starts[-1]
    brace_pos = start + len("\\boxed")
    depth = 0
    started = False
    for idx in range(brace_pos, len(text)):
        ch = text[idx]
        if ch == "{":
            depth += 1
            started = True
        elif ch == "}":
            depth -= 1
            if started and depth == 0:
                return text[start : idx + 1]
    return None


def normalize_ws(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_artifact_lines(text: str) -> str:
    text = BLANK_INLINE_MATH_RE.sub("__BLANK_INLINE_MATH__", text)
    kept: list[str] = []
    for line in text.splitlines():
        if "__BLANK_INLINE_MATH__" in line:
            continue
        kept.append(line.rstrip())
    return normalize_ws("\n".join(kept))


def parse_assistant(assistant_text: str) -> tuple[str, str, list[str]]:
    match = THINK_RE.match(assistant_text)
    if not match:
        raise ValueError("assistant missing canonical <think>...</think> envelope")
    think_body = normalize_ws(match.group(1))
    outside = normalize_ws(match.group(2))
    metas = META_BLOCK_RE.findall(think_body)
    return think_body, outside, metas


def split_reasoning_on_meta(think_body: str) -> list[str]:
    segments: list[str] = []
    last_end = 0
    for match in META_BLOCK_RE.finditer(think_body):
        segment = normalize_ws(think_body[last_end : match.start()])
        if segment:
            segments.append(segment)
        last_end = match.end()
    tail = normalize_ws(think_body[last_end:])
    if tail:
        segments.append(tail)
    return segments


def contains_any(text: str, needles: Iterable[str]) -> bool:
    lower = text.lower()
    return any(needle in lower for needle in needles)


def is_strong_verify(assistant_text: str) -> bool:
    think_body, outside, metas = parse_assistant(assistant_text)
    if len(metas) != 1 or not metas[0].strip():
        return False
    if "<|meta|>" in outside or "<|/meta|>" in outside:
        return False
    boxed = extract_last_boxed(outside)
    if boxed is None:
        return False
    confs = [float(x) for x in CONF_RE.findall(metas[0])]
    if not confs or confs[0] < 0.65:
        return False
    meta_text = metas[0].lower()
    if "study_need:" in meta_text or "what is missing" in meta_text:
        return False
    if not contains_any(meta_text, VERIFY_META_PATTERNS):
        return False
    return True


def is_strong_redirect(assistant_text: str) -> bool:
    think_body, outside, metas = parse_assistant(assistant_text)
    if not (1 <= len(metas) <= 2):
        return False
    if any(not meta.strip() for meta in metas):
        return False
    if "<|meta|>" in outside or "<|/meta|>" in outside:
        return False
    if extract_last_boxed(outside) is None:
        return False
    if assistant_text.count("A first thought is") >= 2:
        return False
    confs = [float(x) for x in CONF_RE.findall(think_body)]
    if not confs or min(confs) > 0.45:
        return False
    meta_text = " ".join(metas).lower()
    if not contains_any(meta_text, REDIRECT_SWITCH_PATTERNS):
        return False
    if not contains_any(meta_text, REDIRECT_DIAGNOSIS_PATTERNS):
        return False
    return True


def should_keep_row(row: pd.Series, assistant_text: str) -> bool:
    scenario = str(row.get("scenario", "")).strip().lower()
    if scenario == "verify":
        return is_strong_verify(assistant_text)
    if scenario == "redirect":
        return is_strong_redirect(assistant_text)
    return False


def filter_lines(lines: Iterable[str], *, scenario: str) -> list[str]:
    kept: list[str] = []
    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            if kept and kept[-1] != "":
                kept.append("")
            continue
        if scenario == "verify":
            if any(pattern.search(stripped) for pattern in VERIFY_LINE_PATTERNS):
                continue
        if scenario == "redirect":
            if any(pattern.search(stripped) for pattern in REDIRECT_TRANSITION_PATTERNS):
                continue
            if any(pattern.search(stripped) for pattern in BASE_ROUTE_PATTERNS):
                continue
            lower_line = line.lower()
            cut_positions = [
                lower_line.find(token)
                for token in BASE_FORBIDDEN_SUBSTRINGS
                if token in lower_line
            ]
            if cut_positions:
                cut_at = min(pos for pos in cut_positions if pos >= 0)
                prefix = line[:cut_at].rstrip()
                if prefix:
                    kept.append(prefix)
                continue
        kept.append(line)
    while kept and kept[0] == "":
        kept.pop(0)
    while kept and kept[-1] == "":
        kept.pop()
    return kept


def dedupe_consecutive_blocks(text: str) -> str:
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    deduped: list[str] = []
    for block in blocks:
        if deduped and deduped[-1] == block:
            continue
        deduped.append(block)
    return "\n\n".join(deduped)


def build_base_assistant(meta_assistant: str, scenario: str) -> str:
    think_body, outside, _ = parse_assistant(meta_assistant)
    boxed = extract_last_boxed(outside)
    if boxed is None:
        raise ValueError("missing boxed answer outside think")

    raw_segments = split_reasoning_on_meta(think_body)
    if not raw_segments:
        raise ValueError("no reasoning segments found")

    if scenario == "verify":
        candidate_segments = raw_segments
    elif scenario == "redirect":
        candidate_segments = raw_segments[1:] if len(raw_segments) > 1 else raw_segments
    else:
        candidate_segments = raw_segments

    lines: list[str] = []
    for segment in candidate_segments:
        lines.extend(segment.splitlines())
    filtered = filter_lines(lines, scenario=scenario)
    if not filtered:
        filtered = filter_lines(
            "\n\n".join(raw_segments).splitlines(),
            scenario="verify" if scenario == "verify" else "redirect",
        )
    base_think = normalize_ws("\n".join(filtered))
    if not base_think:
        raise ValueError("empty base reasoning after strict rewrite")
    base_think = dedupe_consecutive_blocks(base_think)
    base_think = remove_artifact_lines(base_think)
    if not base_think:
        raise ValueError("empty base reasoning after artifact cleanup")

    return f"<think>\n{base_think}\n</think>\n\nThe answer is ${boxed}$."


def sanitize_meta_assistant(meta_assistant: str) -> str:
    think_body, outside, _ = parse_assistant(meta_assistant)
    boxed = extract_last_boxed(outside)
    if boxed is None:
        raise ValueError("missing boxed answer outside think")

    rebuilt: list[str] = []
    last_end = 0
    for match in META_BLOCK_RE.finditer(think_body):
        reasoning = remove_artifact_lines(think_body[last_end : match.start()])
        if reasoning:
            rebuilt.append(reasoning)
        rebuilt.append(match.group(0).strip())
        last_end = match.end()
    tail = remove_artifact_lines(think_body[last_end:])
    if tail:
        rebuilt.append(tail)

    clean_think = normalize_ws("\n\n".join(rebuilt))
    if not clean_think:
        raise ValueError("empty meta reasoning after artifact cleanup")
    return f"<think>\n{clean_think}\n</think>\n\nThe answer is ${boxed}$."


def build_strict_pair(meta_row: pd.Series, base_row: pd.Series) -> tuple[str, str]:
    meta_msgs = load_messages(meta_row["messages"])
    base_msgs = load_messages(base_row["messages"])

    if len(meta_msgs) != 2 or len(base_msgs) != 2:
        raise ValueError("messages must have exactly user+assistant")
    if meta_msgs[0]["content"] != base_msgs[0]["content"]:
        raise ValueError("meta/base user prompt mismatch")

    meta_assistant = sanitize_meta_assistant(str(meta_msgs[1]["content"]))
    if not should_keep_row(meta_row, meta_assistant):
        raise ValueError("row failed strict meta filter")

    scenario = str(meta_row.get("scenario", "")).lower()
    base_assistant = build_base_assistant(meta_assistant, scenario)

    meta_outside = parse_assistant(meta_assistant)[1]
    meta_boxed = extract_last_boxed(meta_outside)
    base_boxed = extract_last_boxed(base_assistant)
    if meta_boxed != base_boxed:
        raise ValueError("base rewrite changed final boxed answer")

    meta_messages = [
        {"role": "user", "content": str(meta_msgs[0]["content"])},
        {"role": "assistant", "content": meta_assistant},
    ]
    base_messages = [
        {"role": "user", "content": str(meta_msgs[0]["content"])},
        {"role": "assistant", "content": base_assistant},
    ]
    return dump_messages(meta_messages), dump_messages(base_messages)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build strict paired V8 meta/base data")
    parser.add_argument("--meta-input", default="data/v8_meta_inside_think.parquet")
    parser.add_argument("--base-input", default="data/v8_base_matched_clean.parquet")
    parser.add_argument("--meta-output", default="data/v8_meta_inside_strict.parquet")
    parser.add_argument("--base-output", default="data/v8_base_matched_strict.parquet")
    parser.add_argument("--summary-json", default="results/strict_data/v8_strict_build_summary.json")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    meta_input = root / args.meta_input
    base_input = root / args.base_input
    meta_output = root / args.meta_output
    base_output = root / args.base_output
    summary_json = root / args.summary_json

    meta_df = pd.read_parquet(meta_input)
    base_df = pd.read_parquet(base_input)
    if len(meta_df) != len(base_df):
        raise ValueError(f"length mismatch: {len(meta_df)} vs {len(base_df)}")

    kept_meta_rows = []
    kept_base_rows = []
    kept_counter = Counter()
    reject_counter = Counter()

    for idx in meta_df.index:
        meta_row = meta_df.loc[idx]
        base_row = base_df.loc[idx]
        scenario = str(meta_row.get("scenario", "")).lower()
        try:
            meta_messages, base_messages = build_strict_pair(meta_row, base_row)
        except Exception as exc:
            reject_counter[f"{scenario}:{type(exc).__name__}"] += 1
            continue

        meta_out = meta_row.copy()
        meta_out["messages"] = meta_messages
        kept_meta_rows.append(meta_out)

        base_out = base_row.copy()
        base_out["messages"] = base_messages
        kept_base_rows.append(base_out)
        kept_counter[scenario] += 1

    meta_strict_df = pd.DataFrame(kept_meta_rows).reset_index(drop=True)
    base_strict_df = pd.DataFrame(kept_base_rows).reset_index(drop=True)

    meta_output.parent.mkdir(parents=True, exist_ok=True)
    base_output.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    meta_strict_df.to_parquet(meta_output, index=False)
    base_strict_df.to_parquet(base_output, index=False)

    summary = {
        "meta_input_rows": int(len(meta_df)),
        "base_input_rows": int(len(base_df)),
        "kept_rows": int(len(meta_strict_df)),
        "kept_by_scenario": dict(sorted(kept_counter.items())),
        "rejected_by_reason": dict(sorted(reject_counter.items())),
        "meta_output": str(meta_output),
        "base_output": str(base_output),
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
