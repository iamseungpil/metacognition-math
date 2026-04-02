"""Redirect-triggered retrieval helpers for inference-time control.

This module implements a lightweight no-training framework:
1. Solve once.
2. Inspect meta blocks for low-confidence / anomaly / diagnosis signals.
3. Retrieve a similar solved example.
4. Retry the original problem with the retrieved example in context.

It intentionally uses TF-IDF retrieval so smoke tests do not require
extra heavyweight dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    import torch
except ImportError:  # pragma: no cover - retrieval-only utilities should still import
    torch = None

def _parse_confidence(text: str) -> float | None:
    match = re.search(r"confidence[:\s]+(\d+\.\d+|\d+)", text, re.IGNORECASE)
    if not match:
        return None
    value = float(match.group(1))
    if value > 1:
        value /= 100.0
    return max(0.0, min(1.0, value))


def _parse_meta_blocks(text: str) -> list[dict[str, Any]]:
    blocks = []
    parts = re.split(r"<\|meta\|>", text)
    for part in parts[1:]:
        end_idx = part.find("<|/meta|>")
        block_text = part[:end_idx] if end_idx != -1 else part[:200]
        blocks.append({
            "text": block_text.strip(),
            "confidence": _parse_confidence(block_text),
            "length": len(block_text.split()),
        })
    return blocks


def _has_uncertainty_signal(text: str) -> bool:
    return bool(re.search(r"\b(wait|hmm|not sure|uncertain|stuck|hold on|let me think|I should check)\b", text, re.IGNORECASE))


def _has_conflict_trigger(text: str) -> bool:
    return bool(re.search(
        r"\b(something feels off|this feels off|that seems off|this seems off|"
        r"contradiction|inconsistent|doesn't satisfy|does not satisfy|fails|mismatch|"
        r"too large|too small|cannot be|can't be|unsupported|forcing|overcommitted)\b",
        text,
        re.IGNORECASE,
    ))


def _has_failure_diagnosis(text: str) -> bool:
    return bool(re.search(
        r"\b(the issue is|the problem is|current route is weak|this route fails because|"
        r"I may be forcing|I am forcing|I committed too early|I overcommitted|"
        r"missing piece|does not control|does not explain|too indirect|too complicated)\b",
        text,
        re.IGNORECASE,
    ))


def _has_failure_decomposition(text: str) -> bool:
    return bool(re.search(
        r"\b(missing skill|missing perspective|missing structure|missing piece|"
        r"the bottleneck is|the blocker is|this is not a calculation problem|"
        r"this is not an algebra problem|need a structural view|need a constraint-based view|"
        r"need an invariant|need a different object of study)\b",
        text,
        re.IGNORECASE,
    ))


def _parse_study_need(text: str) -> str:
    match = re.search(r"study_need:\s*(.+)", text, re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _has_next_strategy(text: str) -> bool:
    return bool(re.search(
        r"\b(switch to|different method|alternative approach|instead I'll|instead I will|"
        r"case split|use a parity|use an invariant|use a direct check|reframe|"
        r"switch methods|switch method|different approach)\b",
        text,
        re.IGNORECASE,
    ))


def _has_strategy_switch_signal(text: str) -> bool:
    return _has_next_strategy(text)


def _has_low_confidence(text: str, threshold: float = 0.55) -> bool:
    confidences = []
    for block in _parse_meta_blocks(text):
        conf = block["confidence"]
        if conf is not None:
            confidences.append(conf)
    return any(conf <= threshold for conf in confidences)


@dataclass
class ExampleRecord:
    question: str
    solution: str
    answer: str = ""
    source: str = ""
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["metadata"] = self.metadata or {}
        return payload


def render_messages_as_text(messages: list[dict[str, str]], add_generation_prompt: bool = False) -> str:
    """Fallback renderer for tokenizers without a chat template."""
    rendered = []
    for message in messages:
        role = message.get("role", "user").strip().capitalize()
        content = message.get("content", "").strip()
        rendered.append(f"{role}: {content}")
    if add_generation_prompt:
        rendered.append("Assistant:")
    return "\n\n".join(rendered)


def build_model_inputs(
    tokenizer,
    messages: list[dict[str, str]],
    *,
    device: torch.device | str | None = None,
    add_generation_prompt: bool = True,
    max_prompt_tokens: int = 2048,
):
    """Render a chat prompt with a safe fallback."""
    prompt_text = None
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except Exception:
            prompt_text = None

    if prompt_text is None:
        prompt_text = render_messages_as_text(messages, add_generation_prompt=add_generation_prompt)

    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_tokens,
    )
    if device is not None:
        inputs = {k: v.to(device) for k, v in inputs.items()}
    return prompt_text, inputs


def generate_from_messages(
    model,
    tokenizer,
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_prompt_tokens: int = 2048,
) -> tuple[str, str, int, int]:
    """Generate one completion from chat-style messages."""
    if torch is None:
        raise ImportError("torch is required for generation utilities")
    prompt_text, inputs = build_model_inputs(
        tokenizer,
        messages,
        device=model.device,
        add_generation_prompt=True,
        max_prompt_tokens=max_prompt_tokens,
    )
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
        )
    prompt_len_tokens = int(inputs["input_ids"].shape[1])
    completion_ids = output[0][prompt_len_tokens:]
    completion_len_tokens = int(completion_ids.shape[0])
    completion = tokenizer.decode(completion_ids, skip_special_tokens=False)
    return completion, prompt_text, prompt_len_tokens, completion_len_tokens


def _first_nonempty(row: dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = row.get(key, "")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def load_example_bank(paths: list[str] | list[Path], require_solution: bool = True) -> list[ExampleRecord]:
    """Load solved examples from parquet/json/jsonl files."""
    records: list[ExampleRecord] = []
    for path in paths:
        file_path = Path(path)
        if not file_path.exists():
            continue

        if file_path.suffix == ".parquet":
            df = pd.read_parquet(file_path)
            rows = df.to_dict(orient="records")
        elif file_path.suffix == ".jsonl":
            rows = [json.loads(line) for line in file_path.read_text().splitlines() if line.strip()]
        elif file_path.suffix == ".json":
            payload = json.loads(file_path.read_text())
            rows = payload["results"] if isinstance(payload, dict) and "results" in payload else payload
        else:
            continue

        for row in rows:
            question = _first_nonempty(row, ["full_question", "question", "problem", "text"])
            solution = _first_nonempty(row, ["completion", "solution", "response", "assistant"])
            answer = _first_nonempty(row, ["full_gold_answer", "gold_answer", "answer"])
            if not question or (require_solution and not solution):
                continue
            metadata = {
                "benchmark": row.get("benchmark", ""),
                "source": row.get("source", file_path.stem),
                "topic": row.get("topic", ""),
                "difficulty": row.get("difficulty", ""),
                "is_correct": row.get("is_correct", True),
            }
            records.append(
                ExampleRecord(
                    question=question,
                    solution=solution,
                    answer=answer,
                    source=str(metadata["source"]),
                    metadata=metadata,
                )
            )
    return records


class TfidfExampleRetriever:
    """Simple lexical retriever for solved exemplars without heavy dependencies."""

    def __init__(self, records: list[ExampleRecord]):
        self.records = records
        self.corpus_tokens = [self._tokenize(self._record_to_text(record)) for record in records]

    @staticmethod
    def _record_to_text(record: ExampleRecord) -> str:
        meta = record.metadata or {}
        meta_bits = [
            meta.get("benchmark", ""),
            meta.get("source", ""),
            meta.get("topic", ""),
            meta.get("difficulty", ""),
        ]
        return "\n".join([record.question] + [bit for bit in meta_bits if bit])

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(re.findall(r"[a-zA-Z0-9_]+", text.lower()))

    def search(self, query: str, top_k: int = 1) -> list[dict[str, Any]]:
        if not self.records:
            return []
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = []
        for idx, doc_tokens in enumerate(self.corpus_tokens):
            if not doc_tokens:
                score = 0.0
            else:
                overlap = len(query_tokens & doc_tokens)
                norm = np.sqrt(len(query_tokens) * len(doc_tokens))
                score = overlap / norm if norm else 0.0
            scores.append(score)

        order = np.argsort(np.asarray(scores))[::-1][:top_k]
        results = []
        for idx in order:
            score = float(scores[idx])
            if score <= 0:
                continue
            record = self.records[idx]
            results.append({
                "score": score,
                "record": record,
            })
        return results


def analyze_completion_for_rag(completion: str) -> dict[str, Any]:
    """Inspect a completion and decide whether redirect-time retrieval is justified."""
    blocks = _parse_meta_blocks(completion)
    diagnosis_blocks = []
    min_conf = None
    for block in blocks:
        text = block.get("text", "").strip()
        conf = block.get("confidence")
        if conf is not None:
            min_conf = conf if min_conf is None else min(min_conf, conf)
        if (
            (conf is not None and conf <= 0.55)
            or _has_failure_diagnosis(text)
            or _has_failure_decomposition(text)
            or bool(_parse_study_need(text))
            or _has_next_strategy(text)
        ):
            clean_lines = []
            for line in text.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.lower().startswith("confidence:"):
                    continue
                if stripped.lower().startswith("study_need:"):
                    continue
                clean_lines.append(stripped)
            if clean_lines:
                diagnosis_blocks.append("\n".join(clean_lines))

    has_trigger = _has_conflict_trigger(completion) or _has_uncertainty_signal(completion)
    has_diagnosis = _has_failure_diagnosis(completion)
    has_decomposition = _has_failure_decomposition(completion)
    has_next_strategy = _has_next_strategy(completion)
    has_switch = _has_strategy_switch_signal(completion)
    has_low_conf = _has_low_confidence(completion)
    study_needs = [need for need in (_parse_study_need(block.get("text", "")) for block in blocks) if need]
    study_need = study_needs[0] if study_needs else ""

    should_retrieve = (has_trigger or has_low_conf or bool(study_need)) and (
        has_diagnosis or has_decomposition or has_next_strategy or has_switch or bool(study_need)
    )
    diagnosis_text = "\n".join(dict.fromkeys([b for b in diagnosis_blocks if b]))

    return {
        "meta_blocks": blocks,
        "meta_count": len(blocks),
        "min_confidence": min_conf,
        "has_trigger": has_trigger,
        "has_low_confidence": has_low_conf,
        "has_diagnosis": has_diagnosis,
        "has_decomposition": has_decomposition,
        "has_next_strategy": has_next_strategy,
        "has_switch": has_switch,
        "study_need": study_need,
        "should_retrieve": should_retrieve,
        "diagnosis_text": diagnosis_text,
    }


def build_retrieval_query(question: str, analysis: dict[str, Any]) -> str:
    parts = [question.strip()]
    diagnosis = analysis.get("diagnosis_text", "").strip()
    if diagnosis:
        parts.append(diagnosis)
    study_need = analysis.get("study_need", "").strip()
    if study_need:
        parts.append(f"missing skill or perspective: {study_need}")
    if analysis.get("has_decomposition"):
        parts.append("needs failure diagnosis and the right missing skill")
    if analysis.get("has_next_strategy") or analysis.get("has_switch"):
        parts.append("needs a different solution method")
    return "\n".join(parts)


def format_retrieved_example(record: ExampleRecord, score: float, rank: int) -> str:
    meta = record.metadata or {}
    header_bits = [f"score={score:.3f}"]
    if meta.get("topic"):
        header_bits.append(f"topic={meta['topic']}")
    if meta.get("difficulty"):
        header_bits.append(f"difficulty={meta['difficulty']}")
    header = ", ".join(header_bits)
    answer = f"\nFinal answer: {record.answer}" if record.answer else ""
    return (
        f"[Retrieved Example {rank}] ({header})\n"
        f"Problem:\n{record.question}\n\n"
        f"Solved approach:\n{record.solution}{answer}\n"
    )


def build_incontext_user_prompt(
    question: str,
    analysis: dict[str, Any],
    retrieved: list[dict[str, Any]],
) -> str:
    diagnosis = analysis.get("diagnosis_text", "").strip()
    diagnosis_section = diagnosis if diagnosis else "The previous route looked unreliable and confidence dropped."
    study_need = analysis.get("study_need", "").strip()
    study_need_section = study_need if study_need else "None stated."
    examples = "\n\n".join(
        format_retrieved_example(item["record"], item["score"], i + 1)
        for i, item in enumerate(retrieved)
    )
    return (
        "You are retrying a math problem after a metacognitive redirect signal.\n"
        "Use the retrieved solved example as a hint about strategy, not as something to copy blindly.\n"
        "If the previous route was weak, switch methods and solve the original problem correctly.\n"
        "Use <|meta|> blocks only if they genuinely change your behavior, and end with a final \\boxed{answer}.\n\n"
        f"Why the previous route looked weak:\n{diagnosis_section}\n\n"
        f"Missing skill or perspective to recover:\n{study_need_section}\n\n"
        f"{examples}\n"
        f"Original problem:\n{question}\n"
    )


def run_redirect_rag_pass(
    model,
    tokenizer,
    question: str,
    first_completion: str,
    retriever: TfidfExampleRetriever,
    *,
    top_k: int = 1,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> dict[str, Any]:
    """Retry a question with retrieval when the first completion indicates redirect."""
    analysis = analyze_completion_for_rag(first_completion)
    result = {
        "rag_used": False,
        "analysis": analysis,
        "retrieved": [],
        "rag_prompt": "",
        "rag_completion": "",
    }
    if not analysis["should_retrieve"]:
        return result

    query = build_retrieval_query(question, analysis)
    retrieved = retriever.search(query, top_k=top_k)
    if not retrieved:
        return result

    rag_prompt = build_incontext_user_prompt(question, analysis, retrieved)
    rag_completion, _, _, _ = generate_from_messages(
        model,
        tokenizer,
        [{"role": "user", "content": rag_prompt}],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    result.update({
        "rag_used": True,
        "retrieved": [
            {
                "score": item["score"],
                "question": item["record"].question,
                "source": item["record"].source,
                "answer": item["record"].answer,
            }
            for item in retrieved
        ],
        "rag_prompt": rag_prompt,
        "rag_completion": rag_completion,
    })
    return result
