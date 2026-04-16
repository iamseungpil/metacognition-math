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


def _infer_difficulty(row: dict[str, Any]) -> str:
    explicit = _first_nonempty(row, ["difficulty"])
    if explicit:
        return explicit
    benchmark = _first_nonempty(row, ["benchmark"]).lower()
    if benchmark == "gsm8k":
        return "easy"
    if benchmark == "math500":
        return "medium"
    if benchmark == "aime2024":
        return "hard"
    return ""


def _classify_study_need_family(text: str) -> str:
    lowered = (text or "").lower()
    if not lowered.strip():
        return ""
    families = [
        ("exponential_growth", ["exponential", "factor-power", "multiplicative structure", "decay", "compounding", "geometric sequence", "growth factor"]),
        ("arithmetic_translation", ["rate", "unit", "percent", "time component", "translate word", "direct isolation", "sample-space"]),
        ("probability_counting", ["probability", "counting", "combin", "sample-space", "stars-and-bars", "inclusion-exclusion"]),
        ("algebraic_structure", ["factor", "identity", "substitution", "common denominator", "index substitution", "functional equation", "binomial", "remainder", "synthetic division"]),
        ("geometry", ["geometric", "coordinate", "circle", "angle", "power of a point", "incenter", "triangle"]),
        ("invariant_modular", ["invariant", "modular", "parity", "residue", "mod "]),
        ("game_dp", ["alternating-turn", "game state", "recurrence", "target-based", "dp", "state analysis"]),
    ]
    for label, keywords in families:
        if any(keyword in lowered for keyword in keywords):
            return label
    return "other"


def _prefer_easy_example(study_need: str, strategy_hint: str = "") -> bool:
    family = _classify_study_need_family("\n".join(part for part in [study_need, strategy_hint] if part))
    return family in {"arithmetic_translation", "exponential_growth", "probability_counting"}


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


@dataclass
class RetrievalQuery:
    """Structured retrieval query for meta-conditioned example lookup."""

    problem: str
    diagnosis: str = ""
    study_need: str = ""
    strategy_hint: str = ""
    study_need_family: str = ""
    prefer_easy: bool = False

    def to_text(self) -> str:
        parts = [self.problem.strip()]
        if self.diagnosis.strip():
            parts.append(self.diagnosis.strip())
        if self.study_need.strip():
            parts.append(f"missing skill or perspective: {self.study_need.strip()}")
        if self.study_need_family.strip():
            parts.append(f"strategy family: {self.study_need_family.strip()}")
        if self.strategy_hint.strip():
            parts.append(self.strategy_hint.strip())
        return "\n".join(part for part in parts if part)


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
            nested_meta = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
            question = _first_nonempty(row, ["full_question", "question", "problem", "text"])
            solution = _first_nonempty(row, ["completion", "solution", "response", "assistant"])
            answer = _first_nonempty(row, ["full_gold_answer", "gold_answer", "answer"])
            if not question or (require_solution and not solution):
                continue
            metadata = {
                "benchmark": row.get("benchmark", nested_meta.get("benchmark", "")),
                "source": row.get("source", nested_meta.get("source", file_path.stem)),
                "topic": row.get("topic", nested_meta.get("topic", "")),
                "difficulty": _infer_difficulty({**nested_meta, **row}),
                "is_correct": row.get("is_correct", nested_meta.get("is_correct", True)),
                "study_need": (
                    _first_nonempty(row, ["study_need", "missing_skill", "missing_perspective"])
                    or _first_nonempty(nested_meta, ["study_need", "missing_skill", "missing_perspective"])
                    or _parse_study_need(solution)
                ),
                "strategy_tags": row.get("strategy_tags", nested_meta.get("strategy_tags", row.get("method_tags", nested_meta.get("method_tags", [])))),
                "method": _first_nonempty(row, ["method", "strategy", "approach"]) or _first_nonempty(nested_meta, ["method", "strategy", "approach"]),
            }
            for key, value in nested_meta.items():
                metadata.setdefault(key, value)
            metadata["study_need_family"] = nested_meta.get("study_need_family") or _classify_study_need_family(metadata["study_need"])
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
        self.problem_tokens = [self._tokenize(self._record_problem_text(record)) for record in records]
        self.solution_tokens = [self._tokenize(self._record_solution_text(record)) for record in records]
        self.strategy_tokens = [self._tokenize(self._record_strategy_text(record)) for record in records]
        self.study_families = [
            str((record.metadata or {}).get("study_need_family", "")).strip()
            or _classify_study_need_family(str((record.metadata or {}).get("study_need", "")))
            or _classify_study_need_family(record.solution)
            for record in records
        ]
        self.difficulties = [str((record.metadata or {}).get("difficulty", "")).strip().lower() for record in records]
        self.dynamic_flags = [
            bool((record.metadata or {}).get("from_lane")) or "dynamic" in str(record.source).lower()
            for record in records
        ]
        self.has_strategy_signal = [bool(tokens) for tokens in self.strategy_tokens]

    @staticmethod
    def _join_nonempty(parts: Iterable[Any]) -> str:
        return "\n".join(str(part).strip() for part in parts if str(part).strip())

    @staticmethod
    def _record_problem_text(record: ExampleRecord) -> str:
        meta = record.metadata or {}
        meta_bits = [
            meta.get("benchmark", ""),
            meta.get("source", ""),
            meta.get("topic", ""),
            meta.get("difficulty", ""),
        ]
        return TfidfExampleRetriever._join_nonempty([record.question] + meta_bits)

    @staticmethod
    def _record_solution_text(record: ExampleRecord) -> str:
        return TfidfExampleRetriever._join_nonempty([record.solution, record.answer])

    @staticmethod
    def _record_strategy_text(record: ExampleRecord) -> str:
        meta = record.metadata or {}
        tags = meta.get("strategy_tags", [])
        if isinstance(tags, str):
            tags = [tags]
        return TfidfExampleRetriever._join_nonempty([
            meta.get("method", ""),
            meta.get("topic", ""),
            meta.get("study_need", ""),
            " ".join(tag for tag in tags if isinstance(tag, str)),
            _parse_study_need(record.solution),
        ])

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(re.findall(r"[a-zA-Z0-9_]+", text.lower()))

    @staticmethod
    def _normalized_overlap(query_tokens: set[str], doc_tokens: set[str]) -> float:
        if not query_tokens or not doc_tokens:
            return 0.0
        overlap = len(query_tokens & doc_tokens)
        norm = np.sqrt(len(query_tokens) * len(doc_tokens))
        return overlap / norm if norm else 0.0

    def _score_structured_query(
        self,
        query: RetrievalQuery,
        idx: int,
    ) -> tuple[float, dict[str, float]]:
        weights = {
            "problem_similarity": 0.45,
            "diagnosis_to_solution": 0.20,
            "study_need_to_strategy": 0.25,
            "strategy_hint": 0.10,
        }
        query_problem = self._tokenize(query.problem)
        query_diagnosis = self._tokenize(query.diagnosis)
        query_study_need = self._tokenize(query.study_need)
        query_strategy = self._tokenize(query.strategy_hint)

        problem_score = self._normalized_overlap(query_problem, self.problem_tokens[idx])
        diagnosis_score = self._normalized_overlap(
            query_diagnosis,
            self.solution_tokens[idx] | self.strategy_tokens[idx],
        )
        study_need_score = self._normalized_overlap(
            query_study_need,
            self.strategy_tokens[idx] | self.solution_tokens[idx],
        )
        strategy_score = self._normalized_overlap(
            query_strategy,
            self.strategy_tokens[idx] | self.solution_tokens[idx],
        )
        family_score = 0.0
        if (
            query.study_need_family
            and query.study_need_family != "other"
            and query.study_need_family == self.study_families[idx]
        ):
            family_score = 1.0
        dynamic_bonus = 1.0 if self.dynamic_flags[idx] else 0.0
        typed_strategy_bonus = 1.0 if self.has_strategy_signal[idx] else 0.0
        difficulty = self.difficulties[idx]
        easy_bonus = 0.0
        if query.prefer_easy and difficulty == "easy" and (
            family_score > 0.0
            or study_need_score > 0.0
            or typed_strategy_bonus > 0.0
        ):
            easy_bonus = 1.0
        generic_penalty = 0.0
        if query.study_need.strip() and not self.has_strategy_signal[idx]:
            generic_penalty += 0.10
        if query.prefer_easy and difficulty == "easy" and easy_bonus == 0.0:
            generic_penalty += 0.08
        if (
            query.study_need_family
            and query.study_need_family != "other"
            and not self.study_families[idx]
        ):
            generic_penalty += 0.06
        breakdown = {
            "problem_similarity": problem_score,
            "diagnosis_to_solution": diagnosis_score,
            "study_need_to_strategy": study_need_score,
            "strategy_hint": strategy_score,
            "study_need_family_match": family_score,
            "dynamic_bonus": dynamic_bonus,
            "typed_strategy_bonus": typed_strategy_bonus,
            "easy_bonus": easy_bonus,
            "generic_penalty": generic_penalty,
        }
        total = 0.0
        active_weight = 0.0
        for key, value in breakdown.items():
            if key == "study_need_family_match":
                if value > 0.0:
                    total += 0.15 * value
                    active_weight += 0.15
                continue
            if key == "dynamic_bonus":
                if value > 0.0:
                    total += 0.10 * value
                    active_weight += 0.10
                continue
            if key == "typed_strategy_bonus":
                if value > 0.0:
                    total += 0.10 * value
                    active_weight += 0.10
                continue
            if key == "easy_bonus":
                if value > 0.0:
                    total += 0.08 * value
                    active_weight += 0.08
                continue
            if key == "generic_penalty":
                continue
            if value > 0.0:
                total += weights[key] * value
                active_weight += weights[key]
        if active_weight > 0:
            total /= active_weight
        total = max(0.0, total - generic_penalty)
        breakdown["total"] = total
        return total, breakdown

    def _candidate_indices(self, query: RetrievalQuery | str) -> list[int]:
        indices = list(range(len(self.records)))
        if not isinstance(query, RetrievalQuery):
            return indices

        if query.study_need_family:
            family_indices = [idx for idx in indices if self.study_families[idx] == query.study_need_family]
            if len(family_indices) >= 5:
                indices = family_indices

        if query.prefer_easy:
            easy_indices = [idx for idx in indices if self.difficulties[idx] == "easy"]
            if len(easy_indices) >= 5:
                indices = easy_indices

        dynamic_indices = [idx for idx in indices if self.dynamic_flags[idx]]
        if len(dynamic_indices) >= 3:
            indices = dynamic_indices + [idx for idx in indices if idx not in set(dynamic_indices)]
        return indices

    def search(self, query: str | RetrievalQuery, top_k: int = 1) -> list[dict[str, Any]]:
        if not self.records:
            return []

        scores = []
        candidate_indices = self._candidate_indices(query)
        if isinstance(query, RetrievalQuery):
            if not query.to_text().strip():
                return []
            for idx in candidate_indices:
                score, breakdown = self._score_structured_query(query, idx)
                scores.append((score, breakdown))
        else:
            query_tokens = self._tokenize(query)
            if not query_tokens:
                return []
            for idx in candidate_indices:
                doc_tokens = self.problem_tokens[idx]
                score = self._normalized_overlap(query_tokens, doc_tokens)
                scores.append((score, {"problem_similarity": score, "total": score}))

        order = np.argsort(np.asarray([score for score, _ in scores]))[::-1][:top_k]
        results = []
        for idx in order:
            score, breakdown = scores[idx]
            record_idx = candidate_indices[idx]
            score = float(score)
            if score <= 0:
                continue
            record = self.records[record_idx]
            results.append({
                "score": score,
                "score_breakdown": breakdown,
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
    trigger_reasons = []
    if has_trigger:
        trigger_reasons.append("surface_uncertainty_or_conflict")
    if has_low_conf:
        trigger_reasons.append("low_confidence")
    if has_diagnosis:
        trigger_reasons.append("failure_diagnosis")
    if has_decomposition:
        trigger_reasons.append("failure_decomposition")
    if has_next_strategy or has_switch:
        trigger_reasons.append("next_strategy")
    if study_need:
        trigger_reasons.append("study_need")

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
        "trigger_reasons": trigger_reasons,
        "should_retrieve": should_retrieve,
        "diagnosis_text": diagnosis_text,
        "retrieval_mode": "diagnosis_triggered_retry" if should_retrieve else "none",
    }


def build_retrieval_query_bundle(question: str, analysis: dict[str, Any]) -> RetrievalQuery:
    diagnosis = analysis.get("diagnosis_text", "").strip()
    study_need = analysis.get("study_need", "").strip()
    strategy_hints = []
    if analysis.get("has_decomposition"):
        strategy_hints.append("needs failure diagnosis and the right missing skill")
    if analysis.get("has_next_strategy") or analysis.get("has_switch"):
        strategy_hints.append("needs a different solution method")
    study_need_family = _classify_study_need_family("\n".join([study_need, diagnosis]))
    return RetrievalQuery(
        problem=question.strip(),
        diagnosis=diagnosis,
        study_need=study_need,
        strategy_hint="\n".join(strategy_hints),
        study_need_family=study_need_family,
        prefer_easy=_prefer_easy_example(study_need, "\n".join(strategy_hints)),
    )


def build_retrieval_query(question: str, analysis: dict[str, Any]) -> str:
    return build_retrieval_query_bundle(question, analysis).to_text()


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
        f"Retrieved example policy:\nUse the example only as strategic evidence. Adapt it to the current problem rather than copying surface details.\n\n"
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
