"""On-policy self-distill artifact builders (vLLM batched).

Supported paths:

1. `question_only_best_of_n`: claim-bearing mainline.
   Sample `n` completions from the original question prompt in one vLLM pass,
   then select a teacher without repair-prompt confounds.
2. `fixed_k_repair`: side-evidence lane with root -> diagnosis/retrieval ->
   repair prompt -> K candidates -> selector.
3. `sdpo_regen`: trigger-gated, feedback-conditioned side-evidence collection.
   This path performs a real two-phase root->feedback->regeneration rollout and
   can optionally feed teacher top-k / meta-only KL preparation downstream.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from src.curriculum.control_rag import (
    TfidfExampleRetriever,
    analyze_completion_for_rag,
    build_incontext_user_prompt,
    build_retrieval_query_bundle,
    load_example_bank,
)
from src.curriculum.rq3_pipeline import (
    evaluate_meta_transition,
    judge_completion,
)
from src.metacot.prompt import META_END, META_START, parse_meta_blocks
from src.training.self_distill.builders import (
    build_sdpo_regen_user_prompt,
    build_self_distill_dataframe,
    summarize_self_distill_dataframe,
)
from src.training.self_distill.trace import NormalizedTeacherTrace
from src.training.meta_quality import score_meta_commit_quality
from src.training.rewards import (
    confidence_omission_floor,
    confidence_revision_reward_v2,
    correctness_reward,
    redirect_execution_reward_v2,
    verify_execution_reward_v2,
)


@dataclass
class OnlineSdpoProblem:
    question: str
    gold_answer: str
    benchmark: str = ""
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["metadata"] = self.metadata or {}
        return payload


def load_online_problems(
    *,
    question: str | None = None,
    gold_answer: str | None = None,
    input_path: str | Path | None = None,
    benchmark_names: list[str] | None = None,
    max_problems: int = 30,
) -> list[OnlineSdpoProblem]:
    problems: list[OnlineSdpoProblem] = []
    if question is not None:
        if not gold_answer:
            raise ValueError("Single-question mode requires --gold_answer")
        problems.append(OnlineSdpoProblem(question=question.strip(), gold_answer=gold_answer.strip()))
        return problems

    if input_path is not None:
        path = Path(input_path)
        if path.suffix == ".parquet":
            rows = pd.read_parquet(path).to_dict(orient="records")
        elif path.suffix == ".jsonl":
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        elif path.suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = payload["results"] if isinstance(payload, dict) and "results" in payload else payload
        else:
            raise ValueError(f"Unsupported problem file: {path}")
        for row in rows:
            q = str(row.get("question") or row.get("problem") or row.get("full_question") or "").strip()
            a = str(row.get("gold_answer") or row.get("answer") or row.get("full_gold_answer") or "").strip()
            if not q or not a:
                continue
            problems.append(
                OnlineSdpoProblem(
                    question=q,
                    gold_answer=a,
                    benchmark=str(row.get("benchmark", "")).strip(),
                    metadata={k: v for k, v in row.items() if k not in {"question", "problem", "full_question", "gold_answer", "answer", "full_gold_answer", "benchmark"}},
                )
            )
            if len(problems) >= max_problems:
                break
        return problems

    if benchmark_names:
        from src.eval.eval_hf import load_benchmarks

        loaded = load_benchmarks(benchmark_names, max_problems=max_problems)
        return [
            OnlineSdpoProblem(
                question=str(row["question"]).strip(),
                gold_answer=str(row["gold_answer"]).strip(),
                benchmark=str(row.get("benchmark", "")).strip(),
            )
            for row in loaded
        ]

    raise ValueError("Provide one of: question+gold_answer, input_path, or benchmark_names")


def load_retriever(example_bank_paths: list[str] | None) -> TfidfExampleRetriever | None:
    if not example_bank_paths:
        return None
    records = load_example_bank(example_bank_paths)
    if not records:
        return None
    return TfidfExampleRetriever(records)


# ---------------------------------------------------------------------------
# Private helpers (vLLM + streaming + resume)
# ---------------------------------------------------------------------------


def _render_chat_prompt(tokenizer, content: str) -> str:
    """Render single user message as chat-templated string (vLLM-ready)."""
    messages = [{"role": "user", "content": content}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"


def _stable_problem_id(benchmark: str, question: str) -> str:
    """SHA1[:16] of (benchmark + '\\n' + question.strip()) for resume tracking."""
    blob = (benchmark or "") + "\n" + (question or "").strip()
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def _load_completed_ids(traces_path: Path) -> set[str]:
    """Read existing jsonl, return set of problem_ids. Tolerant of malformed trailing line."""
    done: set[str] = set()
    if not traces_path.exists():
        return done
    with traces_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                pid = row.get("problem_id")
                if pid:
                    done.add(pid)
            except json.JSONDecodeError:
                continue
    return done


def _append_trace(handle, row: dict[str, Any]) -> None:
    """Write one trace row + flush + fsync (preempt-safe)."""
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()
    try:
        os.fsync(handle.fileno())
    except OSError:
        pass  # not all FS support fsync


def _truncate_for_repair(text: str, max_chars: int = 2000) -> str:
    """Middle-truncate to preserve both head reasoning context and tail \\boxed{}.

    Keeps head + tail, drops middle. Defends against small max_chars where
    naive head/tail computation would go negative (Python slicing does the
    wrong thing with negative indices).
    """
    if len(text) <= max_chars:
        return text
    marker = "\n... [truncated middle] ...\n"
    budget = max_chars - len(marker)
    if budget <= 10:
        # Degenerate: keep tail only (where \\boxed{} lives)
        return text[-max_chars:]
    head = max(10, budget // 2)
    tail = max(10, budget - head)
    return text[:head] + marker + text[-tail:]


def _chunks(items: list, size: int) -> list[list]:
    """Yield list slices of given size."""
    return [items[i:i + size] for i in range(0, len(items), size)]


def _log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def _safe_generate(llm, prompts: list[str], sampling) -> list:
    """Batch generate with per-prompt fallback on batch failure.

    vLLM raises if ANY prompt exceeds max_model_len. This helper catches the
    batch failure and retries prompts one at a time, logging which prompt
    failed and emitting a placeholder RequestOutput for it so downstream
    indexing stays aligned with `prompts`.
    """
    try:
        return llm.generate(prompts, sampling)
    except Exception as exc:
        _log(f"  Batch generate failed ({exc}); falling back to per-prompt")
    outputs = []
    for i, p in enumerate(prompts):
        try:
            out = llm.generate([p], sampling)
            outputs.extend(out)
        except Exception as exc:
            _log(f"    Prompt {i} failed ({exc}); emitting placeholder")
            # Placeholder: empty outputs, so downstream sees empty completion
            class _Stub:
                prompt_token_ids = []
                outputs = []
            outputs.append(_Stub())
    return outputs


# ---------------------------------------------------------------------------
# Selection + prompt builders (CPU-only, shared with offline variants)
# ---------------------------------------------------------------------------


def _score_completion_for_selection(
    completion: str,
    gold_answer: str,
    *,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    active_weights = weights or {
        "correctness": 1.0,
        "confidence_revision": 0.35,
        "redirect_execution": 0.30,
        "verify_execution": 0.15,
        "meta_floor": 0.15,
        "meta_commit_quality": 0.25,
    }
    payload = [{"content": completion}]
    ground_truth = [gold_answer]
    meta_commit_quality = _score_meta_commit_quality(completion)
    components = {
        "correctness": correctness_reward(payload, ground_truth)[0],
        "confidence_revision": confidence_revision_reward_v2(payload, ground_truth)[0],
        "redirect_execution": redirect_execution_reward_v2(payload, ground_truth)[0],
        "verify_execution": verify_execution_reward_v2(payload, ground_truth)[0],
        "meta_floor": confidence_omission_floor(payload, ground_truth)[0],
        "meta_commit_quality": meta_commit_quality["total"],
    }
    for key, value in meta_commit_quality.items():
        components[f"meta_commit_quality_{key}"] = value
    total = 0.0
    for key, weight in active_weights.items():
        total += float(weight) * float(components.get(key, 0.0))
    components["total"] = total
    return components


def _score_meta_commit_quality(completion: str) -> dict[str, float]:
    return score_meta_commit_quality(completion)


def _is_correct_judgment(judgment: dict[str, Any] | None) -> bool:
    return bool((judgment or {}).get("is_correct"))


def _select_best_candidate(
    candidates: list[dict[str, Any]],
    gold_answer: str,
    *,
    selector_mode: str = "reward_weighted",
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    scored: list[dict[str, Any]] = []
    for idx, candidate in enumerate(candidates):
        completion = str(candidate.get("completion", "")).strip()
        if not completion:
            continue
        breakdown = _score_completion_for_selection(completion, gold_answer, weights=weights)
        scored.append({
            **candidate,
            "candidate_id": str(candidate.get("candidate_id", f"repair_{idx}")),
            "selector_breakdown": breakdown,
            "selection_score": float(breakdown["total"]),
        })
    if not scored:
        raise ValueError("No non-empty repair candidates to score")
    if selector_mode == "correctness_only":
        ranked = sorted(
            scored,
            key=lambda row: (
                bool((row.get("judgment") or {}).get("is_correct")),
                -len(str(row.get("completion", ""))),
            ),
            reverse=True,
        )
    elif selector_mode == "correct_then_meta":
        ranked = sorted(
            scored,
            key=lambda row: (
                bool((row.get("judgment") or {}).get("is_correct")),
                float((row.get("selector_breakdown") or {}).get("meta_commit_quality", 0.0)),
                float(row["selection_score"]),
                -len(str(row.get("completion", ""))),
            ),
            reverse=True,
        )
    elif selector_mode == "correctness_first":
        ranked = sorted(
            scored,
            key=lambda row: (
                bool((row.get("judgment") or {}).get("is_correct")),
                float(row["selection_score"]),
                -len(str(row.get("completion", ""))),
            ),
            reverse=True,
        )
    else:
        ranked = sorted(
            scored,
            key=lambda row: (
                float(row["selection_score"]),
                bool((row.get("judgment") or {}).get("is_correct")),
                -len(str(row.get("completion", ""))),
            ),
            reverse=True,
        )
    best = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None
    return {
        "selected": best,
        "ranked_candidates": ranked,
        "selected_candidate_id": best["candidate_id"],
        "selected_score": float(best["selection_score"]),
        "selected_breakdown": best["selector_breakdown"],
        "score_margin": float(best["selection_score"] - runner_up["selection_score"]) if runner_up else None,
        "selector_mode": selector_mode,
    }


def _build_fixed_repair_prompt(
    *,
    question: str,
    root_completion: str,
    root_analysis: dict[str, Any],
    retrieved: list[dict[str, Any]],
) -> str:
    if retrieved:
        return build_incontext_user_prompt(question, root_analysis, retrieved) + (
            "\nPrevious attempt to repair:\n"
            f"{root_completion.strip()}\n\n"
            "Do not repeat the same unsupported route. Re-solve from the original problem and end with \\boxed{answer}."
        )
    diagnosis = str(root_analysis.get("diagnosis_text", "")).strip() or "The earlier attempt may be unreliable."
    study_need = str(root_analysis.get("study_need", "")).strip() or "Re-check the controlling constraint before committing."
    return (
        "You are revising a previous solution attempt.\n"
        "Use the previous attempt only as something to audit, not something to copy.\n"
        "If the route was weak, replace it with a corrected solution and end with a final \\boxed{answer}.\n\n"
        f"Original problem:\n{question.strip()}\n\n"
        f"Previous attempt:\n{root_completion.strip()}\n\n"
        f"Observed weakness:\n{diagnosis}\n\n"
        f"What to recover:\n{study_need}\n"
    )


def _retrieve_examples_for_fixed_k(
    *,
    question: str,
    root_analysis: dict[str, Any],
    retriever: TfidfExampleRetriever | None,
    rag_top_k: int,
    retrieval_query_mode: str,
) -> tuple[list[dict[str, Any]], str]:
    if retriever is None or rag_top_k <= 0 or retrieval_query_mode == "none":
        return [], "none"
    if retrieval_query_mode == "triggered":
        if not root_analysis.get("should_retrieve", False):
            return [], "triggered_none"
        query = build_retrieval_query_bundle(question, root_analysis)
        retrieved = retriever.search(query, top_k=rag_top_k)
        return retrieved, "triggered"
    if retrieval_query_mode == "question_only":
        retrieved = retriever.search(question, top_k=rag_top_k)
        return retrieved, "question_only"
    query = build_retrieval_query_bundle(question, root_analysis)
    if retrieval_query_mode == "analysis_or_question" and not query.to_text().strip():
        retrieved = retriever.search(question, top_k=rag_top_k)
        return retrieved, "question_only_fallback"
    retrieved = retriever.search(query if retrieval_query_mode == "analysis_or_question" else question, top_k=rag_top_k)
    return retrieved, retrieval_query_mode


# ---------------------------------------------------------------------------
# Batched vLLM rollout loops
# ---------------------------------------------------------------------------


def run_online_sdpo_rollouts(*args, **kwargs):
    """Run feedback-conditioned regeneration with on-policy root attempts."""
    return _run_online_sdpo_rollouts_impl(*args, **kwargs)


def _build_sdpo_feedback_context(
    *,
    root_analysis: dict[str, Any],
    root_judgment: dict[str, Any],
    retrieved: list[dict[str, Any]],
    retrieval_mode_used: str,
) -> tuple[str, dict[str, Any]]:
    evidence_items = []
    for item in retrieved:
        if not isinstance(item, dict):
            continue
        record = item.get("record")
        evidence_items.append({
            "question": str(getattr(record, "question", "")),
            "source": str(getattr(record, "source", "")),
            "score": item.get("score"),
            "score_breakdown": item.get("score_breakdown", {}),
        })

    feedback_kind = "teacher_only_rag" if evidence_items else "teacher_feedback_only"
    context = {
        "lane": "sdpo_regen",
        "retrieval_mode": retrieval_mode_used,
        "evidence_items": evidence_items,
        "failure_signals": {
            "root_correct": bool(root_judgment.get("is_correct")),
            "should_retrieve": bool(root_analysis.get("should_retrieve")),
            "has_low_confidence": bool(root_analysis.get("has_low_confidence")),
            "has_diagnosis": bool(root_analysis.get("has_diagnosis")),
            "has_next_strategy": bool(root_analysis.get("has_next_strategy")),
        },
    }
    return feedback_kind, context


def _build_sdpo_regen_prompt(
    *,
    question: str,
    gold_answer: str,
    benchmark: str,
    root_completion: str,
    root_analysis: dict[str, Any],
    feedback_kind: str,
    feedback_context: dict[str, Any],
) -> str:
    trace = NormalizedTeacherTrace(
        question=question,
        teacher_completion="",
        gold_answer=gold_answer,
        benchmark=benchmark,
        origin="sdpo_regen_prompt",
        root_completion=root_completion,
        diagnosis_text=str(root_analysis.get("diagnosis_text", "")).strip(),
        study_need=str(root_analysis.get("study_need", "")).strip(),
        intervention_summary="sdpo_regen",
        teacher_feedback_kind=feedback_kind,
        teacher_feedback_context=feedback_context,
    )
    return build_sdpo_regen_user_prompt(trace)


def _run_online_sdpo_rollouts_impl(
    *,
    llm,
    tokenizer,
    problems: list[OnlineSdpoProblem],
    output_dir: str | Path,
    retriever: TfidfExampleRetriever | None = None,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.95,
    seed: int = 42,
    rag_top_k: int = 1,
    repair_candidates: int = 4,
    retrieval_query_mode: str = "analysis_or_question",
    selector_mode: str = "reward_weighted",
    selector_weights: dict[str, float] | None = None,
    require_correct_teacher: bool = False,
    chunk_size: int = 64,
    resume: bool = True,
) -> list[dict[str, Any]]:
    from vllm import SamplingParams

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    traces_path = outdir / "online_sdpo_traces.jsonl"

    done_ids: set[str] = set()
    existing_rows: list[dict[str, Any]] = []
    if resume and traces_path.exists():
        with traces_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    pid = row.get("problem_id")
                    if pid:
                        done_ids.add(pid)
                        existing_rows.append(row)
                except json.JSONDecodeError:
                    continue
        _log(f"Resume: {len(done_ids)} problems already done; will skip.")

    to_do = [p for p in problems if _stable_problem_id(p.benchmark, p.question) not in done_ids]
    _log(
        "To process: "
        f"{len(to_do)} of {len(problems)} (chunk_size={chunk_size}, K={repair_candidates}, mode=sdpo_regen)"
    )
    if not to_do:
        return existing_rows

    handle = traces_path.open("a", encoding="utf-8")
    new_rows: list[dict[str, Any]] = []
    skipped_problems: list[dict[str, Any]] = []

    try:
        chunks = _chunks(to_do, chunk_size)
        for chunk_i, chunk in enumerate(chunks):
            _log(f"Chunk {chunk_i + 1}/{len(chunks)} ({len(chunk)} problems)")
            root_prompts = [_render_chat_prompt(tokenizer, p.question) for p in chunk]
            root_sampling = SamplingParams(
                n=1,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_new_tokens,
                seed=seed,
                skip_special_tokens=False,
            )
            t0 = datetime.now()
            root_outputs = _safe_generate(llm, root_prompts, root_sampling)
            _log(f"  Phase 1 root: {len(chunk)} prompts in {(datetime.now()-t0).total_seconds():.1f}s")

            root_data: list[dict[str, Any]] = []
            for p, root_prompt, root_out in zip(chunk, root_prompts, root_outputs):
                pid = _stable_problem_id(p.benchmark, p.question)
                if not root_out.outputs:
                    skipped_problems.append({
                        "problem_id": pid,
                        "benchmark": p.benchmark,
                        "question_preview": p.question[:80],
                        "phase": "root",
                        "reason": "no_outputs",
                    })
                    _log(f"  Skipping problem {pid[:8]}: root returned no outputs")
                    continue

                root_completion = root_out.outputs[0].text
                root_completion_tokens = len(root_out.outputs[0].token_ids)
                root_prompt_tokens = len(root_out.prompt_token_ids)
                root_analysis = analyze_completion_for_rag(root_completion)
                root_judgment = judge_completion(root_completion, p.gold_answer)
                trigger_fired = bool(root_analysis.get("should_retrieve")) or not bool(root_judgment.get("is_correct"))
                if not trigger_fired:
                    skipped_problems.append({
                        "problem_id": pid,
                        "benchmark": p.benchmark,
                        "question_preview": p.question[:80],
                        "phase": "feedback_gate",
                        "reason": "root_correct_without_trigger",
                    })
                    continue

                retrieved, retrieval_mode_used = _retrieve_examples_for_fixed_k(
                    question=p.question,
                    root_analysis=root_analysis,
                    retriever=retriever,
                    rag_top_k=rag_top_k,
                    retrieval_query_mode=retrieval_query_mode,
                )
                feedback_kind, feedback_context = _build_sdpo_feedback_context(
                    root_analysis=root_analysis,
                    root_judgment=root_judgment,
                    retrieved=retrieved,
                    retrieval_mode_used=retrieval_mode_used,
                )
                sdpo_prompt = _build_sdpo_regen_prompt(
                    question=p.question,
                    gold_answer=p.gold_answer,
                    benchmark=p.benchmark,
                    root_completion=_truncate_for_repair(root_completion),
                    root_analysis=root_analysis,
                    feedback_kind=feedback_kind,
                    feedback_context=feedback_context,
                )
                root_data.append({
                    "problem": p,
                    "root_prompt": root_prompt,
                    "root_prompt_tokens": root_prompt_tokens,
                    "root_completion": root_completion,
                    "root_completion_tokens": root_completion_tokens,
                    "root_analysis": root_analysis,
                    "root_judgment": root_judgment,
                    "trigger_fired": trigger_fired,
                    "retrieved": retrieved,
                    "retrieval_mode_used": retrieval_mode_used,
                    "selected_feedback_kind": feedback_kind,
                    "selected_feedback_context": feedback_context,
                    "sdpo_prompt": sdpo_prompt,
                })

            if not root_data:
                _log(f"  Chunk {chunk_i + 1}: no eligible feedback-gated problems, skipping regen phase")
                continue

            sdpo_prompts = [_render_chat_prompt(tokenizer, d["sdpo_prompt"]) for d in root_data]
            sdpo_sampling = SamplingParams(
                n=repair_candidates,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_new_tokens,
                skip_special_tokens=False,
            )
            t0 = datetime.now()
            regen_outputs = _safe_generate(llm, sdpo_prompts, sdpo_sampling)
            _log(
                f"  Phase 2 sdpo_regen: {len(root_data)} × {repair_candidates} candidates in "
                f"{(datetime.now()-t0).total_seconds():.1f}s"
            )

            for d, regen_out in zip(root_data, regen_outputs):
                pid = _stable_problem_id(d["problem"].benchmark, d["problem"].question)
                if not regen_out.outputs:
                    skipped_problems.append({
                        "problem_id": pid,
                        "benchmark": d["problem"].benchmark,
                        "question_preview": d["problem"].question[:80],
                        "phase": "sdpo_regen",
                        "reason": "no_outputs",
                    })
                    _log(f"  Skipping problem {pid[:8]}: sdpo_regen returned no outputs")
                    continue

                candidates: list[dict[str, Any]] = []
                for ci, cand in enumerate(regen_out.outputs):
                    completion = cand.text
                    analysis = analyze_completion_for_rag(completion)
                    judgment = judge_completion(completion, d["problem"].gold_answer)
                    meta_transition = evaluate_meta_transition(
                        root_analysis=d["root_analysis"],
                        retry_completion=completion,
                        retry_analysis=analysis,
                        retry_judgment=judgment,
                    )
                    candidates.append({
                        "candidate_id": f"regen_{ci}",
                        "prompt": d["sdpo_prompt"],
                        "prompt_tokens": len(regen_out.prompt_token_ids),
                        "completion_tokens": len(cand.token_ids),
                        "completion": completion,
                        "analysis": analysis,
                        "judgment": judgment,
                        "meta_transition": meta_transition,
                    })

                try:
                    selector = _select_best_candidate(
                        candidates,
                        d["problem"].gold_answer,
                        selector_mode=selector_mode,
                        weights=selector_weights,
                    )
                except ValueError as exc:
                    skipped_problems.append({
                        "problem_id": pid,
                        "benchmark": d["problem"].benchmark,
                        "question_preview": d["problem"].question[:80],
                        "phase": "selector",
                        "reason": f"selector_failed: {exc}",
                    })
                    _log(f"  Selector failed for problem {pid[:8]}: {exc}")
                    continue

                selected = selector["selected"]
                if require_correct_teacher and not _is_correct_judgment(selected.get("judgment")):
                    skipped_problems.append({
                        "problem_id": pid,
                        "benchmark": d["problem"].benchmark,
                        "question_preview": d["problem"].question[:80],
                        "phase": "selected_incorrect",
                        "reason": "best_regen_candidate_incorrect",
                    })
                    _log(f"  Skipping problem {pid[:8]}: selected sdpo_regen teacher is incorrect")
                    continue
                row = {
                    "problem_id": pid,
                    "question": d["problem"].question,
                    "gold_answer": d["problem"].gold_answer,
                    "benchmark": d["problem"].benchmark,
                    "metadata": d["problem"].metadata or {},
                    "generation_mode": "sdpo_regen",
                    "evidence_class": "side_evidence",
                    "source": "online_sdpo_regen",
                    "root_prompt": d["root_prompt"],
                    "root_prompt_tokens": d["root_prompt_tokens"],
                    "root_completion_tokens": d["root_completion_tokens"],
                    "root_completion": d["root_completion"],
                    "root_analysis": d["root_analysis"],
                    "root_judgment": d["root_judgment"],
                    "trigger_fired": d["trigger_fired"],
                    "repair_prompt": d["sdpo_prompt"],
                    "repair_budget": int(max(1, repair_candidates)),
                    "retrieval_mode_requested": retrieval_query_mode,
                    "retrieval_mode_used": d["retrieval_mode_used"],
                    "retrieved": [
                        {
                            "score": item["score"],
                            "score_breakdown": item.get("score_breakdown", {}),
                            "question": item["record"].question,
                            "source": item["record"].source,
                            "answer": item["record"].answer,
                        }
                        for item in d["retrieved"]
                    ],
                    "repair_candidates": selector["ranked_candidates"],
                    "selector": {
                        "selected_candidate_id": selector["selected_candidate_id"],
                        "selected_score": selector["selected_score"],
                        "selected_breakdown": selector["selected_breakdown"],
                        "score_margin": selector["score_margin"],
                        "selector_mode": selector["selector_mode"],
                        "weights": selector_weights or {
                            "correctness": 1.0,
                            "confidence_revision": 0.35,
                            "redirect_execution": 0.30,
                            "verify_execution": 0.15,
                            "meta_floor": 0.15,
                            "meta_commit_quality": 0.25,
                        },
                    },
                    "selected_completion": selected["completion"],
                    "selected_judgment": selected["judgment"],
                    "selected_analysis": selected["analysis"],
                    "selected_meta_transition": selected["meta_transition"],
                    "selected_prompt_kind": "sdpo_regen",
                    "selected_feedback_kind": d["selected_feedback_kind"],
                    "selected_feedback_context": d["selected_feedback_context"],
                    "retriever_active": retriever is not None,
                    "retrieval_enabled": rag_top_k > 0 and retrieval_query_mode != "none",
                    "retrieval_nonempty": bool(d["retrieved"]),
                }
                _append_trace(handle, row)
                new_rows.append(row)

            _log(f"  Chunk {chunk_i + 1} done: {len(new_rows)} new rows total")
    finally:
        handle.close()

    if skipped_problems:
        skipped_path = outdir / "skipped_problems.jsonl"
        with skipped_path.open("a", encoding="utf-8") as sh:
            for skip in skipped_problems:
                sh.write(json.dumps(skip, ensure_ascii=False) + "\n")
        _log(f"Skipped {len(skipped_problems)} problems (see {skipped_path})")

    return existing_rows + new_rows


def run_online_question_only_best_of_n_rollouts(
    *,
    llm,
    tokenizer,
    problems: list[OnlineSdpoProblem],
    output_dir: str | Path,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.95,
    seed: int = 42,
    num_candidates: int = 4,
    selector_mode: str = "correctness_only",
    selector_weights: dict[str, float] | None = None,
    require_correct_teacher: bool = False,
    chunk_size: int = 64,
    resume: bool = True,
) -> list[dict[str, Any]]:
    """Run question-only best-of-N with a single vLLM pass per chunk."""
    from vllm import SamplingParams

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    traces_path = outdir / "online_sdpo_traces.jsonl"

    done_ids: set[str] = set()
    existing_rows: list[dict[str, Any]] = []
    if resume and traces_path.exists():
        with traces_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    pid = row.get("problem_id")
                    if pid:
                        done_ids.add(pid)
                        existing_rows.append(row)
                except json.JSONDecodeError:
                    continue
        _log(f"Resume: {len(done_ids)} problems already done; will skip.")

    to_do = [p for p in problems if _stable_problem_id(p.benchmark, p.question) not in done_ids]
    _log(
        "To process: "
        f"{len(to_do)} of {len(problems)} (chunk_size={chunk_size}, N={num_candidates}, selector={selector_mode})"
    )
    if not to_do:
        return existing_rows

    handle = traces_path.open("a", encoding="utf-8")
    new_rows: list[dict[str, Any]] = []
    skipped_problems: list[dict[str, Any]] = []

    try:
        chunks = _chunks(to_do, chunk_size)
        for chunk_i, chunk in enumerate(chunks):
            _log(f"Chunk {chunk_i + 1}/{len(chunks)} ({len(chunk)} problems)")
            prompts = [_render_chat_prompt(tokenizer, p.question) for p in chunk]
            sampling = SamplingParams(
                n=max(1, int(num_candidates)),
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_new_tokens,
                skip_special_tokens=False,
            )
            t0 = datetime.now()
            outputs = _safe_generate(llm, prompts, sampling)
            _log(
                f"  Question-only pass: {len(chunk)} × {num_candidates} candidates in "
                f"{(datetime.now() - t0).total_seconds():.1f}s"
            )

            for p, prompt_text, output in zip(chunk, prompts, outputs):
                pid = _stable_problem_id(p.benchmark, p.question)
                if not output.outputs:
                    skipped_problems.append({
                        "problem_id": pid,
                        "benchmark": p.benchmark,
                        "question_preview": p.question[:80],
                        "phase": "question_only",
                        "reason": "no_outputs",
                    })
                    _log(f"  Skipping problem {pid[:8]}: question-only generation returned no outputs")
                    continue

                candidates: list[dict[str, Any]] = []
                for ci, candidate_output in enumerate(output.outputs):
                    completion = candidate_output.text
                    analysis = analyze_completion_for_rag(completion)
                    judgment = judge_completion(completion, p.gold_answer)
                    candidates.append({
                        "candidate_id": f"sample_{ci}",
                        "prompt": prompt_text,
                        "prompt_tokens": len(output.prompt_token_ids),
                        "completion_tokens": len(candidate_output.token_ids),
                        "completion": completion,
                        "analysis": analysis,
                        "judgment": judgment,
                        "meta_transition": {},
                    })

                try:
                    selector = _select_best_candidate(
                        candidates,
                        p.gold_answer,
                        selector_mode=selector_mode,
                        weights=selector_weights,
                    )
                except ValueError as exc:
                    skipped_problems.append({
                        "problem_id": pid,
                        "benchmark": p.benchmark,
                        "question_preview": p.question[:80],
                        "phase": "selector",
                        "reason": f"selector_failed: {exc}",
                    })
                    _log(f"  Selector failed for problem {pid[:8]}: {exc}")
                    continue

                selected = selector["selected"]
                if require_correct_teacher and not _is_correct_judgment(selected.get("judgment")):
                    skipped_problems.append({
                        "problem_id": pid,
                        "benchmark": p.benchmark,
                        "question_preview": p.question[:80],
                        "phase": "selected_incorrect",
                        "reason": "best_question_only_candidate_incorrect",
                    })
                    _log(f"  Skipping problem {pid[:8]}: selected question-only teacher is incorrect")
                    continue
                row = {
                    "problem_id": pid,
                    "question": p.question,
                    "gold_answer": p.gold_answer,
                    "benchmark": p.benchmark,
                    "metadata": p.metadata or {},
                    "generation_mode": "question_only_best_of_n",
                    "evidence_class": "mainline",
                    "source": "online_question_only_best_of_n",
                    "root_prompt": prompt_text,
                    "root_prompt_tokens": len(output.prompt_token_ids),
                    "root_completion_tokens": 0,
                    "root_completion": "",
                    "root_analysis": {},
                    "root_judgment": {},
                    "repair_prompt": "",
                    "repair_budget": int(max(1, num_candidates)),
                    "retrieval_mode_requested": "none",
                    "retrieval_mode_used": "none",
                    "retrieved": [],
                    "repair_candidates": selector["ranked_candidates"],
                    "selector": {
                        "selected_candidate_id": selector["selected_candidate_id"],
                        "selected_score": selector["selected_score"],
                        "selected_breakdown": selector["selected_breakdown"],
                        "score_margin": selector["score_margin"],
                        "selector_mode": selector["selector_mode"],
                        "weights": selector_weights or {
                            "correctness": 1.0,
                            "confidence_revision": 0.35,
                            "redirect_execution": 0.30,
                            "verify_execution": 0.15,
                            "meta_floor": 0.15,
                            "meta_commit_quality": 0.25,
                        },
                    },
                    "selected_completion": selected["completion"],
                    "selected_judgment": selected["judgment"],
                    "selected_analysis": selected["analysis"],
                    "selected_meta_transition": selected["meta_transition"],
                    "selected_prompt_kind": "question_only",
                    "selected_feedback_kind": "",
                    "selected_feedback_context": {},
                    "retriever_active": False,
                    "retrieval_enabled": False,
                    "retrieval_nonempty": False,
                }
                _append_trace(handle, row)
                new_rows.append(row)

            _log(f"  Chunk {chunk_i + 1} done: {len(new_rows)} new rows total")
    finally:
        handle.close()

    if skipped_problems:
        skipped_path = outdir / "skipped_problems.jsonl"
        with skipped_path.open("a", encoding="utf-8") as sh:
            for skip in skipped_problems:
                sh.write(json.dumps(skip, ensure_ascii=False) + "\n")
        _log(f"Skipped {len(skipped_problems)} problems (see {skipped_path})")

    return existing_rows + new_rows


def run_online_fixed_k_repair_rollouts(
    *,
    llm,  # vllm.LLM
    tokenizer,
    problems: list[OnlineSdpoProblem],
    output_dir: str | Path,
    retriever: TfidfExampleRetriever | None = None,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.95,
    seed: int = 42,
    rag_top_k: int = 1,
    repair_candidates: int = 4,
    retrieval_query_mode: str = "question_only",
    selector_mode: str = "reward_weighted",
    selector_weights: dict[str, float] | None = None,
    require_correct_teacher: bool = False,
    chunk_size: int = 64,
    resume: bool = True,
) -> list[dict[str, Any]]:
    """Run fixed-K repair via batched vLLM inference with streaming jsonl writes.

    Two-phase per chunk:
      Phase 1: batch root attempts (n=1)
      CPU: analyze, judge, retrieve, build repair prompts
      Phase 2: batch repair candidates (n=repair_candidates)
      CPU: select best, assemble trace row, append to jsonl

    Returns full list of rows (including resumed-from-jsonl).
    """
    from vllm import SamplingParams

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    traces_path = outdir / "online_sdpo_traces.jsonl"

    # Resume: load done IDs and pre-existing rows
    done_ids: set[str] = set()
    existing_rows: list[dict[str, Any]] = []
    if resume and traces_path.exists():
        with traces_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    pid = row.get("problem_id")
                    if pid:
                        done_ids.add(pid)
                        existing_rows.append(row)
                except json.JSONDecodeError:
                    continue
        _log(f"Resume: {len(done_ids)} problems already done; will skip.")

    # Filter to-do problems
    to_do = [p for p in problems if _stable_problem_id(p.benchmark, p.question) not in done_ids]
    _log(f"To process: {len(to_do)} of {len(problems)} (chunk_size={chunk_size}, K={repair_candidates})")

    if not to_do:
        return existing_rows

    # Open jsonl in append mode
    handle = traces_path.open("a", encoding="utf-8")
    new_rows: list[dict[str, Any]] = []
    skipped_problems: list[dict[str, Any]] = []

    try:
        chunks = _chunks(to_do, chunk_size)
        for chunk_i, chunk in enumerate(chunks):
            _log(f"Chunk {chunk_i + 1}/{len(chunks)} ({len(chunk)} problems)")

            # Phase 1: root attempts
            root_prompts = [_render_chat_prompt(tokenizer, p.question) for p in chunk]
            root_sampling = SamplingParams(
                n=1,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_new_tokens,
                seed=seed,
                skip_special_tokens=False,  # preserve <|meta|>...<|/meta|> blocks in output
            )
            t0 = datetime.now()
            root_outputs = _safe_generate(llm, root_prompts, root_sampling)
            _log(f"  Phase 1 root: {len(chunk)} prompts in {(datetime.now()-t0).total_seconds():.1f}s")

            # CPU: analyse + retrieve + build repair prompts
            root_data: list[dict[str, Any]] = []
            for p, root_prompt, root_out in zip(chunk, root_prompts, root_outputs):
                # Skip problems where root generation failed (placeholder from _safe_generate)
                if not root_out.outputs:
                    pid = _stable_problem_id(p.benchmark, p.question)
                    skipped_problems.append({
                        "problem_id": pid,
                        "benchmark": p.benchmark,
                        "question_preview": p.question[:80],
                        "phase": "root",
                        "reason": "no_outputs",
                    })
                    _log(f"  Skipping problem {pid[:8]}: root returned no outputs")
                    continue
                root_completion = root_out.outputs[0].text
                root_completion_tokens = len(root_out.outputs[0].token_ids)
                root_prompt_tokens = len(root_out.prompt_token_ids)
                root_analysis = analyze_completion_for_rag(root_completion)
                root_judgment = judge_completion(root_completion, p.gold_answer)

                retrieved, retrieval_mode_used = _retrieve_examples_for_fixed_k(
                    question=p.question,
                    root_analysis=root_analysis,
                    retriever=retriever,
                    rag_top_k=rag_top_k,
                    retrieval_query_mode=retrieval_query_mode,
                )

                # Truncate long root for repair prompt
                truncated_root = _truncate_for_repair(root_completion)
                repair_prompt = _build_fixed_repair_prompt(
                    question=p.question,
                    root_completion=truncated_root,
                    root_analysis=root_analysis,
                    retrieved=retrieved,
                )

                root_data.append({
                    "problem": p,
                    "root_prompt": root_prompt,
                    "root_prompt_tokens": root_prompt_tokens,
                    "root_completion": root_completion,
                    "root_completion_tokens": root_completion_tokens,
                    "root_analysis": root_analysis,
                    "root_judgment": root_judgment,
                    "retrieved": retrieved,
                    "retrieval_mode_used": retrieval_mode_used,
                    "repair_prompt": repair_prompt,
                })

            # Phase 2: repair candidates with n=K
            repair_prompts = [_render_chat_prompt(tokenizer, d["repair_prompt"]) for d in root_data]
            repair_sampling = SamplingParams(
                n=repair_candidates,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_new_tokens,
                skip_special_tokens=False,  # preserve <|meta|>...<|/meta|> blocks in output
                # seed intentionally omitted: vLLM duplicates all n samples when seed is set,
                # which would defeat the K-way diversity required for selector-based ranking.
            )
            t0 = datetime.now()
            repair_outputs = _safe_generate(llm, repair_prompts, repair_sampling)
            _log(
                f"  Phase 2 repair: {len(chunk)} × {repair_candidates} candidates in "
                f"{(datetime.now()-t0).total_seconds():.1f}s"
            )

            # CPU: per problem, build candidates, select best, assemble row
            for d, repair_out in zip(root_data, repair_outputs):
                # Skip problems where repair generation failed entirely (placeholder from _safe_generate)
                if not repair_out.outputs:
                    pid = _stable_problem_id(d["problem"].benchmark, d["problem"].question)
                    skipped_problems.append({
                        "problem_id": pid,
                        "benchmark": d["problem"].benchmark,
                        "question_preview": d["problem"].question[:80],
                        "phase": "repair",
                        "reason": "no_outputs",
                    })
                    _log(f"  Skipping problem {pid[:8]}: repair returned no outputs")
                    continue
                candidates: list[dict[str, Any]] = []
                for ci in range(len(repair_out.outputs)):
                    c = repair_out.outputs[ci]
                    completion = c.text
                    analysis = analyze_completion_for_rag(completion)
                    judgment = judge_completion(completion, d["problem"].gold_answer)
                    meta_transition = evaluate_meta_transition(
                        root_analysis=d["root_analysis"],
                        retry_completion=completion,
                        retry_analysis=analysis,
                        retry_judgment=judgment,
                    )
                    candidates.append({
                        "candidate_id": f"repair_{ci}",
                        "prompt": d["repair_prompt"],
                        "prompt_tokens": len(repair_out.prompt_token_ids),
                        "completion_tokens": len(c.token_ids),
                        "completion": completion,
                        "analysis": analysis,
                        "judgment": judgment,
                        "meta_transition": meta_transition,
                    })

                try:
                    selector = _select_best_candidate(
                        candidates,
                        d["problem"].gold_answer,
                        selector_mode=selector_mode,
                        weights=selector_weights,
                    )
                except ValueError as exc:
                    pid = _stable_problem_id(d["problem"].benchmark, d["problem"].question)
                    skipped_problems.append({
                        "problem_id": pid,
                        "benchmark": d["problem"].benchmark,
                        "question_preview": d["problem"].question[:80],
                        "phase": "selector",
                        "reason": f"selector_failed: {exc}",
                    })
                    _log(f"  Selector failed for problem {pid[:8]}: {exc}")
                    continue
                selected = selector["selected"]
                pid = _stable_problem_id(d["problem"].benchmark, d["problem"].question)
                if require_correct_teacher and not _is_correct_judgment(selected.get("judgment")):
                    skipped_problems.append({
                        "problem_id": pid,
                        "benchmark": d["problem"].benchmark,
                        "question_preview": d["problem"].question[:80],
                        "phase": "selected_incorrect",
                        "reason": "best_fixed_k_candidate_incorrect",
                    })
                    _log(f"  Skipping problem {pid[:8]}: selected fixed_k teacher is incorrect")
                    continue

                selected_feedback_context: dict[str, Any] = {}
                selected_feedback_kind = ""
                if d["retrieved"]:
                    selected_feedback_kind = "forced_rag"
                    selected_feedback_context = {
                        "lane": "fixed_k_repair",
                        "retrieval_mode": d["retrieval_mode_used"],
                        "evidence_items": [
                            {
                                "question": item["record"].question,
                                "source": item["record"].source,
                                "score": item["score"],
                                "score_breakdown": item.get("score_breakdown", {}),
                            }
                            for item in d["retrieved"]
                        ],
                    }

                row = {
                    "problem_id": pid,
                    "question": d["problem"].question,
                    "gold_answer": d["problem"].gold_answer,
                    "benchmark": d["problem"].benchmark,
                    "metadata": d["problem"].metadata or {},
                    "generation_mode": "fixed_k_repair",
                    "evidence_class": "side_evidence",
                    "source": "online_fixed_k_repair",
                    "root_prompt": d["root_prompt"],
                    "root_prompt_tokens": d["root_prompt_tokens"],
                    "root_completion_tokens": d["root_completion_tokens"],
                    "root_completion": d["root_completion"],
                    "root_analysis": d["root_analysis"],
                    "root_judgment": d["root_judgment"],
                    "repair_prompt": d["repair_prompt"],
                    "repair_budget": int(max(1, repair_candidates)),
                    "retrieval_mode_requested": retrieval_query_mode,
                    "retrieval_mode_used": d["retrieval_mode_used"],
                    "retrieved": [
                        {
                            "score": item["score"],
                            "score_breakdown": item.get("score_breakdown", {}),
                            "question": item["record"].question,
                            "source": item["record"].source,
                            "answer": item["record"].answer,
                        }
                        for item in d["retrieved"]
                    ],
                    "repair_candidates": selector["ranked_candidates"],
                    "selector": {
                        "selected_candidate_id": selector["selected_candidate_id"],
                        "selected_score": selector["selected_score"],
                        "selected_breakdown": selector["selected_breakdown"],
                        "score_margin": selector["score_margin"],
                        "selector_mode": selector["selector_mode"],
                        "weights": selector_weights or {
                            "correctness": 1.0,
                            "confidence_revision": 0.35,
                            "redirect_execution": 0.30,
                            "verify_execution": 0.15,
                            "meta_floor": 0.15,
                            "meta_commit_quality": 0.25,
                        },
                    },
                    "selected_completion": selected["completion"],
                    "selected_judgment": selected["judgment"],
                    "selected_analysis": selected["analysis"],
                    "selected_meta_transition": selected["meta_transition"],
                    "selected_prompt_kind": "fixed_k_repair",
                    "selected_feedback_kind": selected_feedback_kind,
                    "selected_feedback_context": selected_feedback_context,
                    # Retrieval bookkeeping (mirroring HF path's metadata)
                    "retriever_active": retriever is not None,
                    "retrieval_enabled": rag_top_k > 0 and retrieval_query_mode != "none",
                    "retrieval_nonempty": bool(d["retrieved"]),
                }
                _append_trace(handle, row)
                new_rows.append(row)

            _log(f"  Chunk {chunk_i + 1} done: {len(new_rows)} new rows total")
    finally:
        handle.close()

    # Persist skipped-problem log alongside traces for later audit + summary reconciliation
    if skipped_problems:
        skipped_path = outdir / "skipped_problems.jsonl"
        with skipped_path.open("a", encoding="utf-8") as sh:
            for skip in skipped_problems:
                sh.write(json.dumps(skip, ensure_ascii=False) + "\n")
        _log(f"Skipped {len(skipped_problems)} problems (see {skipped_path})")

    return existing_rows + new_rows


def write_online_sdpo_outputs(
    *,
    rows: list[dict[str, Any]],
    output_dir: str | Path,
    source_tag: str = "online_sdpo_regen",
    mode: str = "sdpo_regen",
    claim_bearing: bool = False,
    traces_already_written: bool = False,
) -> dict[str, Any]:
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    traces_path = outdir / "online_sdpo_traces.jsonl"
    if not traces_already_written:
        with traces_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    df = build_self_distill_dataframe(
        rows,
        mode=mode,
        source_tag=source_tag,
        claim_bearing=claim_bearing,
    )
    parquet_path = outdir / "online_sdpo_regen.parquet"
    if not df.empty:
        df.to_parquet(parquet_path, index=False)
    elif claim_bearing:
        raise ValueError(
            f"Claim-bearing online self-distill build produced 0 rows for mode={mode}. "
            "This would silently invalidate the base/meta comparison."
        )

    summary = summarize_self_distill_dataframe(df)
    evidence_class = "mainline" if mode == "epistemic" and source_tag == "online_question_only_best_of_n" else "side_evidence"

    # Reconcile skipped-problem log (written by run_online_fixed_k_repair_rollouts)
    # into the summary so downstream analyses see attempt vs success counts.
    skipped_path = outdir / "skipped_problems.jsonl"
    skipped_count = 0
    skipped_by_phase: dict[str, int] = {}
    if skipped_path.exists():
        with skipped_path.open("r", encoding="utf-8") as sh:
            for line in sh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                skipped_count += 1
                phase = str(entry.get("phase", "unknown"))
                skipped_by_phase[phase] = skipped_by_phase.get(phase, 0) + 1

    payload = {
        "num_rollouts": len(rows),
        "num_dataset_rows": int(len(df)),
        "num_sdpo_regen_rows": int(len(df)),
        "dataset_mode": mode,
        "evidence_class": evidence_class,
        "claim_bearing": bool(claim_bearing),
        "trace_path": str(traces_path),
        "parquet_path": str(parquet_path) if not df.empty else "",
        "summary": summary,
        "skipped_count": skipped_count,
        "skipped_by_phase": skipped_by_phase,
        "retrieval": {
            "rows_with_retriever": int(sum(bool(row.get("retriever_active", False)) for row in rows)),
            "rows_with_retrieval_enabled": int(sum(bool(row.get("retrieval_enabled", False)) for row in rows)),
            "rows_with_nonempty_retrieval": int(sum(bool(row.get("retrieval_nonempty", False)) for row in rows)),
            "retrieval_nonempty_rate": (
                float(sum(bool(row.get("retrieval_nonempty", False)) for row in rows)) / float(len(rows))
                if rows else 0.0
            ),
            "requested_modes": sorted({
                str(row.get("retrieval_mode_requested", ""))
                for row in rows
                if row.get("retrieval_mode_requested")
            }),
            "used_modes": sorted({
                str(row.get("retrieval_mode_used", ""))
                for row in rows
                if row.get("retrieval_mode_used")
            }),
        },
    }
    (outdir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


__all__ = [
    "OnlineSdpoProblem",
    "load_online_problems",
    "load_retriever",
    "run_online_question_only_best_of_n_rollouts",
    "run_online_fixed_k_repair_rollouts",
    "run_online_sdpo_rollouts",
    "write_online_sdpo_outputs",
]
