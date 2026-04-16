"""On-policy self-distill artifact builders (vLLM batched).

Two paths are supported:

1. `sdpo_regen`: trigger-gated, feedback-conditioned side-evidence collection.
   (Currently stubbed for vLLM; P4 lane — see `run_online_sdpo_rollouts`.)
2. `fixed_k_repair`: fair claim-bearing path with the same root/repair budget for
   base and meta models, plus reward-ranked candidate selection. Runs as batched
   vLLM with streaming jsonl writes and resume support.
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
from src.training.self_distill.builders import (
    build_self_distill_dataframe,
    summarize_self_distill_dataframe,
)
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
    }
    payload = [{"content": completion}]
    ground_truth = [gold_answer]
    components = {
        "correctness": correctness_reward(payload, ground_truth)[0],
        "confidence_revision": confidence_revision_reward_v2(payload, ground_truth)[0],
        "redirect_execution": redirect_execution_reward_v2(payload, ground_truth)[0],
        "verify_execution": verify_execution_reward_v2(payload, ground_truth)[0],
        "meta_floor": confidence_omission_floor(payload, ground_truth)[0],
    }
    total = 0.0
    for key, weight in active_weights.items():
        total += float(weight) * float(components.get(key, 0.0))
    components["total"] = total
    return components


def _select_best_candidate(
    candidates: list[dict[str, Any]],
    gold_answer: str,
    *,
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
    """SDPO-regen mode (P4 lane). Not implemented in vLLM path yet.

    For triggered-retry logic, port the on-policy retry lane from
    `rq3_pipeline.run_curriculum_retry_lane` using batched vLLM. This is a P4
    lane and not blocking P1 (fixed_k_repair).
    """
    raise NotImplementedError(
        "sdpo_regen vLLM path not implemented yet (P4 deferred). Use fixed_k_repair mode."
    )


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
    selector_weights: dict[str, float] | None = None,
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
                    _log(f"  Skipping problem {_stable_problem_id(d['problem'].benchmark, d['problem'].question)[:8]}: repair returned no outputs")
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
                    selector = _select_best_candidate(candidates, d["problem"].gold_answer, weights=selector_weights)
                except ValueError as exc:
                    _log(f"  Selector failed for problem {_stable_problem_id(d['problem'].benchmark, d['problem'].question)[:8]}: {exc}")
                    continue
                selected = selector["selected"]

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
                    "problem_id": _stable_problem_id(d["problem"].benchmark, d["problem"].question),
                    "question": d["problem"].question,
                    "gold_answer": d["problem"].gold_answer,
                    "benchmark": d["problem"].benchmark,
                    "metadata": d["problem"].metadata or {},
                    "generation_mode": "fixed_k_repair",
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
                        "weights": selector_weights or {
                            "correctness": 1.0,
                            "confidence_revision": 0.35,
                            "redirect_execution": 0.30,
                            "verify_execution": 0.15,
                            "meta_floor": 0.15,
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
    payload = {
        "num_rollouts": len(rows),
        "num_dataset_rows": int(len(df)),
        "num_sdpo_regen_rows": int(len(df)),
        "dataset_mode": mode,
        "claim_bearing": bool(claim_bearing),
        "trace_path": str(traces_path),
        "parquet_path": str(parquet_path) if not df.empty else "",
        "summary": summary,
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
    "run_online_fixed_k_repair_rollouts",
    "run_online_sdpo_rollouts",
    "write_online_sdpo_outputs",
]
