"""On-policy self-distill artifact builders.

Two paths are supported:

1. `sdpo_regen`: trigger-gated, feedback-conditioned side-evidence collection.
2. `fixed_k_repair`: fair claim-bearing path with the same root/repair budget for
   base and meta models, plus reward-ranked candidate selection.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.curriculum.control_rag import (
    TfidfExampleRetriever,
    analyze_completion_for_rag,
    build_incontext_user_prompt,
    build_retrieval_query_bundle,
    generate_from_messages,
    load_example_bank,
)
from src.curriculum.rq3_pipeline import (
    evaluate_meta_transition,
    judge_completion,
    run_curriculum_retry_lane,
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


def run_online_sdpo_case(
    *,
    model,
    tokenizer,
    problem: OnlineSdpoProblem,
    retriever: TfidfExampleRetriever | None = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
    rag_top_k: int = 1,
) -> dict[str, Any]:
    root_completion, root_prompt, root_prompt_tokens, root_completion_tokens = generate_from_messages(
        model,
        tokenizer,
        [{"role": "user", "content": problem.question}],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    root_analysis = analyze_completion_for_rag(root_completion)
    root_judgment = judge_completion(root_completion, problem.gold_answer)

    curriculum_retry = run_curriculum_retry_lane(
        question=problem.question,
        gold_answer=problem.gold_answer,
        root_completion=root_completion,
        retriever=retriever,
        model=model,
        tokenizer=tokenizer,
        top_k=rag_top_k,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    if curriculum_retry.retry_judgment is not None:
        curriculum_retry.improved_over_root = (
            curriculum_retry.retry_judgment["is_correct"] and not root_judgment["is_correct"]
        )

    return {
        "question": problem.question,
        "gold_answer": problem.gold_answer,
        "benchmark": problem.benchmark,
        "metadata": problem.metadata or {},
        "root_prompt": root_prompt,
        "root_prompt_tokens": root_prompt_tokens,
        "root_completion_tokens": root_completion_tokens,
        "root_completion": root_completion,
        "root_analysis": root_analysis,
        "root_judgment": root_judgment,
        "trigger_fired": bool(root_analysis.get("should_retrieve", False)),
        "curriculum_retry": curriculum_retry.to_dict(),
    }


def run_online_fixed_k_repair_case(
    *,
    model,
    tokenizer,
    problem: OnlineSdpoProblem,
    retriever: TfidfExampleRetriever | None = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
    rag_top_k: int = 0,
    repair_candidates: int = 4,
    retrieval_query_mode: str = "question_only",
    selector_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    root_completion, root_prompt, root_prompt_tokens, root_completion_tokens = generate_from_messages(
        model,
        tokenizer,
        [{"role": "user", "content": problem.question}],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    root_analysis = analyze_completion_for_rag(root_completion)
    root_judgment = judge_completion(root_completion, problem.gold_answer)
    retrieved, retrieval_mode_used = _retrieve_examples_for_fixed_k(
        question=problem.question,
        root_analysis=root_analysis,
        retriever=retriever,
        rag_top_k=rag_top_k,
        retrieval_query_mode=retrieval_query_mode,
    )
    repair_prompt = _build_fixed_repair_prompt(
        question=problem.question,
        root_completion=root_completion,
        root_analysis=root_analysis,
        retrieved=retrieved,
    )

    candidates: list[dict[str, Any]] = []
    for idx in range(max(1, repair_candidates)):
        completion, prompt_text, prompt_tokens, completion_tokens = generate_from_messages(
            model,
            tokenizer,
            [{"role": "user", "content": repair_prompt}],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        analysis = analyze_completion_for_rag(completion)
        judgment = judge_completion(completion, problem.gold_answer)
        meta_transition = evaluate_meta_transition(
            root_analysis=root_analysis,
            retry_completion=completion,
            retry_analysis=analysis,
            retry_judgment=judgment,
        )
        candidates.append({
            "candidate_id": f"repair_{idx}",
            "prompt": prompt_text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "completion": completion,
            "analysis": analysis,
            "judgment": judgment,
            "meta_transition": meta_transition,
        })

    selector = _select_best_candidate(candidates, problem.gold_answer, weights=selector_weights)
    selected = selector["selected"]
    selected_feedback_context = {}
    selected_feedback_kind = ""
    if retrieved:
        selected_feedback_kind = "forced_rag"
        selected_feedback_context = {
            "lane": "fixed_k_repair",
            "retrieval_mode": retrieval_mode_used,
            "evidence_items": [
                {
                    "question": item["record"].question,
                    "source": item["record"].source,
                    "score": item["score"],
                    "score_breakdown": item.get("score_breakdown", {}),
                }
                for item in retrieved
            ],
        }

    return {
        "question": problem.question,
        "gold_answer": problem.gold_answer,
        "benchmark": problem.benchmark,
        "metadata": problem.metadata or {},
        "generation_mode": "fixed_k_repair",
        "source": "online_fixed_k_repair",
        "root_prompt": root_prompt,
        "root_prompt_tokens": root_prompt_tokens,
        "root_completion_tokens": root_completion_tokens,
        "root_completion": root_completion,
        "root_analysis": root_analysis,
        "root_judgment": root_judgment,
        "repair_prompt": repair_prompt,
        "repair_budget": int(max(1, repair_candidates)),
        "retriever_active": bool(retriever is not None),
        "retrieval_enabled": bool(retriever is not None and rag_top_k > 0 and retrieval_query_mode != "none"),
        "retrieval_nonempty": bool(retrieved),
        "retrieval_mode_requested": retrieval_query_mode,
        "retrieval_mode_used": retrieval_mode_used,
        "retrieved": [
            {
                "score": item["score"],
                "score_breakdown": item.get("score_breakdown", {}),
                "question": item["record"].question,
                "source": item["record"].source,
                "answer": item["record"].answer,
            }
            for item in retrieved
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
    }


def run_online_sdpo_rollouts(
    *,
    model,
    tokenizer,
    problems: list[OnlineSdpoProblem],
    retriever: TfidfExampleRetriever | None = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
    rag_top_k: int = 1,
) -> list[dict[str, Any]]:
    return [
        run_online_sdpo_case(
            model=model,
            tokenizer=tokenizer,
            problem=problem,
            retriever=retriever,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            rag_top_k=rag_top_k,
        )
        for problem in problems
    ]


def run_online_fixed_k_repair_rollouts(
    *,
    model,
    tokenizer,
    problems: list[OnlineSdpoProblem],
    retriever: TfidfExampleRetriever | None = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
    rag_top_k: int = 0,
    repair_candidates: int = 4,
    retrieval_query_mode: str = "question_only",
    selector_weights: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    return [
        run_online_fixed_k_repair_case(
            model=model,
            tokenizer=tokenizer,
            problem=problem,
            retriever=retriever,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            rag_top_k=rag_top_k,
            repair_candidates=repair_candidates,
            retrieval_query_mode=retrieval_query_mode,
            selector_weights=selector_weights,
        )
        for problem in problems
    ]


def write_online_sdpo_outputs(
    *,
    rows: list[dict[str, Any]],
    output_dir: str | Path,
    source_tag: str = "online_sdpo_regen",
    mode: str = "sdpo_regen",
    claim_bearing: bool = False,
) -> dict[str, Any]:
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    traces_path = outdir / "online_sdpo_traces.jsonl"
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
    "run_online_fixed_k_repair_case",
    "run_online_fixed_k_repair_rollouts",
    "run_online_sdpo_case",
    "run_online_sdpo_rollouts",
    "write_online_sdpo_outputs",
]
