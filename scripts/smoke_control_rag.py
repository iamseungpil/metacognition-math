#!/usr/bin/env python3
"""Smoke test for redirect-time RAG and one-example adaptation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.curriculum.control_rag import (
    ExampleRecord,
    TfidfExampleRetriever,
    analyze_completion_for_rag,
    build_incontext_user_prompt,
    build_retrieval_query,
    run_redirect_rag_pass,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--output", default="results/control_rag_smoke.json")
    parser.add_argument("--require_model_smoke", action="store_true")
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Alias for retrieval-only smoke; kept for launcher compatibility.",
    )
    args = parser.parse_args()

    exemplar = ExampleRecord(
        question="Solve x + 4 = 9.",
        solution="Subtract 4 from both sides to isolate x, so x = 5. \\boxed{5}",
        answer="5",
        source="synthetic",
        metadata={"topic": "linear equation", "difficulty": "easy"},
    )
    distractor = ExampleRecord(
        question="Compute 3 + 3.",
        solution="3 + 3 = 6. \\boxed{6}",
        answer="6",
        source="synthetic",
        metadata={"topic": "arithmetic", "difficulty": "easy"},
    )
    retriever = TfidfExampleRetriever([exemplar, distractor])

    question = "Solve x + 7 = 12."
    first_completion = (
        "<|meta|>\n"
        "confidence: 0.38\n"
        "Something feels off. I may be forcing the wrong transformation and should switch back "
        "to isolating the variable directly.\n"
        "<|/meta|>\n"
        "I am not sure about the current route."
    )
    analysis = analyze_completion_for_rag(first_completion)
    query = build_retrieval_query(question, analysis)
    hits = retriever.search(query, top_k=1)
    rag_prompt = build_incontext_user_prompt(question, analysis, hits)
    if not analysis["should_retrieve"] or not hits:
        raise RuntimeError("RAG smoke failed: retrieval was not triggered")
    if "Solve x + 4 = 9." not in rag_prompt:
        raise RuntimeError("RAG smoke failed: retrieved example was not injected into the retry prompt")

    payload = {
        "model": args.model,
        "question": question,
        "rag_run": {
            "rag_used": True,
            "retrieved": [{"question": item["record"].question, "score": item["score"]} for item in hits],
            "rag_prompt": rag_prompt,
        },
        "model_smoke": {"attempted": False, "skipped_reason": ""},
    }

    if args.skip_model:
        payload["model_smoke"]["skipped_reason"] = "skip_model_flag"
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(json.dumps({
            "rag_used": True,
            "retrieved_questions": payload["rag_run"]["retrieved"],
            "model_smoke_skipped": payload["model_smoke"]["skipped_reason"],
            "output": str(output_path),
        }, indent=2, ensure_ascii=False))
        return

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from src.curriculum.one_example_adapt import run_one_example_adaptation
    except ImportError as exc:
        if args.require_model_smoke:
            raise
        # Retrieval/query construction smoke still passes without model dependencies.
        payload["model_smoke"]["skipped_reason"] = f"missing_dependency:{type(exc).__name__}"
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(json.dumps({
            "rag_used": True,
            "retrieved_questions": payload["rag_run"]["retrieved"],
            "model_smoke_skipped": payload["model_smoke"]["skipped_reason"],
            "output": str(output_path),
        }, indent=2, ensure_ascii=False))
        return

    payload = {
        "model": args.model,
        "question": question,
        "rag_run": payload["rag_run"],
        "model_smoke": {"attempted": True, "skipped_reason": ""},
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()

    rag_run = run_redirect_rag_pass(
        model,
        tokenizer,
        question,
        first_completion,
        retriever,
        top_k=1,
        max_new_tokens=48,
    )
    one_example_messages = [
        {"role": "user", "content": exemplar.question},
        {"role": "assistant", "content": exemplar.solution},
    ]
    adapt_summary = run_one_example_adaptation(
        model_name_or_path=args.model,
        example_messages=one_example_messages,
        target_question=question,
        output_dir="results/one_example_adapt_smoke",
        max_steps=1,
        max_new_tokens=48,
        device="cpu",
    )
    if not Path(adapt_summary["output_dir"]).exists():
        raise RuntimeError("One-example adaptation smoke failed: output directory was not created")
    if not (Path(adapt_summary["output_dir"]) / "one_example_summary.json").exists():
        raise RuntimeError("One-example adaptation smoke failed: summary file missing")
    payload["rag_run"] = rag_run
    payload["one_example_adapt"] = adapt_summary

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(json.dumps({
        "rag_used": rag_run["rag_used"],
        "retrieved_questions": rag_run["retrieved"],
        "adapt_output_dir": adapt_summary["output_dir"],
        "output": str(output_path),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
