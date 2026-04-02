#!/usr/bin/env python3
"""Smoke test for redirect-time RAG and one-example adaptation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

from src.curriculum.control_rag import (
    ExampleRecord,
    TfidfExampleRetriever,
    run_redirect_rag_pass,
)
from src.curriculum.one_example_adapt import run_one_example_adaptation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--output", default="results/control_rag_smoke.json")
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

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()

    question = "Solve x + 7 = 12."
    first_completion = (
        "<|meta|>\n"
        "confidence: 0.38\n"
        "Something feels off. I may be forcing the wrong transformation and should switch back "
        "to isolating the variable directly.\n"
        "<|/meta|>\n"
        "I am not sure about the current route."
    )
    rag_run = run_redirect_rag_pass(
        model,
        tokenizer,
        question,
        first_completion,
        retriever,
        top_k=1,
        max_new_tokens=48,
    )
    if not rag_run["rag_used"]:
        raise RuntimeError("RAG smoke failed: retrieval was not triggered")
    if "Solve x + 4 = 9." not in rag_run["rag_prompt"]:
        raise RuntimeError("RAG smoke failed: retrieved example was not injected into the retry prompt")

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

    payload = {
        "model": args.model,
        "question": question,
        "rag_run": rag_run,
        "one_example_adapt": adapt_summary,
    }
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
