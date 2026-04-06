#!/usr/bin/env python3
"""Generate probe-training rollouts with prefix-conditioned targets."""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
import sys
from typing import Callable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.curriculum.control_rag import build_model_inputs
from src.metacot.prompt import META_START, META_END
from src.training.rewards import _check_correctness


def _shard_output_path(output_path: Path, shard_index: int, num_shards: int) -> Path:
    if num_shards <= 1:
        return output_path
    return output_path.with_suffix(f".shard{shard_index}of{num_shards}.parquet")


def _summary_path(output_path: Path) -> Path:
    return output_path.with_suffix(".summary.json")


def _progress_path(output_path: Path) -> Path:
    return output_path.with_suffix(".progress.json")


def _partial_path(output_path: Path) -> Path:
    return output_path.with_suffix(".partial.parquet")


def _select_shard(items: list[dict], shard_index: int, num_shards: int) -> list[dict]:
    if num_shards <= 1:
        return list(items)
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"invalid shard {shard_index}/{num_shards}")
    return [item for idx, item in enumerate(items) if idx % num_shards == shard_index]


def _extract_math_answer(row: dict) -> str:
    answer = row.get("answer")
    if answer:
        return str(answer)
    solution = str(row.get("solution", ""))
    boxed = re.findall(r'\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}', solution)
    if boxed:
        return boxed[-1].strip()
    return solution


def load_probe_problems(gsm_n: int, math_n: int, seed: int) -> list[dict]:
    from datasets import load_dataset

    random.seed(seed)
    problems: list[dict] = []

    gsm = load_dataset("openai/gsm8k", "main", split="train")
    for idx, row in enumerate(gsm):
        if idx >= gsm_n:
            break
        answer = row["answer"].split("####")[-1].strip() if "####" in row["answer"] else row["answer"]
        problems.append({
            "problem_id": f"gsm8k_train_{idx}",
            "question": row["question"],
            "gold_answer": answer,
            "category": "gsm8k",
            "difficulty": "easy_medium",
            "source": "gsm8k_train",
        })

    math_rows = []
    for cfg in [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    ]:
        ds = load_dataset("EleutherAI/hendrycks_math", cfg, split="train")
        for row in ds:
            gt = _extract_math_answer(row)
            if not gt:
                continue
            math_rows.append({
                "problem_id": f"{cfg}_{len(math_rows)}",
                "question": row["problem"],
                "gold_answer": gt,
                "category": cfg,
                "difficulty": "math_train",
                "source": "hendrycks_math_train",
            })

    random.shuffle(math_rows)
    problems.extend(math_rows[:math_n])
    random.shuffle(problems)
    return problems


def _iter_meta_completion_prefixes(completion: str) -> list[tuple[int, str]]:
    pattern = re.compile(
        rf"{re.escape(META_START)}(.*?){re.escape(META_END)}",
        re.IGNORECASE | re.DOTALL,
    )
    return [(idx, completion[:match.end()]) for idx, match in enumerate(pattern.finditer(completion))]


def _empirical_prefix_success(
    prefix_payloads: list[str],
    *,
    gold_answer: str,
    continuation_sampler: Callable[[str], list[str]],
) -> list[float]:
    probs: list[float] = []
    for payload in prefix_payloads:
        continuations = continuation_sampler(payload)
        if not continuations:
            probs.append(0.0)
            continue
        correct = sum(1 for text in continuations if _check_correctness(text, gold_answer))
        probs.append(correct / len(continuations))
    return probs


def _sample_continuations(
    model,
    tokenizer,
    prefix_text: str,
    *,
    n: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    max_prompt_tokens: int,
) -> list[str]:
    import torch

    encoded = tokenizer(
        prefix_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_tokens,
    )
    encoded = {k: v.to(model.device) for k, v in encoded.items()}
    outputs = []
    with torch.no_grad():
        for _ in range(n):
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
            )
            prompt_len = int(encoded["input_ids"].shape[1])
            outputs.append(tokenizer.decode(generated[0][prompt_len:], skip_special_tokens=False))
    return outputs


def _build_row(
    *,
    prob: dict,
    prompt_text: str,
    completion: str,
    is_correct: bool,
    meta_prefix_target_probs: list[float],
    continuations_per_prefix: int,
) -> dict:
    return {
        **prob,
        "prompt_text": prompt_text,
        "completion": completion,
        "is_correct": bool(is_correct),
        "completion_length_chars": len(completion),
        "meta_prefix_count": len(meta_prefix_target_probs),
        "meta_prefix_target_probs": meta_prefix_target_probs,
        "probe_target_kind": "empirical_p_correct_given_prefix",
        "continuations_per_prefix": continuations_per_prefix,
    }


def _write_rollout_sidecars(rows: list[dict], output_path: Path, continuations_per_prefix: int, *, final: bool) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    target_path = output_path if final else _partial_path(output_path)
    df.to_parquet(target_path, index=False)

    correct = sum(bool(r["is_correct"]) for r in rows)
    summary = {
        "n_rows": len(rows),
        "n_correct": correct,
        "n_incorrect": len(rows) - correct,
        "n_prefix_targets": sum(int(r.get("meta_prefix_count", 0) or 0) for r in rows),
        "continuations_per_prefix": continuations_per_prefix,
        "status": "final" if final else "partial",
    }
    _summary_path(output_path).write_text(json.dumps(summary, indent=2))

    progress = {
        "completed_problem_ids": sorted(str(r["problem_id"]) for r in rows),
        "n_completed": len(rows),
    }
    _progress_path(output_path).write_text(json.dumps(progress, indent=2))


def merge_rollout_shards(output_path: Path, num_shards: int) -> dict:
    shard_frames = []
    for shard_index in range(num_shards):
        shard_path = _shard_output_path(output_path, shard_index, num_shards)
        if not shard_path.exists():
            raise FileNotFoundError(shard_path)
        shard_frames.append(pd.read_parquet(shard_path))

    merged = pd.concat(shard_frames, ignore_index=True)
    if "problem_id" in merged.columns:
        merged = merged.drop_duplicates(subset=["problem_id"], keep="last").reset_index(drop=True)
    merged.to_parquet(output_path, index=False)

    rows = merged.to_dict("records")
    continuations_per_prefix = int(rows[0]["continuations_per_prefix"]) if rows else 0
    _write_rollout_sidecars(rows, output_path, continuations_per_prefix, final=True)

    summary = {
        "n_rows": len(rows),
        "n_correct": int(sum(bool(r["is_correct"]) for r in rows)),
        "n_incorrect": int(len(rows) - sum(bool(r["is_correct"]) for r in rows)),
        "n_prefix_targets": int(sum(int(r.get("meta_prefix_count", 0) or 0) for r in rows)),
        "continuations_per_prefix": continuations_per_prefix,
        "num_shards": num_shards,
    }
    _summary_path(output_path).write_text(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--gsm-n", type=int, default=256)
    parser.add_argument("--math-n", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument("--continuations-per-prefix", type=int, default=4)
    parser.add_argument("--flush-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--merge-shards", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.merge_shards:
        summary = merge_rollout_shards(output_path, args.num_shards)
        print(f"merged:{output_path}")
        print(json.dumps(summary, indent=2))
        return 0

    if not args.model_path:
        raise ValueError("--model-path is required unless --merge-shards is set")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed)
    problems = load_probe_problems(args.gsm_n, args.math_n, args.seed)
    problems = _select_shard(problems, args.shard_index, args.num_shards)
    shard_output_path = _shard_output_path(output_path, args.shard_index, args.num_shards)
    partial_path = _partial_path(shard_output_path)

    use_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if use_cuda else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    if use_cuda:
        model = model.to("cuda")
    model.eval()

    rows = []
    completed_problem_ids: set[str] = set()
    total_prefixes = 0

    if args.resume and (partial_path.exists() or shard_output_path.exists()):
        resume_path = partial_path if partial_path.exists() else shard_output_path
        existing = pd.read_parquet(resume_path)
        rows = existing.to_dict("records")
        completed_problem_ids = {str(row["problem_id"]) for row in rows}
        total_prefixes = sum(int(row.get("meta_prefix_count", 0) or 0) for row in rows)
        print(
            f"resuming from {resume_path}: "
            f"{len(rows)} rows, {len(completed_problem_ids)} completed problems, "
            f"{total_prefixes} prefix targets"
        )

    pending = [prob for prob in problems if str(prob["problem_id"]) not in completed_problem_ids]

    for local_idx, prob in enumerate(pending, start=1):
        messages = [{"role": "user", "content": prob["question"]}]
        prompt_text, inputs = build_model_inputs(
            tokenizer,
            messages,
            device=model.device,
            add_generation_prompt=True,
            max_prompt_tokens=args.max_prompt_tokens,
        )
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tokenizer.pad_token_id,
            )
        prompt_len = int(inputs["input_ids"].shape[1])
        completion = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=False)
        is_correct = _check_correctness(completion, prob["gold_answer"])

        completion_prefixes = _iter_meta_completion_prefixes(completion)
        prefix_payloads = [prompt_text + prefix for _, prefix in completion_prefixes]
        meta_prefix_target_probs = _empirical_prefix_success(
            prefix_payloads,
            gold_answer=prob["gold_answer"],
            continuation_sampler=lambda payload: _sample_continuations(
                model,
                tokenizer,
                payload,
                n=args.continuations_per_prefix,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                max_prompt_tokens=args.max_prompt_tokens,
            ),
        )

        rows.append(_build_row(
            prob=prob,
            prompt_text=prompt_text,
            completion=completion,
            is_correct=is_correct,
            meta_prefix_target_probs=meta_prefix_target_probs,
            continuations_per_prefix=args.continuations_per_prefix,
        ))
        total_prefixes += len(meta_prefix_target_probs)

        global_idx = len(rows)
        if global_idx % args.flush_every == 0:
            _write_rollout_sidecars(rows, shard_output_path, args.continuations_per_prefix, final=False)

        if global_idx % 25 == 0:
            correct = sum(r["is_correct"] for r in rows)
            prefix_targets = sum(r["meta_prefix_count"] for r in rows)
            print(
                f"{global_idx}/{len(problems)} generated, "
                f"acc={correct/len(rows):.3f}, prefix_targets={prefix_targets}"
            )

    _write_rollout_sidecars(rows, shard_output_path, args.continuations_per_prefix, final=True)
    correct = sum(r["is_correct"] for r in rows)
    summary = {
        "n_rows": len(rows),
        "n_correct": correct,
        "n_incorrect": len(rows) - correct,
        "n_prefix_targets": total_prefixes,
        "continuations_per_prefix": args.continuations_per_prefix,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
    }
    _summary_path(shard_output_path).write_text(json.dumps(summary, indent=2))
    print(f"saved:{shard_output_path}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
