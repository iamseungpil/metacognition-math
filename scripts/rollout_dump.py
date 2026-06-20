"""Stage-0a: dump the STUDENT's real rollouts for the confidence-redirect/verify
teacher-distill build (spec 2026-06-20-confidence-rv-stage-checks-and-datagen §A).

The local teacher build (Stage-0b) needs the STUDENT's REAL wrong attempts (to
anchor redirect) and per-problem self-consistency (the calibrated confidence label)
— neither is saved by pg0_yield_pilot. This script rolls the SFT-init student out
n times on the easy/medium problems and dumps, per problem:

    question, gold, difficulty, rollouts (list[str]), grades (list[int]),
    answers (list[str]), pass_rate

to a parquet, then uploads it to HF so the teacher build can run LOCALLY (TRAPI on
the host) with no further GPU. easy/medium only (PG0: redirect hurts hard).

GPU/vLLM wiring mirrors scripts/pg0_yield_pilot.py (the proven rollout path); the
pure pieces (_load_pool, _render) are reused from it.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pg0_yield_pilot import _load_pool, _render
from src.training.rewards import _check_correctness

_ANCHOR_DIFFICULTIES = ("easy", "medium")
_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
_ANSWER_IS_RE = re.compile(r"(?i)answer\s+is\s*[:=]?\s*\$?([^\s$.]+)")


def _extract_answer(text: str) -> str:
    """Best-effort final-answer string for confidently-wrong / majority gating.
    The boxed value if present, else the 'answer is X' tail, else ''."""
    if not text:
        return ""
    boxed = _BOXED_RE.findall(text)
    if boxed:
        return boxed[-1].strip()
    m = list(_ANSWER_IS_RE.finditer(text))
    return m[-1].group(1).strip() if m else ""


def main() -> None:  # pragma: no cover - GPU wiring; pure helpers are unit-tested elsewhere
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/scratch/models/v8_meta_inside_strict_sft")
    parser.add_argument("--train_parquet",
                        default="/scratch/metacognition/data/verl_train_meta_mix.parquet")
    parser.add_argument("--n_problems", type=int, default=300,
                        help="easy/medium problems to roll out (PG-build dev slice)")
    parser.add_argument("--rollout_k", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--tp_size", type=int, default=4)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_path", default="/scratch/eval_results/rv_rollout_dump.parquet")
    parser.add_argument("--hf_repo", default="iamseungpil/metacot")
    parser.add_argument("--hf_path", default="data/rv_rollout_dump.parquet")
    args = parser.parse_args()

    # _load_pool over-reads, then we keep easy/medium up to n_problems.
    raw = _load_pool(args.train_parquet, pool_size=10 * args.n_problems)
    pool = [p for p in raw
            if str((p.get("tags") or {}).get("difficulty", "")).lower() in _ANCHOR_DIFFICULTIES]
    pool = pool[: args.n_problems]
    if not pool:
        raise SystemExit(f"No easy/medium problems loaded from {args.train_parquet}")
    print(f"[dump] {len(pool)} easy/medium problems from {args.train_parquet}")

    print(f"[dump] loading vLLM: {args.model_path} (tp={args.tp_size})")
    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model_path, tensor_parallel_size=args.tp_size,
              gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=args.max_model_len, trust_remote_code=True,
              dtype="bfloat16", seed=args.seed)
    tokenizer = llm.get_tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = [_render(tokenizer, p["question"]) for p in pool]
    sp = SamplingParams(n=args.rollout_k, temperature=0.8, top_p=0.95,
                        max_tokens=args.max_new_tokens, seed=args.seed,
                        skip_special_tokens=False)
    print(f"[dump] rollout n={args.rollout_k} temp=0.8 on {len(prompts)} problems")
    outs = llm.generate(prompts, sp)

    rows = []
    for prob, out in zip(pool, outs):
        texts = [s.text for s in out.outputs]
        grades = [1 if _check_correctness(t, prob["gold"]) else 0 for t in texts]
        answers = [_extract_answer(t) for t in texts]
        rows.append({
            "question": prob["question"],
            "gold": prob["gold"],
            "difficulty": str((prob.get("tags") or {}).get("difficulty", "")).lower(),
            "rollouts": texts,
            "grades": grades,
            "answers": answers,
            "pass_rate": sum(grades) / len(grades) if grades else 0.0,
        })

    import pandas as pd
    df = pd.DataFrame(rows)
    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out_path)
    pr = df["pass_rate"]
    in_band = ((pr >= 0.125) & (pr <= 0.5)).mean()
    print(f"[dump] wrote {args.out_path}: {len(df)} problems | "
          f"mean pass_rate={pr.mean():.3f} | in-band[0.125,0.5]={in_band:.3f}")

    from huggingface_hub import HfApi
    HfApi(token=os.environ["HF_TOKEN"]).upload_file(
        path_or_fileobj=args.out_path, path_in_repo=args.hf_path,
        repo_id=args.hf_repo, repo_type="dataset")
    print(f"[dump] uploaded -> hf://{args.hf_repo}/{args.hf_path}")


if __name__ == "__main__":
    main()
