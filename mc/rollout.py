#!/usr/bin/env python
r"""생성 공용 — 문제 적재(`load_problems`)·단일 패스 12k 평가(`single_pass`/`run_eval`, mc/eval.py 가 부른다)·
vLLM 엔진(`build_engine`)·FSDP 병합(`merged_model`). 2턴(H1 알림 셀) 생성 경로는 0924 에 뺐다(수정 8 에서 강등,
`_gen` 이 사라진 뒤로 돌 수 없던 죽은 경로 — 백업 /hdd_data/seungpil/tmp/mc_backup_0924_v1p2/rollout.py).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# vLLM 0.20 의 DeepGEMM FP8 워밍업은 이 박스에 없다 — 학습(trainer._ray_env)과 같은 스위치를 생성·평가에도 건다.
os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0"); os.environ.setdefault("VLLM_MOE_USE_DEEP_GEMM", "0")
from mc import context as ctx


def load_problems(dataset: str, limit: int = 0) -> list[dict]:
    """`parquet:<path>[:<level>]` · `hf:<id>:<split>`(전이 평가 — HMMT25 는 split 이 "train"만
    있다) · 맨 HF id(기존 규약, split 고정 "test"). math_rollout.py 와 같은 규약."""
    if dataset.startswith("parquet:"):
        import pandas as pd  # noqa: PLC0415
        parts = dataset.split(":", 2)
        df = pd.read_parquet(parts[1])
        lvl = parts[2] if len(parts) > 2 else None
        rows = [{"problem": r["problem"], "gold": str(r["gold"])}
                for _, r in df.iterrows()
                if not lvl or (r["extra_info"] or {}).get("level") == lvl]
    elif dataset.startswith("hf:"):
        from datasets import load_dataset  # noqa: PLC0415
        _, hf_id, split = dataset.split(":", 2)
        ds = load_dataset(hf_id, split=split)
        rows = [{"problem": r["problem"], "gold": str(r["answer"])} for r in ds]
    else:
        from datasets import load_dataset  # noqa: PLC0415
        ds = load_dataset(dataset, split="test")
        rows = [{"problem": r["problem"], "gold": str(r["answer"])} for r in ds]
    return rows[:limit] if limit else rows


def single_pass(llm, tok, problems, *, k, max_tokens, temperature, top_p, seed,
                variant=None) -> list[dict]:
    """단일 패스(12,288토큰) — `mc.eval.summarize_single` 스키마의 행을 낸다."""
    from vllm import SamplingParams  # noqa: PLC0415
    prompts = [ctx.turn1_prompt(tok, p["problem"], variant) for p in problems]
    outs = llm.generate(prompts, SamplingParams(n=k, temperature=temperature, top_p=top_p,
                                                max_tokens=max_tokens, seed=seed))
    rows = []
    for gi, (src, o) in enumerate(zip(problems, outs)):
        for c in o.outputs:
            rows.append({"group_id": f"g{gi}", "problem_id": gi, "gold": src["gold"],
                         "text": c.text, "final_answer": ctx.turn1_answer(c.text),
                         "n_tok": len(c.token_ids),
                         "truncated": int(c.finish_reason == "length")})
    return rows


def run_eval(a) -> list[dict]:
    """mc/eval.py 의 생성 경로 — 단일 패스(`--max_tokens`, 기본 12,288). 같은 문제·seed·K 로 짝비교한다."""
    max_tokens = getattr(a, "max_tokens", None) or 12288
    llm, tok = build_engine(a.model_path, max_tokens=max_tokens, gpu_util=a.gpu_util, seed=a.seed)
    return single_pass(llm, tok, load_problems(a.dataset, getattr(a, "limit", 0) or 0), k=max(1, a.k),
                       max_tokens=max_tokens, temperature=1.0, top_p=1.0, seed=a.seed)


def build_engine(model_path: str, *, max_tokens: int, gpu_util: float = 0.8, seed: int = 11):
    from vllm import LLM  # noqa: PLC0415
    # MC_EAGER=0 → CUDA 그래프(작은 배치 해독 ~3배, 수정 41). 기본 1 = 옛 평가와 같은 엔진 — 짝비교는 같은 값끼리만.
    llm = LLM(model=model_path, dtype="bfloat16", seed=seed,
              gpu_memory_utilization=gpu_util,
              max_model_len=max_tokens + 2048, enforce_eager=os.environ.get("MC_EAGER", "1") != "0")
    return llm, llm.get_tokenizer()


def merged_model(path: str) -> str:
    """raw FSDP `…/<계보>/global_step_N/actor` → `$WORK/models/merged_<계보>_step<N>`(있으면 재사용) —
    병합의 유일한 자리(mc/run.sh·mc/probe.py). 그 밖의 경로는 그대로."""
    p = Path(path)
    if p.name != "actor" or not p.parent.name.startswith("global_step_"):
        return path
    out = (Path(os.environ.get("WORK") or "/hdd_data/seungpil/scratch") / "models"
           / f"merged_{p.parent.parent.name}_step{p.parent.name.split('_')[-1]}")
    if not (out / "config.json").is_file():
        out.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, "-m", "verl.model_merger", "merge", "--backend", "fsdp",
                        "--local_dir", str(p), "--target_dir", str(out)],
                       check=True, stdout=sys.stderr)
    if not (out / "config.json").is_file() or not list(out.glob("*safetensors*")):
        raise RuntimeError(f"[MC] 병합 산출물이 비었다: {out}")
    return str(out)


