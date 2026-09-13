#!/usr/bin/env python
"""math_rollout — 수학 벤치마크 롤아웃 + 채점 + (기존) 메타 텔레메트리.

사용자 지시(2026-09-12): "예전 자를 기준으로 math 로 옮겨서 다시 테스트". 그래서
출력 스키마와 메타 블록 문법을 Countdown 쪽과 **같게** 맞춘다 — `countdown_rewards`
의 자·텔레메트리(emit_rate / meta_position / boilerplate / decision / confidence /
group_emit)와 `src/eval/pmi_shift_signal.py` 가 그대로 붙는다.

행 스키마(텍스트 한 줄 = 롤아웃 하나):
    group_id, problem_id, problem, gold, r_corr, text, final_answer, truncated, n_tok

채점은 `math_verify`(parse+verify). ⚠이 저장소에는 math_verify 의 timeout 래퍼가
워커 스레드에서 **정답을 조용히 오답으로 만드는** 알려진 함정이 있다
(`scripts/patch_math_verify.py`). 그래서 시작할 때 자가검사를 돌리고, 깨져 있으면
조용히 0 을 흘리지 않고 **즉사**한다.

사용법:
  math_rollout.py --dataset math500 --model_path $WORK/models/Qwen3.5-4B \
      --variant math_opt --num_samples 8 --out_dir $WORK/eval/math500_qwen35_opt
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.metacot.math_meta_prompt import MATH_PROMPT_VARIANTS, build_math_prompt  # noqa: E402
from src.training import countdown_rewards as cdr  # noqa: E402

# (hf id, config, split, 문제 컬럼, 정답 컬럼)
DATASETS = {
    "math500":  ("HuggingFaceH4/MATH-500", None, "test", "problem", "answer"),
    "aime24":   ("HuggingFaceH4/aime_2024", None, "train", "problem", "answer"),
    "aime25":   ("math-ai/aime25", None, "test", "problem", "answer"),
    # ★LiveMathBench 는 gated(인증 필요) — 같은 «최신·비오염» 역할을 하는 공개 대안으로
    #   HMMT 2025-02 를 쓴다(MathArena, 30문제). livemath 라는 이름은 남기지 않는다:
    #   실제로 무엇을 돌렸는지가 파일명에 남아야 한다.
    "hmmt25":   ("MathArena/hmmt_feb_2025", None, "train", "problem", "answer"),
}


def _grade(pred_text: str, gold: str) -> int:
    from math_verify import parse, verify
    try:
        return int(verify(parse(str(gold)), parse(pred_text)))
    except Exception:
        return 0


def _selftest_math_verify() -> None:
    """조용한 오채점 방지 — 깨져 있으면 즉사한다."""
    cases = [("\\boxed{42}", "42", 1), ("\\boxed{\\frac{1}{2}}", "0.5", 1),
             ("\\boxed{7}", "42", 0)]
    got = [_grade(p, g) for p, g, _ in cases]
    want = [e for *_, e in cases]
    if got != want:
        raise RuntimeError(
            f"math_verify 자가검사 실패: got={got} want={want} — "
            "scripts/patch_math_verify.py 를 먼저 적용하라(조용한 오채점 방지).")


def _final_answer(text: str) -> str:
    """마지막 \\boxed{...} 안의 문자열(없으면 "")."""
    i = text.rfind("\\boxed{")
    if i < 0:
        return ""
    j, depth = i + len("\\boxed{"), 1
    while j < len(text) and depth:
        depth += (text[j] == "{") - (text[j] == "}")
        j += 1
    return text[i + len("\\boxed{"): j - 1].strip() if depth == 0 else ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    ap.add_argument("--model_path", required=True)
    # ★0913 수리: 선택지를 하드코딩했다가 math_new 추가 후 CLI 가 거부해 롤아웃 2건이
    #   죽었다. 단일 진실 원천(MATH_PROMPT_VARIANTS)에서 끌어와 다시 어긋나지 않게 한다.
    ap.add_argument("--variant", default="math_opt", choices=sorted(MATH_PROMPT_VARIANTS))
    ap.add_argument("--num_samples", type=int, default=8)
    ap.add_argument("--max_tokens", type=int, default=3072)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--gpu_util", type=float, default=0.4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()

    _selftest_math_verify()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset
    from vllm import LLM, SamplingParams

    ds_id, cfg, split, qcol, acol = DATASETS[a.dataset]
    ds = load_dataset(ds_id, cfg, split=split) if cfg else load_dataset(ds_id, split=split)
    rows_in = [{"problem": r[qcol], "gold": str(r[acol])} for r in ds]
    if a.limit:
        rows_in = rows_in[: a.limit]
    print(f"[math] {a.dataset}: {len(rows_in)} 문제 x {a.num_samples} 롤아웃 "
          f"· variant={a.variant}", flush=True)

    llm = LLM(model=a.model_path, dtype="bfloat16", seed=a.seed,
              gpu_memory_utilization=a.gpu_util,
              max_model_len=a.max_tokens + 2048, enforce_eager=True)
    tok = llm.get_tokenizer()

    def chat(msgs) -> str:
        try:
            return tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True)

    prompts = [chat(build_math_prompt(r["problem"], a.variant)) for r in rows_in]
    outs = llm.generate(prompts, SamplingParams(
        n=a.num_samples, temperature=a.temperature, top_p=a.top_p,
        max_tokens=a.max_tokens, seed=a.seed))

    rows, n_trunc = [], 0
    for gi, (src, o) in enumerate(zip(rows_in, outs)):
        for x in o.outputs:
            trunc = int(x.finish_reason == "length")
            n_trunc += trunc
            rows.append({
                "group_id": f"g{gi}",
                "problem_id": gi,
                "problem": src["problem"],
                "gold": src["gold"],
                "text": x.text,
                "r_corr": _grade(x.text, src["gold"]),
                "final_answer": _final_answer(x.text),
                "truncated": trunc,
                "n_tok": len(x.token_ids),
            })

    with (out / "texts.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ── 기존(Countdown) 메타 텔레메트리를 그대로 재사용한다 ────────────────────
    n = len(rows)
    groups: dict[str, list] = {}
    for r in rows:
        groups.setdefault(r["group_id"], []).append(r)
    tel = {
        "dataset": a.dataset, "model_path": a.model_path, "variant": a.variant,
        "n_rows": n, "n_groups": len(groups),
        "acc": sum(r["r_corr"] for r in rows) / max(1, n),
        "pass_at_n": sum(int(any(x["r_corr"] for x in g)) for g in groups.values()) / max(1, len(groups)),
        "trunc_rate": n_trunc / max(1, n),
        "no_answer_rate": sum(1 for r in rows if not r["final_answer"]) / max(1, n),
        "len_mean": sum(r["n_tok"] for r in rows) / max(1, n),
        "emit_rate": cdr.emit_rate(rows, form="new"),
        "meta_position": cdr.meta_position_stats(rows, form="new"),
        "boilerplate": cdr.boilerplate_rate(rows, form="new"),
    }
    # 오답 분포(디코이 후보) — PMI-shift 는 «모델이 실제로 낸 오답»을 디코이로 써야 한다
    # (규칙기반 오답은 구세대 수학 판에서 AUC 0.539 로 포화됐다 — pmi_shift.py 주석).
    dec = {}
    for gid, g in groups.items():
        wrong = Counter(x["final_answer"] for x in g
                        if x["final_answer"] and not x["r_corr"])
        if wrong:
            dec[gid] = wrong.most_common(1)[0][0]
    tel["decoy_available_rate"] = len(dec) / max(1, len(groups))
    (out / "decoys.json").write_text(json.dumps(dec, ensure_ascii=False, indent=2))
    (out / "telemetry.json").write_text(json.dumps(tel, ensure_ascii=False, indent=2,
                                                   default=float))
    for k, v in tel.items():
        if k not in ("model_path", "meta_position", "boilerplate"):
            print(f"  {k:22s} {v}")
    print(f"  meta_position.p50      {tel['meta_position'].get('p50')}")
    print(f"[out] {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
