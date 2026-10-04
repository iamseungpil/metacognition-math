#!/usr/bin/env python
r"""math_critique_eval — M_CRIT(비평 정보이득 팔)의 held-out 판정.

무엇을 재는가. M_CRIT 은 «문맥 안의 재시도»를 버린 팔이다(cd9 정박 진단: 같은 문맥 이어쓰기의
구제율 0.000). 그래서 판정도 «두 번째 시도가 답을 고쳤는가»가 아니라 **비평이 새 풀이를
실제로 돕는가**여야 한다. 세 층으로 잰다:

  1. 첫 답 정확도(first_acc)와 형식 지표(발화·누출·다중 블록·비평 단어 수) — 세금과 형식 붕괴.
  2. **답 없는 재풀이**(answer-blind re-solve): 첫 답이 **틀린** 샘플마다, 그 비평만 들고
     다시 K 개 푼다(crit) vs 아무것도 없이 다시 K 개 푼다(blind).
         rescue_crit − rescue_blind = rescue_delta   ← 이것이 «비평의 값»의 직접 측정
     ★blind 대조가 필수다. 오답을 독립 재표본하면 그 자체로 49% 가 살아난다(cd9 실측) —
       blind 없이 rescue_crit 만 보면 «다시 풀기»의 값을 «비평»의 값으로 오독한다.
  3. **정보 이득**(IG): 얼어붙은 채점기(src/training/critique_scorer.py)로 학습 항과 **같은
     정의**를 held-out 에서 재계산한다(ig / ig_donor / ig_delta = 그 차, 내용 대조).
     학습 중 텔레메트리와 held-out 이 같은 함수(math_meta.crit_telemetry)를 쓴다.

  + `decision` 의 **혼합 문제 안 선택성**(selectivity_mixed = redirect|wrong − redirect|right,
    0<first_pass_rate<1 인 문제로만 제한) — 난이도와 «문제 안 판단»을 가른다(retry_metrics).

비평 재풀이 프롬프트는 `math_critique_resolve_gate.resolve_prompt` 를 **그대로 쓴다** — 그
게이트와 이 평가가 한 글자라도 다른 문맥을 쓰면 두 산출물을 나란히 읽을 수 없다.
생성기는 주입 가능하다(단위 테스트는 모의 생성기로 돈다 — GPU 불필요).

사용법:
  math_critique_eval.py --dataset math500 --model_path $WORK/merged/cd9_M_CRIT_s2/step_30 \
      --num_samples 8 --resolve_k 4 --max_tokens 8192 \
      --out_dir $WORK/eval/cd9_M_CRIT_s2/step_30/math500_crit_8k
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Callable, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from math_critique_resolve_gate import resolve_prompt  # noqa: E402
from src.metacot.math_meta_prompt import MATH_PROMPT_VARIANTS, build_math_prompt  # noqa: E402
from src.training import math_meta as MM  # noqa: E402
from src.training import retry_metrics as RMET  # noqa: E402

VARIANT = "math_crit"
assert VARIANT in MATH_PROMPT_VARIANTS

_NAN = float("nan")
RESOLVE_CONDS = ("crit", "blind")

# 1차 생성기: chat 메시지 리스트들 -> 프롬프트마다 [(text, truncated, n_tok), ...] K 개.
Generator = Callable[[Sequence[list[dict]]], Sequence[Sequence[tuple[str, int, int]]]]
# 재풀이 생성기: **문자열 프롬프트**들 -> 프롬프트마다 [text, ...] K 개(잘림 정보는 안 쓴다).
ResolveGenerator = Callable[[Sequence[str]], Sequence[Sequence[str]]]


# ── 1) 첫 패스: 풀이 + 비평 블록 ────────────────────────────────────────────────
def evaluate(problems: Sequence[Mapping], generate: Generator, *, num_samples: int,
             variant: str = VARIANT) -> tuple[list[dict], dict]:
    """행(롤아웃)과 요약. problems = [{problem, gold}, ...].
    ★행은 `MM.parse_crit_row` 로 뽑는다 — 학습 텔레메트리와 **같은 파서**(첫 답·비평·누출)."""
    if variant not in MATH_PROMPT_VARIANTS:
        raise ValueError(f"[crit-eval] variant={variant!r} 를 모른다")
    prompts = [build_math_prompt(p["problem"], variant) for p in problems]
    outs = generate(prompts)
    if len(outs) != len(problems):
        raise RuntimeError(f"[crit-eval] 생성기가 {len(outs)} 개를 돌려줬다(문제 {len(problems)})")
    rows = []
    for gi, (src, samples) in enumerate(zip(problems, outs)):
        for text, trunc, n_tok in samples:
            r = MM.parse_crit_row(text, str(src["gold"]), src["problem"], truncated=int(trunc))
            r.update({"group_id": f"g{gi}", "problem_id": gi, "n_tok": int(n_tok),
                      "uid": f"g{gi}",
                      "multi_block": int(int(r["n_blocks"]) > 1)})
            r["crit_ok"] = int(bool(r["emitted"]) and not r["multi_block"]
                               and bool(r["critique"]) and not r["leaked"])
            rows.append(r)
    n = max(1, len(rows))
    groups: dict = {}
    for r in rows:
        groups.setdefault(r["group_id"], []).append(r)
    tel = MM.crit_telemetry(rows)
    tel.update({
        "variant": variant, "num_samples": num_samples,
        "n_rows": len(rows), "n_groups": len(groups),
        "first_acc": sum(int(r["first_correct"]) for r in rows) / n,
        "emit_rate": sum(int(r["emitted"]) for r in rows) / n,
        "multi_block_rate": sum(int(r["multi_block"]) for r in rows) / n,
        "boxed_in_meta": sum(int(r["boxed_in_meta"]) for r in rows) / n,
        "trunc_rate": sum(int(r["truncated"]) for r in rows) / n,
        "mean_tokens": sum(int(r["n_tok"]) for r in rows) / n,
        "pass_at_n_first": sum(int(any(x["first_correct"] for x in g))
                               for g in groups.values()) / max(1, len(groups)),
        "decision_rate": (sum(1 for r in rows if r["decision"] in ("verify", "redirect"))
                          / max(1, sum(1 for r in rows if r["emitted"]))),
    })
    # ★혼합 문제 안 `decision` 선택성 — «틀린 자기 풀이를 redirect 로 표시하는가»(통제 축).
    tel.update(RMET.all_metrics(rows, group_key="group_id"))
    return rows, tel


# ── 2) 답 없는 재풀이: 비평 vs blind ────────────────────────────────────────────
def wrong_rows(rows: Sequence[Mapping]) -> list[int]:
    """재풀이 대상 = 첫 답이 **틀린**, 비평이 성립한(발화·단일 블록·비어있지 않음·누출 아님),
    잘리지 않은 행. ★누출 비평을 재풀이에 쓰면 «답을 다시 보여 준 값»을 재게 된다."""
    return [i for i, r in enumerate(rows)
            if not int(MM._cdr._bool01(r.get("first_correct", 0)))
            and int(r.get("crit_ok", 0)) and not int(MM._cdr._bool01(r.get("truncated", 0)))]


def build_resolve_requests(rows: Sequence[Mapping], tok, variant: str,
                           idxs: Sequence[int]) -> tuple[list[str], list[tuple[int, str]]]:
    """(프롬프트들, [(행 인덱스, 조건)]). 두 조건의 유일한 차이는 note 문자열이다
    (math_critique_resolve_gate.resolve_prompt 규약)."""
    prompts, index = [], []
    for i in idxs:
        r = rows[i]
        for cond in RESOLVE_CONDS:
            note = r["critique"] if cond == "crit" else None
            prompts.append(resolve_prompt(tok, variant, r["problem"], note))
            index.append((i, cond))
    return prompts, index


def resolve_eval(rows: Sequence[Mapping], tok, variant: str, generate: ResolveGenerator,
                 *, idxs: Sequence[int] | None = None) -> dict:
    """비평/blind 재풀이를 돌려 행마다 p_crit·p_blind 를 붙이고 짝지은 Δ 를 요약한다."""
    idxs = list(idxs) if idxs is not None else wrong_rows(rows)
    if not idxs:
        return {"n_resolved": 0, "rescue_crit": _NAN, "rescue_blind": _NAN,
                "rescue_delta": _NAN, "frac_crit_gt_blind": _NAN}
    prompts, index = build_resolve_requests(rows, tok, variant, idxs)
    outs = generate(prompts)
    if len(outs) != len(prompts):
        raise RuntimeError(f"[crit-eval] 재풀이 생성기가 {len(outs)} 개를 돌려줬다(요청 {len(prompts)})")
    agg: dict = {}
    for (i, cond), texts in zip(index, outs):
        gold = rows[i]["gold"]
        vals = [int(MM.grade_math(t, gold)) for t in texts]
        agg[(i, cond)] = (sum(vals) / len(vals)) if vals else _NAN
    d = []
    for i in idxs:
        rows[i]["p_crit"] = agg.get((i, "crit"), _NAN)
        rows[i]["p_blind"] = agg.get((i, "blind"), _NAN)
        if MM._cdr._finite(rows[i]["p_crit"]) and MM._cdr._finite(rows[i]["p_blind"]):
            d.append(rows[i]["p_crit"] - rows[i]["p_blind"])
    _mean = lambda xs: (sum(xs) / len(xs)) if xs else _NAN
    pc = [rows[i]["p_crit"] for i in idxs if MM._cdr._finite(rows[i].get("p_crit"))]
    pb = [rows[i]["p_blind"] for i in idxs if MM._cdr._finite(rows[i].get("p_blind"))]
    return {
        "n_resolved": len(idxs),
        "rescue_crit": _mean(pc),
        "rescue_blind": _mean(pb),
        "rescue_delta": _mean(d),
        "frac_crit_gt_blind": (sum(1 for x in d if x > 0) / len(d)) if d else _NAN,
    }


# ── 3) 정보 이득(얼어붙은 채점기) ────────────────────────────────────────────────
def ig_eval(rows: Sequence[Mapping], scorer, *, variant: str = VARIANT, seed: int = 11) -> dict:
    """학습 항과 **같은 함수**로 held-out IG 를 잰다(math_meta.annotate_crit_ig → crit_telemetry).
    그룹 키는 group_id(= held-out 의 K 샘플 묶음; 학습은 uid — 같은 개념, 다른 키)."""
    keys = [r["group_id"] for r in rows]
    MM.annotate_crit_ig(rows, keys, scorer=scorer, variant=variant, rng=random.Random(seed))
    tel = MM.crit_telemetry(rows)
    return {k: tel[k] for k in ("crit_defined_rate", "ig_mean", "ig_donor_mean", "ig_delta_mean")}


# ── 요약 출력 ───────────────────────────────────────────────────────────────────
_SUMMARY_KEYS = ("first_acc", "pass_at_n_first", "emit_rate", "decision_rate", "leak_rate",
                 "crit_rows", "crit_defined_rate", "crit_words_mean", "multi_block_rate",
                 "boxed_in_meta", "second_attempt_rate", "trunc_rate", "mean_tokens",
                 "n_resolved", "rescue_crit", "rescue_blind", "rescue_delta",
                 "frac_crit_gt_blind", "ig_mean", "ig_donor_mean", "ig_delta_mean",
                 "n_mixed_problems", "redirect_rate_given_wrong_mixed",
                 "redirect_rate_given_right_mixed", "selectivity_mixed",
                 "mean_within_problem_auc", "judgment_acc_mixed")


def format_summary(tel: Mapping) -> str:
    def _f(x):
        try:
            return f"{float(x):.3f}"
        except (TypeError, ValueError):
            return "nan"
    return "\n".join(f"  {k:32s} {_f(tel.get(k))}" for k in _SUMMARY_KEYS)


# ── 생성기(vLLM) ────────────────────────────────────────────────────────────────
def vllm_generators(model_path: str, *, num_samples: int, resolve_k: int, max_tokens: int,
                    temperature: float, top_p: float, seed: int, gpu_util: float):
    """(1차 생성기, 재풀이 생성기, tokenizer) — 엔진 하나를 둘이 공유한다."""
    from vllm import LLM, SamplingParams  # noqa: PLC0415

    from src.metacot.math_meta_prompt import render_chat_messages  # noqa: PLC0415
    llm = LLM(model=model_path, dtype="bfloat16", seed=seed, gpu_memory_utilization=gpu_util,
              max_model_len=max_tokens + 2048, enforce_eager=True)
    tok = llm.get_tokenizer()

    def gen(prompts):
        outs = llm.generate([render_chat_messages(tok, m) for m in prompts], SamplingParams(
            n=num_samples, temperature=temperature, top_p=top_p, max_tokens=max_tokens, seed=seed))
        return [[(x.text, int(x.finish_reason == "length"), len(x.token_ids)) for x in o.outputs]
                for o in outs]

    def gen_resolve(prompts):
        outs = llm.generate(list(prompts), SamplingParams(
            n=resolve_k, temperature=temperature, top_p=top_p, max_tokens=max_tokens, seed=seed))
        return [[x.text for x in o.outputs] for o in outs]

    return gen, gen_resolve, tok


def main(argv=None) -> int:
    from math_retry_eval import load_problems  # noqa: PLC0415  (데이터 로딩 정의를 한 곳에)

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--variant", default=VARIANT, choices=[VARIANT],
                    help="math_crit 고정 — 첫 답 뒤 비평 블록 구조에서만 이 지표들이 정의된다")
    ap.add_argument("--num_samples", type=int, default=8)
    ap.add_argument("--resolve_k", type=int, default=4, help="답 없는 재풀이 K (조건마다)")
    ap.add_argument("--max_tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--gpu_util", type=float, default=0.4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shuffle_seed", type=int, default=0)
    ap.add_argument("--max_resolve_rows", type=int, default=200,
                    help="재풀이 대상 상한(0=전부) — 오답 행이 많으면 생성 비용이 폭발한다")
    ap.add_argument("--scorer_path", default=None,
                    help="IG 채점기(얼어붙은 초기 정책). 미지정이면 MATH_CRIT_SCORER_PATH, "
                         "그것도 없으면 IG 를 건너뛴다")
    ap.add_argument("--no_ig", action="store_true", help="IG 채점을 아예 건너뛴다")
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args(argv)

    MM.selftest_math_verify()
    problems = load_problems(a.dataset, limit=a.limit, shuffle_seed=a.shuffle_seed)
    print(f"[crit-eval] {a.dataset}: {len(problems)} 문제 x {a.num_samples} · variant={a.variant}",
          flush=True)
    gen, gen_resolve, tok = vllm_generators(
        a.model_path, num_samples=a.num_samples, resolve_k=a.resolve_k, max_tokens=a.max_tokens,
        temperature=a.temperature, top_p=a.top_p, seed=a.seed, gpu_util=a.gpu_util)
    rows, tel = evaluate(problems, gen, num_samples=a.num_samples, variant=a.variant)

    idxs = wrong_rows(rows)
    if a.max_resolve_rows and len(idxs) > a.max_resolve_rows:
        random.Random(a.seed).shuffle(idxs)
        idxs = idxs[:a.max_resolve_rows]
    print(f"[crit-eval] 답 없는 재풀이: 오답·비평 성립 행 {len(idxs)}개 x 2조건 x K={a.resolve_k}",
          flush=True)
    tel.update(resolve_eval(rows, tok, a.variant, gen_resolve, idxs=idxs))

    scorer_path = a.scorer_path or None
    if not a.no_ig:
        try:
            from src.training.critique_scorer import CritiqueScorer  # noqa: PLC0415
            sc = CritiqueScorer(scorer_path)      # 경로가 없으면 env, 그것도 없으면 RuntimeError
            tel.update(ig_eval(rows, sc, variant=a.variant, seed=a.seed))
        except RuntimeError as e:
            print(f"[crit-eval] IG 건너뜀: {e}", flush=True)

    tel.update({"dataset": a.dataset, "model_path": a.model_path, "max_tokens": a.max_tokens,
                "resolve_k": a.resolve_k, "scorer_path": scorer_path})
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "texts.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "telemetry.json").write_text(json.dumps(tel, ensure_ascii=False, indent=2, default=float))
    print(format_summary(tel))
    print(f"[out] {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
