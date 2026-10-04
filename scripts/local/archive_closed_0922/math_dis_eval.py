#!/usr/bin/env python
r"""math_dis_eval — M_DIS(불일치 진단 팔)의 held-out 판정.

무엇을 재는가. M_DIS 는 «한 풀이 안의 메타»를 버리고 **자기 샘플들의 불일치** 위에서 판단을
돌리는 팔이다. 그래서 판정도 «메타를 썼는가»가 아니라 **진단이 다수결보다 나은가**여야 한다 —
그리고 그 비교는 **같은 계산 예산**에서 해야 한다(후보 4개를 뽑는 비용은 두 쪽 모두 낸다).

세 단계(체크포인트 하나에 대해):
  (a) 후보 표집 — 문제마다 plain `math_opt` 프롬프트로 **신선한** 후보 5개(temperature 1.0,
      8192 토큰). 앞 **4개**가 진단 프롬프트에 들어갈 후보이고, 5번째는 acc_majority5 전용이다
      (= 진단 대신 «한 번 더 뽑아 다수결»에 쓰는 대등 비용 대안).
  (b) 진단 턴 조립 — `math_dis.build_dis_user_turn`(학습과 **같은 함수**).
  (c) 진단 응답 N=4 표집 → 채점.

보고:
  acc_dis            진단 응답의 gold 정오 평균(N 개 전부)          ← 이 팔의 성적
  acc_majority4      후보 4개의 다수답 정오                         ← 진단 없이 같은 후보로
  acc_majority5      후보 5개의 다수답 정오                         ← **대등 비용** 대안
  pass_at_4          후보 4개 중 하나라도 맞은 비율(천장)
  minority_rescue    후보 다수답이 **틀린** 문제에서 진단 응답이 맞은 비율 ← 다수결을 이긴 몫
  majority_break     후보 다수답이 **맞은** 문제에서 진단 응답이 틀린 비율 ← 그 대가(짝지어 읽는다)
  commit_parsed / cites_two_plus / diag_words / dis_emit_rate  형식 지표(학습 텔레메트리와 같은 정의)
  tokens_per_problem_{dis,majority4,majority5}  문제당 토큰 — 성적을 이 값과 **같이** 읽어야 한다.

생성기는 주입 가능하다(단위 테스트는 모의 생성기로 돈다 — GPU 불필요).

사용법:
  math_dis_eval.py --dataset math500 --model_path $WORK/merged/cd9_M_DIS_s2/step_30 \
      --num_samples 4 --max_tokens 8192 --out_dir $WORK/eval/cd9_M_DIS_s2/step_30/math500_dis_8k
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.metacot.math_meta_prompt import build_math_prompt  # noqa: E402
from src.training import math_dis as MD  # noqa: E402
from src.training import math_meta as MM  # noqa: E402

VARIANT = "math_dis"
CAND_VARIANT = "math_opt"          # ★후보는 **plain 정책**으로 뽑는다(진단 프롬프트가 아니다)
_NAN = float("nan")

# 생성기: chat 메시지 리스트들 -> 프롬프트마다 [(text, truncated, n_tok), ...] K 개.
Generator = Callable[[Sequence[list[dict]]], Sequence[Sequence[tuple[str, int, int]]]]


# ── (a) 후보 표집 ───────────────────────────────────────────────────────────────
def sample_candidates(problems: Sequence[Mapping], generate: Generator) -> list[list[dict]]:
    """문제마다 후보 목록(dict: text/final_answer/r_corr/n_tok/truncated). 순서는 생성 순서 —
    앞 N_CAND 개가 진단 프롬프트에 들어간다."""
    prompts = [build_math_prompt(p["problem"], CAND_VARIANT) for p in problems]
    outs = generate(prompts)
    if len(outs) != len(problems):
        raise RuntimeError(f"[dis-eval] 후보 생성기가 {len(outs)} 개를 돌려줬다(문제 {len(problems)})")
    cands = []
    for src, samples in zip(problems, outs):
        gold = str(src["gold"])
        cands.append([{"text": t, "final_answer": MM.last_boxed(t),
                       "r_corr": MM.grade_math(t, gold), "truncated": int(tr), "n_tok": int(nt)}
                      for t, tr, nt in samples])
    return cands


def majority_correct(cands: Sequence[Mapping], gold: str, *, k: int) -> float:
    r"""앞 k 개 후보의 **다수답**이 gold 와 맞는가(0/1). 다수 미정(동률·무답)이면 NaN —
    «못 쟀다»이지 «틀렸다»가 아니다(0 으로 읽으면 다수결 기준선이 부당하게 낮아진다)."""
    ans = MD.plurality_answer([c.get("final_answer") for c in list(cands)[:k]])
    if ans is None:
        return _NAN
    return float(MM.grade_math(f"\\boxed{{{ans}}}", gold))


# ── (b)(c) 진단 표집 + 채점 ──────────────────────────────────────────────────────
def evaluate(problems: Sequence[Mapping], cands: Sequence[Sequence[Mapping]],
             generate: Generator, *, n_cand: int = MD.N_CAND) -> tuple[list[dict], dict]:
    """진단 응답 행과 요약. `cands` 는 sample_candidates 의 출력(문제당 ≥ n_cand+1 권장).

    ★0914 리뷰 D3: 잘렸거나(\boxed 없음) 답이 빈 후보는 **후보로 세지 않는다** — 그런 후보가
      뽑히면 `math_dis.make_sketch` 가 답 없는 후보에 즉사한다(학습 빌더와 같은 필터,
      build_math_dis_parquet.py 참조 — «학습 데이터와 게이트가 같은 것을 재야 한다»는 규약을
      평가도 지킨다)."""
    if len(cands) != len(problems):
        raise RuntimeError(f"[dis-eval] 후보 묶음 {len(cands)} != 문제 {len(problems)}")
    prompts, used = [], []
    for src, cs in zip(problems, cands):
        usable = [x for x in cs if not int(x.get("truncated", 0) or 0)
                 and str(x.get("final_answer") or "").strip()]
        c = usable[:n_cand]
        if len(c) != n_cand:
            raise RuntimeError(f"[dis-eval] 후보가 {len(c)} 개인 문제가 있다(필요 {n_cand}, "
                               f"잘렸거나 답 없는 후보는 제외)")
        turn = MD.build_dis_user_turn(src["problem"], [MD.make_sketch(x["text"]) for x in c])
        prompts.append(build_math_prompt(turn, VARIANT))
        used.append(c)
    outs = generate(prompts)
    if len(outs) != len(problems):
        raise RuntimeError(f"[dis-eval] 진단 생성기가 {len(outs)} 개를 돌려줬다(문제 {len(problems)})")
    rows: list[dict] = []
    for gi, (src, c, samples) in enumerate(zip(problems, used, outs)):
        gold = str(src["gold"])
        answers = [str(x.get("final_answer") or "") for x in c]
        correct = [int(x.get("r_corr", 0)) for x in c]
        for text, trunc, n_tok in samples:
            r = MD.parse_dis_row(text, gold, src["problem"], answers, correct, truncated=trunc)
            r.update(MD.dis_row_flags(r))
            r.update({"group_id": f"g{gi}", "problem_id": gi, "uid": f"g{gi}", "n_tok": int(n_tok)})
            rows.append(r)
    # ★크레딧 자체(dis_credit)는 held-out 에서 **학습과 같은 함수**로 다시 잰다 — 그룹은 이
    #   문제의 N 개 진단 응답(학습의 uid 그룹과 같은 개념).
    by_g: dict = {}
    for r in rows:
        by_g.setdefault(r["group_id"], []).append(r)
    for group in by_g.values():
        for r in group:
            credit, defined = MD.dis_row_credit(r, group)
            r["dis_credit"] = credit
            r["dis_defined"] = int(defined)
            r["meta_defined"] = int(defined)
    tel = MM.dis_telemetry(rows)
    n = max(1, len(rows))
    ng = max(1, len(problems))
    _mean = lambda xs: (sum(xs) / len(xs)) if xs else _NAN
    maj4 = [majority_correct(cs, str(p["gold"]), k=n_cand) for p, cs in zip(problems, cands)]
    maj5 = [majority_correct(cs, str(p["gold"]), k=n_cand + 1) for p, cs in zip(problems, cands)]
    # ★소수 구제 / 다수 파괴는 **후보 다수답이 맞았는가**로 문제를 가른 뒤 진단 응답의 정오를 본다.
    #   둘은 짝으로 읽는다 — 구제만 보면 «의심을 늘리면 오르는 지표»가 되어 굿하트가 열린다.
    resc, brk = [], []
    for gi, m in enumerate(maj4):
        vals = [int(r["r_corr"]) for r in by_g.get(f"g{gi}", [])]
        if not MM._cdr._finite(m) or not vals:
            continue
        p_dis = sum(vals) / len(vals)
        if float(m) >= 0.5:
            brk.append(1.0 - p_dis)      # 다수가 맞았는데 진단이 틀린 비율
        else:
            resc.append(p_dis)           # 다수가 틀렸는데 진단이 맞은 비율
    tok_cand = [sum(int(x.get("n_tok", 0)) for x in list(cs)[:n_cand]) for cs in cands]
    tok_cand5 = [sum(int(x.get("n_tok", 0)) for x in list(cs)[:n_cand + 1]) for cs in cands]
    tok_dis = [sum(int(r.get("n_tok", 0)) for r in by_g.get(f"g{gi}", []))
               for gi in range(len(problems))]
    tel.update({
        "variant": VARIANT, "n_rows": len(rows), "n_groups": len(problems),
        "acc_dis": sum(int(r["r_corr"]) for r in rows) / n,
        "acc_majority4": _mean([m for m in maj4 if MM._cdr._finite(m)]),
        "acc_majority5": _mean([m for m in maj5 if MM._cdr._finite(m)]),
        "pass_at_4": _mean([float(any(int(x.get("r_corr", 0)) for x in list(cs)[:n_cand]))
                            for cs in cands]),
        "cand_acc_mean": _mean([_mean([float(x.get("r_corr", 0)) for x in list(cs)[:n_cand]])
                                for cs in cands]),
        "all_agree_frac": _mean([float(MD.all_agree([x.get("final_answer")
                                                     for x in list(cs)[:n_cand]])) for cs in cands]),
        "multi_block": sum(int(r.get("multi_block", 0)) for r in rows) / n,
        "boxed_in_meta": sum(int(r.get("boxed_in_meta", 0)) for r in rows) / n,
        "minority_rescue": _mean(resc),
        "majority_break": _mean(brk),
        "n_minority_problems": len(resc),
        "trunc_rate": sum(int(r.get("truncated", 0)) for r in rows) / n,
        # ★문제당 토큰 — acc_dis 를 이 값과 **같이** 읽어야 «진단이 나은가»가 공정한 비교가 된다.
        "tokens_per_problem_majority4": sum(tok_cand) / ng,
        "tokens_per_problem_majority5": sum(tok_cand5) / ng,
        "tokens_per_problem_dis": (sum(tok_cand) + sum(tok_dis)) / ng,
    })
    return rows, tel


# ── 요약 출력 ───────────────────────────────────────────────────────────────────
_SUMMARY_KEYS = ("acc_dis", "acc_majority4", "acc_majority5", "pass_at_4", "cand_acc_mean",
                 "minority_rescue", "majority_break", "n_minority_problems", "all_agree_frac",
                 "dis_emit_rate", "commit_parsed", "dis_defined_rate", "dis_credit_mean",
                 "commit_is_minority_rate", "cites_two_plus_rate", "diag_words_mean",
                 "commit_cand_correct_rate", "multi_block", "boxed_in_meta", "trunc_rate",
                 "tokens_per_problem_dis", "tokens_per_problem_majority4",
                 "tokens_per_problem_majority5")


def format_summary(tel: Mapping) -> str:
    def _f(x):
        try:
            return f"{float(x):.3f}"
        except (TypeError, ValueError):
            return "nan"
    return "\n".join(f"  {k:32s} {_f(tel.get(k))}" for k in _SUMMARY_KEYS)


# ── 생성기(vLLM) ────────────────────────────────────────────────────────────────
def vllm_generators(model_path: str, *, n_cand: int, num_samples: int, max_tokens: int,
                    temperature: float, top_p: float, seed: int, gpu_util: float):
    """(후보 생성기, 진단 생성기, tokenizer) — 엔진 하나를 둘이 공유한다.
    ★후보는 n_cand+1 개 뽑는다(5번째는 acc_majority5 전용, 대등 비용 대안)."""
    from vllm import LLM, SamplingParams  # noqa: PLC0415

    from src.metacot.math_meta_prompt import render_chat_messages  # noqa: PLC0415
    llm = LLM(model=model_path, dtype="bfloat16", seed=seed, gpu_memory_utilization=gpu_util,
              max_model_len=max_tokens + 4096, enforce_eager=True)
    tok = llm.get_tokenizer()

    def _gen(n):
        def g(prompts):
            outs = llm.generate([render_chat_messages(tok, m) for m in prompts], SamplingParams(
                n=n, temperature=temperature, top_p=top_p, max_tokens=max_tokens, seed=seed))
            return [[(x.text, int(x.finish_reason == "length"), len(x.token_ids)) for x in o.outputs]
                    for o in outs]
        return g

    return _gen(n_cand + 1), _gen(num_samples), tok


def main(argv=None) -> int:
    from math_retry_eval import load_problems  # noqa: PLC0415  (데이터 로딩 정의를 한 곳에)

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help="math500 등 math_rollout.DATASETS 키, 또는 parquet:<path>[:<level>] "
                         "(L5 평가 parquet 를 그대로 쓴다)")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--variant", default=VARIANT, choices=[VARIANT],
                    help="math_dis 고정 — 후보 4개 + 진단 구조에서만 이 지표들이 정의된다")
    ap.add_argument("--num_samples", type=int, default=4, help="진단 응답 N")
    ap.add_argument("--max_tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--gpu_util", type=float, default=0.4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shuffle_seed", type=int, default=0)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args(argv)

    MM.selftest_math_verify()
    problems = load_problems(a.dataset, limit=a.limit, shuffle_seed=a.shuffle_seed)
    print(f"[dis-eval] {a.dataset}: {len(problems)} 문제 · 후보 {MD.N_CAND + 1} + 진단 {a.num_samples}",
          flush=True)
    gen_cand, gen_dis, _tok = vllm_generators(
        a.model_path, n_cand=MD.N_CAND, num_samples=a.num_samples, max_tokens=a.max_tokens,
        temperature=a.temperature, top_p=a.top_p, seed=a.seed, gpu_util=a.gpu_util)
    cands = sample_candidates(problems, gen_cand)
    rows, tel = evaluate(problems, cands, gen_dis)
    tel.update({"dataset": a.dataset, "model_path": a.model_path, "max_tokens": a.max_tokens,
                "num_samples": a.num_samples})
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
