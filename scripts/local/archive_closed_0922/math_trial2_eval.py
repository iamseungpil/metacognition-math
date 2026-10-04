#!/usr/bin/env python
"""math_trial2_eval — S3 2-시도 팔(`docs/DESIGN_S3_trial2_0915.md` §6)의 두 관문 채점기.

설계가 요구하는 두 숫자는 **서로 다른 모집단**에서 나온다. 그래서 두 단계를 한 스크립트에
두되 생성은 한 줄도 새로 쓰지 않는다 — 기존 기계를 부른다.

  `--stage attempt1` (기본)
      홀드아웃 MATH-500 **시도-1 pass@1**. `math_rollout.py` 의 main 을 그대로 호출한다
      (같은 프롬프트·같은 채점기·같은 `telemetry.json` 스키마) → M_G0 `.740` / M_G1 `.744`
      와 **비교 가능**하다. run_math_arm.sh 가 다른 팔에 쓰는 경로와 바이트 동일.

  `--stage two_trial`
      G8 이 `.617`(사실만 재시도)을 잰 것과 **같은 모집단**(난이도 5 오답 롤아웃, MIXED 그룹,
      문제당 2개, max_sites 150, K=8, 8,192 토큰)에서 2-시도 정확도를 잰다. 각 오답 행마다
        ① 반성문 사전 패스(`trial2.note_ask_prompt`, ≤ NOTE_MAX_TOKENS, 누출 가드 `clean_note`)
        ② `trial2.attempt2_prompt` 로 **문맥을 버린** 재시도 K회
      를 note 모드 `self`(반성문 있음)와 `none`(= `.617` 참조와 바이트 동일) 두 팔로 돌리고
      rescue·(self − none) 짝 차이를 `math_cited_site_gate.bootstrap_ci` 로 낸다. 정답 행에는
      같은 두 조건을 걸어 **거짓 경보**(정답이 오답으로 뒤집히는 비율)를 본다.

      ⚠️여기서 재는 것은 «구제율»(오답 행이 재시도로 살아나는 비율)이다. 설계 §6 의
      «2-시도 정확도 R1∨R2»는 이 모집단이 정의상 R1=0 이므로 구제율과 같은 수다.

사용 예:
  math_trial2_eval.py --stage attempt1 --dataset math500 --model_path $CKPT \
      --num_samples 8 --max_tokens 8192 --out_dir $EVAL/math500_8k
  math_trial2_eval.py --stage two_trial --rollouts $EVAL/l5_rollouts/texts.jsonl \
      --model_path $CKPT --out_dir $EVAL/trial2_two_8k
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from math_activation_gate import select_correct_rollouts  # noqa: E402
from math_cited_site_gate import (  # noqa: E402  (선별·통계 정의를 한 곳에 둔다)
    bootstrap_ci, select_wrong_rollouts, sign_test_p,
)
from src.training import trial2 as t2  # noqa: E402
from src.training.math_meta import (  # noqa: E402
    grade_math, last_boxed, selftest_math_verify,
)

_NAN = float("nan")
# G8(`docs/RESULTS_cd9.md` 0915 11:00) 사실만 고지 재시도의 구제율 — 설계 §6 의 기준선.
FACT_ONLY_REFERENCE = 0.6167
MODES = ("self", "none")


# ── 집계(순수 함수 — GPU 없이 테스트한다) ──────────────────────────────────────
def summarize_two_trial(wrong_recs: Sequence[dict], correct_recs: Sequence[dict] = (),
                        *, k: int = 0, seed: int = 0, n_boot: int = 2000,
                        reference: float = FACT_ONLY_REFERENCE,
                        modes: Sequence[str] = MODES) -> dict:
    """행 단위 기록 → 관문 ②의 보고표.

    `wrong_recs` 의 각 원소는 한 **오답 롤아웃**이고 모드마다 `p_{mode}`(그 행의 K 표본
    구제율)를 갖는다. `correct_recs` 는 정답 롤아웃으로 `flip_{mode}`(정답 → 오답 비율)를
    갖는다. 두 모드가 다 있는 행만 짝 차이에 쓴다 — 짝이 아니면 (self − none) 이
    «모집단 차이»를 재게 된다.
    """
    modes = list(modes)
    # 짝 차이의 «처치» 팔 — none 이 아닌 첫 모드(self|switch|notx|switch_notx).
    treat = next((m for m in modes if m != "none"), "")
    out: dict = {"n_wrong": len(wrong_recs), "n_correct": len(correct_recs),
                 "k": k, "reference_fact_only": reference, "modes": modes,
                 "treat_mode": treat}
    for i, m in enumerate(modes):
        vals = [r[f"p_{m}"] for r in wrong_recs if f"p_{m}" in r]
        out[f"rescue_{m}"] = bootstrap_ci(vals, seed=seed + i, n_boot=n_boot)
        out[f"vs_reference_{m}"] = bootstrap_ci([v - reference for v in vals],
                                                seed=seed + 10 + i, n_boot=n_boot)
    paired = [r for r in wrong_recs if all(f"p_{m}" in r for m in modes)]
    d = ([r[f"p_{treat}"] - r["p_none"] for r in paired]
         if treat and "none" in modes else [])
    out[f"paired_{treat or 'treat'}_minus_none"] = bootstrap_ci(d, seed=seed + 20,
                                                               n_boot=n_boot)
    out[f"sign_p_{treat or 'treat'}_minus_none"] = sign_test_p(d)
    out["n_paired"] = len(paired)
    for i, m in enumerate(modes):
        fl = [r[f"flip_{m}"] for r in correct_recs if f"flip_{m}" in r]
        out[f"false_alarm_{m}"] = bootstrap_ci(fl, seed=seed + 30 + i, n_boot=n_boot)
        tr = [r[f"trunc_{m}"] for r in wrong_recs if f"trunc_{m}" in r]
        out[f"trunc_{m}"] = (sum(tr) / len(tr)) if tr else _NAN
        ng = [r[f"note_generic_{m}"] for r in wrong_recs if f"note_generic_{m}" in r]
        out[f"note_generic_{m}"] = (sum(ng) / len(ng)) if ng else _NAN
    return out


def two_trial_pass(summary: dict, *, reference: float = FACT_ONLY_REFERENCE,
                   treat: str = "") -> tuple[bool, list]:
    """관문 ② 판정: (a) 처치 팔의 구제율 CI 하한이 `.617` 이상이고 (b) 짝 차이
    (처치 − none) 의 CI 하한이 0 초과. 둘 다여야 «노트가 기여했다»가 성립한다.
    처치 팔은 기본적으로 요약의 `treat_mode`(없으면 self) — NOTE_MODE 확장(switch 계열)
    에서도 같은 판정선을 쓴다."""
    t = treat or summary.get("treat_mode") or "self"
    misses = []
    rs = summary.get(f"rescue_{t}") or {}
    if not (math.isfinite(rs.get("lo", _NAN)) and rs["lo"] >= reference):
        misses.append(f"rescue_{t}.lo={rs.get('lo')} < {reference}")
    pd = summary.get(f"paired_{t}_minus_none") or {}
    if not (math.isfinite(pd.get("lo", _NAN)) and pd["lo"] > 0.0):
        misses.append(f"paired_{t}_minus_none.lo={pd.get('lo')} <= 0")
    return (not misses), misses


# ── 단계 1: 시도-1 pass@1 (생성은 math_rollout 에 위임) ────────────────────────
def run_attempt1(a) -> int:
    """`math_rollout.main()` 을 **그대로** 부른다. 인자를 argv 로 넘기는 이유: 그 스크립트의
    단일 진입점이 argparse 이고, 여기서 로직을 복제하면 두 채점 경로가 갈라진다."""
    import math_rollout  # noqa: PLC0415

    argv = ["math_rollout.py", "--dataset", a.dataset, "--model_path", a.model_path,
            "--variant", a.variant, "--num_samples", str(a.k),
            "--max_tokens", str(a.max_tokens), "--seed", str(a.seed),
            "--gpu_util", str(a.gpu_util), "--out_dir", a.out_dir]
    if a.limit:
        argv += ["--limit", str(a.limit)]
    old, sys.argv = sys.argv, argv
    try:
        rc = math_rollout.main()
    finally:
        sys.argv = old
    tel = json.loads((Path(a.out_dir) / "telemetry.json").read_text())
    print(f"[trial2-eval] 관문① 시도-1 pass@1 acc={tel['acc']:.4f} "
          f"(M_G0 .740 / M_G1 .744 와 같은 자)", flush=True)
    return rc


# ── 단계 2: 2-시도 정확도 ─────────────────────────────────────────────────────
def run_two_trial(a) -> int:
    from vllm import LLM, SamplingParams  # noqa: PLC0415

    selftest_math_verify()
    rng = random.Random(a.seed)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rolls = [json.loads(line) for line in open(a.rollouts)]
    wrong = select_wrong_rollouts(rolls, per_problem=a.per_problem)
    rng.shuffle(wrong)
    wrong = wrong[: a.max_sites]
    correct = select_correct_rollouts(rolls, per_problem=1)
    rng.shuffle(correct)
    correct = correct[: len(wrong)]
    if not wrong:
        raise SystemExit("[trial2-eval] 오답 후보가 없다 — MIXED 오답이 있는 롤아웃인지 확인하라.")
    print(f"[trial2-eval] 오답 {len(wrong)} / 정답(거짓-경보) {len(correct)} 롤아웃", flush=True)

    llm = LLM(model=a.model_path, dtype="bfloat16", seed=a.seed,
              gpu_memory_utilization=a.gpu_util,
              max_model_len=a.max_tokens + a.max_prefix_tokens + 1024, enforce_eager=True)
    tok = llm.get_tokenizer()
    lim = a.max_prefix_tokens + 512

    modes = [t2.check_note_mode(m) for m in (a.modes or "self,none").split(",") if m.strip()]
    # 노트 단계가 있는 모드가 하나라도 있으면 사전 패스를 돈다(질문은 그 모드의 것 —
    # self=NOTE_ASK, switch 계열=LABEL_ASK). 두 종류를 한 판에서 섞지는 않는다.
    stage_modes = [m for m in modes if t2.note_stage_on(m)]
    if len({"self" if m == "self" else "label" for m in stage_modes}) > 1:
        raise SystemExit("[trial2-eval] 한 판에 self 와 switch 계열을 같이 둘 수 없다 "
                         "(사전 패스 질문이 다르다) — 따로 돌려라.")
    stage_mode = stage_modes[0] if stage_modes else ""

    # ── 노트 사전 패스(오답 본문이 들어가는 **유일한** 자리) ────────────────────
    notes = {("wrong", i): "" for i in range(len(wrong))}
    notes.update({("correct", i): "" for i in range(len(correct))})
    reasons: dict = {}
    reqs, ix = [], []
    for pop, rows in (("wrong", wrong), ("correct", correct)) if stage_mode else ():
        for si, s in enumerate(rows):
            q = t2.note_stage_prompt(tok, a.variant, s["problem"], s["text"], stage_mode)
            if len(tok.encode(q)) > lim:
                reasons[(pop, si)] = "too_long_prompt"
                continue
            reqs.append(q)
            ix.append((pop, si))
    print(f"[trial2-eval] 노트 사전 패스({stage_mode or '없음'}) 요청 {len(reqs)}개", flush=True)
    nouts = llm.generate(reqs, SamplingParams(n=1, temperature=0.0, top_p=1.0,
                                              max_tokens=a.note_max_tokens,
                                              seed=a.seed)) if reqs else []
    note_rows = []
    for (pop, si), o in zip(ix, nouts):
        s = (wrong if pop == "wrong" else correct)[si]
        n, why = t2.clean_stage_text(stage_mode, o.outputs[0].text, last_boxed(s["text"]))
        notes[(pop, si)] = n
        reasons[(pop, si)] = why
        note_rows.append({"population": pop, "roll_id": s["roll_id"],
                          "raw": o.outputs[0].text, "note": n, "reason": why})
    with (out / "notes.jsonl").open("w") as fh:
        for r in note_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[trial2-eval] 반성문 누출 통계 "
          f"{t2.leak_stats([reasons[kk] for kk in reasons])}", flush=True)

    # ── 재시도(문맥 폐기) — `--modes` 의 팔들 ─────────────────────────────────
    reqs, ix = [], []
    for pop, rows in (("wrong", wrong), ("correct", correct)):
        for si, s in enumerate(rows):
            for m in modes:
                # ★정답 행에서는 notx 계열을 **돌리지 않는다** — 그 행의 `\boxed` 는 gold
                #   이므로 «답은 X 가 아니다»가 곧 gold 누출(반대 방향 힌트)이 된다.
                #   오답 행에서는 X 가 «검증기가 이미 0 이라 한 답»이라 누출이 아니다.
                if pop == "correct" and m in ("notx", "switch_notx"):
                    continue
                nt = notes[(pop, si)]
                q = t2.attempt2_prompt(tok, a.variant, s["problem"], nt, mode=m,
                                       label=nt, notx=last_boxed(s["text"]))
                if len(tok.encode(q)) > lim:
                    continue
                reqs.append(q)
                ix.append((pop, si, m))
    print(f"[trial2-eval] 재시도 요청 {len(reqs)}개 x K={a.k} (예산 {a.max_tokens})", flush=True)
    outs = llm.generate(reqs, SamplingParams(n=a.k, temperature=1.0, top_p=1.0,
                                             max_tokens=a.max_tokens, seed=a.seed))
    agg: dict = {}
    for (pop, si, m), o in zip(ix, outs):
        s = (wrong if pop == "wrong" else correct)[si]
        agg[(pop, si, m)] = [{"r_corr": grade_math(x.text, s["gold"]),
                              "trunc": int(x.finish_reason == "length")}
                             for x in o.outputs]

    wrong_recs, correct_recs = [], []
    for pop, rows, sink in (("wrong", wrong, wrong_recs), ("correct", correct, correct_recs)):
        for si, s in enumerate(rows):
            rec = {"roll_id": s["roll_id"], "group_id": s["group_id"]}
            for m in modes:
                g = agg.get((pop, si, m))
                if not g:
                    continue
                p = sum(int(x["r_corr"]) for x in g) / len(g)
                if pop == "wrong":
                    rec[f"p_{m}"] = p
                    rec[f"note_generic_{m}"] = float(
                        m == "self" and reasons.get((pop, si), "ok") != "ok")
                else:
                    rec[f"flip_{m}"] = 1.0 - p
                rec[f"trunc_{m}"] = sum(int(x["trunc"]) for x in g) / len(g)
            sink.append(rec)

    summary = summarize_two_trial(wrong_recs, correct_recs, k=a.k, seed=a.seed,
                                  n_boot=a.n_boot, reference=a.reference, modes=modes)
    ok, misses = two_trial_pass(summary, reference=a.reference)
    summary["gate2_pass"] = ok
    summary["gate2_misses"] = misses
    with (out / "rows.jsonl").open("w") as fh:
        for r in wrong_recs + correct_recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                                                 default=float))
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=float))
    print(f"[out] {out}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("attempt1", "two_trial"), default="attempt1")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--variant", default="math_opt")
    ap.add_argument("--k", type=int, default=8, help="재시도/표집 표본 수 K")
    ap.add_argument("--max_tokens", type=int, default=8192, help="EVAL_MAX_TOKENS 규약")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--gpu_util", type=float, default=0.45)
    # stage attempt1
    ap.add_argument("--dataset", default="math500")
    ap.add_argument("--limit", type=int, default=0)
    # stage two_trial
    ap.add_argument("--rollouts", default=None, help="math_rollout 산출물 texts.jsonl (난이도 5)")
    ap.add_argument("--per_problem", type=int, default=2)
    ap.add_argument("--max_sites", type=int, default=150)
    ap.add_argument("--max_prefix_tokens", type=int, default=8192)
    ap.add_argument("--note_max_tokens", type=int, default=64)
    ap.add_argument("--modes", default="self,none",
                    help="비교할 NOTE_MODE 목록(쉼표). 예: switch,none / switch_notx,none")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--reference", type=float, default=FACT_ONLY_REFERENCE)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    if a.stage == "two_trial":
        if not a.rollouts:
            raise SystemExit("[trial2-eval] --stage two_trial 은 --rollouts 가 필요하다.")
        return run_two_trial(a)
    return run_attempt1(a)


if __name__ == "__main__":
    sys.exit(main())
