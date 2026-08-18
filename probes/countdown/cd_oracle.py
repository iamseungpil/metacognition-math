"""Countdown 오라클 판 — **기전이 있나, 그리고 PMI-shift 가 품질 순서를 아는가.**

여덟 판이 전부 "모델이 만든 메타"만 넣었다. 그래서 널이 **"기전이 없다"** 인지
**"우리 메타가 나쁘다"** 인지 한 번도 못 갈랐다. 이 판이 그 둘을 가른다.

★그리고 수학에서는 이걸 할 수 없다 — *"완벽한 힌트"* 가 무엇인지 모르니까.
Countdown 은 증인 풀이가 있어서 **실행 가능한 조언을 정확히 만들 수 있다.**

═══════════════════════════════════════════════════════════════════════
팔 여섯 — 품질이 **구성으로 정해진 사다리**
═══════════════════════════════════════════════════════════════════════
    N    메타 없음                                  기준 (모집단 정의용)
    N′   메타 없음, 디코딩 시드만 다름                ★재추첨 잡음바닥
    R    모델 자신의 자기점검                        지금까지 재던 것
    OF   거짓 오라클 — 같은 문장 틀, **틀린 결합**    "구체적 수가 있으면 돕는다" 를 죽인다
    O1   오라클 — 증인의 **첫 결합**을 알려줌         실행 가능한 조언의 상한
    O3   오라클 — **전체 식**을 알려줌                ★배선검사: SAVE ~100% 여야 한다

  참 품질 순서 (구성으로 알려짐):   O3 > O1 > R ≈ OF ≈ N′

═══════════════════════════════════════════════════════════════════════
사전등록 — 판정 둘
═══════════════════════════════════════════════════════════════════════

★판정 A · 기전이 있나
    배선   P(SAVE|O3) ≥ 0.80.  미달이면 **실험 자체가 고장**이고 나머지를 읽지 않는다.
    주지표 P(SAVE|O1) − P(SAVE|N′)      실행 가능한 조언이 재추첨보다 나은가
    통제   P(SAVE|O1) − P(SAVE|OF)      ★맞는 조언이 **틀린** 조언보다 나은가
                                         (같은 틀·같은 구체성 — 길이 차 중앙값 1자)
    여지   P(SAVE|O1) − P(SAVE|R)       우리 메타가 오라클에 얼마나 모자라나

    ⇒ O1 이 N′ 와 OF 를 모두 이긴다  →  ★기전 있음. 병목 = 메타 품질. 학습으로 간다.
      O1 이 널                        →  얼어붙은 모델에는 채널이 없다. 주입식 실험 종료.

★판정 B · PMI-shift 가 품질 순서를 아는가  (2단계 `cd_rank.py` 가 채점)
    ★이 판이 **PMI-shift 의 정답지**를 처음으로 준다 — 품질 순서가 **구성으로** 알려져 있고,
      다섯 메타가 **같은 prefix** 위에 있어 난이도가 상쇄된다. n = 문제수 × 5.
      지금까지의 검정은 전부 (a) 잡음 섞인 결과를 과녁으로 삼거나 (b) 문제 간 비교였다.
    판정  문제 안 Spearman(PMI-shift, 참 품질순위) 의 평균이 치환 귀무를 넘는가
          + 음성 대조군(메타 길이)을 이기는가

설계 수리 둘 (직전 판이 드러낸 것):
    frac 0.6 → 0.3      직전 판은 prefix 가 탐색예산의 절반 이상을 태웠다(시도 7.8 → 3.76)
    예산 8 → 16회       메타가 "남은 예산을 잘 쓰게" 할 여지를 만든다

중단  ① 형식 < 80%  ② acc(N)=0 후보 < 100  ③ O3 배선 미달
⚠오라클은 **진단이지 학습 방법이 아니다.** 정답을 쓴다. 성능 주장 불가.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics as st

from countdown import (gen_instance, oracle_metas, grade, parse_ok, n_attempts,
                       SOLVE_SYS, CHECK_SYS, user_msg)
from splice_probe import splice, boot_ci

ARMS = ["N", "Np", "R", "OF", "O1", "O3"]
DOC = {"N": "메타 없음", "Np": "메타 없음·시드만 (잡음바닥)", "R": "모델 자신의 자기점검",
       "OF": "거짓 오라클 (틀린 결합)", "O1": "오라클 (첫 결합)", "O3": "오라클 (전체 식)"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--n_problems", type=int, default=400)
    ap.add_argument("--n_nums", type=int, default=5)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--frac", type=float, default=0.3)     # 수리: 0.6 → 0.3
    ap.add_argument("--budget", type=int, default=16)      # 수리: 8 → 16
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_tokens", type=int, default=3072)
    ap.add_argument("--gpu_util", type=float, default=0.55)
    ap.add_argument("--out", default="cd_oracle.json")
    args = ap.parse_args()

    # 탐색예산 상한을 프롬프트에서 올린다 — 치환이 세 곳 다 걸렸는지 확인한다
    sys_solve = (SOLVE_SYS.replace("AT MOST 8", f"AT MOST {args.budget}")
                 .replace("past 8 attempts", f"past {args.budget} attempts")
                 .replace("within 8 attempts", f"within {args.budget} attempts"))
    assert sys_solve.count(str(args.budget)) == 3, "예산 치환이 세 곳 다 안 걸렸다"
    print(f"[0] 탐색예산 {args.budget}회 · splice frac {args.frac}")

    from vllm import LLM, SamplingParams
    rng = random.Random(args.seed)

    items = []
    while len(items) < args.n_problems:
        it = gen_instance(args.n_nums, rng)
        if it is None:
            continue
        om = oracle_metas(it["witness"], it["nums"], rng)
        if not all(om[k] for k in ("O1", "O3", "OF")):
            continue
        it.update(om)
        items.append(it)
    print(f"    문제 {len(items)} · 오라클 3종 전부 생성됨")

    llm = LLM(model=args.model, dtype="bfloat16", seed=args.seed,
              gpu_memory_utilization=args.gpu_util, max_model_len=4096)
    tok = llm.get_tokenizer()

    def chat(sm, u):
        m = [{"role": "system", "content": sm}, {"role": "user", "content": u}]
        try:
            return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)

    o1 = llm.generate([chat(sys_solve, user_msg(it)) for it in items],
                      SamplingParams(n=1, temperature=0.7, top_p=0.95,
                                     max_tokens=args.max_tokens, seed=args.seed))
    keep = []
    for it, o in zip(items, o1):
        pre = splice(o.outputs[0].text, frac=args.frac)
        if pre:
            it["prompt"], it["prefix"] = o.prompt, pre
            keep.append(it)
    items = keep
    print(f"[1] prefix {len(items)}")

    o2 = llm.generate(
        [chat(CHECK_SYS, f"Puzzle:\n{user_msg(it)}\n\nWork so far:\n{it['prefix']}")
         for it in items],
        SamplingParams(n=1, temperature=0.7, top_p=0.95, max_tokens=160, seed=args.seed))
    keep = []
    for it, o in zip(items, o2):
        t = " ".join(o.outputs[0].text.split())
        if re.search(r"confidence:\s*[0-9.]+", t) and re.search(r"decision:\s*\w+", t):
            it["R"] = t[:400]
            it["N"] = it["Np"] = None
            keep.append(it)
    print(f"[2] self-check {len(keep)}/{len(items)} = {len(keep)/max(1,len(items)):.1%}")
    items = keep

    reqs, idx = [], []
    for c in ARMS:
        for k, it in enumerate(items):
            m = it[c]
            reqs.append(it["prompt"] + it["prefix"]
                        + ("" if m is None else f"\n<meta>{m}</meta>") + "\n")
            idx.append((c, k))
    print(f"[3] 재생성 {len(reqs)} = {len(ARMS)}팔 x {len(items)} x k={args.k}")

    is_np = [c == "Np" for c, _ in idx]
    o3 = [None] * len(reqs)
    for flag, sd in ((False, args.seed), (True, args.seed + 977)):
        sel = [i for i, f in enumerate(is_np) if f == flag]
        out = llm.generate([reqs[i] for i in sel],
                           SamplingParams(n=args.k, temperature=0.7, top_p=0.95,
                                          max_tokens=args.max_tokens, seed=sd))
        for i, o in zip(sel, out):
            o3[i] = o

    acc = {c: [0.0] * len(items) for c in ARMS}
    recs = {c: [None] * len(items) for c in ARMS}
    for (c, k), o in zip(idx, o3):
        it = items[k]
        per = [{"graded": grade(x.text, it["nums"], it["target"]),
                "fmt": parse_ok(x.text), "finish": x.finish_reason,
                "ntok": len(x.token_ids), "tries": n_attempts(x.text),
                "text": x.text if si == 0 else None}
               for si, x in enumerate(o.outputs)]
        acc[c][k] = st.mean(r["graded"] for r in per)
        recs[c][k] = per

    fmt = st.mean(r["fmt"] for c in ARMS for P in recs[c] for r in P)
    print(f"\n[4] 형식 {fmt:.1%} · 절단 "
          f"{st.mean(1 if r['finish']=='length' else 0 for c in ARMS for P in recs[c] for r in P):.1%}")
    print(f"\n{'팔':4s} {'정확도':>7s} {'시도':>6s}   설명")
    for c in ARMS:
        print(f"{c:4s} {st.mean(acc[c]):7.4f} "
              f"{st.mean(r['tries'] for P in recs[c] for r in P):6.2f}   {DOC[c]}")

    POOL = [i for i in range(len(items)) if acc["N"][i] <= 1e-9]
    print(f"\n[모집단] acc(N)=0 : {len(POOL)}문제")
    sv = {c: [1 if acc[c][i] > 1e-9 else 0 for i in POOL] for c in ARMS}
    print(f"\n{'팔':4s} {'SAVE율':>8s}")
    for c in ARMS:
        print(f"{c:4s} {st.mean(sv[c])*100:7.1f}%")

    print("\n[판정 A]")
    o3r = st.mean(sv["O3"])
    print(f"  ★배선  P(SAVE|O3) = {o3r*100:.1f}%   "
          f"{'✅ 통과' if o3r >= .80 else '⛔미달 — 실험 자체가 고장. 나머지를 읽지 않는다.'}")
    res = {}
    for a, b, lab in (("O1", "Np", "★주지표 오라클 vs 재추첨"),
                      ("O1", "OF", "★통제 맞는 조언 vs 틀린 조언"),
                      ("O1", "R", "여지 오라클 vs 모델 자신"),
                      ("R", "Np", "참고 모델 자신 vs 재추첨")):
        d, lo, hi = boot_ci(list(zip(sv[a], sv[b])))
        ex = lo > 0 or hi < 0
        res[f"{a}-{b}"] = {"delta": d, "ci": [lo, hi], "excludes_zero": ex}
        print(f"  {lab:28s} {d*100:+6.2f}pp [{lo*100:+6.2f},{hi*100:+6.2f}]  {'유의' if ex else '널'}")
    a1, a2 = res["O1-Np"], res["O1-OF"]
    live = a1["excludes_zero"] and a1["delta"] > 0 and a2["excludes_zero"] and a2["delta"] > 0
    print(f"\n  ⇒ {'★기전 있음 — 병목은 메타 품질. 학습으로 간다.' if live else '기전 없음 — 얼어붙은 모델에 채널이 없다.'}")

    json.dump({"model": args.model, "arms": ARMS, "frac": args.frac, "budget": args.budget,
               "fmt": fmt, "mean_acc": {c: st.mean(acc[c]) for c in ARMS},
               "pool": POOL, "save_rate": {c: st.mean(sv[c]) for c in ARMS},
               "verdict_A": res, "o3_wiring": o3r,
               "per_problem": acc, "per_sample": recs, "items": items},
              open(args.out, "w"))
    print(f"\n[5] wrote {args.out}")


if __name__ == "__main__":
    main()
