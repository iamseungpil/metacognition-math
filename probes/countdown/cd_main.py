"""Countdown 본 판 1단계 — SAVE 라벨을 만든다 (생성).

묻는 것 하나: ***자기점검이 "못 풀던 문제를 풀게" 만드는가 — 재추첨보다 나은가.***

왜 이 과제인가. 수학에서 여섯 판이 전부 널이었는데 **"방법이 나쁜가"와 "이 과제에 여지가
없는가"를 한 번도 못 갈랐다.** Gandhi(2503.01307)가 Countdown 에서 자기교정이 정답률을
올린다고 보였으니, 여기서도 널이면 방법이 문제고 양성이면 수학이 잘못된 시험장이었다.

파일럿 3회차(0818)로 보정 완료:
    수 5개 · cap 3072 · 탐색예산 8회 강제  →  형식 99.5% · 절단 0.2% · acc=0 58%
  1·2차는 사전등록 중단조건 ②(형식 <80%)에 걸려 판정하지 않았다. 원인은 캡이 아니라
  **답을 안 맺는 것**이었다(3,790 토큰 쓰고 \\boxed 없음). 그래서 acc=0 이 "못 찾음"만
  뜻하게 됐다 — 이게 없으면 SAVE 가 "안 맺다가 맺음"을 재게 된다.

═══════════════════════════════════════════════════════════════════════
사전등록 (실행 전에 적는다 — 결과를 보고 고치지 않는다)
═══════════════════════════════════════════════════════════════════════

팔 다섯. 각각이 하나의 대안설명을 죽인다.
    N    메타 없음                              기준
    N′   메타 없음, **디코딩 시드만 다름**        ★"그냥 다시 뽑아도 풀린다" ← 진짜 잡음바닥
    B    정형문 (문제정보 0, 일반 재시도 권고)     "아무 재시도 권고나 도움이 된다"
    SH   R 의 어절 섞기 (어휘·길이 보존, 의미 파괴) "어휘·길이가 도움이 된다"
    R    진짜 자기점검                           —

모집단   acc(N) = 0 인 문제   (메타 없이 k 번 전부 실패)
SAVE     그 문제에서 해당 팔이 한 번이라도 맞힘

★주 지표  P(SAVE|R) − P(SAVE|N′)     문제 단위 쌍대 부트스트랩.  **단 하나.**
보조     P(SAVE|B)−P(SAVE|N′) · P(SAVE|R)−P(SAVE|B) · P(SAVE|R)−P(SAVE|SH)   Holm 보정

판정밴드  주 지표 CI 가 0 배제 & 양수 → ★기전 있음. 2단계(격자 신호 채점)로 간다.
         CI 가 0 포함              → 기전 없음. R_meta 프로그램을 접는다.

중단     ① 형식 준수율 < 80%  → 판정하지 않는다 (파일럿에서 두 번 발동했다)
         ② acc(N)=0 문제가 100 미만 → 표본 부족, 판정하지 않는다
         ③ 메타가 정답 식을 그대로 포함 → 그 행 제외 (답 누출)
음성     SH 가 R 을 이기면 전부 폐기

⚠2단계를 위해 **팔마다 전 샘플**의 graded·pred·finish·ntok·시도수와 sample0 텍스트,
  메타 세 종, witness, 연산자교체 decoy 를 전부 저장한다. 어제 `cont_R` 만 저장해서
  판정이 세 번 막혔다(참경보 n=5).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics as st

from countdown import (gen_instance, swap_op_decoy, grade, parse_ok, n_attempts,
                       SOLVE_SYS, CHECK_SYS, BOILER, user_msg)
from splice_probe import splice, shuffle_words, boot_ci

ARMS = ["N", "Np", "B", "SH", "R"]
ARM_DOC = {"N": "메타 없음", "Np": "메타 없음·시드만 다름 (잡음바닥)",
           "B": "정형문", "SH": "어절 섞기", "R": "진짜 자기점검"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--n_problems", type=int, default=400)
    ap.add_argument("--n_nums", type=int, default=5)      # 파일럿 3회차가 정한 값
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_tokens", type=int, default=3072)
    ap.add_argument("--gpu_util", type=float, default=0.55)
    ap.add_argument("--out", default="cd_main.json")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    rng = random.Random(args.seed)

    items = []
    while len(items) < args.n_problems:
        it = gen_instance(args.n_nums, rng)
        if it is None:
            continue
        it["decoy"] = swap_op_decoy(it["witness"], it["nums"], it["target"], rng)
        items.append(it)
    print(f"[0] 문제 {len(items)} (수 {args.n_nums}개) · "
          f"연산자교체 오답 생성 {sum(1 for i in items if i['decoy'])}/{len(items)}")

    llm = LLM(model=args.model, dtype="bfloat16", seed=args.seed,
              gpu_memory_utilization=args.gpu_util, max_model_len=4096)
    tok = llm.get_tokenizer()

    def chat(sys_msg, user):
        msgs = [{"role": "system", "content": sys_msg},
                {"role": "user", "content": user}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True)

    # 1 ─ prefix (부분 탐색)
    o1 = llm.generate([chat(SOLVE_SYS, user_msg(it)) for it in items],
                      SamplingParams(n=1, temperature=0.7, top_p=0.95,
                                     max_tokens=args.max_tokens, seed=args.seed))
    keep = []
    for it, o in zip(items, o1):
        pre = splice(o.outputs[0].text)
        if pre:
            it["prompt"] = o.prompt
            it["prefix"] = pre
            keep.append(it)
    items = keep
    print(f"[1] prefix {len(items)}")

    # 2 ─ 진짜 자기점검 R
    o2 = llm.generate(
        [chat(CHECK_SYS, f"Puzzle:\n{user_msg(it)}\n\nWork so far:\n{it['prefix']}")
         for it in items],
        SamplingParams(n=1, temperature=0.7, top_p=0.95, max_tokens=160, seed=args.seed))
    keep = []
    for it, o in zip(items, o2):
        t = " ".join(o.outputs[0].text.split())
        if re.search(r"confidence:\s*[0-9.]+", t) and re.search(r"decision:\s*\w+", t):
            it["R"] = t[:400]
            keep.append(it)
    yr = len(keep) / max(1, len(items))
    print(f"[2] self-check {len(keep)}/{len(items)} = {yr:.1%}")
    items = keep

    # 3 ─ 다섯 팔
    for it in items:
        it["N"] = it["Np"] = None
        it["B"] = BOILER
        it["SH"] = shuffle_words(it["R"], rng)
        # 답 누출 검사 — 메타가 증인 식을 그대로 담고 있나 (중단조건 ③)
        it["leak"] = 1 if it["witness"].replace(" ", "") in it["R"].replace(" ", "") else 0

    reqs, idx = [], []
    for c in ARMS:
        for k, it in enumerate(items):
            m = it[c]
            block = "" if m is None else f"\n<meta>{m}</meta>"
            reqs.append(it["prompt"] + it["prefix"] + block + "\n")
            idx.append((c, k))
    print(f"[3] 재생성 {len(reqs)} = {len(ARMS)}팔 x {len(items)}문제 x k={args.k}")

    # N′ 만 다른 시드 — 같은 프롬프트에서 **순수 디코딩 잡음**만 뽑는다
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
        per = []
        for si, x in enumerate(o.outputs):
            per.append({"graded": grade(x.text, it["nums"], it["target"]),
                        "fmt": parse_ok(x.text), "finish": x.finish_reason,
                        "ntok": len(x.token_ids), "tries": n_attempts(x.text),
                        "text": x.text if si == 0 else None})
        acc[c][k] = st.mean(r["graded"] for r in per)
        recs[c][k] = per

    # 4 ─ 판정
    fmt = st.mean(r["fmt"] for c in ARMS for P in recs[c] for r in P)
    trunc = st.mean(1 if r["finish"] == "length" else 0
                    for c in ARMS for P in recs[c] for r in P)
    print(f"\n[4] 형식 준수 {fmt:.1%} · 절단 {trunc:.1%} · 답누출 {sum(i['leak'] for i in items)}")
    if fmt < 0.80:
        print("  ⛔중단조건 ① — 형식 <80%. 판정하지 않는다.")

    print(f"\n{'팔':4s} {'정확도':>7s} {'시도':>6s}   설명")
    for c in ARMS:
        print(f"{c:4s} {st.mean(acc[c]):7.4f} "
              f"{st.mean(r['tries'] for P in recs[c] for r in P):6.2f}   {ARM_DOC[c]}")

    POOL = [i for i in range(len(items)) if acc["N"][i] <= 1e-9 and not items[i]["leak"]]
    print(f"\n[모집단] acc(N)=0 · 누출 없음 : {len(POOL)}문제")
    if len(POOL) < 100:
        print("  ⛔중단조건 ② — 후보 100 미만. 표본 부족.")

    sv = {c: [1 if acc[c][i] > 1e-9 else 0 for i in POOL] for c in ARMS}
    print(f"\n{'팔':4s} {'SAVE율':>8s}   n={len(POOL)}")
    for c in ARMS:
        print(f"{c:4s} {st.mean(sv[c])*100:7.1f}%")

    print("\n[판정]")
    res = {}
    for a, b, lab, prim in (("R", "Np", "★주 지표 자기점검 vs 재추첨", True),
                            ("B", "Np", "정형문 vs 재추첨", False),
                            ("R", "B", "내용 vs 일반권고", False),
                            ("R", "SH", "의미 vs 어휘", False)):
        o, lo, hi = boot_ci(list(zip(sv[a], sv[b])))
        excl = lo > 0 or hi < 0
        res[f"{a}-{b}"] = {"delta": o, "ci": [lo, hi], "excludes_zero": excl}
        star = "★" if (prim and excl) else " "
        print(f" {star}{lab:28s} {o*100:+6.2f}pp [{lo*100:+6.2f},{hi*100:+6.2f}]"
              f"  {'유의' if excl else '널'}")
    p = res["R-Np"]
    print(f"\n  ⇒ {'★기전이 있다 — 2단계(격자 신호)로' if p['excludes_zero'] and p['delta']>0 else '기전이 없다 — R_meta 프로그램 종료'}")

    json.dump({"model": args.model, "n_nums": args.n_nums, "k": args.k,
               "arms": ARMS, "fmt": fmt, "trunc": trunc,
               "mean_acc": {c: st.mean(acc[c]) for c in ARMS},
               "pool": POOL, "save_rate": {c: st.mean(sv[c]) for c in ARMS},
               "verdict": res, "per_problem": acc, "per_sample": recs,
               "items": items}, open(args.out, "w"))
    print(f"\n[5] wrote {args.out}")


if __name__ == "__main__":
    main()
