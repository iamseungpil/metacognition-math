"""Countdown 파일럿 — **난이도 보정만** 한다. 판정은 안 한다.

왜 파일럿이 먼저인가. MATH-500 에서 `acc_N = 0` 인 문제가 47/500(9.4%)뿐이라
SAVE 후보가 없어서 아무것도 판정 못 했다. 같은 실수를 반복하지 않으려면
**본 판을 발사하기 전에 난이도를 맞춰야** 한다.

목표: `acc_N = 0` 인 문제가 **40~60%** 가 되는 숫자 개수를 찾는다.
  너무 낮으면(=쉬우면) SAVE 후보가 없다.  너무 높으면(=절망적이면) 회수가 안 된다.

같이 재는 것:
  형식 준수율      < 80% 면 프롬프트부터 고친다 (사전등록 중단 조건 ②)
  응답 길이        절단이 정답률을 깎았던 전례 — Countdown 은 짧아야 정상
  시도 횟수        ★기전 지표. 수학에서는 못 재는 것.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st

from countdown import (gen_instance, grade, parse_ok, n_attempts,
                       SOLVE_SYS, user_msg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--n_per", type=int, default=50)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--sizes", default="4,5,6")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_tokens", type=int, default=1536)
    ap.add_argument("--gpu_util", type=float, default=0.55)   # 공유 A100 — 비켜간다
    ap.add_argument("--out", default="cd_pilot.json")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    rng = random.Random(args.seed)
    sizes = [int(x) for x in args.sizes.split(",")]

    probs = []
    for N in sizes:
        for _ in range(args.n_per):
            it = gen_instance(N, rng)
            if it:
                it["n_nums"] = N
                probs.append(it)
    print(f"[0] 문제 {len(probs)}  (크기별 {args.n_per})")

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

    outs = llm.generate([chat(SOLVE_SYS, user_msg(p)) for p in probs],
                        SamplingParams(n=args.k, temperature=0.7, top_p=0.95,
                                       max_tokens=args.max_tokens, seed=args.seed))
    for p, o in zip(probs, outs):
        g = [grade(x.text, p["nums"], p["target"]) for x in o.outputs]
        p["acc"] = st.mean(g)
        p["fmt"] = st.mean(parse_ok(x.text) for x in o.outputs)
        p["ntok"] = st.mean(len(x.token_ids) for x in o.outputs)
        p["trunc"] = st.mean(1 if x.finish_reason == "length" else 0 for x in o.outputs)
        p["tries"] = st.mean(n_attempts(x.text) for x in o.outputs)
        p["sample"] = o.outputs[0].text[:1200]

    print(f"\n{'수 개수':>7s} {'정확도':>7s} {'acc=0':>8s} {'acc=1':>7s} "
          f"{'형식':>6s} {'토큰':>6s} {'절단':>6s} {'시도':>5s}")
    res = {}
    for N in sizes:
        P = [p for p in probs if p["n_nums"] == N]
        z = sum(1 for p in P if p["acc"] <= 1e-9)
        one = sum(1 for p in P if p["acc"] >= 1 - 1e-9)
        res[N] = {"n": len(P), "acc": st.mean(p["acc"] for p in P),
                  "zero_rate": z / len(P), "one_rate": one / len(P),
                  "fmt": st.mean(p["fmt"] for p in P),
                  "ntok": st.mean(p["ntok"] for p in P),
                  "trunc": st.mean(p["trunc"] for p in P),
                  "tries": st.mean(p["tries"] for p in P)}
        r = res[N]
        print(f"{N:7d} {r['acc']:7.3f} {r['zero_rate']*100:7.1f}% {r['one_rate']*100:6.1f}% "
              f"{r['fmt']*100:5.1f}% {r['ntok']:6.0f} {r['trunc']*100:5.1f}% {r['tries']:5.1f}")

    # ── 보정 판정
    print("\n[판정]")
    good = [N for N in sizes if 0.40 <= res[N]["zero_rate"] <= 0.60]
    fmt_bad = [N for N in sizes if res[N]["fmt"] < 0.80]
    if fmt_bad:
        print(f"  ⛔형식 준수율 < 80% : 크기 {fmt_bad}  → 프롬프트부터 고친다 (중단조건 ②)")
    if good:
        pick = good[0]
        print(f"  ✅본 판 난이도 = 수 {pick}개  (acc=0 비율 {res[pick]['zero_rate']*100:.0f}%)")
        need = int(400 * 0.5 / max(0.05, res[pick]["zero_rate"]))
        print(f"     SAVE 후보 200개를 얻으려면 문제 {int(200/res[pick]['zero_rate'])}개 필요")
    else:
        z = {N: round(res[N]["zero_rate"], 3) for N in sizes}
        print(f"  ⚠40~60% 대역에 든 크기가 없다: {z}")
        print("     → 숫자 범위(hi)나 min_target 을 조절해 재보정한다 (중단조건 ①)")

    json.dump({"model": args.model, "k": args.k, "res": {str(k): v for k, v in res.items()},
               "probs": probs}, open(args.out, "w"))
    print(f"\n[1] wrote {args.out}")


if __name__ == "__main__":
    main()
