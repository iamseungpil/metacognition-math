"""B′ — 등가 마진 δ_eq 를 데이터로 만든다. 그리고 난이도 층화.

EXP-0816h-A2 사전등록: `δ_eq := 2 × (acc(B) − acc(B′) 의 95% CI 반폭)`.
B′ 는 **B 와 글자까지 동일한 정형문, 시드만 다름** -- 처치가 아니라 순수 디코드 잡음이다.
이것이 없으면 `acc(R) − acc(B) ≈ 0` 이 "내용이 무의미하다"인지 "우리 자가 못 읽는다"인지
구별할 수 없다. C-031 에서 정확히 그 실수를 했다(MDE 1.84pp, 등가 마진 미선언).

난이도 층화도 같이 한다. PG0(0619)가 easy +0.082 / **hard −0.034** 로 부호가 갈렸으므로,
전체 평균 하나로 보고하면 그 구조를 덮는다.
"""
from __future__ import annotations

import argparse
import json
import statistics as st

from splice_probe import BOILER, SOLVE_SYS, boot_ci, grade


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splice", default="splice_8b.json")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--seed", type=int, default=777, help="B 와 달라야 한다 (B 는 0)")
    ap.add_argument("--gpu_util", type=float, default=0.55)
    ap.add_argument("--levels", default="levels_math500.json")
    ap.add_argument("--out", default="bprime.json")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    blob = json.load(open(args.splice))
    items = blob["items"]
    acc = blob["per_problem"]

    llm = LLM(model=args.model, dtype="bfloat16", seed=args.seed,
              gpu_memory_utilization=args.gpu_util, max_model_len=4096)
    tok = llm.get_tokenizer()

    # 진행 중이던 splice 런은 `prompt` 를 저장하지 않는 판이었다(저장 패치가 실행 시작 뒤에
    # 들어갔다). 프롬프트는 (SOLVE_SYS, problem) 의 결정론적 함수이므로 정확히 재구성된다 --
    # 같은 토크나이저·같은 시스템 프롬프트. 근사가 아니라 동일 문자열이다.
    def chat(sys_msg, user):
        msgs = [{"role": "system", "content": sys_msg}, {"role": "user", "content": user}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True)

    for it in items:
        it.setdefault("prompt", chat(SOLVE_SYS, it["problem"]))

    reqs = [it["prompt"] + it["prefix"] + f"\n<meta>{BOILER}</meta>\n" for it in items]
    outs = llm.generate(reqs, SamplingParams(n=args.k, temperature=0.7, top_p=0.95,
                                             max_tokens=700, seed=args.seed))
    accBp = [st.mean(grade(x.text, it["gold"]) for x in o.outputs)
             for it, o in zip(items, outs)]

    # δ_eq — 같은 문자열, 다른 시드. 이 폭이 우리 자의 눈금이다.
    d_noise, lo, hi = boot_ci(list(zip(acc["B"], accBp)))
    half = (hi - lo) / 2.0
    delta_eq = 2.0 * half

    res = {"n": len(items), "k": args.k,
           "acc_B": st.mean(acc["B"]), "acc_Bprime": st.mean(accBp),
           "noise_B_minus_Bprime": {"delta": d_noise, "ci": [lo, hi], "half_width": half},
           "DELTA_EQ": delta_eq}

    # 사전등록 주 통계: 순서 가설 D ≤ S ≤ B ≤ R 에 대한 1 d.f. 선형 대비
    W = {"D": -1.5, "S": -0.5, "B": +0.5, "R": +1.5}
    per_problem = [sum(W[c] * acc[c][i] for c in W) for i in range(len(items))]
    obs, clo, chi = boot_ci([(v, 0.0) for v in per_problem])
    res["PRIMARY_ordered_contrast"] = {
        "delta": obs, "ci": [clo, chi],
        "excludes_zero": clo > 0 or chi < 0,
        "within_delta_eq": abs(obs) < delta_eq and clo > -delta_eq and chi < delta_eq,
    }

    # 난이도 층화 — PG0 가 easy 와 hard 에서 부호가 갈렸다
    try:
        lv = {r["problem"]: r["level"] for r in json.load(open(args.levels))}
        strata = {}
        for i, it in enumerate(items):
            L = lv.get(it["problem"])
            if L is None:
                continue
            key = "easy(1-2)" if L <= 2 else ("med(3)" if L == 3 else "hard(4-5)")
            strata.setdefault(key, []).append(i)
        res["by_level"] = {}
        for key, idx in sorted(strata.items()):
            row = {"n": len(idx)}
            for a, b in (("R", "N"), ("R", "B"), ("R", "S"), ("R", "D")):
                o, l, h = boot_ci([(acc[a][i], acc[b][i]) for i in idx], n=4000)
                row[f"{a}-{b}"] = {"delta": round(o, 4), "ci": [round(l, 4), round(h, 4)]}
            row["acc"] = {c: round(st.mean(acc[c][i] for i in idx), 4) for c in ("N", "B", "S", "R", "D")}
            res["by_level"][key] = row
    except FileNotFoundError:
        res["by_level"] = "levels file missing"

    json.dump({**res, "accBprime_per_problem": accBp}, open(args.out, "w"), indent=2)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
