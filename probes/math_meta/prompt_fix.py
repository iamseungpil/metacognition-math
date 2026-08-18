"""2단계 — 헛경보의 얼마가 **지시문 탓**인가. (보상을 고치기 전에 반드시 먼저)

0817 실측: 잘 풀고 있는 208문제 중 **115개(55%)** 에서 자기점검이 없는 오류를 주장했고,
그 헛경보가 −5.9pp 를 먹었다. 주장 없는 확인 권고는 −1.4pp 로 거의 무해했다.

그런데 지금 지시문(splice_probe.CHECK_SYS)이 이렇게 되어 있다:

    "Name the single quantity, step, or assumption in the work so far that is
     **most likely to be wrong** ... Then give your confidence and a decision."

★**오류를 하나 지목하라고 강제한다.** 빠져나갈 문장이 없다. 그러면 55% 헛경보율은
모델의 성향이 아니라 **지시문이 만든 수**일 수 있고, 그 상태로 보상을 설계하면
지시문의 결함을 보상으로 덮으려 드는 꼴이 된다.

묻는 것 하나: ***"문제 없으면 없다고 말해도 된다"고 허락하면 헛경보율이 얼마나 떨어지나.***

네 갈래를 같은 300 prefix 위에서 나란히 생성한다(본문 고정, 지시문만 교체):

    P0  현행          강제 지목                         (기준선 — 헛경보 55% 재현 확인)
    P1  탈출구        "문제 없으면 no issue 라고 하라"
    P2  증거요구      "구체적 수치로 반증할 수 있을 때만 오류라고 하라"
    P3  탈출구+증거    둘 다

각 갈래마다 재생성·재채점까지 해서 **헛경보율**과 **실제 손익(acc − acc_N)** 을 같이 낸다.
지시문만 바꿔서 손해가 사라지면, 그것은 보상 문제가 아니었다는 뜻이다.

⛔EXP 트랙. 헤드라인 금지.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st

from splice_probe import SOLVE_SYS, grade
from alarm_probe import claims_error, get_conf

# ── 지시문 네 갈래. 출력 형식은 **바이트 단위로 동일**하게 유지한다 --
#    형식이 달라지면 파서가 달라지고, 그러면 갈래 비교가 아니라 파서 비교가 된다.
_FORM = ("\n\nAnswer in exactly this form and nothing else:\n"
         "confidence: <0 to 1> | <your one sentence> | decision: verify\n"
         "  (use `decision: redirect` instead if the approach should be abandoned)")

P0 = ("You are reviewing a partial solution to a math problem. Name the single "
      "quantity, step, or assumption in the work so far that is most likely to "
      "be wrong, in one sentence, referring to this problem's own numbers. Then "
      "give your confidence and a decision." + _FORM)

_ESCAPE = (" If the work so far contains no actual error, say so plainly instead "
           "of inventing one -- write `no issue found` as your sentence.")

_EVIDENCE = (" Only call something wrong if you can point to the specific number "
             "or step that contradicts it. A step you merely have not checked is "
             "not an error.")

P1 = P0.replace(_FORM, _ESCAPE + _FORM)
P2 = P0.replace(_FORM, _EVIDENCE + _FORM)
P3 = P0.replace(_FORM, _ESCAPE + _EVIDENCE + _FORM)

ARMS = {"P0_current": P0, "P1_escape": P1, "P2_evidence": P2, "P3_both": P3}

_NOISSUE = re.compile(r"no issue|nothing (?:is )?wrong|appears? correct|"
                      r"no error|all steps? (?:are )?correct", re.I)


def boot_ci(xs, n=2000, seed=0):
    import random
    rng = random.Random(seed)
    if not xs:
        return (float("nan"), float("nan"))
    v = sorted(st.mean(rng.choice(xs) for _ in xs) for _ in range(n))
    return (v[int(.025 * n)], v[int(.975 * n)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splice", default="splice_4b.json")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--k", type=int, default=6, help="갈래당 재생성 표본")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu_util", type=float, default=0.45,
                    help="⚠로컬 A100 은 공유 자원 -- 남의 잡을 밀어내지 않는다")
    ap.add_argument("--out", default="prompt_fix.json")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    blob = json.load(open(args.splice))
    items = blob["items"]
    accN = blob["per_problem"]["N"]

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

    user_of = [f"Problem:\n{it['problem']}\n\nWork so far:\n{it['prefix']}"
               for it in items]

    # ── 1. 갈래별 자기점검 생성. temp 0.7 · 한 문제당 하나 (0단계 데이터와 같은 조건).
    metas = {}
    for name, sys_msg in ARMS.items():
        outs = llm.generate([chat(sys_msg, u) for u in user_of],
                            SamplingParams(n=1, temperature=0.7, top_p=0.95,
                                           max_tokens=160, seed=args.seed))
        metas[name] = [" ".join(o.outputs[0].text.split())[:400] for o in outs]
        ok = sum(1 for m in metas[name]
                 if re.search(r"confidence:\s*[0-9.]+", m) and "decision:" in m)
        print(f"[1] {name:12s} 형식 통과 {ok}/{len(items)}")

    # ── 2. 각 갈래 메타를 본문에 접합하고 다시 풀린다 (본문 고정 -- matched)
    reqs, idx = [], []
    for name in ARMS:
        for i, it in enumerate(items):
            reqs.append(it["prompt"] + it["prefix"] + f"\n<meta>{metas[name][i]}</meta>\n")
            idx.append((name, i))
    print(f"[2] 재생성 {len(reqs)} × k={args.k}")
    o2 = llm.generate(reqs, SamplingParams(n=args.k, temperature=0.7, top_p=0.95,
                                           max_tokens=700, seed=args.seed))
    acc = {n: [0.0] * len(items) for n in ARMS}
    for (name, i), o in zip(idx, o2):
        acc[name][i] = st.mean(grade(x.text, items[i]["gold"]) for x in o.outputs)

    # ── 3. 판정. 층은 acc_N 으로만 가른다(처치군 성적으로 층을 가르면 선택 편향).
    easy = [i for i in range(len(items)) if accN[i] >= 0.999]
    hard = [i for i in range(len(items)) if accN[i] <= 0.001]
    res = {}
    print(f"\n{'갈래':12s} {'헛경보율':>9s} {'노이슈율':>9s} "
          f"{'잘푸는 손익':>13s} {'틀리는 손익':>13s} {'전체':>9s}")
    for name in ARMS:
        cl = [claims_error(m) for m in metas[name]]
        fa = st.mean(cl[i] for i in easy)                       # 헛경보율
        ni = st.mean(1 if _NOISSUE.search(m) else 0 for m in metas[name])
        de = [acc[name][i] - accN[i] for i in easy]
        dh = [acc[name][i] - accN[i] for i in hard]
        dall = [acc[name][i] - accN[i] for i in range(len(items))]
        lo_e, hi_e = boot_ci(de)
        res[name] = {
            "false_alarm_rate_easy": round(fa, 4),
            "no_issue_rate": round(ni, 4),
            "claim_rate_hard": round(st.mean(cl[i] for i in hard), 4),
            "delta_easy": round(st.mean(de), 4), "delta_easy_ci": [round(lo_e, 4), round(hi_e, 4)],
            "delta_hard": round(st.mean(dh), 4),
            "delta_all": round(st.mean(dall), 4),
            "mean_conf": round(st.mean([c for c in (get_conf(m) for m in metas[name])
                                        if c is not None] or [float("nan")]), 4),
        }
        print(f"{name:12s} {fa*100:8.1f}% {ni*100:8.1f}% "
              f"{st.mean(de)*100:+12.2f}pp {st.mean(dh)*100:+12.2f}pp "
              f"{st.mean(dall)*100:+8.2f}pp")

    base = res["P0_current"]["false_alarm_rate_easy"]
    best = min(res, key=lambda k: res[k]["false_alarm_rate_easy"])
    print(f"\n[판정] 헛경보율 {base*100:.1f}% -> {res[best]['false_alarm_rate_easy']*100:.1f}% "
          f"({best})  ·  잘 푸는 문제 손해 "
          f"{res['P0_current']['delta_easy']*100:+.2f}pp -> {res[best]['delta_easy']*100:+.2f}pp")
    print("   ⇒ 손해가 지시문만으로 사라지면 이것은 보상 문제가 아니었다.")

    json.dump({"model": args.model, "k": args.k, "arms": res,
               "metas": metas, "acc": acc, "accN": accN},
              open(args.out, "w"))
    print(f"[4] wrote {args.out}")


if __name__ == "__main__":
    main()
