"""★SFT 없이 프롬프트만으로 모델이 메타를 **인라인**으로 내는가.

사용자 지적: "SFT 없이 진행해도 되지 않아? 그냥 프롬프트만 주고? 지금 그렇게 테스트한 거 아니야?"

⚠**아니다 — 지금까지의 프로브는 다르다.**
    프로브     ① 푼다 (메타 없음)  ② **별도 턴**에서 CHECK_SYS 로 자기점검을 받는다  ③ 주입한다
    RL 이 필요한 것  ★**한 번의 생성 안에서** 모델이 스스로 `<meta>...</meta>` 를 낸다

그 둘은 다르다. 프로브의 자기점검 수율 100% 는 **별도 턴**의 수율이지 인라인 발화율이 아니다.
그리고 기록에 불길한 실측이 있다: "Qwen3-4B 프롬프트만 redirect 0.7%(2/300)".

이 파일이 답하는 것 하나: ***프롬프트만으로 인라인 메타 발화율이 얼마인가.***
  높으면(≥0.6)  →  SFT 불필요. 오늘 바로 RL 발사.
  낮으면        →  SFT 1.5h 가 필요하거나, 2턴 구조로 가야 한다.

세 프롬프트를 겨룬다 (형식 강제 강도가 다르다).
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st

from countdown import gen_instance, grade, user_msg

BASE = ("You are playing the Countdown numbers game. You are given a list of numbers "
        "and a target. Use EACH given number EXACTLY ONCE with + - * / and parentheses "
        "to make the target. Every intermediate value must be a positive integer.\n\n"
        "Work through AT MOST 8 candidate groupings, backtracking when one fails. "
        "Keep each attempt to one short line.\n")

END = ("\nYour response MUST end with the final arithmetic expression in \\boxed{...} "
       "— the expression itself, not its value. Example: \\boxed{(3+7)*8-25}\n"
       "If you cannot find an exact solution, still end with \\boxed{...} containing "
       "your closest expression that uses each number once.")

# P0 — 메타 없음 (기준: 발화율 0 이어야 정상)
P0 = BASE + END

# P1 — 약한 요구: 한 번 자기점검하라
P1 = (BASE + "\nAfter your third attempt, pause once and write a one-line self-check "
      "inside <meta>...</meta> naming the grouping most likely to be blocking you.\n" + END)

# P2 — 형식 명시
P2 = (BASE + "\nAfter your third attempt, pause once and write EXACTLY this line:\n"
      "<meta>confidence: <0 to 1> | <one sentence about which grouping is blocking you> "
      "| decision: verify</meta>\n"
      "Use `decision: redirect` instead if the current branch should be abandoned. "
      "Then continue searching.\n" + END)

# P3 — 형식 명시 + 예시 (few-shot 한 줄)
P3 = (BASE + "\nAfter your third attempt, pause once and write EXACTLY one line of this "
      "form:\n<meta>confidence: 0.6 | The pairing 22*8=176 overshoots and leaves no way "
      "back. | decision: redirect</meta>\n"
      "Fill it with THIS puzzle's own numbers. Then continue searching.\n" + END)

PROMPTS = {"P0_none": P0, "P1_weak": P1, "P2_format": P2, "P3_fewshot": P3}

_META = re.compile(r"<meta>(.*?)</meta>", re.S)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--n_nums", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_tokens", type=int, default=3072)
    ap.add_argument("--gpu_util", type=float, default=0.55)
    ap.add_argument("--out", default="emit_test.json")
    args = ap.parse_args()

    import random
    from vllm import LLM, SamplingParams
    rng = random.Random(args.seed)
    probs = [gen_instance(args.n_nums, rng) for _ in range(args.n)]
    probs = [p for p in probs if p]

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

    res = {}
    print(f"{'프롬프트':12s} {'발화율':>7s} {'형식준수':>9s} {'정확도':>7s} {'토큰':>6s}   "
          f"(발화율 = <meta> 블록이 한 번 이상)")
    for name, sm in PROMPTS.items():
        outs = llm.generate([chat(sm, user_msg(p)) for p in probs],
                            SamplingParams(n=args.k, temperature=0.7, top_p=0.95,
                                           max_tokens=args.max_tokens, seed=args.seed))
        emit, wf, gr, ntok, samples = [], [], [], [], []
        for p, o in zip(probs, outs):
            for x in o.outputs:
                m = _META.findall(x.text)
                emit.append(1 if m else 0)
                wf.append(1 if (m and re.search(r"confidence:\s*[0-9.]+", m[0])
                                and "decision:" in m[0]) else 0)
                gr.append(grade(x.text, p["nums"], p["target"]))
                ntok.append(len(x.token_ids))
            if len(samples) < 2 and _META.findall(o.outputs[0].text):
                samples.append(_META.findall(o.outputs[0].text)[0][:150])
        res[name] = {"emit": st.mean(emit), "wellformed": st.mean(wf),
                     "acc": st.mean(gr), "ntok": st.mean(ntok), "samples": samples}
        r = res[name]
        print(f"{name:12s} {r['emit']*100:6.1f}% {r['wellformed']*100:8.1f}% "
              f"{r['acc']:7.4f} {r['ntok']:6.0f}")
        for s in samples[:1]:
            print(f"             예: {' '.join(s.split())[:100]}")

    best = max((k for k in res if k != "P0_none"), key=lambda k: res[k]["wellformed"])
    print(f"\n[판정] 최고 = {best} · 형식준수 발화율 {res[best]['wellformed']*100:.1f}%")
    print(f"  {'✅ SFT 불필요 — 프롬프트만으로 RL 발사 가능' if res[best]['wellformed'] >= 0.60 else '⛔ SFT 또는 2턴 구조가 필요하다'}")
    print(f"  기준선 P0(메타 요구 없음) 발화율 {res['P0_none']['emit']*100:.1f}% "
          f"(0 에 가까워야 정상)")
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"[1] wrote {args.out}")


if __name__ == "__main__":
    main()
