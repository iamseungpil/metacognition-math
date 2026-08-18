"""조타 프로브 A단계 (vLLM) — 본문 고정, 메타만 여덟 갈래, 각각 실제로 다시 풀린다.

앞선 접합 프로브의 결함 둘을 고친다.

  결함 1 · 모집단.  250 문제 중 163 개(65%)를 모델이 이미 6/6 맞혔다. 거기서 메타는
    정의상 도울 수 없다. 전체 평균 −3.1pp 는 그 층이 끌어내린 값이다.
    → 여기서는 **prefix 안에서** 메타 여덟을 비교하므로, 정답률이 갈리지 않는 prefix 는
      Spearman 이 정의되지 않아 자동으로 빠진다. 모집단 선택이 설계에 내장된다.

  결함 2 · 비교 대상.  학습 중 그룹 8 개는 **각자 다른 prefix** 를 갖는다. 90% 푼 롤아웃과
    막힌 롤아웃의 shift 는 애초에 비교 대상이 아니다.
    → **prefix 를 고정**하고 메타만 바꾼다. matched 비교다.

묻는 것: ★**prefix 안에서, 어떤 점수가 "실제로 도움이 된 메타"를 위로 올리는가.**
그 점수가 있으면 R_meta 로 쓸 자격이 생긴다. 하나도 못 올리면, 정형문을 고른 이유가 설명된다.

이 파일은 생성만 한다(vLLM). 점수 계산은 guidance_score.py 가 이어받는다 --
두 프레임워크가 동시에 GPU 를 쥐지 않도록 프로세스를 나눈다.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st

from splice_probe import CHECK_SYS, SOLVE_SYS, grade


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splice", default="splice_8b.json")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--n_prefix", type=int, default=160)
    ap.add_argument("--K", type=int, default=8, help="prefix 당 메타 후보 수")
    ap.add_argument("--k", type=int, default=4, help="메타당 재생성 표본")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prefilter", action="store_true",
                    help="독립 표본으로 0<acc_N<1 인 위험 prefix 만 남긴다")
    ap.add_argument("--k_filter", type=int, default=8, help="선별용 표본 (채점용과 분리)")
    ap.add_argument("--gpu_util", type=float, default=0.55)
    ap.add_argument("--out", default="guidance_gen.json")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    items = json.load(open(args.splice))["items"][:args.n_prefix]
    llm = LLM(model=args.model, dtype="bfloat16", seed=args.seed,
              gpu_memory_utilization=args.gpu_util, max_model_len=4096)
    tok = llm.get_tokenizer()

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

    # 0 ─ ★위험 prefix 만 남긴다. 1차 실측: 160 중 정답률이 갈리는 곳이 69, 채점까지 되는
    #     곳이 30 뿐이었다. 나머지는 모델이 무엇을 넣든 같은 답을 낸다 -- 거기서 메타를
    #     비교하는 것은 해상도가 0 인 자로 재는 일이다.
    #     ⚠선별용 표본과 채점용 표본을 **분리**한다. 같은 표본으로 고르고 채점하면
    #     평균회귀만으로 처치가 좋아 보인다(EXP-0816h-A 에서 실제로 그 실수를 했다).
    if args.prefilter:
        p0 = [it["prompt"] + it["prefix"] + "\n" for it in items]
        o0 = llm.generate(p0, SamplingParams(n=args.k_filter, temperature=0.7, top_p=0.95,
                                             max_tokens=700, seed=args.seed + 101))
        for it, o in zip(items, o0):
            it["acc_N_filter"] = st.mean(grade(x.text, it["gold"]) for x in o.outputs)
        before = len(items)
        items = [it for it in items if 0.001 < it["acc_N_filter"] < 0.999]
        print(f"[0] 위험 prefix 선별(독립 표본 k={args.k_filter}): "
              f"{len(items)}/{before} 유지 — 0<acc_N<1")

    # 1 ─ prefix 당 메타 후보 K 개. temp 1.0 으로 뽑아 **다양성을 확보**한다 --
    #     후보가 서로 같으면 "어느 메타가 나은가"라는 질문 자체가 성립하지 않는다.
    p1 = [chat(CHECK_SYS, f"Problem:\n{it['problem']}\n\nWork so far:\n{it['prefix']}")
          for it in items]
    o1 = llm.generate(p1, SamplingParams(n=args.K, temperature=1.0, top_p=1.0,
                                         max_tokens=160, seed=args.seed))
    for it, o in zip(items, o1):
        cands = []
        for c in o.outputs:
            t = " ".join(c.text.split())
            if re.search(r"confidence:\s*[0-9.]+", t) and re.search(r"decision:\s*\w+", t):
                cands.append(t[:400])
        it["metas"] = cands
    items = [it for it in items if len(it["metas"]) >= 4]
    n_meta = sum(len(it["metas"]) for it in items)
    print(f"[1] prefix {len(items)} · 메타 후보 {n_meta} "
          f"(prefix 당 평균 {n_meta / max(1, len(items)):.1f})")

    # 2 ─ 각 후보로 이어풀기 + 메타 없음(N) 도 같은 prefix 에서
    reqs, idx = [], []
    for i, it in enumerate(items):
        reqs.append(it["prompt"] + it["prefix"] + "\n")
        idx.append((i, -1))                       # -1 = 메타 없음
        for j, m in enumerate(it["metas"]):
            reqs.append(it["prompt"] + it["prefix"] + f"\n<meta>{m}</meta>\n")
            idx.append((i, j))
    print(f"[2] 재생성 요청 {len(reqs)} × k={args.k}")
    o2 = llm.generate(reqs, SamplingParams(n=args.k, temperature=0.7, top_p=0.95,
                                           max_tokens=700, seed=args.seed))

    for it in items:
        it["acc"] = [0.0] * len(it["metas"])
        it["cont"] = [""] * len(it["metas"])
    for (i, j), o in zip(idx, o2):
        it = items[i]
        a = st.mean(grade(x.text, it["gold"]) for x in o.outputs)
        if j < 0:
            it["acc_N"] = a
            it["cont_N"] = o.outputs[0].text
        else:
            it["acc"][j] = a
            it["cont"][j] = o.outputs[0].text   # 점수 단계가 "메타 뒤 본문"으로 쓴다

    # prefix 안에서 정답률이 갈리는 곳이 실제로 있나 -- 없으면 이 실험은 해상도가 없다
    spread = [max(it["acc"]) - min(it["acc"]) for it in items]
    live = sum(1 for s in spread if s > 1e-9)
    print(f"[3] prefix 내 정답률 퍼짐: 살아있는 prefix {live}/{len(items)} · "
          f"평균 퍼짐 {st.mean(spread):.4f} · 중앙 {st.median(spread):.4f}")
    print(f"    acc_N 평균 {st.mean(it['acc_N'] for it in items):.4f} · "
          f"메타 평균 {st.mean(st.mean(it['acc']) for it in items):.4f} · "
          f"prefix 내 최고 메타 평균 {st.mean(max(it['acc']) for it in items):.4f}")

    json.dump({"model": args.model, "K": args.K, "k": args.k,
           "prefilter": args.prefilter, "items": items},
              open(args.out, "w"))
    print(f"[4] wrote {args.out}")


if __name__ == "__main__":
    main()
