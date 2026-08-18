"""좋은 메타 vs 정형문 -- 같은 지점에서 갈라 실제로 다시 풀린다.

이 프로브가 답하는 질문은 프로그램 전체의 전제다:

    ***메타 '내용'이 결과를 바꾸는가?***

바꾸지 않는다면 R_meta 에는 과녁이 없고, 정의를 어떻게 고치든 의미가 없다.
바꾼다면 그 차이의 크기가 R_meta 가 가질 수 있는 신호의 상한이다.

설계 -- 접합 후 재생성(splice-and-regenerate). 대리 지표가 아니라 개입이다.

  1  풀이를 생성하고 답이 나오기 전에서 자른다  ->  prefix (모든 조건이 공유)
  2  그 prefix 에 대해 모델에게 자기점검 한 문장을 받는다  ->  R
  3  같은 prefix 에 메타만 다섯 가지로 갈아 끼우고 **각각 다시 풀게 한다**
  4  최종 답을 채점한다

     N  메타 없음                      -- 개입 없음
     B  정형문 (b4p2 가 수렴한 그 문장)  -- 문제 정보 0
     D  다른 문제의 진짜 메타           -- 진짜 문장이되 이 문제와 무관
     S  R 의 어절을 섞은 것             -- 어휘·길이 보존, 의미 파괴
     R  이 prefix 에 대한 진짜 자기점검

사전등록 예측 (돌리기 전에 적는다)

  H1  acc(R) − acc(N) > 0        메타가 도움이 되나
  H2  acc(R) − acc(B) > 0        ★내용이 중요한가 -- 헤드라인
  H3  acc(B) − acc(N) ≈ 0        아무거나 끼우면 오르는 게 아니다 (플라시보)
  H4  acc(R) − acc(S) > 0        어휘가 아니라 의미다
  H5  acc(D) ≈ acc(B)            남의 메타 = 정형문 (둘 다 이 문제와 무관)

  ⇒ H2 가 0 이면 R_meta 에 과녁이 없다. 그 경우 정의 교체도, 성분 축소도 무의미하고,
    보고할 것은 그 사실 자체다.

선례: PG0(2026-06-19)이 같은 구조로 R / N' / Nc 를 돌려 mean_gap +0.046,
easy +0.082 / hard −0.034, verify +0.020 > redirect +0.007 을 얻었다. 그때는 콜드스타트
SFT 모델이었고 채점 정정이 붙어 있다. 이 프로브는 그 사전확률을 현 모델에서 다시 잰다.

⚠ 저장된 is_correct 는 쓰지 않는다. 완성 텍스트를 gold 로 재채점한다(원장 규율).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics as st

BOILER = ("confidence: 0.88 | The answer looks plausible but should not be accepted "
          "without an independent check, since the result can be verified directly "
          "from the quantities given. | decision: verify")

SOLVE_SYS = ("Solve the competition math problem. Show your work step by step, "
             "then give the final answer as \\boxed{...}.")

# prefix 를 주고 '그 작업에 대한' 한 문장을 받는다. 형식 준수가 아니라 한 문장 쓰기라
# 수율이 높다 -- 프롬프트-only 메타가 막혔던 이유(순서·형식)를 우회한다.
CHECK_SYS = ("You are reviewing a partial solution to a math problem. Name the single "
             "quantity, step, or assumption in the work so far that is most likely to "
             "be wrong, in one sentence, referring to this problem's own numbers. Then "
             "give your confidence and a decision.\n\n"
             "Answer in exactly this form and nothing else:\n"
             "confidence: <0 to 1> | <your one sentence> | decision: verify\n"
             "  (use `decision: redirect` instead if the approach should be abandoned)")

_BOXED = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")


# ------------------------------------------------------------------- 채점

def _extract(text: str):
    m = _BOXED.findall(text)
    if m:
        return m[-1].strip()
    m = re.search(r"####\s*(.+?)(?:\n|$)", text)
    return m.group(1).strip() if m else None
    # last-number 폴백은 **쓰지 않는다** -- 0714 버그픽스가 그것을 버렸다.


def grade(text: str, gold: str) -> int:
    pred = _extract(text)
    if pred is None:
        return 0
    if pred.replace(" ", "") == str(gold).replace(" ", ""):
        return 1
    try:
        from math_verify import parse, verify
        return int(bool(verify(parse(f"${gold}$"), parse(f"${pred}$"))))
    except Exception:
        return 0


# ------------------------------------------------------------------- 접합

def splice(text: str, frac: float = 0.6) -> str | None:
    """답이 나오기 전에서 자른다. 이미 boxed 가 있으면 그 앞에서 자른다.

    줄바꿈 경계로 맞춰 문장 중간에서 끊기지 않게 한다 -- 중간에서 끊으면 이어쓰기가
    부자연스러워지고, 그 부자연스러움이 조건 전체에 공통으로 실려 신호를 덮는다.
    """
    cut = text.find("\\boxed{")
    limit = len(text) if cut < 0 else cut
    target = int(limit * frac)
    nl = text.rfind("\n", 0, target)
    end = nl if nl > 80 else target
    pre = text[:end].rstrip()
    return pre if len(pre) > 60 else None


def shuffle_words(s: str, rng: random.Random) -> str:
    """의미만 파괴하고 어휘·길이는 보존한다. confidence/decision 껍데기는 유지."""
    m = re.match(r"(confidence:\s*[0-9.]+\s*\|)(.*?)(\|\s*decision:\s*\w+)", s, re.S)
    if not m:
        w = s.split()
        rng.shuffle(w)
        return " ".join(w)
    mid = m.group(2).split()
    rng.shuffle(mid)
    return f"{m.group(1)} {' '.join(mid)} {m.group(3)}"


def boot_ci(pairs, n=10000, seed=0):
    """문제 단위 쌍대 부트스트랩 -- 같은 문제의 두 조건이 함께 뽑힌다."""
    rng = random.Random(seed)
    N = len(pairs)
    if N == 0:
        return (0.0, 0.0, 0.0)
    obs = st.mean(a - b for a, b in pairs)
    ds = []
    for _ in range(n):
        s = [pairs[rng.randrange(N)] for _ in range(N)]
        ds.append(st.mean(a - b for a, b in s))
    ds.sort()
    return obs, ds[int(.025 * n)], ds[int(.975 * n)]


# ------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--problems", default="problems_math500.json")
    ap.add_argument("--n_problems", type=int, default=250)
    ap.add_argument("--k", type=int, default=6, help="조건당 재생성 표본")
    ap.add_argument("--seed", type=int, default=0)
    # 이 A100 은 공유된다. 0816 실측: 다른 세션의 잡이 14.7GB 를 쥐고 있어서
    # gpu_memory_utilization=0.85 (68GB) 가 초기화에 실패했다. 기본값을 낮게 두고,
    # 남의 잡을 죽이는 대신 우리가 비켜간다.
    ap.add_argument("--gpu_util", type=float, default=0.55)
    ap.add_argument("--out", default="splice.json")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    rng = random.Random(args.seed)

    pool = json.load(open(args.problems))[:args.n_problems]
    llm = LLM(model=args.model, dtype="bfloat16", seed=args.seed,
              gpu_memory_utilization=args.gpu_util, max_model_len=4096)
    tok = llm.get_tokenizer()

    def chat(sys, user):
        msgs = [{"role": "system", "content": sys}, {"role": "user", "content": user}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True)

    # 1 ─ prefix
    p1 = [chat(SOLVE_SYS, r["problem"]) for r in pool]
    o1 = llm.generate(p1, SamplingParams(n=1, temperature=0.7, top_p=0.95,
                                         max_tokens=900, seed=args.seed))
    items = []
    for r, o in zip(pool, o1):
        pre = splice(o.outputs[0].text)
        if pre:
            items.append({"problem": r["problem"], "gold": str(r["gold"]),
                          "prompt": o.prompt, "prefix": pre})
    print(f"[1] prefix {len(items)}/{len(pool)}")

    # 2 ─ 진짜 자기점검 R
    p2 = [chat(CHECK_SYS, f"Problem:\n{it['problem']}\n\nWork so far:\n{it['prefix']}")
          for it in items]
    o2 = llm.generate(p2, SamplingParams(n=1, temperature=0.7, top_p=0.95,
                                         max_tokens=160, seed=args.seed))
    keep = []
    for it, o in zip(items, o2):
        t = " ".join(o.outputs[0].text.split())
        if re.search(r"confidence:\s*[0-9.]+", t) and re.search(r"decision:\s*\w+", t):
            it["R"] = t[:400]
            keep.append(it)
    items = keep
    print(f"[2] self-check {len(items)}")
    if len(items) < 30:
        raise SystemExit("자기점검 수율이 너무 낮다 -- 프롬프트부터 고친다")

    # 3 ─ 다섯 조건, 같은 prefix, 메타만 교체하고 재생성
    for i, it in enumerate(items):
        j = (i + 1 + rng.randrange(max(1, len(items) - 1))) % len(items)
        it["D"] = items[j]["R"]                       # 다른 문제의 진짜 메타
        it["S"] = shuffle_words(it["R"], rng)         # 의미만 파괴
        it["B"] = BOILER
        it["N"] = None

    CONDS = ["N", "B", "D", "S", "R"]
    reqs, index = [], []
    for ci, c in enumerate(CONDS):
        for k, it in enumerate(items):
            meta = it[c]
            block = "" if meta is None else f"\n<meta>{meta}</meta>"
            # prompt 는 1단계와 동일한 chat 껍데기 -> 조건 간 유일한 차이는 메타 슬롯
            reqs.append(it["prompt"] + it["prefix"] + block + "\n")
            index.append((c, k))
    print(f"[3] 재생성 요청 {len(reqs)} = {len(CONDS)}조건 x {len(items)}문제")

    o3 = llm.generate(reqs, SamplingParams(n=args.k, temperature=0.7, top_p=0.95,
                                           max_tokens=700, seed=args.seed))

    acc = {c: [0.0] * len(items) for c in CONDS}
    for (c, k), o in zip(index, o3):
        g = items[k]["gold"]
        acc[c][k] = st.mean(grade(x.text, g) for x in o.outputs)
        # R 조건의 이어쓰기 한 벌을 남긴다 -- sweep_rmeta.py 가 "메타 뒤 본문"으로 쓴다.
        # (이걸 안 남기면 공식 sweep 을 위해 생성을 통째로 다시 해야 한다)
        if c == "R":
            items[k]["cont_R"] = o.outputs[0].text

    # 4 ─ 판정
    res = {"model": args.model, "n_problems": len(items), "k": args.k,
           "mean_acc": {c: st.mean(acc[c]) for c in CONDS}, "contrasts": {}}
    for a, b in [("R", "N"), ("R", "B"), ("B", "N"), ("R", "S"), ("D", "B"), ("R", "D")]:
        obs, lo, hi = boot_ci(list(zip(acc[a], acc[b])))
        res["contrasts"][f"{a}-{b}"] = {"delta": obs, "ci": [lo, hi],
                                        "excludes_zero": lo > 0 or hi < 0}
    res["HEADLINE_R_minus_B"] = res["contrasts"]["R-B"]
    json.dump({**res, "per_problem": {c: acc[c] for c in CONDS},
               "items": items},
              open(args.out, "w"))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
