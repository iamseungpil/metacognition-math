"""접합 프로브 v2 — **절단을 통제하고 외생 층으로 판정한다.**

왜 v2 인가. v1(`splice_probe.py` → `splice_4b.json`)의 판정이 0817 적대검토에서 무너졌다.

  ⛔**층화 오염.** 층을 `acc_N`(대조군 자신의 점수)으로 정의했다. 그 층을 고르면 회귀 효과만으로
    모든 팔이 오른다 — 실측: `acc_N==0` 층에서 R −N +5.15pp 인데 **어절섞기 S +4.55pp,
    정형문 B +3.64pp, 남의것 D +2.42pp**. 같은 층 `R−B = +1.52pp [−3.03,+6.67]` 로 0 을 품는다.
    ⇒ **층은 외생 기준(사람이 매긴 MATH level)으로만 자른다.**

  ⛔**절단 교란.** `cont_R` 의 no-boxed 비율이 L5 **48.6%** · L1 3.7% 이고 `grade()` 는 boxed 가
    없으면 무조건 0 이다. 진짜 자기점검은 의심을 주입해 재유도를 부르고, 그러면 이어쓰기가
    길어져 캡(700)에 더 자주 걸린다 — 정형문은 의심을 안 주니 짧다. 그래서 외생 층에서 나온
    `R−B | L5 = −5.56pp [−10.19,−1.62]` 가 **"메타가 답을 망친다"인지 "메타가 절단을 늘린다"인지
    분리 불가능**했다. v1 은 R 팔의 이어쓰기만 저장해서 팔별 절단률조차 못 센다.

v2 의 수정 넷.

  ①  캡 상향 700 → 2048 (prefix 단계도 900 → 2048, max_model_len 4096 → 6144).
      "KV 가 모자란다"는 제약은 없다 — `splice_4b.log` 가 available KV 38.55 GiB /
      280,688 tokens 를 찍는다. 6144 에서도 동시성 45x 가 남는다.
  ②  **팔별로** 이어쓰기·`finish_reason`·`has_boxed`·토큰수를 전부 저장한다.
      이 한 칸이 있어야 (i) 팔별 절단률 (ii) 절단 행을 뺀 재집계 (iii) 재채점이
      **GPU 없이** 가능하다. v1 이 이걸 안 남겨서 오늘 판정이 막혔다.
  ③  채점기를 **균형괄호 스캔**으로 교체. 현행 정규식은 2단 중첩을 못 읽어 300건 중 3건을
      구조적으로 0점 처리했고 **셋 다 하드 층**이었다. 래핑도 정본 `robust_grade` 와 맞춘다
      (`$...$` → `\\boxed{...}`).
  ④  팔 구성 변경. **`B'` 신설**(B 와 텍스트 동일·시드만 다름) = 순수 디코딩 잡음 바닥.
      **`E` 신설**(빈 메타 블록) = 태그 삽입만의 효과. **`D` 폐기**(이미 답이 나왔다:
      `D−B −2.06pp [−3.39,−0.72]`). 5팔 → 6팔.

═══════════════════════════════════════════════════════════════════════════
사전등록 (실행 전에 적는다 — 결과를 보고 고치지 않는다)
═══════════════════════════════════════════════════════════════════════════

모집단   MATH-500 전체 500문제. 층은 **사람이 매긴 level** 로만 자른다 (외생).
         `levels_math500.json` 로 조인 (v1 데이터에서 300/300 일치 확인).

주 지표  **acc(R) − acc(B) | level ∈ {4,5}**  ... 단 하나.
         정형문 B 가 대조군인 이유: N 대비는 "무엇이든 끼우면 생기는 변화"를 못 뺀다.
         B 는 길이·형식·어조가 R 과 같고 문제 정보만 0 이다.

잡음 바닥 **δ_eq = 2 × (acc(B') − acc(B) 의 CI 반폭)**.
         |주 지표| < δ_eq 이면 "차이 없음"이 아니라 **"해상도 아래"** 로 적는다.
         (선례 C-031 이 밴드를 선언 안 해서 "not detected, and not zero" 로 남았다.)

판정 밴드
         주 지표 CI 가 0 을 배제 & 음수  →  ★자기점검의 내용이 **해롭다**
         주 지표 CI 가 0 을 배제 & 양수  →  ★내용이 **이롭다**
         CI 가 0 을 품고 |δ| ≥ δ_eq     →   방향 미정, 표본 부족
         CI 가 0 을 품고 |δ| < δ_eq     →   해상도 아래

절단 통제 (주 지표와 **같이** 보고한다 — 하나만 보면 오독한다)
         (a) 팔별 `finish_reason == 'length'` 비율. 팔 간 차이가 2pp 를 넘으면
             주 지표의 부호를 **절단으로 설명 가능**하다고 명시한다.
         (b) **모든 팔의 모든 표본이 종료한 문제**만으로 주 지표를 재집계.
             둘의 부호가 다르면 결론은 "절단 아티팩트"다.

보조 (Holm 보정, 주 지표를 대체할 수 없다)
         acc(R)−acc(S) · acc(B)−acc(E) · acc(E)−acc(N) · level 대역별 R−B

중단     자기점검 수율(형식 통과) < 60% 이면 프롬프트부터 고치고 판정하지 않는다.

⚠EXP 트랙. 헤드라인 금지. 판정문은 codex-sol 게이트 후 CLAIMS.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics as st

from splice_probe import BOILER, SOLVE_SYS, CHECK_SYS, splice, shuffle_words, boot_ci


# ────────────────────────────────────────────────── ③ 채점기 (균형괄호)

def extract_boxed(text: str):
    """마지막 `\\boxed{...}` 의 내용. **중첩 깊이 무제한** — 정규식판은 2단에서 실패한다."""
    out, i = [], 0
    while True:
        j = text.find("\\boxed{", i)
        if j < 0:
            break
        k, dep = j + 7, 1
        while k < len(text) and dep:
            dep += (text[k] == "{") - (text[k] == "}")
            k += 1
        if dep == 0:
            out.append(text[j + 7:k - 1])
        i = j + 7
    if out:
        return out[-1].strip()
    m = re.search(r"####\s*(.+?)(?:\n|$)", text)
    return m.group(1).strip() if m else None


def grade2(text: str, gold: str) -> int:
    pred = extract_boxed(text)
    if pred is None:
        return 0
    if pred.replace(" ", "") == str(gold).replace(" ", ""):
        return 1
    try:
        from math_verify import parse, verify
        return int(bool(verify(parse(f"\\boxed{{{gold}}}"),
                               parse(f"\\boxed{{{pred}}}"))))
    except Exception:
        return 0


# ────────────────────────────────────────────────── level 조인 (외생 층)

def load_levels(path: str):
    raw = json.load(open(path))
    rows = raw if isinstance(raw, list) else raw.get("data", [])
    norm = lambda s: " ".join(str(s).split())

    def lvl(v):
        if isinstance(v, int):
            return v
        m = re.search(r"(\d)", str(v))
        return int(m.group(1)) if m else None
    return {norm(r["problem"]): lvl(r.get("level")) for r in rows}


# ────────────────────────────────────────────────── main

CONDS = ["N", "E", "B", "Bp", "S", "R"]
COND_DOC = {
    "N": "메타 없음",
    "E": "빈 메타 블록 (태그만)",
    "B": "정형문 (문제 정보 0)",
    "Bp": "정형문 B' (텍스트 동일·시드만 다름 = 잡음 바닥)",
    "S": "R 의 어절 섞기 (어휘·길이 보존, 의미 파괴)",
    "R": "진짜 자기점검",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--problems", default="problems_math500.json")
    ap.add_argument("--levels", default="levels_math500.json")
    ap.add_argument("--n_problems", type=int, default=500)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_tokens", type=int, default=2048)   # ① was 700
    ap.add_argument("--max_model_len", type=int, default=6144)
    ap.add_argument("--gpu_util", type=float, default=0.55)   # 공유 A100 — 비켜간다
    ap.add_argument("--out", default="splice2_4b.json")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    rng = random.Random(args.seed)

    pool = json.load(open(args.problems))[:args.n_problems]
    lmap = load_levels(args.levels)
    norm = lambda s: " ".join(str(s).split())
    hit = sum(1 for r in pool if norm(r["problem"]) in lmap)
    print(f"[0] level 조인 {hit}/{len(pool)}  "
          f"(외생 층 — acc_N 으로 자르지 않는다)")
    if hit < len(pool) * 0.95:
        raise SystemExit("level 조인 실패율이 높다 — 층화를 못 하면 판정도 못 한다")

    llm = LLM(model=args.model, dtype="bfloat16", seed=args.seed,
              gpu_memory_utilization=args.gpu_util,
              max_model_len=args.max_model_len)
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

    # 1 ─ prefix
    o1 = llm.generate([chat(SOLVE_SYS, r["problem"]) for r in pool],
                      SamplingParams(n=1, temperature=0.7, top_p=0.95,
                                     max_tokens=args.max_tokens, seed=args.seed))
    items, dropped = [], {}
    for r, o in zip(pool, o1):
        lv = lmap.get(norm(r["problem"]))
        pre = splice(o.outputs[0].text)
        if pre:
            items.append({"problem": r["problem"], "gold": str(r["gold"]),
                          "level": lv, "prompt": o.prompt, "prefix": pre})
        else:
            dropped[lv] = dropped.get(lv, 0) + 1
    print(f"[1] prefix {len(items)}/{len(pool)}  층별 폐기 {dict(sorted(dropped.items(), key=str))}")

    # 2 ─ 진짜 자기점검 R
    o2 = llm.generate(
        [chat(CHECK_SYS, f"Problem:\n{it['problem']}\n\nWork so far:\n{it['prefix']}")
         for it in items],
        SamplingParams(n=1, temperature=0.7, top_p=0.95, max_tokens=160, seed=args.seed))
    keep, drop2 = [], {}
    for it, o in zip(items, o2):
        t = " ".join(o.outputs[0].text.split())
        if re.search(r"confidence:\s*[0-9.]+", t) and re.search(r"decision:\s*\w+", t):
            it["R"] = t[:400]
            keep.append(it)
        else:
            drop2[it["level"]] = drop2.get(it["level"], 0) + 1
    yield_rate = len(keep) / max(1, len(items))
    print(f"[2] self-check {len(keep)}/{len(items)} = {yield_rate:.1%}  "
          f"층별 폐기 {dict(sorted(drop2.items(), key=str))}")
    if yield_rate < 0.60:                       # 사전등록 중단 조건
        raise SystemExit("자기점검 수율 < 60% — 프롬프트부터 고친다 (판정하지 않는다)")
    items = keep

    # 3 ─ ④ 여섯 팔
    for it in items:
        it["E"] = ""
        it["B"] = BOILER
        it["Bp"] = BOILER                       # 텍스트 동일 — 시드만 다르게 요청한다
        it["S"] = shuffle_words(it["R"], rng)
        it["N"] = None

    reqs, index = [], []
    for c in CONDS:
        for k, it in enumerate(items):
            meta = it[c]
            block = "" if meta is None else f"\n<meta>{meta}</meta>"
            reqs.append(it["prompt"] + it["prefix"] + block + "\n")
            index.append((c, k))
    print(f"[3] 재생성 {len(reqs)} = {len(CONDS)}팔 x {len(items)}문제 x k={args.k} "
          f"(cap {args.max_tokens})")

    # B' 만 다른 시드로 — 같은 문자열에서 순수 디코딩 잡음만 뽑는다
    is_bp = [c == "Bp" for c, _ in index]
    o3 = [None] * len(reqs)
    for flag, sd in ((False, args.seed), (True, args.seed + 977)):
        sel = [i for i, f in enumerate(is_bp) if f == flag]
        out = llm.generate([reqs[i] for i in sel],
                           SamplingParams(n=args.k, temperature=0.7, top_p=0.95,
                                          max_tokens=args.max_tokens, seed=sd))
        for i, o in zip(sel, out):
            o3[i] = o

    # ② 팔별 전량 기록 — 이게 있어야 나중에 GPU 없이 재판정할 수 있다
    acc = {c: [0.0] * len(items) for c in CONDS}
    recs = {c: [None] * len(items) for c in CONDS}
    for (c, k), o in zip(index, o3):
        g = items[k]["gold"]
        per = []
        for si, x in enumerate(o.outputs):
            pred = extract_boxed(x.text)
            per.append({"graded": grade2(x.text, g), "pred": pred,
                        "has_boxed": pred is not None,
                        "finish": x.finish_reason, "ntok": len(x.token_ids),
                        "text": x.text if si == 0 else None})
        acc[c][k] = st.mean(r["graded"] for r in per)
        recs[c][k] = per
        if c == "R":
            items[k]["cont_R"] = o.outputs[0].text     # 하위 프로브 호환

    # 4 ─ 판정
    lv_of = [it["level"] for it in items]
    HARD = [i for i, L in enumerate(lv_of) if L in (4, 5)]
    EASY = [i for i, L in enumerate(lv_of) if L in (1, 2, 3)]

    def contrast(a, b, idx):
        return boot_ci([(acc[a][i], acc[b][i]) for i in idx])

    def trunc_rate(c, idx):
        n = sum(len(recs[c][i]) for i in idx)
        t = sum(1 for i in idx for r in recs[c][i] if r["finish"] == "length")
        return t / max(1, n)

    print("\n" + "=" * 68)
    print(f"[4] 팔별 평균 정확도 (n={len(items)})")
    for c in CONDS:
        print(f"    {c:3s} {st.mean(acc[c]):.4f}   절단 {trunc_rate(c, range(len(items)))*100:5.2f}%"
              f"   {COND_DOC[c]}")

    # 잡음 바닥
    o_bp, lo_bp, hi_bp = contrast("Bp", "B", range(len(items)))
    delta_eq = (hi_bp - lo_bp)                      # 2 x 반폭
    print(f"\n[δ_eq] B'−B = {o_bp*100:+.2f}pp [{lo_bp*100:+.2f},{hi_bp*100:+.2f}]"
          f"  ⇒ δ_eq = {delta_eq*100:.2f}pp")

    # ★주 지표
    o, lo, hi = contrast("R", "B", HARD)
    excl = lo > 0 or hi < 0
    band = ("★해롭다" if (excl and o < 0) else "★이롭다" if excl else
            ("방향 미정 (표본 부족)" if abs(o) >= delta_eq else "해상도 아래"))
    print(f"\n★주 지표  acc(R)−acc(B) | L4+L5 (n={len(HARD)})"
          f"  = {o*100:+.2f}pp [{lo*100:+.2f},{hi*100:+.2f}]   ⇒ {band}")

    # 절단 통제 (a)
    print("\n[절단 (a)] 팔별 length-종료 비율, L4+L5")
    for c in CONDS:
        print(f"    {c:3s} {trunc_rate(c, HARD)*100:6.2f}%")
    gap = abs(trunc_rate("R", HARD) - trunc_rate("B", HARD))
    print(f"    R vs B 차이 {gap*100:.2f}pp  ⇒ "
          f"{'⚠주 지표를 절단으로 설명 가능' if gap > 0.02 else '절단으로 설명 불가'}")

    # 절단 통제 (b) — 모든 팔의 모든 표본이 종료한 문제만
    clean = [i for i in HARD
             if all(r["finish"] != "length" for c in CONDS for r in recs[c][i])]
    if len(clean) >= 20:
        oc, loc, hic = contrast("R", "B", clean)
        agree = (oc < 0) == (o < 0)
        print(f"\n[절단 (b)] 전 팔 완주 문제만 (n={len(clean)}/{len(HARD)})"
              f"  R−B = {oc*100:+.2f}pp [{loc*100:+.2f},{hic*100:+.2f}]"
              f"   ⇒ {'부호 일치 — 절단 아티팩트 아님' if agree else '★부호 불일치 — 절단 아티팩트'}")
    else:
        print(f"\n[절단 (b)] 완주 문제 {len(clean)}개 — 재집계 불가")

    # 보조
    print("\n[보조] (Holm 보정 대상, 주 지표를 대체하지 않는다)")
    for a, b in (("R", "S"), ("B", "E"), ("E", "N"), ("R", "N")):
        ob, lb, hb = contrast(a, b, HARD)
        print(f"    {a}−{b} | L4+L5  {ob*100:+6.2f}pp [{lb*100:+6.2f},{hb*100:+6.2f}]")
    print("\n[대역별] R−B")
    for lab, idx in (("L1+L2+L3", EASY), ("L4+L5", HARD)):
        for L in ([None] if lab == "L1+L2+L3" else [None, 4, 5]):
            sel = idx if L is None else [i for i in idx if lv_of[i] == L]
            if len(sel) < 15:
                continue
            ob, lb, hb = contrast("R", "B", sel)
            nm = lab if L is None else f"L{L}"
            print(f"    {nm:9s} n={len(sel):3d}  {ob*100:+6.2f}pp "
                  f"[{lb*100:+6.2f},{hb*100:+6.2f}]")

    json.dump({"model": args.model, "n_problems": len(items), "k": args.k,
               "max_tokens": args.max_tokens, "conds": CONDS,
               "mean_acc": {c: st.mean(acc[c]) for c in CONDS},
               "delta_eq": delta_eq,
               "primary": {"name": "acc(R)-acc(B) | L4+L5", "n": len(HARD),
                           "delta": o, "ci": [lo, hi], "band": band},
               "per_problem": acc, "per_sample": recs,
               "levels": lv_of, "items": items},
              open(args.out, "w"))
    print(f"\n[5] wrote {args.out}")


if __name__ == "__main__":
    main()
