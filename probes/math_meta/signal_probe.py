"""신호 검정 — **정답을 안 쓰는 신호**가 정답을 쓰는 신호를 이기는가.

0817 진단. `logp(gold) − logp(rule_decoy)` 가 포화됐다:

    pmi_open 중앙값   6/6 맞히는 문제 +6.52   6/6 틀리는 문제 +6.38
    AUC(맞힘>틀림) = 0.539                    ← 난이도를 거의 못 안다

원인 후보 둘. (a) 티처포싱 ≠ 생성 도달가능성. (b) ★**decoy 가 경쟁자가 아니다** --
`_rule_based_decoy` 는 gold 를 규칙으로 비튼 문자열이라 모델이 실제로 고려한 적이 없다.
그래서 `logp(decoy)` 는 항상 바닥이고 차이가 +6 에 고정된다.

진짜 경쟁자는 **모델 자신이 낸 다른 답**이다. 이 파일은 그 교체를 잰다:

    PMI_gold(ctx) = logp(gold | ctx)  − logp(rule_decoy | ctx)      현행
    PMI_self(ctx) = logp(a1   | ctx)  − logp(a2        | ctx)      ★제안 — gold 를 안 쓴다
        a1, a2 = 이 문제에서 모델이 실제로 낸 서로 다른 두 답 (빈도 1·2위)

그리고 같은 판에서 나머지 세 축의 수리도 같이 잰다 (0817 격자에서 방향이 나왔다):

    측정위치   메타 경계(P1)  vs  답 직전(P2)          격자: 0.464 -> 0.783 (n=5, 미검정)
    대조       위치(앞/뒤)    vs  문맥(R메타 / B정형문)  격자: 전 칸에서 문맥이 나음
    후속본문   drive = logp(내 본문|내 메타) − logp(같은 본문|정형문)   gold 를 안 쓴다

═══════════════════════════════════════════════════════════════════════
검정 넷 (사전등록 — 실행 전에 적는다)
═══════════════════════════════════════════════════════════════════════

T1  난이도 판별 :  AUC(6/6 맞힘 > 6/6 틀림) at 메타 없는 문맥
      기준선  PMI_gold = 0.539 (실측)  ·  자기일관성 = 0.907 (실측, 36샘플)
      ★판정  PMI_self 가 0.65 를 넘으면 수리 성공

T2  메타 가치 예측 :  문제별 (acc_R − acc_B) 와의 Spearman
      ⚠측정 신뢰도 상한 : 초과분산/전체분산 = 0.309  ->  달성 가능한 최대 rho = 0.556
      기준선  acc_N 0.093 · meta_len 0.093 (실측)
      ★판정  감쇠보정 rho 가 0.30 을 넘으면 실용 신호

T3  음성 대조군 :  meta_len 이 위 전부를 이기면 **전부 폐기**
      실측 현황 : meta_len 0.598 > 현행 pos_full 0.457 -- 이미 지고 있다

T4  gold 의존성 :  gold 를 쓰는 신호와 안 쓰는 신호의 격차
      ★프로젝트의 노벨티가 여기 달려 있다

⛔EXP 트랙. 헤드라인 금지.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics as st
import sys
from collections import Counter

import torch

from abl_pmi import divergent_positions, _sum_logp, body_before_answer

sys.path.insert(0, "/home/v-seungplee/metacognition-math")
from src.training._decoy_utils import _rule_based_decoy  # noqa: E402


def auc(p, n):
    if not p or not n:
        return None
    return sum(1. if a > b else .5 if a == b else 0. for a in p for b in n) / (len(p) * len(n))


def boot_auc(p, n, B=2000, s=0):
    rng = random.Random(s)
    v = sorted(auc([rng.choice(p) for _ in p], [rng.choice(n) for _ in n]) for _ in range(B))
    return v[int(.025 * B)], v[int(.975 * B)]


def spear(x, y):
    def rk(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.] * len(v); i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            for k in range(i, j + 1):
                r[o[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    a, b = rk(x), rk(y); ma, mb = st.mean(a), st.mean(b)
    num = sum((p - ma) * (q - mb) for p, q in zip(a, b))
    den = math.sqrt(sum((p - ma) ** 2 for p in a) * sum((q - mb) ** 2 for q in b))
    return num / den if den else 0.


def perm_p(x, y, B=2000, s=0):
    o = abs(spear(x, y)); rng = random.Random(s); c = 0
    for _ in range(B):
        yy = y[:]; rng.shuffle(yy)
        if abs(spear(x, yy)) >= o:
            c += 1
    return c / B


@torch.inference_mode()
def pair_logp(model, tok, ctx, s1, s2, device="cuda", stats=None):
    """logp(s1|ctx) − logp(s2|ctx).

    ⚠**발산 토큰만 비교하는 방식이 자기후보 쌍에서 자주 무너진다.** `divergent_positions`
    는 접두·접미를 잘라내는데, 한 답이 다른 답의 부분열이면(`'2'` vs `'12'`,
    `'90^\\circ'` vs `'90'`) 한쪽 슬라이스가 **빈다**. 0817 스모크에서 후보 쌍의 다수가
    그 형태였고 전부 버려졌다(n=0).

    그래서 두 경로를 둔다:
      div   발산 토큰만 (원래 방식) — 두 슬라이스가 모두 비지 않을 때
      full  `\\boxed{...}` 전체 문자열 비교 — 위가 무너질 때 폴백

    `full` 은 길이 편향이 있다(긴 문자열이 logp 가 낮다). 그래서 **경로를 행마다 기록**하고,
    판정에서 경로별로도 나눠 본다. 길이 음성 대조군(T3)이 이 편향을 잡는 최종 방어선이다.
    """
    if s1 is None or s2 is None or s1.strip() == s2.strip():
        return None
    g, d, gs, ds = divergent_positions(tok, s1, s2)
    path = "div"
    if gs.stop <= gs.start or ds.stop <= ds.start:
        gs, ds, path = slice(0, len(g)), slice(0, len(d)), "full"
    if stats is not None:
        stats[path] = stats.get(path, 0) + 1
    return (_sum_logp(model, tok, ctx, g, gs, device)
            - _sum_logp(model, tok, ctx, d, ds, device))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splice", default="splice2_4b.json")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--out", default="signal_probe_4b.json")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    blob = json.load(open(args.splice))
    items, ps, acc = blob["items"], blob["per_sample"], blob["per_problem"]
    lv, CONDS = blob["levels"], blob["conds"]
    n = len(items) if not args.limit else min(args.limit, len(items))
    nrm = lambda s: (s or "").replace(" ", "").strip()

    # ── 0. 후보 답 쌍 추출 (gold 를 안 본다). 여섯 팔 전 샘플에서 모은다.
    cands = []
    for i in range(n):
        c = Counter(nrm(r["pred"]) for cd in CONDS for r in ps[cd][i] if r["pred"])
        c.pop("", None)
        top = c.most_common(2)
        cands.append((top[0][0] if top else None,
                      top[1][0] if len(top) > 1 else None,
                      (top[0][1] / sum(c.values())) if top else 0.0))
    n2 = sum(1 for a, b, _ in cands if b is not None)
    print(f"[0] 후보 쌍 있는 문제 {n2}/{n} ({n2/n*100:.1f}%)  — PMI_self 는 여기서만 정의된다")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda").eval()

    rows = []
    PATH = {}
    for i in range(n):
        it = items[i]
        a1, a2, agree = cands[i]
        gold = it["gold"]
        dec = _rule_based_decoy(gold)

        O = it["prompt"] + it["prefix"]                      # 메타 없음
        CR = O + f"\n<meta>{it['R']}</meta>"                 # 진짜 자기점검
        CB = O + f"\n<meta>{it['B']}</meta>"                 # 정형문 (문맥 대조군)

        # 답 직전 문맥 — R 팔 sample0 의 이어쓰기
        t0 = (ps["R"][i][0] or {}).get("text") or ""
        post = body_before_answer(t0) or ""
        pids = tok(post, add_special_tokens=False)["input_ids"][:600]
        has_post = len(pids) >= 5
        sl = slice(0, len(pids)) if has_post else None

        r = {"i": i, "level": lv[i], "agree": agree, "has_pair": a2 is not None,
             "has_post": has_post,
             "accN": acc["N"][i], "accR": acc["R"][i], "accB": acc["B"][i],
             "meta_len": len(tok(it["R"], add_special_tokens=False)["input_ids"])}

        for key, ctx in (("O", O), ("R", CR), ("B", CB)):
            r[f"gold_{key}"] = pair_logp(model, tok, ctx, gold, dec)
            r[f"self_{key}"] = pair_logp(model, tok, ctx, a1, a2, stats=PATH)
            if has_post:
                r[f"lp_{key}"] = _sum_logp(model, tok, ctx, pids, sl)
                r[f"goldp_{key}"] = pair_logp(model, tok, ctx + post, gold, dec)
                r[f"selfp_{key}"] = pair_logp(model, tok, ctx + post, a1, a2, stats=PATH)
        rows.append(r)
        if (i + 1) % 100 == 0:
            print(f"    ... {i+1}/{n}")

    print(f"[1] PMI_self 계산 경로 {PATH}  (full 폴백이 많으면 길이 편향 주의)")

    # ══ T1 난이도 판별 — 메타 없는 문맥에서
    HI = [r for r in rows if r["accN"] >= .999]
    LO = [r for r in rows if r["accN"] <= .001]
    print(f"\n[T1] 난이도 판별  AUC(6/6맞힘 {len(HI)} > 6/6틀림 {len(LO)})")
    T1 = {}
    for nm, f, gold_free in (
        ("PMI_gold  (현행)", lambda r: r["gold_O"], False),
        ("★PMI_self (제안)", lambda r: r["self_O"], True),
        ("자기일관성 (공짜)", lambda r: r["agree"], True),
        ("meta_len  (음성)", lambda r: float(r["meta_len"]), True),
    ):
        p = [f(r) for r in HI if f(r) is not None]
        q = [f(r) for r in LO if f(r) is not None]
        if len(p) < 5 or len(q) < 5:
            print(f"    {nm:20s} 표본부족 {len(p)}/{len(q)}"); continue
        a = auc(p, q); lo, hi = boot_auc(p, q)
        T1[nm] = {"auc": round(a, 4), "ci": [round(lo, 4), round(hi, 4)],
                  "n": [len(p), len(q)], "gold_free": gold_free}
        mark = "★" if (lo > .5 or hi < .5) else " "
        print(f"   {mark}{nm:20s} {a:.4f} [{lo:.3f},{hi:.3f}]  n={len(p)}/{len(q)}"
              f"  {'gold 안 씀' if gold_free else 'gold 씀'}")

    # ══ T2 메타 가치 예측 — 문제별 (acc_R − acc_B)
    RB = [r["accR"] - r["accB"] for r in rows]
    REL = 0.309                     # 초과분산/전체분산 (splice2 실측)
    CEIL = math.sqrt(REL)
    print(f"\n[T2] 메타 가치 예측 (달성 가능 최대 rho = {CEIL:.3f})")
    T2 = {}
    AX = {
        "PMI_gold 위치(R−O)":  lambda r: (None if r["gold_R"] is None or r["gold_O"] is None
                                          else r["gold_R"] - r["gold_O"]),
        "PMI_gold 문맥(R−B)":  lambda r: (None if r["gold_R"] is None or r["gold_B"] is None
                                          else r["gold_R"] - r["gold_B"]),
        "★PMI_self 위치":       lambda r: (None if r["self_R"] is None or r["self_O"] is None
                                          else r["self_R"] - r["self_O"]),
        "★PMI_self 문맥":       lambda r: (None if r["self_R"] is None or r["self_B"] is None
                                          else r["self_R"] - r["self_B"]),
        "★PMI_self 문맥·답직전": lambda r: (None if not r["has_post"] or r.get("selfp_R") is None
                                          or r.get("selfp_B") is None
                                          else r["selfp_R"] - r["selfp_B"]),
        "drive (내 본문)":      lambda r: (None if not r["has_post"]
                                          else (r["lp_R"] - r["lp_B"]) / max(1, len(str(r)))),
        "자기일관성":            lambda r: r["agree"],
        "meta_len (음성)":      lambda r: float(r["meta_len"]),
    }
    for nm, f in AX.items():
        idx = [k for k in range(len(rows)) if f(rows[k]) is not None]
        if len(idx) < 40:
            print(f"    {nm:22s} 표본부족 n={len(idx)}"); continue
        x = [f(rows[k]) for k in idx]; y = [RB[k] for k in idx]
        rho = spear(x, y); p = perm_p(x, y)
        corr = rho / CEIL
        T2[nm] = {"rho": round(rho, 4), "perm_p": round(p, 4), "n": len(idx),
                  "rho_corrected": round(corr, 4)}
        mark = "★" if p < .05 else " "
        print(f"   {mark}{nm:22s} rho {rho:+.4f}  보정후 {corr:+.4f}  p={p:.3f}  n={len(idx)}")

    out = {"model": args.model, "n": n, "n_pair": n2, "reliability": REL,
           "rho_ceiling": CEIL, "T1": T1, "T2": T2, "path": PATH, "rows": rows}
    json.dump(out, open(args.out, "w"))
    print(f"\n[3] wrote {args.out}")


if __name__ == "__main__":
    main()
