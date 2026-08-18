"""격자 프로브 — R_meta 의 설계 공간을 **세 축으로 완전 분해**해서 한 판에 잰다.

PMI-shift 는 사실 세 선택의 곱이었고, 우리는 그중 한 칸만 써 왔다.

  축1 · 무엇의 확률을 보나 (채점 대상)
        T1  logp(gold)                     정답만
        T2  logp(gold) − logp(decoy)       정답 vs 오답후보   ← 현행
        T3  logp(내 후속본문)               정답을 안 본다     ← 내부 신호

  축2 · 무엇과 비교하나 (대조)
        K1  메타 뒤 − 메타 앞               위치 대조          ← 현행
        K2  내 메타 − 남의 메타             문맥 치환

  축3 · 어디서 재나 (측정 위치)
        P1  메타 경계 직후                                    ← 현행
        P2  답 직전 (후속 본문을 다 지난 뒤)

현행 라이브 = **T2 × K1 × P1**. 격자에서 한 칸이다.

왜 나눠야 하나. 0817 실측에서 T2×K1×P1 이 AUC 0.457 로 눈이 멀었는데, **세 선택 중
어느 것이 범인인지** 알 수 없었다. 축을 붙여 놓고 한 칸만 재면 그 칸이 실패해도
"믿음 이동이라는 발상이 틀렸다"인지 "위치 대조가 틀렸다"인지 못 가른다.

★그리고 축2 에는 한 칸이 더 있다 — **의미 파괴 대조(S)**. 진짜 메타와 어절만 섞은
메타를 구별하는가. 못 구별하면 그 신호는 헛경보뿐 아니라 **영향력 자체를 못 잰다**.
0816 실측에서 두 조건의 정답률은 R 0.7428 vs S 0.7572 로 사실상 같았다 -- 즉
"메타 내용이 결과를 바꾼다"는 전제부터 약하다. 신호가 그 약한 차이라도 잡는지 본다.

판정은 앞과 같다: **진짜 경보(주장O·6/6 틀림, n=41) > 헛경보(주장O·6/6 맞힘, n=115)** 를
매기는가. 그리고 정오답을 겹친 **상호작용**도 같이 낸다 (§2.1: 조건부 지표는 파괴 비용을
못 본다 -- 그래서 층별로 따로 찍는다).

행당 17 forward · 300행 ≈ 15분.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys

import torch

from abl_pmi import body_before_answer, divergent_positions, _sum_logp
from alarm_probe import claims_error, get_conf, auc, boot_auc_ci, mean

sys.path.insert(0, "/home/v-seungplee/metacognition-math")
from src.training._decoy_utils import _rule_based_decoy  # noqa: E402


@torch.inference_mode()
def gd(model, tok, ctx, g_ids, d_ids, g_sl, d_sl, device="cuda"):
    return (_sum_logp(model, tok, ctx, g_ids, g_sl, device),
            _sum_logp(model, tok, ctx, d_ids, d_sl, device))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splice", default="splice_4b.json")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--out", default="grid_probe_4b.json")
    args = ap.parse_args()

    blob = json.load(open(args.splice))
    items, accN = blob["items"], blob["per_problem"]["N"]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda").eval()

    rows = []
    for n, it in enumerate(items):
        gold = it["gold"]
        decoy = _rule_based_decoy(gold)
        g_ids, d_ids, g_sl, d_sl = divergent_positions(tok, gold, decoy)

        O = it["prompt"] + it["prefix"]                       # 메타 없음
        C = O + f"\n<meta>{it['R']}</meta>"                   # 내 메타
        Dn = O + f"\n<meta>{it['D']}</meta>"                  # 남의 문제 메타
        Sh = O + f"\n<meta>{it['S']}</meta>"                  # 어절 섞기 (의미 파괴)

        post = body_before_answer(it.get("cont_R") or "") or ""
        post_ids = tok(post, add_special_tokens=False)["input_ids"][:600]
        has_post = len(post_ids) >= 5
        sl = slice(0, len(post_ids)) if has_post else None

        r = {"i": n, "claim": claims_error(it["R"]), "accN": accN[n],
             "conf": get_conf(it["R"]), "has_post": has_post,
             "n_post": len(post_ids) if has_post else 0}

        # ── 축3 P1 : 메타 경계에서 gold/decoy
        for key, ctx in (("O", O), ("C", C), ("D", Dn), ("S", Sh)):
            r[f"g_{key}"], r[f"d_{key}"] = gd(model, tok, ctx, g_ids, d_ids, g_sl, d_sl)

        # ── 축3 P2 : 답 직전 (본문을 다 지난 뒤) + 축1 T3 : 후속본문 자체의 logp
        if has_post:
            for key, ctx in (("O", O), ("C", C), ("D", Dn)):
                r[f"gp_{key}"], r[f"dp_{key}"] = gd(model, tok, ctx + post,
                                                    g_ids, d_ids, g_sl, d_sl)
                r[f"lp_{key}"] = _sum_logp(model, tok, ctx, post_ids, sl)

        rows.append(r)
        if (n + 1) % 75 == 0:
            print(f"    ... {n+1}/{len(items)}  (후속본문 있음 "
                  f"{sum(x['has_post'] for x in rows)})")

    T = [r for r in rows if r["claim"] == 1 and r["accN"] <= 0.001]
    F = [r for r in rows if r["claim"] == 1 and r["accN"] >= 0.999]
    print(f"\n[1] 참 경보 {len(T)} · 헛경보 {len(F)}  "
          f"(후속본문 있는 행: 참 {sum(r['has_post'] for r in T)} / "
          f"헛 {sum(r['has_post'] for r in F)})\n")

    # ── 격자. (채점대상, 대조, 측정위치) -> 스칼라
    def mk(target, contrast, pos):
        pre = "gp_" if pos == "P2" else "g_"
        pred = "dp_" if pos == "P2" else "d_"
        ctrl = "O" if contrast == "K1" else "D"

        def f(r):
            if target == "T1":
                return r[pre + "C"] - r[pre + ctrl]
            return ((r[pre + "C"] - r[pred + "C"])
                    - (r[pre + ctrl] - r[pred + ctrl]))
        return f

    def drive(contrast):
        ctrl = "O" if contrast == "K1" else "D"
        return lambda r: (r["lp_C"] - r["lp_" + ctrl]) / max(1, r["n_post"])

    CELLS = {}
    for t in ("T1", "T2"):
        for k in ("K1", "K2"):
            for p in ("P1", "P2"):
                CELLS[f"{t}·{k}·{p}"] = (mk(t, k, p), p == "P2")
    CELLS["T3·K1 (drive vs 무메타)"] = (drive("K1"), True)
    CELLS["T3·K2 (drive vs 남의메타)"] = (drive("K2"), True)

    NAMES = {"T1": "정답만", "T2": "정답−오답", "T3": "내 본문",
             "K1": "위치(앞/뒤)", "K2": "문맥(내/남)", "P1": "메타경계", "P2": "답직전"}
    print(f"{'칸':30s} {'AUC (참>헛)':>22s}  {'n':>9s}   해석")
    res = {}
    for name, (f, need_post) in CELLS.items():
        Ts = [r for r in T if r["has_post"] or not need_post]
        Fs = [r for r in F if r["has_post"] or not need_post]
        if len(Ts) < 5:
            print(f"{name:30s} {'표본부족':>22s}  {len(Ts):3d}/{len(Fs):3d}")
            continue
        p, q = [f(r) for r in Ts], [f(r) for r in Fs]
        a = auc(p, q)
        lo, hi = boot_auc_ci(p, q)
        live = " ←현행" if name == "T2·K1·P1" else ""
        star = "★" if (lo > 0.5 or hi < 0.5) else " "
        res[name] = {"auc": round(a, 4), "ci": [round(lo, 4), round(hi, 4)],
                     "n_true": len(Ts), "n_false": len(Fs),
                     "mean_true": round(mean(p), 4), "mean_false": round(mean(q), 4)}
        print(f"{star}{name:29s} {a:.4f} [{lo:.3f},{hi:.3f}]  "
              f"{len(Ts):3d}/{len(Fs):3d}   참 {mean(p):+8.3f} vs 헛 {mean(q):+8.3f}{live}")

    # ── ★영향력 검정: 신호가 진짜 메타(C)와 **의미 파괴 메타(S)** 를 구별하는가.
    #    못 구별하면 그 신호는 헛경보뿐 아니라 "메타가 일을 했나"도 못 잰다.
    print("\n[영향력] 같은 본문 위에서 진짜 메타 C 와 어절섞기 S 를 구별하나 "
          "(짝지은 차, 300행 전부)")
    for nm, fc, fs in (
        ("T2·K1·P1 (현행)",
         lambda r: (r["g_C"] - r["d_C"]) - (r["g_O"] - r["d_O"]),
         lambda r: (r["g_S"] - r["d_S"]) - (r["g_O"] - r["d_O"])),
        ("T1·K1·P1 (정답만)",
         lambda r: r["g_C"] - r["g_O"], lambda r: r["g_S"] - r["g_O"]),
        ("T2·K2·P1 (문맥, S를 대조군으로)",
         lambda r: (r["g_C"] - r["d_C"]) - (r["g_S"] - r["d_S"]),
         lambda r: 0.0),
    ):
        dif = [fc(r) - fs(r) for r in rows]
        m = st.mean(dif)
        rng = random.Random(0)
        bs = sorted(st.mean(rng.choice(dif) for _ in dif) for _ in range(2000))
        lo, hi = bs[50], bs[1949]
        mark = "★구별함" if (lo > 0 or hi < 0) else " 구별 못함"
        print(f"  {nm:34s} C−S = {m:+8.4f} [{lo:+.4f},{hi:+.4f}] {mark}")

    # ── 정오답을 겹친다: 층별로 따로 (조건부 지표는 파괴 비용을 못 본다)
    print("\n[층별] 현행 신호(T2·K1·P1)가 층마다 무엇을 매기나")
    live_f = mk("T2", "K1", "P1")
    for lab, sub in (("6/6 맞힘·주장O (헛경보)", F), ("6/6 틀림·주장O (참경보)", T),
                     ("6/6 맞힘·주장X", [r for r in rows if r["claim"] == 0 and r["accN"] >= .999]),
                     ("6/6 틀림·주장X", [r for r in rows if r["claim"] == 0 and r["accN"] <= .001])):
        if sub:
            print(f"  {lab:24s} n={len(sub):3d}  평균 {mean([live_f(r) for r in sub]):+8.4f}")

    json.dump({"model": args.model, "n_true": len(T), "n_false": len(F),
               "cells": res, "rows": rows}, open(args.out, "w"))
    print(f"\n[2] wrote {args.out}")


if __name__ == "__main__":
    main()
