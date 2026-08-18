"""Countdown 본 판 2단계 — **격자 지표가 SAVE 를 찾는가.**

1단계(`cd_main.py`)가 만든 라벨 위에서, PMI 계열 전 축을 티처포싱으로 채점한다.
⚠1단계 주 지표가 널이면 **이 파일을 돌리지 않는다** — 기전이 없으면 신호도 없다.

과녁 (사용자 지정, 0818):
    모집단  acc(N) = 0 인 문제만        ← 메타 없이 못 푸는 문제
    라벨    SAVE (메타로 풀게 됨)  vs  실패 유지
    ⚠전체 문제를 대조군에 넣으면 AUC 가 **난이도 탐지**로 변질된다 — 실측으로 두 번 속았다
      (`−confidence` 0.668→0.552, `자기일관성` 0.085→0.371 로 둘 다 널이 됐다).

★Countdown 이 수학보다 나은 점 — decoy 가 완전히 짝지어진다:
    증인      (((22-2)*(21-12))-24)  = 156
    연산자교체 (((22+2)*(21-12))-24)  = 192      같은 수·같은 구조·연산자 하나만 다름
  수학 판의 `_rule_based_decoy` 는 모델이 고려한 적 없는 문자열이라
  `logp(gold)−logp(decoy)` 가 +6 nats 에서 포화됐다(AUC(맞힘>틀림) 0.539).
  여기서 **포화 가설을 처음으로 제대로 검정한다.**

격자 (축1 × 축2 × 축3) + 무료 지표:
    축1 채점대상  ① 증인 단독  ② 증인−연산자교체  ③ 자기후보 a1 vs a2  ④ 내 후속본문(drive)
    축2 대조      위치(메타 앞 O vs 뒤 R)  ·  문맥(내 메타 R vs 정형문 B)
    축3 위치      메타 경계  ·  답 직전
    무료          confidence · 시도횟수 · 메타길이(★음성 대조군)
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics as st
from collections import Counter

import torch

from abl_pmi import divergent_positions, _sum_logp


def auc(p, q):
    if not p or not q:
        return None
    return sum(1. if a > b else .5 if a == b else 0. for a in p for b in q) / (len(p) * len(q))


def boot_auc(p, q, B=2000, s=0):
    r = random.Random(s)
    v = sorted(auc([r.choice(p) for _ in p], [r.choice(q) for _ in q]) for _ in range(B))
    return v[int(.025 * B)], v[int(.975 * B)]


@torch.inference_mode()
def pair(model, tok, ctx, s1, s2, device="cuda", stats=None):
    """logp(s1|ctx) − logp(s2|ctx). 발산 토큰만, 무너지면 전체 문자열 폴백.

    ⚠부분열 쌍('2' vs '12')에서 발산 슬라이스가 비는 버그를 0818 스모크에서 잡았다.
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


@torch.inference_mode()
def solo(model, tok, ctx, s, device="cuda"):
    ids = tok(f"\\boxed{{{s}}}", add_special_tokens=False)["input_ids"]
    return _sum_logp(model, tok, ctx, ids, slice(0, len(ids)), device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default="cd_main.json")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--out", default="cd_grid.json")
    args = ap.parse_args()

    b = json.load(open(args.main))
    items, acc, ps, ARMS = b["items"], b["per_problem"], b["per_sample"], b["arms"]
    POOL = b["pool"]
    n = len(items)
    print(f"[0] 모집단 acc(N)=0 · 누출없음 : {len(POOL)}/{n}")
    SV = [i for i in POOL if acc["R"][i] > 1e-9]
    NS = [i for i in POOL if acc["R"][i] <= 1e-9]
    print(f"    SAVE {len(SV)}  vs  실패유지 {len(NS)}")
    if len(SV) < 15:
        raise SystemExit("SAVE 표본 15 미만 — 판정 불가. 1단계 규모를 늘려라.")

    # 자기후보 두 개 (gold 를 안 본다) — 다섯 팔 전 샘플의 최종 식에서
    nrm = lambda s: (s or "").replace(" ", "").strip()
    _BX = re.compile(r"\\boxed\{([^{}]*)\}")
    cands = []
    for i in range(n):
        c = Counter()
        for a in ARMS:
            for r in ps[a][i]:
                t = r.get("text")
                if t:
                    for m in _BX.findall(t):
                        if nrm(m):
                            c[nrm(m)] += 1
        top = c.most_common(2)
        cands.append((top[0][0] if top else None, top[1][0] if len(top) > 1 else None))

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda").eval()

    PATH, rows = {}, {}
    for i in POOL:
        it = items[i]
        w, dec = it["witness"], it.get("decoy")
        a1, a2 = cands[i]
        O = it["prompt"] + it["prefix"]
        CR = O + f"\n<meta>{it['R']}</meta>"
        CB = O + f"\n<meta>{it['B']}</meta>"
        # 답 직전 = R 팔 sample0 의 이어쓰기 (boxed 앞까지)
        t0 = (ps["R"][i][0] or {}).get("text") or ""
        j = t0.find("\\boxed{")
        post = t0[:j] if j > 0 else ""
        pid = tok(post, add_special_tokens=False)["input_ids"][:600]
        hp = len(pid) >= 5

        r = {"i": i, "save": 1 if acc["R"][i] > 1e-9 else 0,
             "conf": (lambda m: float(m.group(1)) if m else .5)(
                 re.search(r"confidence:\s*([0-9.]+)", it["R"])),
             "meta_len": len(tok(it["R"], add_special_tokens=False)["input_ids"]),
             "tries_R": st.mean(x["tries"] for x in ps["R"][i]),
             "tries_N": st.mean(x["tries"] for x in ps["N"][i]),
             "has_post": hp, "has_pair": a2 is not None}
        for key, ctx in (("O", O), ("R", CR), ("B", CB)):
            r[f"w_{key}"] = solo(model, tok, ctx, w)                       # 축1-①
            r[f"wd_{key}"] = pair(model, tok, ctx, w, dec, stats=PATH)     # 축1-②
            r[f"sf_{key}"] = pair(model, tok, ctx, a1, a2, stats=PATH)     # 축1-③
            if hp:
                r[f"lp_{key}"] = _sum_logp(model, tok, ctx, pid, slice(0, len(pid)))
                r[f"wdp_{key}"] = pair(model, tok, ctx + post, w, dec, stats=PATH)
        rows[i] = r
    print(f"[1] 원재료 완료 {len(rows)} · PMI 경로 {PATH}")

    # ── 격자 채점
    def mk(t, k, p):
        pre = {"w": "w_", "wd": "wd_" if p == "P1" else "wdp_", "sf": "sf_"}[t]
        ctrl = "O" if k == "K1" else "B"
        return lambda r: (None if r.get(pre + "R") is None or r.get(pre + ctrl) is None
                          else r[pre + "R"] - r[pre + ctrl])

    CELLS = {}
    for t, tn in (("w", "증인단독"), ("wd", "증인−연산자교체"), ("sf", "자기후보")):
        for k, kn in (("K1", "위치"), ("K2", "문맥")):
            for p, pn in (("P1", "경계"), ("P2", "답직전")):
                if p == "P2" and t != "wd":
                    continue                     # 답직전은 증인−교체 축에서만
                CELLS[f"{tn}·{kn}·{pn}"] = (mk(t, k, p), p == "P2")
    CELLS["drive (내 본문)"] = (
        lambda r: (None if not r["has_post"] else (r["lp_R"] - r["lp_B"]) / 100.0), True)
    CELLS["−confidence"] = (lambda r: -r["conf"], False)
    CELLS["시도수 R−N"] = (lambda r: r["tries_R"] - r["tries_N"], False)
    CELLS["★meta_len (음성)"] = (lambda r: float(r["meta_len"]), False)

    print(f"\n[2] ★격자 — SAVE({len(SV)}) vs 실패유지({len(NS)}) 를 가리나")
    res = {}
    for nm, (f, _) in CELLS.items():
        p = [f(rows[i]) for i in SV if f(rows[i]) is not None]
        q = [f(rows[i]) for i in NS if f(rows[i]) is not None]
        if len(p) < 8 or len(q) < 8:
            print(f"    {nm:26s} 표본부족 {len(p)}/{len(q)}")
            continue
        a = auc(p, q)
        lo, hi = boot_auc(p, q)
        res[nm] = {"auc": round(a, 4), "ci": [round(lo, 4), round(hi, 4)],
                   "n": [len(p), len(q)]}
        print(f"   {'★' if lo > .5 or hi < .5 else ' '}{nm:26s} AUC {a:.4f} "
              f"[{lo:.3f},{hi:.3f}]  n={len(p)}/{len(q)}")

    neg = res.get("★meta_len (음성)", {}).get("auc")
    if neg is not None:
        beat = [k for k, v in res.items()
                if k != "★meta_len (음성)" and abs(v["auc"] - .5) > abs(neg - .5)]
        print(f"\n[3] 음성 대조군(길이) AUC {neg:.4f} — 이것을 이긴 칸 {len(beat)}개")
        if not beat:
            print("    ⛔길이가 전부를 이긴다 ⇒ 격자 전체 폐기 (사전등록 음성 조건)")

    json.dump({"model": args.model, "n_save": len(SV), "n_nosave": len(NS),
               "cells": res, "rows": list(rows.values())}, open(args.out, "w"))
    print(f"\n[4] wrote {args.out}")


if __name__ == "__main__":
    main()
