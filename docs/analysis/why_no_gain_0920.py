#!/usr/bin/env python
r"""L5-800x8 롤아웃 진단 — PMI 작은 수정 크레딧 25스텝 후 정확도↑ / 자발 수정률↓ 원인.

실행: /hdd_data/seungpil/envs/qwen35/bin/python docs/analysis/why_no_gain_0920.py
CPU 전용. 결과 숫자는 모두 이 스크립트 출력에서만 인용한다.
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.training.math_meta import grade_math  # noqa: E402
from src.training.revision import revision_zone  # noqa: E402
from src.training.trial2 import agreement_state  # noqa: E402

FILES = {
    "base": "/hdd_data/seungpil/scratch/eval/mathL5_q3i2507_opt_b8k/texts.jsonl",
    "pmi25": "/hdd_data/seungpil/scratch/eval/mathL5_cd9_M_REV_PMI_GOLD_s1_r8192_step25_b8k/texts.jsonl",
    "g1_50": "/hdd_data/seungpil/scratch/eval/mathL5_cd9_M_G1_s1_r8192_step50_b8k/texts.jsonl",
    "oo30": "/hdd_data/seungpil/scratch/eval/mathL5_trial2s7step30_b8k/texts.jsonl",
}
HEDGE = re.compile(r"\bwait\b|hmm|double[- ]check|let me verify|actually", re.I)
CACHE = "/hdd_data/seungpil/scratch/eval/_why_no_gain_0920_cache.json"


def load(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def featurize(rows):
    """행마다 파생 지표. gold 채점은 grade_math(math_verify) 로 한 번만."""
    out = []
    for r in rows:
        text = r.get("text") or ""
        z = revision_zone(text)
        first = z["first_answer"] if z else (r.get("final_answer") or "")
        has_box = bool(z) or bool((r.get("final_answer") or "").strip())
        fc = grade_math("\\boxed{" + str(first) + "}", str(r.get("gold"))) if str(first).strip() else 0
        out.append({
            "pid": int(r["problem_id"]),
            "corr": int(r.get("r_corr") or 0),
            "final": (r.get("final_answer") or ""),
            "first_corr": int(fc),
            "revised": bool(z and z["revised"]),
            "n_boxes": (z["n_boxes"] if z else (1 if has_box else 0)),
            "n_cp": (z["n_change_points"] if z else 0),
            "zone_len": ((z["zone_end"] - z["zone_start"]) if z else 0),
            "has_box": has_box,
            "trunc": bool(r.get("truncated")),
            "n_tok": int(r.get("n_tok") or 0),
            "hedge": bool(HEDGE.search(text)),
        })
    return out


def get(tag):
    path = FILES[tag]
    if not os.path.exists(path):
        return None
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    if tag in cache:
        return cache[tag]
    feats = featurize(load(path))
    cache[tag] = feats
    json.dump(cache, open(CACHE, "w"))
    return feats


def bypid(feats):
    d = defaultdict(list)
    for f in feats:
        d[f["pid"]].append(f)
    return d


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def pct(x):
    return f"{100 * x:.2f}%"


def quant(xs, q):
    xs = sorted(xs)
    if not xs:
        return 0
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def overall(tag, F):
    print(f"[{tag}] n={len(F)} pass@1={mean(f['corr'] for f in F):.4f} "
          f"first_box_acc={mean(f['first_corr'] for f in F):.4f} "
          f"rev_rate={pct(mean(f['revised'] for f in F))} "
          f"multibox_rate={pct(mean(f['n_boxes'] >= 2 for f in F))} "
          f"trunc={pct(mean(f['trunc'] for f in F))} "
          f"nobox={pct(mean(not f['has_box'] for f in F))} "
          f"ntok_mean={mean(f['n_tok'] for f in F):.0f} "
          f"p50={quant([f['n_tok'] for f in F], .5)} p90={quant([f['n_tok'] for f in F], .9)} "
          f"hedge={pct(mean(f['hedge'] for f in F))}")


def main():
    D = {t: get(t) for t in FILES}
    avail = [t for t in D if D[t]]
    print("=== 0) 파일 ===")
    for t in FILES:
        print(f"  {t}: {'OK' if D[t] else 'MISSING'} {FILES[t]}")

    print("\n=== 1) 전체 지표 ===")
    for t in avail:
        overall(t, D[t])

    B, P = bypid(D["base"]), bypid(D["pmi25"])
    pids = sorted(set(B) & set(P))
    print(f"\n=== 1b) 문제별 정확도 변화 base->pmi25 (paired n={len(pids)}) ===")
    st_of = {}
    for pid in pids:
        st_of[pid] = agreement_state([f["final"] for f in B[pid]])["state"]
    buckets = defaultdict(lambda: {"imp": 0, "wor": 0, "same": 0, "d": 0.0, "n": 0,
                                   "base_acc": 0.0, "p_acc": 0.0})
    tot = {"imp": 0, "wor": 0, "same": 0}
    for pid in pids:
        a, b = mean(f["corr"] for f in B[pid]), mean(f["corr"] for f in P[pid])
        k = "imp" if b > a else ("wor" if b < a else "same")
        tot[k] += 1
        s = buckets[st_of[pid]]
        s[k] += 1
        s["n"] += 1
        s["d"] += b - a
        s["base_acc"] += a
        s["p_acc"] += b
    print(f"  전체: improved={tot['imp']} worsened={tot['wor']} same={tot['same']}")
    print("  state        n   imp  wor  same   base_acc  pmi_acc   delta_pp  contrib_pp")
    for s in ["ALL_SAME", "DOMINANT", "SPLIT", "SCATTER", "NOANS"]:
        v = buckets.get(s)
        if not v or not v["n"]:
            continue
        print(f"  {s:10s} {v['n']:4d} {v['imp']:5d}{v['wor']:5d}{v['same']:6d}"
              f"   {v['base_acc']/v['n']:.4f}  {v['p_acc']/v['n']:.4f}"
              f"   {100*v['d']/v['n']:+7.2f}  {100*v['d']/len(pids):+7.2f}")

    print("\n=== 1c) ALL_SAME 손실의 정체 ===")
    asp = [p for p in pids if st_of[p] == "ALL_SAME"]
    asc = [p for p in asp if mean(f["corr"] for f in B[p]) >= 0.999]   # base 전원 정답
    asw = [p for p in asp if mean(f["corr"] for f in B[p]) <= 0.001]   # base 전원 오답
    print(f"  ALL_SAME n={len(asp)} (전원정답 {len(asc)}, 전원오답 {len(asw)})")
    for nm, S in (("base", B), ("pmi25", P)):
        rs = [f for p in asc for f in S[p]]
        print(f"  [{nm}] 전원정답 ALL_SAME: pass@1={mean(f['corr'] for f in rs):.4f} "
              f"first_box={mean(f['first_corr'] for f in rs):.4f} "
              f"nobox={pct(mean(not f['has_box'] for f in rs))} trunc={pct(mean(f['trunc'] for f in rs))} "
              f"rev={pct(mean(f['revised'] for f in rs))} ntok={mean(f['n_tok'] for f in rs):.0f}")
    rsb = [f for p in asc for f in B[p]]
    rsp = [f for p in asc for f in P[p]]
    dnb = mean(not f["has_box"] for f in rsp) - mean(not f["has_box"] for f in rsb)
    print(f"  전원정답 ALL_SAME 손실 {100*(mean(f['corr'] for f in rsp)-mean(f['corr'] for f in rsb)):+.2f}pp 중 "
          f"무박스 증가분 {100*dnb:+.2f}pp (나머지는 박스는 있는데 오답)")

    print("\n=== 2) base 에서 수정이 있던 문제들의 행방 ===")
    rev_pids = [p for p in pids if any(f["revised"] for f in B[p])]
    bn = sum(sum(f["revised"] for f in B[p]) for p in rev_pids)
    pn = sum(sum(f["revised"] for f in P[p]) for p in rev_pids)
    print(f"  base 에 수정 롤아웃 >=1 인 문제: {len(rev_pids)} / {len(pids)}")
    print(f"  그 문제들의 수정 롤아웃 수: base={bn} -> pmi25={pn} ({pn/max(bn,1):.2f}x)")
    for nm, S in (("base", B), ("pmi25", P)):
        rs = [f for p in rev_pids for f in S[p]]
        print(f"  [{nm}] 해당 문제 pass@1={mean(f['corr'] for f in rs):.4f} "
              f"first_box_acc={mean(f['first_corr'] for f in rs):.4f} "
              f"rev_rate={pct(mean(f['revised'] for f in rs))}")
    # 전체에서 first->final 전이
    for t in avail:
        F = D[t]
        w2r = mean(f["first_corr"] == 0 and f["corr"] == 1 for f in F)
        r2w = mean(f["first_corr"] == 1 and f["corr"] == 0 for f in F)
        print(f"  [{t}] 전체 first->final: 구제(0->1)={pct(w2r)} 파손(1->0)={pct(r2w)} "
              f"순이득={pct(w2r - r2w)}")

    print("\n=== 3) 절단/길이 ===")
    for t in avail:
        F = D[t]
        R = [f for f in F if f["revised"]]
        print(f"  [{t}] trunc={pct(mean(f['trunc'] for f in F))} "
              f"trunc|무박스={pct(mean(f['trunc'] for f in F if not f['has_box']))} "
              f"ntok(p50/p90/p99)={quant([f['n_tok'] for f in F],.5)}/"
              f"{quant([f['n_tok'] for f in F],.9)}/{quant([f['n_tok'] for f in F],.99)} "
              f"| revised n={len(R)} zone_len(mean/p50/p90)={mean(f['zone_len'] for f in R):.0f}/"
              f"{quant([f['zone_len'] for f in R],.5)}/{quant([f['zone_len'] for f in R],.9)}")

    print("\n=== 4) 잔존 오답 분해 ===")
    for t in avail:
        F = D[t]
        W = [f for f in F if not f["corr"]]
        c = Counter()
        for f in W:
            if not f["has_box"]:
                c["no_box"] += 1
            elif f["revised"]:
                c["rev_right2wrong" if f["first_corr"] else "rev_still_wrong"] += 1
            else:
                c["first_wrong_no_rev"] += 1
        nrw = [f for f in W if f["has_box"] and not f["revised"]]
        print(f"  [{t}] wrong={len(W)} ({pct(len(W)/len(F))}) : "
              + " ".join(f"{k}={v}({pct(v/max(len(W),1))})" for k, v in
                         [("first_wrong_no_rev", c["first_wrong_no_rev"]),
                          ("rev_still_wrong", c["rev_still_wrong"]),
                          ("rev_right2wrong", c["rev_right2wrong"]),
                          ("no_box", c["no_box"])]))
        print(f"       '틀렸고 수정 안 함' 안의 hedge 텍스트 비율={pct(mean(f['hedge'] for f in nrw))} "
              f"(그 중 다중박스(같은 답 재확인)={pct(mean(f['n_boxes']>=2 for f in nrw))}) "
              f"| 전체 대비 누락질량={pct(len(nrw)/len(F))}")

    print("\n=== 5) outcome-only 비교 (위 [oo30]/[g1_50] 행 참조) ===")
    for t in avail:
        if t == "base":
            continue
        F, Fb = bypid(D[t]), B
        pp = sorted(set(F) & set(Fb))
        d = mean(mean(f["corr"] for f in F[p]) for p in pp) - mean(mean(f["corr"] for f in Fb[p]) for p in pp)
        dr = mean(f["revised"] for p in pp for f in F[p]) - mean(f["revised"] for p in pp for f in Fb[p])
        dfb = mean(f["first_corr"] for p in pp for f in F[p]) - mean(f["first_corr"] for p in pp for f in Fb[p])
        print(f"  [{t}] vs base: d_pass@1={100*d:+.2f}pp d_rev_rate={100*dr:+.2f}pp d_first_box={100*dfb:+.2f}pp")

    print("\n=== 6) 사라진 수정 구간 예시 ===")
    raw_b = {(r["problem_id"], i): r for i, r in enumerate(load(FILES["base"]))}
    base_rows = load(FILES["base"])
    pmi_rows = load(FILES["pmi25"])
    gone = [p for p in rev_pids if not any(f["revised"] for f in P[p])]
    print(f"  base 에 수정 있고 pmi25 에 전무한 문제: {len(gone)}")
    shown = 0
    for r in base_rows:
        if shown >= 3 or r["problem_id"] not in set(gone):
            continue
        z = revision_zone(r.get("text") or "")
        if not (z and z["revised"]):
            continue
        zone = (r["text"][z["zone_start"]:z["zone_end"]]).strip().replace("\n", " ")[:400]
        ok = grade_math(r["text"], str(r["gold"]))
        print(f"  --- base pid={r['problem_id']} {z['first_answer']}->{z['last_answer']} final_corr={ok}\n      {zone}")
        shown += 1
    shown = 0
    for r in pmi_rows:
        if shown >= 2:
            break
        z = revision_zone(r.get("text") or "")
        if not (z and z["revised"]):
            continue
        zone = (r["text"][z["zone_start"]:z["zone_end"]]).strip().replace("\n", " ")[:400]
        ok = grade_math(r["text"], str(r["gold"]))
        print(f"  --- pmi25 pid={r['problem_id']} {z['first_answer']}->{z['last_answer']} final_corr={ok}\n      {zone}")
        shown += 1


if __name__ == "__main__":
    main()
