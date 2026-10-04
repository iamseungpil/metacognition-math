#!/usr/bin/env python
"""habit_vs_pmi_0919 — 자발적 답 수정 구간의 텍스트 습관(recheck/switch/doubt/...)이
PMI 신념 이동(SHIFT_real)과 훈련 크레딧(pmi_shift_reward)을 얼마나 가져가는지 잰다.

입력:
  rows.jsonl  = /hdd_data/seungpil/scratch/eval/pmishift_full_L5/rows.jsonl
                (scripts/local/math_pmi_shift_probe.py 산출, roll_id = f"{group_id}#{j}")
  texts.jsonl = /hdd_data/seungpil/scratch/eval/mathL5_q3i2507_opt_b8k/texts.jsonl
                (그룹 안 등장 순서가 roll_id 의 j 다 — plan_rows 의 group_rollouts 순서와 동일)

zone = src.training.revision.revision_zone(text) 의 [zone_start:zone_end] 문자 구간.
카테고리 정규식은 과제 명세 그대로. 출력은 docs/analysis/HABIT_vs_PMI_0919.md 로 인쇄만 함
(이 스크립트는 표준출력에 그대로 마크다운을 낸다 — 리다이렉트해서 저장했다).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "/home/ubuntu/seungpil/metacognition-math")

import numpy as np

from src.training.revision import revision_zone
from src.training.dcpo_pmi_shift import pmi_shift_reward

ROWS = "/hdd_data/seungpil/scratch/eval/pmishift_full_L5/rows.jsonl"
TEXTS = "/hdd_data/seungpil/scratch/eval/mathL5_q3i2507_opt_b8k/texts.jsonl"
PAIRS = ("gold", "self", "goldx")

CATS = {
    "recheck": re.compile(r"double[- ]check|re-?check|verify|let me check|confirm|re-?examine|recompute|re-?calculate|plug|substitut", re.I),
    "switch": re.compile(r"alternatively|another (way|approach|method)|instead|different approach|other approach|let'?s try|try (a|another)", re.I),
    "doubt": re.compile(r"\bwait\b|\bhmm\b|hold on|actually|but wait|oops", re.I),
    "error_naming": re.compile(r"mistake|error|wrong|incorrect|typo|miscalculat|misread|misinterpret", re.I),
    "reread": re.compile(r"re-?read|the (problem|question) (says|asks|states)|going back to the problem", re.I),
    "constraint": re.compile(r"constraint|condition|domain|case|does(n'?t| not) satisf|check (if|whether)|boundary", re.I),
}


def tag(zone_text: str) -> list[str]:
    tags = [name for name, rx in CATS.items() if rx.search(zone_text)]
    return tags if tags else ["no_marker"]


def exclusive(tags: list[str]) -> str:
    has_switch = "switch" in tags
    has_recheck = "recheck" in tags
    if has_switch and has_recheck:
        return "both"
    if has_switch:
        return "switch_only"
    if has_recheck:
        return "recheck_only"
    return "neither"


def bootstrap_ci(vals, n_boot=2000, seed=0):
    vals = np.asarray([v for v in vals if v is not None and np.isfinite(v)], dtype=float)
    if len(vals) == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = len(vals)
    boots = np.array([rng.choice(vals, size=n, replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return (float(vals.mean()), float(lo), float(hi))


def spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return float("nan"), len(x)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan"), len(x)
    r = np.corrcoef(rx, ry)[0, 1]
    return float(r), len(x)


def main():
    texts_by_group: dict[str, list[dict]] = {}
    with open(TEXTS) as f:
        for line in f:
            r = json.loads(line)
            gid = r.get("group_id", r.get("problem_id"))
            texts_by_group.setdefault(gid, []).append(r)

    rows = []
    with open(ROWS) as f:
        for line in f:
            rows.append(json.loads(line))

    n_total = len(rows)
    n_revised = sum(1 for r in rows if r.get("revised"))

    recs = []
    n_no_text_zone = 0
    for r in rows:
        if not r.get("revised"):
            continue
        gid, roll_id = r["group_id"], r["roll_id"]
        j = int(roll_id.split("#")[-1])
        glist = texts_by_group.get(gid)
        if glist is None or j >= len(glist):
            n_no_text_zone += 1
            continue
        text = glist[j].get("text", "")
        z = revision_zone(text)
        if z is None:
            n_no_text_zone += 1
            continue
        zone_text = text[z["zone_start"]:z["zone_end"]]
        tags = tag(zone_text)
        excl = exclusive(tags)
        n_words = len(zone_text.split())
        rec = {"roll_id": roll_id, "tags": tags, "excl": excl, "n_words": n_words,
               "first_correct": r.get("first_correct", 0), "last_correct": r.get("last_correct", 0)}
        for p in PAIRS:
            if r.get(f"{p}_skip", "skip"):
                if r.get(f"{p}_skip") != "":
                    continue
            rec[f"{p}_shift_real"] = r.get(f"{p}_shift_real")
            rec[f"{p}_shift_placebo"] = r.get(f"{p}_shift_placebo")
            po, pc = r.get(f"{p}_pmi_open"), r.get(f"{p}_pmi_close")
            rec[f"{p}_reward"] = pmi_shift_reward(po, pc) if (po is not None and pc is not None) else None
        recs.append(rec)

    out = []
    out.append("# HABIT vs PMI (0919)")
    out.append("")
    out.append(f"스크립트: `docs/analysis/habit_vs_pmi_0919.py`")
    out.append("")
    out.append(f"- 전체 행 n={n_total}, revised=True n={n_revised}")
    out.append(f"- revised 중 텍스트/zone 매칭 실패(제외) n={n_no_text_zone}")
    out.append(f"- 분석 대상(revised & zone 있음) n={len(recs)}")
    out.append("")

    cat_names = list(CATS.keys()) + ["no_marker"]

    for p in PAIRS:
        out.append(f"## 앵커 쌍: {p}")
        out.append("")
        out.append("| category | n | mean SHIFT_real [95%CI] | mean SHIFT_placebo | mean(real-placebo) | frac w->r | mean reward |")
        out.append("|---|---|---|---|---|---|---|")
        for cat in cat_names:
            sub = [r for r in recs if cat in r["tags"] and r.get(f"{p}_shift_real") is not None]
            n = len(sub)
            if n == 0:
                out.append(f"| {cat} | 0 | - | - | - | - | - |")
                continue
            sr = [r[f"{p}_shift_real"] for r in sub]
            sp_ = [r[f"{p}_shift_placebo"] for r in sub]
            diff = [a - b for a, b in zip(sr, sp_) if a is not None and b is not None]
            m, lo, hi = bootstrap_ci(sr)
            mp = np.nanmean([v for v in sp_ if v is not None])
            mdiff = np.nanmean(diff) if diff else float("nan")
            wr = [r for r in sub if r["first_correct"] == 0]
            frac_wr = (sum(1 for r in wr if r["last_correct"] == 1) / len(wr)) if wr else float("nan")
            rewards = [r[f"{p}_reward"] for r in sub if r.get(f"{p}_reward") is not None]
            mrew = np.mean(rewards) if rewards else float("nan")
            out.append(f"| {cat} | {n} | {m:.3f} [{lo:.3f},{hi:.3f}] | {mp:.3f} | {mdiff:.3f} | {frac_wr:.3f} | {mrew:.3f} |")
        out.append("")

        out.append("### 배타적 분할 (switch_only/recheck_only/both/neither)")
        out.append("")
        out.append("| split | n | mean SHIFT_real [95%CI] | mean SHIFT_placebo | mean(real-placebo) | frac w->r | mean reward |")
        out.append("|---|---|---|---|---|---|---|")
        for excl in ("switch_only", "recheck_only", "both", "neither"):
            sub = [r for r in recs if r["excl"] == excl and r.get(f"{p}_shift_real") is not None]
            n = len(sub)
            if n == 0:
                out.append(f"| {excl} | 0 | - | - | - | - | - |")
                continue
            sr = [r[f"{p}_shift_real"] for r in sub]
            sp_ = [r[f"{p}_shift_placebo"] for r in sub]
            diff = [a - b for a, b in zip(sr, sp_) if a is not None and b is not None]
            m, lo, hi = bootstrap_ci(sr)
            mp = np.nanmean([v for v in sp_ if v is not None])
            mdiff = np.nanmean(diff) if diff else float("nan")
            wr = [r for r in sub if r["first_correct"] == 0]
            frac_wr = (sum(1 for r in wr if r["last_correct"] == 1) / len(wr)) if wr else float("nan")
            rewards = [r[f"{p}_reward"] for r in sub if r.get(f"{p}_reward") is not None]
            mrew = np.mean(rewards) if rewards else float("nan")
            out.append(f"| {excl} | {n} | {m:.3f} [{lo:.3f},{hi:.3f}] | {mp:.3f} | {mdiff:.3f} | {frac_wr:.3f} | {mrew:.3f} |")
        out.append("")

    out.append("## Zone 길이(단어수) per category, 그리고 SHIFT_real과의 상관")
    out.append("")
    out.append("| category | n | mean n_words | median n_words |")
    out.append("|---|---|---|---|")
    for cat in cat_names:
        sub = [r for r in recs if cat in r["tags"]]
        n = len(sub)
        if n == 0:
            out.append(f"| {cat} | 0 | - | - |")
            continue
        ws = [r["n_words"] for r in sub]
        out.append(f"| {cat} | {n} | {np.mean(ws):.1f} | {np.median(ws):.1f} |")
    out.append("")

    out.append("### Spearman(zone n_words, SHIFT_real) — 앵커별, 전체 revised 행")
    out.append("")
    out.append("| pair | rho | n |")
    out.append("|---|---|---|")
    for p in PAIRS:
        sub = [r for r in recs if r.get(f"{p}_shift_real") is not None]
        rho, n = spearman([r["n_words"] for r in sub], [r[f"{p}_shift_real"] for r in sub])
        out.append(f"| {p} | {rho:.3f} | {n} |")
    out.append("")

    print("\n".join(out))


if __name__ == "__main__":
    main()
