#!/usr/bin/env python
r"""자발적 답 수정(revision) 행위 계측 — CPU 전용.

입력은 롤아웃 jsonl(필드: text, gold, r_corr, n_tok, group_id/problem_id, final_answer).
«무엇이 수정인가»는 `src.training.revision.revision_zone` 이 정한다(학습기와 같은 정의).

    python scripts/local/math_revision_eval.py --rollouts texts.jsonl --out DIR [--compare base.jsonl]

산출: DIR/summary.json + DIR/summary.md — 행 수·정확도·수정 수/비율·w→r·r→w·정밀도
(w→r / 수정)·순 pp·토큰/행·수정 행의 길이 초과·합의 상태별 표·구간 텍스트의 습관 표식 표.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.training.math_meta import answers_equivalent, grade_math  # noqa: E402
from src.training.revision import revision_zone  # noqa: E402
from src.training.trial2 import agreement_state  # noqa: E402

# 습관 표식(구간 텍스트, 대소문자 무시). 어느 것도 안 맞으면 no_marker.
MARKERS: dict[str, str] = {
    "recheck": r"double[- ]check|re-?check|verify|let me check|confirm|re-?examine|recompute|re-?calculate|plug|substitut",
    "switch": r"alternatively|another (way|approach|method)|instead|different approach|other approach|let'?s try|try (a|another)",
    "doubt": r"\bwait\b|\bhmm\b|hold on|actually|but wait|oops",
    "error_naming": r"mistake|error|wrong|incorrect|typo|miscalculat|misread|misinterpret",
    "reread": r"re-?read|the (problem|question) (says|asks|states)|going back to the problem",
    "constraint": r"constraint|condition|domain|case|does(n'?t| not) satisf|check (if|whether)|boundary",
}
_MARKER_RE = {k: re.compile(v, re.IGNORECASE) for k, v in MARKERS.items()}


def load(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def analyze(rows: list[dict]) -> dict:
    n = len(rows)
    groups: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        gid = str(r.get("group_id") or r.get("problem_id") or i)
        groups.setdefault(gid, []).append(i)
    g_state = {}
    for gid, idxs in groups.items():
        finals = [str(rows[i].get("final_answer") or "") for i in idxs]
        g_state[gid] = str(agreement_state(finals)["state"])

    acc = sum(float(r.get("r_corr") or 0) for r in rows) / max(1, n)
    toks = [float(r.get("n_tok") or 0) for r in rows]

    revised, revised_strict = [], 0
    # ★0920(WHY_NO_GAIN_0920 §4): 박스를 두 번 이상 쓰고도 **같은 답**을 재확인한 행.
    #   누락 질량의 63.45% 가 여기다 — 의심은 하는데 못 바꾼다. 그 중 첫 박스가 오답인
    #   몫이 «가짜 확인» 질량이다.
    n_confirmed = n_confirmed_first_wrong = 0
    w2r = r2w = same = 0
    marker_counts: Counter = Counter()
    state_tab: dict[str, dict] = {}
    for i, r in enumerate(rows):
        gid = str(r.get("group_id") or r.get("problem_id") or i)
        st = g_state[gid]
        cell = state_tab.setdefault(st, {"rows": 0, "revised": 0, "w2r": 0, "r2w": 0})
        cell["rows"] += 1
        z = revision_zone(str(r.get("text") or ""))
        if z is None:
            continue
        gold0 = str(r.get("gold") or "")
        if not z["revised"]:
            n_confirmed += 1
            if not grade_math("\\boxed{%s}" % z["first_answer"], gold0):
                n_confirmed_first_wrong += 1
            continue
        revised.append(i)
        cell["revised"] += 1
        gold = str(r.get("gold") or "")
        if not answers_equivalent(z["first_answer"], z["last_answer"]):
            revised_strict += 1
        fc = bool(grade_math("\\boxed{%s}" % z["first_answer"], gold))
        lc = bool(grade_math("\\boxed{%s}" % z["last_answer"], gold))
        if lc and not fc:
            w2r += 1
            cell["w2r"] += 1
        elif fc and not lc:
            r2w += 1
            cell["r2w"] += 1
        else:
            same += 1
        zone_text = str(r.get("text") or "")[z["zone_start"]:z["zone_end"]]
        hit = False
        for name, rx in _MARKER_RE.items():
            if rx.search(zone_text):
                marker_counts[name] += 1
                hit = True
        if not hit:
            marker_counts["no_marker"] += 1

    nrev = len(revised)
    rev_tok = [toks[i] for i in revised] if revised else []
    other_tok = [toks[i] for i in range(n) if i not in set(revised)]
    return {
        "rows": n,
        "acc": acc,
        "revised": nrev,
        "revised_pct": 100.0 * nrev / max(1, n),
        "revised_strict": revised_strict,
        "revised_strict_pct": 100.0 * revised_strict / max(1, n),
        "w2r": w2r,
        "r2w": r2w,
        "unchanged_correctness": same,
        "precision": (w2r / nrev) if nrev else 0.0,
        "net_pp": 100.0 * (w2r - r2w) / max(1, n),
        "tok_per_row": (sum(toks) / max(1, n)),
        "tok_per_revised_row": (sum(rev_tok) / len(rev_tok)) if rev_tok else 0.0,
        "revised_len_excess": ((sum(rev_tok) / len(rev_tok)) - (sum(other_tok) / len(other_tok)))
        if rev_tok and other_tok else 0.0,
        "confirmed_same_answer": n_confirmed,
        "confirmed_same_answer_pct": 100.0 * n_confirmed / max(1, n),
        "confirmed_first_wrong": n_confirmed_first_wrong,
        "confirmed_first_wrong_share": (n_confirmed_first_wrong / n_confirmed)
        if n_confirmed else 0.0,
        "by_state": state_tab,
        "markers": dict(marker_counts),
    }


def to_md(s: dict, cmp: dict | None = None) -> str:
    def d(key, fmt="{:.4f}"):
        if cmp is None:
            return ""
        return "  (Δ " + fmt.format(s[key] - cmp[key]) + ")"

    out = ["# revision eval", ""]
    out += [
        f"- rows: {s['rows']}",
        f"- acc: {s['acc']:.4f}{d('acc')}",
        f"- revised (loose): {s['revised']} ({s['revised_pct']:.2f}%){d('revised_pct', '{:+.2f}pp')}",
        f"- revised (strict answers_equivalent): {s['revised_strict']} ({s['revised_strict_pct']:.2f}%)",
        f"- w->r: {s['w2r']}   r->w: {s['r2w']}   correctness unchanged: {s['unchanged_correctness']}",
        f"- precision (w->r / revised): {s['precision']:.4f}{d('precision')}",
        f"- net: {s['net_pp']:+.3f} pp",
        f"- confirmed same answer (>=2 boxes, unchanged): {s['confirmed_same_answer']} "
        f"({s['confirmed_same_answer_pct']:.2f}%), of which first box wrong "
        f"{s['confirmed_first_wrong']} ({100.0 * s['confirmed_first_wrong_share']:.1f}%) "
        f"= false-confirmation mass",
        f"- tok/row: {s['tok_per_row']:.1f} (revised rows {s['tok_per_revised_row']:.1f}, "
        f"excess {s['revised_len_excess']:+.1f})",
        "",
        "## by agreement state",
        "",
        "| state | rows | revised | rev% | w->r | r->w |",
        "|---|---|---|---|---|---|",
    ]
    for st in sorted(s["by_state"], key=lambda k: -s["by_state"][k]["rows"]):
        c = s["by_state"][st]
        pct = 100.0 * c["revised"] / max(1, c["rows"])
        out.append(f"| {st} | {c['rows']} | {c['revised']} | {pct:.2f}% | {c['w2r']} | {c['r2w']} |")
    out += ["", "## habit markers (zone text, multi-label)", "",
            "| marker | n | % of revised |", "|---|---|---|"]
    for k in list(MARKERS) + ["no_marker"]:
        v = s["markers"].get(k, 0)
        out.append(f"| {k} | {v} | {100.0 * v / max(1, s['revised']):.1f}% |")
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--compare", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    s = analyze(load(a.rollouts))
    cmp = analyze(load(a.compare)) if a.compare else None
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump({"summary": s, "compare": cmp}, fh, indent=2, ensure_ascii=False)
    md = to_md(s, cmp)
    with open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
