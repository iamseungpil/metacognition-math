#!/usr/bin/env python
r"""build_revision_traces — «습관 증류» 축 2단계의 **데이터 단계**.

교사도 학생도 **같은 정책**이다. 교사는 그 정책이 스스로 답을 고쳐서 **성공한** 롤아웃이고,
학생은 그 정책 자신이다. 가르치려는 것은 **답이 무엇인가**가 아니라 **언제 멈추고 다시
유도하는가** 이므로, 첫 박스까지의 머리는 손실에서 빼고(`wrong_prefix` + `scenario="redirect"`
→ `sft._should_mask_prefix`) **수정 구간과 그 뒤에만** 손실이 떨어진다.

왜 이 필터들인가(0916 계측)
  · 자발적 수정은 롤아웃의 ~2.9% — 그중 문제 안에서 첫 답이 틀렸던 비율 82.6%(비수정 30.3%),
    순이득 +0.95pp.
  · 이득은 ALL_SAME(+1.21pp) · DOMINANT(+0.80)에서 나오고 **SPLIT 에서는 −1.25pp 로 해롭다**
    → `--states` 기본값이 ALL_SAME,DOMINANT 인 이유(SPLIT/SCATTER/NOANS 제외).
  · **두 번째 수정**(변경점 2개 이상)은 정밀도 .535 로 동전던지기 → 변경점이 정확히 하나인
    롤아웃만 교사로 쓴다(나머지는 `second_revisions` 로 따로 센다).

라벨(`--label`)
  gold      첫 박스 오답 · 마지막 박스 정답(`grade_math`). gold 를 읽는다.
  majority  마지막 답 == 그 문제 8개 `final_answer` 의 다수결이고 첫 답은 아니다.
            ★이 모드에서는 gold 를 **데이터 경로 어디에서도 읽지 않는다** — `--audit`
            보고서에서만 읽고 parquet 에는 절대 쓰지 않는다.

출력
  DIR/traces.parquet  messages / wrong_prefix / scenario / kind / extra_info (sft.py 컬럼).
                      ★gold·r_corr 컬럼은 **없다**.
  DIR/stats.json      필터 단계별 개수(깔때기) · 상태별 유지 수 · 마커율 · 응답 길이 분위 ·
                      대표 문제 수 · (--audit 시) 상태별 gold 정밀도.

사용(예):
  python scripts/local/build_revision_traces.py \
      --rollouts /hdd_data/seungpil/scratch/eval/mathL5_q3i2507_opt_b8k/texts.jsonl \
      --out_dir  /hdd_data/seungpil/scratch/data/revision_v1 --label majority --audit
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence

os.environ.setdefault("TMPDIR", "/hdd_data/seungpil/tmp")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from math_protocol_eval import load_attempt1  # noqa: E402
from src.metacot.math_meta_prompt import build_math_prompt  # noqa: E402
from src.training.math_meta import answers_equivalent, boxed_spans, grade_math  # noqa: E402
from src.training.trial2 import agreement_state  # noqa: E402

VARIANT = "math_opt"
LABEL_MODES = ("gold", "majority")
DEFAULT_STATES = ("ALL_SAME", "DOMINANT")
DEFAULT_TOKENIZER = "/hdd_data/seungpil/scratch/models/Qwen3-4B-Instruct-2507"
#: 문서화된 «수정 신호» 어휘 — 첫 박스와 마지막 박스 **사이** 구간에서 소문자로 찾는다.
#: (`--require_marker` 가 없어도 마커 유무는 언제나 세어 stats 에 남긴다.)
REVISION_MARKERS = ("wait", "actually", "re-check", "recheck", "double-check", "mistake",
                    "wrong", "hold on", "oops", "typo", "hmm", "try again")
#: 깔때기 단계 — 이 순서대로 떨어뜨리고 이 순서대로 보고한다.
FUNNEL_STEPS = ("n_rollouts", "drop_truncated", "drop_lt2_boxes", "drop_first_equiv_last",
                "drop_multi_change", "drop_not_improved", "drop_state", "drop_no_marker",
                "drop_cap", "n_kept")


# ── 답 계열 ───────────────────────────────────────────────────────────────────
def answer_runs(answers: Sequence[str]) -> list[str]:
    """연속 동치 답을 하나로 접은 **순서 있는 군집 계열**. 길이−1 = 변경점 수.

    4,4,5,5 → [4,5](변경점 1) · 4,5,4 → [4,5,4](변경점 2, 두 번째 수정이라 기각)."""
    out: list[str] = []
    for a in answers:
        s = str(a or "").strip()
        if out and answers_equivalent(out[-1], s):
            continue
        out.append(s)
    return out


def find_marker(segment: str) -> str:
    """`segment` 에 처음 나타나는 REVISION_MARKERS 원소(없으면 "")."""
    low = (segment or "").lower()
    hits = [(low.find(m), m) for m in REVISION_MARKERS if m in low]
    return min(hits)[1] if hits else ""


# ── 문제 하나 ─────────────────────────────────────────────────────────────────
def build_problem_rows(prob: dict, *, label_mode: str = "majority",
                       states: Sequence[str] = DEFAULT_STATES, max_per_problem: int = 2,
                       require_marker: bool = False, tokenizer=None) -> tuple[list[dict], dict]:
    """문제 하나 → (교사 행 목록, 깔때기 진단).

    ★`label_mode == "majority"` 면 `prob["gold"]` 를 **한 번도 읽지 않는다**."""
    if label_mode not in LABEL_MODES:
        raise ValueError(f"[REVISION] --label {label_mode!r} 는 {'|'.join(LABEL_MODES)} 중 하나.")
    rows = prob["rows"]
    answers = [r["answer"] for r in rows]
    agree = agreement_state(answers, k=len(rows))
    st = agree["state"]
    gold = str(prob["gold"]) if label_mode == "gold" else ""
    # ★majority_answer 중복 제거(0918) — agreement_state 의 dominant_answer 와 동일한
    #   군집·동률 규칙(최대 군집, 동률 → 첫 등장, 빈 답 제외)이므로 로컬 복사본을 없앤다.
    maj = agree["dominant_answer"] if label_mode == "majority" else ""
    diag = {k: 0 for k in FUNNEL_STEPS}
    diag.update({"problem_id": prob["problem_id"], "state": st,
                 "second_revisions": 0, "n_candidates": 0, "n_candidates_with_marker": 0})
    diag["n_rollouts"] = len(rows)
    state_ok = st in tuple(states)

    cands: list[dict] = []
    for r in rows:
        text = r["text"] or ""
        if int(r.get("truncated", 0) or 0):
            diag["drop_truncated"] += 1
            continue
        boxes = boxed_spans(text)
        if len(boxes) < 2:
            diag["drop_lt2_boxes"] += 1
            continue
        first, last = boxes[0], boxes[-1]
        if answers_equivalent(first[0], last[0]):
            diag["drop_first_equiv_last"] += 1
            continue
        runs = answer_runs([b[0] for b in boxes])
        if len(runs) != 2:
            diag["drop_multi_change"] += 1
            diag["second_revisions"] += 1
            continue
        if label_mode == "gold":
            improved = (grade_math(f"\\boxed{{{first[0]}}}", gold) == 0
                        and grade_math(f"\\boxed{{{last[0]}}}", gold) == 1)
        else:
            improved = bool(maj) and answers_equivalent(last[0], maj) \
                and not answers_equivalent(first[0], maj)
        if not improved:
            diag["drop_not_improved"] += 1
            continue
        if not state_ok:
            diag["drop_state"] += 1
            continue
        marker = find_marker(text[first[2]:last[1]])
        diag["n_candidates"] += 1
        diag["n_candidates_with_marker"] += int(bool(marker))
        if require_marker and not marker:
            diag["drop_no_marker"] += 1
            continue
        cands.append({"row": r, "text": text, "first": first, "last": last,
                      "marker": marker, "n_boxes": len(boxes)})

    # 상한: **전체 응답이 짧은 것부터**(문자 수; 동률은 roll_id) — 긴 궤적이 코퍼스를
    # 독식하지 않게, 그리고 어느 GPU 에서 돌려도 같은 결과가 나오게 결정적으로 고른다.
    cands.sort(key=lambda c: (len(c["text"]), int(c["row"]["roll_id"])))
    keep = cands[:max(0, int(max_per_problem))]
    diag["drop_cap"] = len(cands) - len(keep)
    diag["n_kept"] = len(keep)

    out: list[dict] = []
    for c in keep:
        text, first, last = c["text"], c["first"], c["last"]
        prefix = text[:first[2]]                       # ★첫 박스 닫는 중괄호까지 포함
        seg = text[first[2]:]                          # 손실이 떨어지는 구간(수정 + 그 뒤)
        extra = {"problem_id": prob["problem_id"], "roll_id": int(c["row"]["roll_id"]),
                 "agree_state": st, "n_boxes": int(c["n_boxes"]),
                 "first_answer": first[0], "last_answer": last[0],
                 "has_marker": bool(c["marker"]), "marker": c["marker"],
                 "label_mode": label_mode}
        if tokenizer is not None:
            extra["seg_tokens"] = int(len(tokenizer.encode(seg, add_special_tokens=False)))
        else:
            # ★토크나이저가 없으면 토큰 수를 **추정하지 않는다**(문자 수는 토큰이 아니다).
            extra["seg_chars"] = int(len(seg))
        messages = [dict(m) for m in build_math_prompt(prob["problem"], VARIANT)]
        messages.append({"role": "assistant", "content": text})   # 원문 그대로
        out.append({"messages": messages, "wrong_prefix": prefix, "scenario": "redirect",
                    "kind": "revision", "extra_info": extra})
    return out, diag


# ── 통계·감사 ─────────────────────────────────────────────────────────────────
def build_stats(diags: Sequence[dict], rows: Sequence[dict], *, unit: str = "chars",
                lengths: Sequence[int] = ()) -> dict:
    funnel = {k: sum(int(d[k]) for d in diags) for k in FUNNEL_STEPS}
    kept_by_state: dict = {}
    for r in rows:
        kept_by_state[r["extra_info"]["agree_state"]] = \
            kept_by_state.get(r["extra_info"]["agree_state"], 0) + 1
    n_cand = sum(int(d["n_candidates"]) for d in diags)
    n_cand_marker = sum(int(d["n_candidates_with_marker"]) for d in diags)
    xs = sorted(int(x) for x in lengths)

    def q(p: float):
        return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else float("nan")

    return {
        "n_problems": len(diags),
        "n_problems_represented": len({r["extra_info"]["problem_id"] for r in rows}),
        "n_rows": len(rows),
        "funnel": funnel,
        "problems_by_state": dict(Counter(d["state"] for d in diags)),
        "kept_by_state": kept_by_state,
        "second_revisions": sum(int(d["second_revisions"]) for d in diags),
        # 마커 필터를 켰든 껐든 **양쪽 수**를 남긴다 — 필터의 대가를 언제나 읽을 수 있게.
        "marker": {"n_candidates": n_cand, "n_with_marker": n_cand_marker,
                   "rate": (n_cand_marker / n_cand) if n_cand else float("nan"),
                   "n_kept_if_required": n_cand_marker, "n_kept_if_not_required": n_cand,
                   "markers": list(REVISION_MARKERS)},
        "response_len": {"unit": unit, "p10": q(0.10), "p50": q(0.50), "p90": q(0.90)},
    }


def audit_stats(problems: Sequence[dict], rows: Sequence[dict]) -> dict:
    """★gold 를 읽는 **유일한** 곳 — 보고 전용(parquet 에는 안 들어간다).

    유지된 교사 궤적의 «정밀도» = 마지막 답이 gold 와 맞는 비율(상태별). first_wrong_last_right
    는 «정말로 오답→정답 으로 고쳤는가» 로 더 엄한 잣대다."""
    gold = {p["problem_id"]: p["gold"] for p in problems}
    per_state: dict = {}
    for r in rows:
        e = r["extra_info"]
        g = gold.get(e["problem_id"], "")
        last_ok = bool(answers_equivalent(e["last_answer"], g))
        first_ok = bool(answers_equivalent(e["first_answer"], g))
        c = per_state.setdefault(e["agree_state"],
                                 {"n": 0, "n_last_correct": 0, "n_first_correct": 0,
                                  "n_first_wrong_last_right": 0})
        c["n"] += 1
        c["n_last_correct"] += int(last_ok)
        c["n_first_correct"] += int(first_ok)
        c["n_first_wrong_last_right"] += int(last_ok and not first_ok)
    for c in per_state.values():
        c["precision"] = c["n_last_correct"] / c["n"] if c["n"] else float("nan")
        c["true_improvement"] = c["n_first_wrong_last_right"] / c["n"] if c["n"] else float("nan")
    n = sum(c["n"] for c in per_state.values())
    n_ok = sum(c["n_last_correct"] for c in per_state.values())
    return {"by_state": per_state, "precision": (n_ok / n) if n else float("nan"), "n_rows": n}


def funnel_line(stats: dict, *, label_mode: str, require_marker: bool) -> str:
    f = stats["funnel"]
    return (f"[revision] rollouts={f['n_rollouts']} -trunc={f['drop_truncated']} "
            f"-boxes<2={f['drop_lt2_boxes']} -same={f['drop_first_equiv_last']} "
            f"-2ndrev={f['drop_multi_change']} -notimproved={f['drop_not_improved']} "
            f"-state={f['drop_state']} -nomarker={f['drop_no_marker']} -cap={f['drop_cap']} "
            f"→ kept={f['n_kept']} over {stats['n_problems_represented']} problems "
            f"(label={label_mode}, require_marker={int(require_marker)}, "
            f"marker_rate={stats['marker']['rate']:.3f})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True, help="math_rollout 산출 texts.jsonl")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--label", default="majority", choices=list(LABEL_MODES))
    ap.add_argument("--states", default=",".join(DEFAULT_STATES),
                    help="쉼표 목록. 기본 ALL_SAME,DOMINANT — SPLIT 에서 수정은 순해롭다.")
    ap.add_argument("--max_per_problem", type=int, default=2)
    ap.add_argument("--require_marker", action="store_true")
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER,
                    help="seg_tokens 용. 못 열면 seg_chars 로 떨어뜨리고 stats 에 적는다.")
    ap.add_argument("--limit_problems", type=int, default=0)
    ap.add_argument("--audit", action="store_true")
    a = ap.parse_args()

    import pandas as pd  # noqa: PLC0415

    tok = None
    try:
        from transformers import AutoTokenizer  # noqa: PLC0415
        tok = AutoTokenizer.from_pretrained(a.tokenizer)
    except Exception as exc:                     # noqa: BLE001
        print(f"[revision][warn] tokenizer 없음({exc}) → seg_chars 로 기록한다.", flush=True)

    states = tuple(s.strip() for s in a.states.split(",") if s.strip())
    problems = load_attempt1(a.rollouts, a.limit_problems)
    rows: list[dict] = []
    diags: list[dict] = []
    lengths: list[int] = []
    for p in problems:
        rs, d = build_problem_rows(p, label_mode=a.label, states=states,
                                   max_per_problem=a.max_per_problem,
                                   require_marker=a.require_marker, tokenizer=tok)
        for r in rs:
            body = r["messages"][-1]["content"]
            lengths.append(len(tok.encode(body, add_special_tokens=False)) if tok else len(body))
        rows.extend(rs)
        diags.append(d)

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stats = build_stats(diags, rows, unit=("tokens" if tok else "chars"), lengths=lengths)
    stats["meta"] = {"rollouts": a.rollouts, "label": a.label, "states": list(states),
                     "max_per_problem": a.max_per_problem, "require_marker": a.require_marker,
                     "variant": VARIANT, "seg_len_unit": ("tokens" if tok else "chars"),
                     "tokenizer": (a.tokenizer if tok else "")}
    if a.audit:
        stats["audit"] = audit_stats(problems, rows)
    pd.DataFrame(rows).to_parquet(out / "traces.parquet", index=False)
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    print(funnel_line(stats, label_mode=a.label, require_marker=a.require_marker), flush=True)
    print(f"[out] {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
