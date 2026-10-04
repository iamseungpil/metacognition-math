#!/usr/bin/env python
r"""build_self_traces — S6 «자기 프로토콜 증류»의 **데이터 단계**.

두 패스 프로토콜(시도 1 K개 → 합의 상태 게이트 → 리셋 재시도)이 얻은 것을 **한 패스**
궤적으로 접는다. gold 는 **데이터 경로 어디에도 쓰지 않는다** — 정답 자리에 정책 자신의
표본이 뽑은 **의사 라벨**(시도-1 답 ∪ 재시도 답의 다수결)을 둔다. gold 는 `--audit`
보고서(품질 계측)에서만 읽고 parquet 에는 **절대** 쓰지 않는다.

입력
  --attempt1 texts.jsonl   math_rollout 산출(문제당 K=8행; group_id/problem/text/final_answer…)
  --gens gens.jsonl        math_protocol_eval 산출(재시도 생성; protocol/problem_id/roll_id/…)

문제마다
  (i)   합의 상태 = `src.training.trial2.agreement_state`(gold 없음)
  (ii)  의사 라벨 = 시도-1 답 ∪ 그 프로토콜의 재시도 답의 다수결(`answers_equivalent` 군집,
        동률 → 재시도 표가 더 많은 군집, 그래도 같으면 먼저 나온 군집)
  (iii) 학생 궤적
        ALL_SAME(만장일치): 의사 라벨과 같은 시도-1 풀이 하나(절단 안 된 것 중 최단),
                            메타 구간 없음 → kind="direct"
        그 밖:              «자기 구조» 행 — 시도-1 답 ≠ 의사 라벨 이고 그 행의 재시도 답
                            == 의사 라벨 인 행 → 시도-1 텍스트(그대로) + 다리(BRIDGE_TMPL)
                            + 재시도 텍스트 → kind="rescue" (문제당 최대
                            --max_rescue_per_problem 개, 전체 길이가 짧은 것부터).
                            같은 문제에서 이미 의사 라벨을 맞힌 시도-1 행이 있으면 direct
                            한 줄도 함께 낸다 — 어려운 문제에서 두 행동을 모두 본다.

출력
  DIR/traces.parquet  messages / wrong_prefix / scenario / extra_info … (src/training/sft.py 가
                      읽는 컬럼). gold·r_corr 컬럼은 없다.
  DIR/stats.json      상태 × kind 개수, 의사 라벨 동의 비율 분포, (--audit 시) gold 대비 계측.

마스킹(`--mask_prefix`)
  none(기본) wrong_prefix="" · scenario="" → sft.py 는 프롬프트만 마스크한다. 학생은 «틀린
             시도 1 을 쓰고 → 알아채고 → 다시 푼다» 전체를 배운다(한 패스 재현이 목표).
  a1         rescue 행만 wrong_prefix=시도-1 텍스트 · scenario="redirect" → sft.py 의
             `_should_mask_prefix` 가 그 머리를 손실에서 뺀다. «틀린 풀이를 생산하는 것»은
             배우지 않고 다리 + 회복만 배운다. (E-093/0812c 의 교훈: 전부 마스크하면 메타
             앞 추론을 한 행도 가르치지 않는다 — 그래서 기본은 none 이다.)

사용(예):
  python scripts/local/build_self_traces.py \
      --attempt1 /hdd_data/.../train_pool/texts.jsonl \
      --gens     /hdd_data/.../proto_train_s1/gens.jsonl \
      --protocol gated_notx --out_dir /hdd_data/.../self_traces_v1 --audit
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
from src.metacot.prompt import META_END, META_START  # noqa: E402
from src.training.math_meta import answers_equivalent  # noqa: E402
from src.training.trial2 import agreement_state  # noqa: E402

VARIANT = "math_opt"
#: 다리 문장 — 자기 답 X 를 부정하고 문맥을 버린다는 **고정** 템플릿(내용 생성 없음).
BRIDGE_TMPL = ("\n\n{open}My answer \\boxed{{{x}}} may be wrong. Discarding the work above "
               "and starting over from scratch, and the answer is not {x}.{close}\n\n")
MASK_MODES = ("none", "a1")


# ── 의사 라벨 ─────────────────────────────────────────────────────────────────
def pseudo_label(a1_answers: Sequence[str], a2_answers: Sequence[str]) -> dict:
    """시도-1 답 ∪ 재시도 답의 다수결. 빈 답은 세지 않는다.

    동률이면 **재시도 표가 더 많은 군집**, 그래도 같으면 먼저 나온 군집.
    반환 {label, n_total, n_a1, n_a2, share} — share 는 전체 유효 답 중 이긴 군집의 몫,
    n_a1/n_a2 는 그 군집의 시도-1/재시도 구성원 수."""
    clusters: list[list] = []           # [대표, 총, 시도1, 재시도, 첫 등장]
    seq = [(str(x or "").strip(), 0) for x in a1_answers] + \
          [(str(x or "").strip(), 1) for x in a2_answers]
    n_total = 0
    for i, (s, side) in enumerate(seq):
        if not s:
            continue
        n_total += 1
        for cl in clusters:
            if answers_equivalent(cl[0], s):
                cl[1] += 1
                cl[2 + side] += 1
                break
        else:
            cl = [s, 1, 0, 0, i]
            cl[2 + side] = 1
            clusters.append(cl)
    if not clusters:
        return {"label": "", "n_total": 0, "n_a1": 0, "n_a2": 0, "share": 0.0}
    best = min(clusters, key=lambda cl: (-cl[1], -cl[3], cl[4]))
    return {"label": best[0], "n_total": n_total, "n_a1": best[2], "n_a2": best[3],
            "share": best[1] / n_total}


# ── 궤적 조립 ─────────────────────────────────────────────────────────────────
def bridge_text(x: str, meta_open: str = META_START, meta_close: str = META_END) -> str:
    return BRIDGE_TMPL.format(open=meta_open, close=meta_close, x=str(x or "").strip())


def _row(problem: str, target: str, *, kind: str, wrong_prefix: str, extra: dict) -> dict:
    """sft.py 가 읽는 행. gold 는 들어가지 않는다."""
    messages = [dict(m) for m in build_math_prompt(problem, VARIANT)]
    messages.append({"role": "assistant", "content": target})
    return {
        "messages": messages,
        "wrong_prefix": wrong_prefix,
        # ★scenario 는 sft._should_mask_prefix 가 보는 유일한 스위치. wrong_prefix 가 비면
        #   ""(= 프롬프트만 마스크)이어도 마스킹은 일어나지 않는다.
        "scenario": "redirect" if wrong_prefix else "",
        "kind": kind,
        "extra_info": extra,
    }


def build_problem_traces(prob: dict, retry_by_roll: dict, *, min_a2_share: float = 0.5,
                         max_rescue_per_problem: int = 2,
                         meta_open: str = META_START, meta_close: str = META_END,
                         mask_prefix: str = "none") -> tuple:
    """문제 하나 → (행 목록, 진단 dict). gold 는 읽지 않는다."""
    rows = prob["rows"]
    a1_answers = [r["answer"] for r in rows]
    a2_answers = [str(v.get("final_answer", "") or "").strip()
                  for _, v in sorted(retry_by_roll.items())]
    st = agreement_state(a1_answers, k=len(rows))["state"]
    pl = pseudo_label(a1_answers, a2_answers)
    diag = {"problem_id": prob["problem_id"], "state": st, **pl,
            "n_a1_rows": len(rows), "n_a2_rows": len(a2_answers),
            "n_rescue": 0, "n_direct": 0, "a2_share": 0.0}
    if not pl["label"]:
        return [], diag
    label = pl["label"]
    # 재시도 표 중 의사 라벨의 몫 — rescue 를 낼지의 문턱(재시도가 없으면 0).
    a2_hits = sum(1 for x in a2_answers if x and answers_equivalent(x, label))
    a2_valid = sum(1 for x in a2_answers if x)
    diag["a2_share"] = (a2_hits / a2_valid) if a2_valid else 0.0

    def _extra(kind: str) -> dict:
        return {"kind": kind, "agree_state": st, "pseudo_label": label,
                "n_a1": int(pl["n_a1"]), "n_a2": int(pl["n_a2"]),
                "share": float(pl["share"])}

    hits = [r for r in rows if r["answer"] and answers_equivalent(r["answer"], label)]
    direct_pool = sorted(hits, key=lambda r: (int(r.get("truncated", 0) or 0), len(r["text"])))
    out: list[dict] = []
    if st == "ALL_SAME":
        if direct_pool:
            out.append(_row(prob["problem"], direct_pool[0]["text"], kind="direct",
                            wrong_prefix="", extra=_extra("direct")))
            diag["n_direct"] = 1
        return out, diag

    # 비-만장일치: 자기 구조 행 + (있으면) direct 한 줄
    if diag["a2_share"] >= float(min_a2_share):
        cands = []
        for r in rows:
            a1 = r["answer"]
            if a1 and answers_equivalent(a1, label):
                continue                      # 이미 맞은 행은 구조가 아니다
            got = retry_by_roll.get(int(r["roll_id"]))
            if not got:
                continue
            a2 = str(got.get("final_answer", "") or "").strip()
            if not a2 or not answers_equivalent(a2, label):
                continue
            target = r["text"] + bridge_text(a1, meta_open, meta_close) + got.get("text", "")
            cands.append((len(target), int(r["roll_id"]), target, r))
        cands.sort(key=lambda t: (t[0], t[1]))
        for _, _, target, r in cands[:max(0, int(max_rescue_per_problem))]:
            out.append(_row(prob["problem"], target, kind="rescue",
                            wrong_prefix=(r["text"] if mask_prefix == "a1" else ""),
                            extra=_extra("rescue")))
        diag["n_rescue"] = len(out)
    if direct_pool:
        out.append(_row(prob["problem"], direct_pool[0]["text"], kind="direct",
                        wrong_prefix="", extra=_extra("direct")))
        diag["n_direct"] = 1
    return out, diag


# ── gens 읽기 ─────────────────────────────────────────────────────────────────
def load_retries(path: str, protocol: str) -> dict:
    """gens.jsonl → {problem_id: {roll_id: row}} — 그 프로토콜의 **답 행**만
    (mode == "label" 인 사전 패스는 답이 아니므로 버린다)."""
    out: dict = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("protocol") != protocol or r.get("mode") == "label":
                continue
            out.setdefault(r["problem_id"], {})[int(r["roll_id"])] = r
    return out


# ── 통계·감사 ─────────────────────────────────────────────────────────────────
def build_stats(diags: Sequence[dict], rows: Sequence[dict]) -> dict:
    by_state: dict = {}
    for d in diags:
        c = by_state.setdefault(d["state"], {"n_problems": 0, "direct": 0, "rescue": 0})
        c["n_problems"] += 1
        c["direct"] += d["n_direct"]
        c["rescue"] += d["n_rescue"]
    shares = sorted(d["share"] for d in diags if d["n_total"])
    def q(p):
        return shares[min(len(shares) - 1, int(p * len(shares)))] if shares else float("nan")
    return {
        "n_problems": len(diags),
        "n_rows": len(rows),
        "by_state": by_state,
        "kind_counts": dict(Counter(r["kind"] for r in rows)),
        "share": {"mean": (sum(shares) / len(shares)) if shares else float("nan"),
                  "p10": q(0.10), "p50": q(0.50), "p90": q(0.90),
                  "hist": dict(Counter(round(s, 1) for s in shares))},
    }


def audit_stats(problems: Sequence[dict], diags: Sequence[dict], rows: Sequence[dict],
                retries: dict) -> dict:
    """★gold 를 읽는 **유일한** 곳 — 보고서 전용. parquet 에는 들어가지 않는다."""
    gold = {p["problem_id"]: p["gold"] for p in problems}
    per_state: dict = {}
    for d in diags:
        c = per_state.setdefault(d["state"], {"n": 0, "n_correct": 0})
        c["n"] += 1
        g = gold.get(d["problem_id"], "")
        if d["label"] and answers_equivalent(d["label"], g):
            c["n_correct"] += 1
    for c in per_state.values():
        c["pseudo_acc"] = (c["n_correct"] / c["n"]) if c["n"] else float("nan")
    tc: dict = {}
    for r in rows:
        g = gold.get(r["extra_info"]["problem_id"], "")
        ok = bool(r["extra_info"]["pseudo_label"]
                  and answers_equivalent(r["extra_info"]["pseudo_label"], g))
        c = tc.setdefault(r["kind"], {"n": 0, "n_correct": 0})
        c["n"] += 1
        c["n_correct"] += int(ok)
    for c in tc.values():
        c["target_true_correct"] = (c["n_correct"] / c["n"]) if c["n"] else float("nan")
    n_retry_rows = sum(len(v) for v in retries.values())
    return {"pseudo_label_acc_by_state": per_state, "target_true_correct_by_kind": tc,
            "n_retry_rows": n_retry_rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempt1", required=True)
    ap.add_argument("--gens", required=True)
    ap.add_argument("--protocol", default="gated_notx")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--min_a2_share", type=float, default=0.5)
    ap.add_argument("--max_rescue_per_problem", type=int, default=2)
    ap.add_argument("--meta_open", default=META_START)
    ap.add_argument("--meta_close", default=META_END)
    ap.add_argument("--mask_prefix", default="none", choices=list(MASK_MODES))
    ap.add_argument("--limit_problems", type=int, default=0)
    ap.add_argument("--audit", action="store_true")
    a = ap.parse_args()

    import pandas as pd  # noqa: PLC0415

    problems = load_attempt1(a.attempt1, a.limit_problems)
    retries = load_retries(a.gens, a.protocol)
    rows: list[dict] = []
    diags: list[dict] = []
    for p in problems:
        rs, d = build_problem_traces(
            p, retries.get(p["problem_id"], {}), min_a2_share=a.min_a2_share,
            max_rescue_per_problem=a.max_rescue_per_problem,
            meta_open=a.meta_open, meta_close=a.meta_close, mask_prefix=a.mask_prefix)
        for r in rs:
            r["extra_info"]["problem_id"] = p["problem_id"]
        rows.extend(rs)
        diags.append(d)

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stats = build_stats(diags, rows)
    stats["meta"] = {"attempt1": a.attempt1, "gens": a.gens, "protocol": a.protocol,
                     "min_a2_share": a.min_a2_share,
                     "max_rescue_per_problem": a.max_rescue_per_problem,
                     "mask_prefix": a.mask_prefix, "variant": VARIANT}
    if a.audit:
        stats["audit"] = audit_stats(problems, diags, rows, retries)
    df = pd.DataFrame(rows)
    df.to_parquet(out / "traces.parquet", index=False)
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in stats.items() if k != "share"},
                     ensure_ascii=False, indent=2))
    print(f"[out] {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
