#!/usr/bin/env python
r"""build_decision_traces — 설계 C «발화 = 결정, 실행 = 리셋» 의 **데이터 단계**.

정책의 K-표본 합의(내부 신호, gold 없음)를 **한 줄의 메타 발화**로 증류한다. 학생이
배우는 것은 풀이가 아니라 **자기 풀이를 읽고 내리는 판단**이다:

  target = 시도-1 텍스트(그대로) + `decision.utterance(label, x=그 행의 자기 답)`

그리고 시도-1 텍스트는 `wrong_prefix` 로 넣고 `scenario="redirect"` 로 표시해
`sft._should_mask_prefix` 가 그 머리를 **손실에서 뺀다** — 손실은 오직 발화 토큰에만
떨어진다. ★이것이 이 코퍼스의 핵심 설계다: «자기 풀이를 재생산하는 법»이 아니라
«자기 풀이를 읽는 법»을 가르친다(build_self_traces 의 기본 mask_prefix=none 과 정반대).

라벨(gold 없음)
  의사 라벨 = `build_self_traces.pseudo_label`(시도-1 답 ∪ 그 프로토콜의 재시도 답 다수결).
  재시도가 한 줄도 없는 문제(만장일치라 게이트가 안 열린 경우)는 시도-1 다수결이 그대로
  의사 라벨이 된다. 행 라벨 = `decision.decision_label(자기 답, 의사 라벨)`.
  **모든** 시도-1 행이 한 줄씩 나온다(만장일치 문제 포함) — 발화는 «가끔 하는 것»이 아니라
  매 풀이 끝의 습관이어야 하고, 만장일치에서 commit 을 못 배우면 발화가 항상 restart 로
  퇴화한다.

출력
  DIR/traces.parquet  messages / wrong_prefix / scenario / kind / extra_info (sft.py 컬럼).
                      ★gold·r_corr 컬럼은 **없다**.
  DIR/stats.json      상태 × 라벨 개수, share 분포, (--audit 시) 상태별 라벨-대-gold 일치.

--balance per_state
  상태마다 commit 행을 **그 상태의 restart 행 수의 3배**까지로 줄인다(결정적 표집).
  왜: 만장일치 문제는 ~99% 가 commit 이라 그대로 두면 발화의 사전확률이 commit 으로
  쏠려 «항상 commit» 이 최적해가 된다(판별력 AUC = .5).
  ⚠️한 상태에 restart 행이 하나도 없으면 그 상태는 **통째로 비워진다**(3×0 = 0).
  ALL_SAME 은 정의상 K 개 답이 모두 동치라 restart 가 0 이므로 이 옵션은 ALL_SAME 을
  전부 버린다 — 그래서 기본값은 `none` 이고, stats 에 버려진 수를 남긴다.

--balance global
  commit 총량을 restart **총량**의 `--balance_ratio`(기본 3.0)배로 **전역** 상한을 걸고,
  그 예산을 상태별 commit 점유율에 비례해 나눈 뒤 문제 단위로 고르게(라운드-로빈) 채운다.
  `per_state` 와 달리 ALL_SAME 처럼 자기 상태의 restart 가 0 이어도 다른 상태의 restart 가
  채워주는 전역 예산 안에서는 계속 살아남는다 — 다만 원래 점유율이 가장 크므로 절대량으로는
  가장 많이 깎인다.

사용(예):
  python scripts/local/build_decision_traces.py \
      --attempt1 /hdd_data/seungpil/scratch/eval/mathTRAINscreen_q3i2507_opt_b8k/texts.jsonl \
      --gens     /hdd_data/seungpil/scratch/eval/s6_train_retries_b1/gens.jsonl \
      --protocol gated_notx --out_dir /hdd_data/seungpil/scratch/data/decision_v1 --audit
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence

os.environ.setdefault("TMPDIR", "/hdd_data/seungpil/tmp")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_self_traces import VARIANT, load_retries, pseudo_label  # noqa: E402
from math_protocol_eval import load_attempt1  # noqa: E402
from src.metacot.math_meta_prompt import build_math_prompt  # noqa: E402
from src.training.decision import decision_label, utterance  # noqa: E402
from src.training.math_meta import answers_equivalent  # noqa: E402
from src.training.trial2 import agreement_state  # noqa: E402

BALANCE_MODES = ("none", "per_state", "global")
#: per_state 균형의 상한 배수 — commit ≤ COMMIT_CAP_RATIO × restart.
COMMIT_CAP_RATIO = 3
#: global 균형의 기본 상한 배수 — 총 commit ≤ DEFAULT_GLOBAL_RATIO × 총 restart.
DEFAULT_GLOBAL_RATIO = 3.0


def build_problem_rows(prob: dict, retry_by_roll: dict) -> tuple[list[dict], dict]:
    """문제 하나 → (행 목록, 진단). gold 는 읽지 않는다. 시도-1 행마다 정확히 한 줄."""
    rows = prob["rows"]
    a1_answers = [r["answer"] for r in rows]
    a2_answers = [str(v.get("final_answer", "") or "").strip()
                  for _, v in sorted(retry_by_roll.items())]
    st = agreement_state(a1_answers, k=len(rows))["state"]
    pl = pseudo_label(a1_answers, a2_answers)
    diag = {"problem_id": prob["problem_id"], "state": st, "label": pl["label"],
            "share": float(pl["share"]), "n_a1_rows": len(rows),
            "n_a2_rows": len(a2_answers), "commit": 0, "restart": 0}
    out: list[dict] = []
    for r in rows:
        x = str(r["answer"] or "").strip()
        lab = decision_label(x, pl["label"])
        # ★불변: decision_label 은 빈 답을 절대 commit 으로 매기지 않는다 — commit 은
        # 빈 x 를 utterance() 가 ValueError 로 거부하므로, 여기서 어긋나면 즉시 죽는다.
        assert not (lab == "commit" and not x), \
            f"[DECISION] invariant broken: empty answer labeled commit ({prob['problem_id']})"
        diag[lab] += 1
        target = r["text"] + utterance(lab, x)
        messages = [dict(m) for m in build_math_prompt(prob["problem"], VARIANT)]
        messages.append({"role": "assistant", "content": target})
        out.append({
            "messages": messages,
            # ★시도-1 본문을 손실에서 빼는 유일한 배선 — 이 둘이 같이 있어야 한다.
            "wrong_prefix": r["text"],
            "scenario": "redirect",
            "kind": lab,
            "extra_info": {"agree_state": st, "pseudo_label": pl["label"],
                           "share": float(pl["share"]), "problem_id": prob["problem_id"],
                           "label": lab, "roll_id": int(r["roll_id"])},
        })
    return out, diag


def _apportion(counts: dict[str, int], cap: int) -> dict[str, int]:
    """`cap` 을 `counts` 의 점유율에 비례해 정수로 나눈다(최대 나머지법).

    나머지 배분의 동점은 상태 이름으로 끊어 시드와 무관하게 결정적이다 — 시드는
    (몇 개를 남기느냐 가 아니라) *어느 행을* 남기느냐에만 쓴다."""
    total = sum(counts.values())
    if total <= 0 or cap <= 0:
        return {st: 0 for st in counts}
    ideal = {st: cap * (n / total) for st, n in counts.items()}
    alloc = {st: int(ideal[st]) for st in counts}
    remainder = cap - sum(alloc.values())
    order = sorted(counts, key=lambda st: (-(ideal[st] - alloc[st]), st))
    for st in order[:remainder]:
        alloc[st] += 1
    return {st: min(alloc[st], counts[st]) for st in counts}


def _spread_by_problem(rows: Sequence[dict], keep_n: int, *, seed_key: str) -> list[dict]:
    """`rows`(한 상태의 commit 행들) 에서 `keep_n` 개를, 문제 단위로 고르게(라운드-로빈)
    골라낸다 — 모든 문제가 1개씩 받은 뒤에야 어떤 문제가 2개째를 받는다. 문제 처리
    순서는 `seed_key` 로 섞어 시드가 바뀌면 어느 문제가 먼저 채워지는지도 바뀐다."""
    by_problem: dict[str, list[dict]] = {}
    for r in rows:
        by_problem.setdefault(str(r["extra_info"]["problem_id"]), []).append(r)
    for pid, rs in by_problem.items():
        rs.sort(key=lambda r: int(r["extra_info"].get("roll_id", 0)))
    prob_order = sorted(by_problem)
    random.Random(seed_key).shuffle(prob_order)
    max_len = max((len(rs) for rs in by_problem.values()), default=0)
    flattened = [by_problem[pid][i] for i in range(max_len) for pid in prob_order
                 if i < len(by_problem[pid])]
    return flattened[:keep_n]


def balance_rows(rows: Sequence[dict], mode: str = "none", *, seed: int = 0,
                 ratio: int = COMMIT_CAP_RATIO,
                 balance_ratio: float = DEFAULT_GLOBAL_RATIO) -> tuple[list[dict], dict]:
    """`per_state` 면 상태별 commit 을 restart 의 `ratio` 배까지, `global` 이면 **전체**
    commit 을 **전체** restart 의 `balance_ratio` 배까지 **결정적으로** 줄인다.
    반환 (남은 행, 균형 보고서)."""
    if mode not in BALANCE_MODES:
        raise ValueError(f"[DECISION] --balance {mode!r} 는 {'|'.join(BALANCE_MODES)} 중 하나.")
    if mode == "none":
        return list(rows), {}
    if mode == "global":
        commit_rows = [r for r in rows if r["kind"] == "commit"]
        restart_rows = [r for r in rows if r["kind"] == "restart"]
        by_state: dict[str, list[dict]] = {}
        for r in commit_rows:
            by_state.setdefault(r["extra_info"]["agree_state"], []).append(r)
        cap = int(balance_ratio * len(restart_rows))
        alloc = _apportion({st: len(rs) for st, rs in by_state.items()}, cap) \
            if len(commit_rows) > cap else {st: len(rs) for st, rs in by_state.items()}
        kept_commit: list[dict] = []
        by_state_report: dict[str, dict] = {}
        restart_by_state = Counter(r["extra_info"]["agree_state"] for r in restart_rows)
        for st, rs in by_state.items():
            picked = _spread_by_problem(rs, alloc[st], seed_key=f"{seed}|global|{st}")
            kept_commit.extend(picked)
            by_state_report[st] = {"commit_before": len(rs), "commit_kept": len(picked),
                                   "restart": restart_by_state.get(st, 0)}
        for st, n in restart_by_state.items():
            by_state_report.setdefault(st, {"commit_before": 0, "commit_kept": 0, "restart": n})
        report = {"ratio": balance_ratio, "n_commit_before": len(commit_rows),
                 "n_commit_kept": len(kept_commit), "n_restart": len(restart_rows),
                 "by_state": by_state_report}
        keep_ids = {id(r) for r in kept_commit} | {id(r) for r in restart_rows}
        return [r for r in rows if id(r) in keep_ids], report
    by_state: dict[str, dict[str, list[dict]]] = {}
    for r in rows:
        st = r["extra_info"]["agree_state"]
        by_state.setdefault(st, {"commit": [], "restart": []})[r["kind"]].append(r)
    keep_ids: set[int] = set()
    report: dict = {}
    for st, d in by_state.items():
        cap = ratio * len(d["restart"])
        # 결정적 표집: (problem_id, roll_id) 로 정렬한 뒤 상태별 시드로 뽑는다.
        pool = sorted(d["commit"],
                      key=lambda r: (str(r["extra_info"]["problem_id"]),
                                     int(r["extra_info"].get("roll_id", 0))))
        pick = pool if len(pool) <= cap else random.Random(f"{seed}|{st}").sample(pool, cap)
        keep_ids.update(id(r) for r in pick)
        keep_ids.update(id(r) for r in d["restart"])
        report[st] = {"restart": len(d["restart"]), "kept_commit": len(pick),
                      "dropped_commit": len(pool) - len(pick)}
    return [r for r in rows if id(r) in keep_ids], report


def build_stats(diags: Sequence[dict], rows: Sequence[dict]) -> dict:
    by_state: dict = {}
    for d in diags:
        c = by_state.setdefault(d["state"], {"n_problems": 0, "commit": 0, "restart": 0})
        c["n_problems"] += 1
        c["commit"] += d["commit"]
        c["restart"] += d["restart"]
    # 실제로 parquet 에 남은 행(균형 이후)의 상태 × 라벨
    kept: dict = {}
    for r in rows:
        c = kept.setdefault(r["extra_info"]["agree_state"], {"commit": 0, "restart": 0})
        c[r["kind"]] += 1
    shares = sorted(float(d["share"]) for d in diags if d["label"])

    def q(p: float) -> float:
        return shares[min(len(shares) - 1, int(p * len(shares)))] if shares else float("nan")

    return {
        "n_problems": len(diags),
        "n_rows": len(rows),
        "label_counts_by_state": by_state,
        "kept_rows_by_state": kept,
        "kind_counts": dict(Counter(r["kind"] for r in rows)),
        "share": {"mean": (sum(shares) / len(shares)) if shares else float("nan"),
                  "p10": q(0.10), "p50": q(0.50), "p90": q(0.90)},
    }


def audit_stats(problems: Sequence[dict], rows: Sequence[dict]) -> dict:
    """★gold 를 읽는 **유일한** 곳 — 보고 전용(parquet 에는 안 들어간다).

    «참 라벨» = 자기 답이 gold 와 같으면 commit, 아니면 restart. 우리 라벨과의 일치율을
    상태별로 낸다 — 라벨 잡음의 상한을 읽는 자다."""
    gold = {p["problem_id"]: p["gold"] for p in problems}
    a1: dict = {}
    for p in problems:
        for r in p["rows"]:
            a1[(p["problem_id"], int(r["roll_id"]))] = str(r["answer"] or "").strip()
    per_state: dict = {}
    for r in rows:
        e = r["extra_info"]
        x = a1.get((e["problem_id"], int(e.get("roll_id", 0))), "")
        truth = "commit" if (x and answers_equivalent(x, gold.get(e["problem_id"], ""))) else \
            "restart"
        c = per_state.setdefault(e["agree_state"],
                                 {"n": 0, "n_agree": 0, "n_true_commit": 0, "n_pred_commit": 0})
        c["n"] += 1
        c["n_agree"] += int(truth == r["kind"])
        c["n_true_commit"] += int(truth == "commit")
        c["n_pred_commit"] += int(r["kind"] == "commit")
    for c in per_state.values():
        c["label_vs_gold_agreement"] = (c["n_agree"] / c["n"]) if c["n"] else float("nan")
    n = sum(c["n"] for c in per_state.values())
    n_ok = sum(c["n_agree"] for c in per_state.values())
    return {"by_state": per_state,
            "label_vs_gold_agreement": (n_ok / n) if n else float("nan"), "n_rows": n}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempt1", required=True)
    ap.add_argument("--gens", required=True)
    ap.add_argument("--protocol", default="gated_notx")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--balance", default="none", choices=list(BALANCE_MODES))
    ap.add_argument("--balance_ratio", type=float, default=DEFAULT_GLOBAL_RATIO,
                    help="--balance global 전용: 총 commit ≤ ratio × 총 restart.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit_problems", type=int, default=0)
    ap.add_argument("--audit", action="store_true")
    a = ap.parse_args()

    import pandas as pd  # noqa: PLC0415

    problems = load_attempt1(a.attempt1, a.limit_problems)
    retries = load_retries(a.gens, a.protocol)
    rows: list[dict] = []
    diags: list[dict] = []
    for p in problems:
        rs, d = build_problem_rows(p, retries.get(p["problem_id"], {}))
        rows.extend(rs)
        diags.append(d)
    rows, bal = balance_rows(rows, a.balance, seed=a.seed, balance_ratio=a.balance_ratio)

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stats = build_stats(diags, rows)
    if a.balance == "global":
        stats["balance"] = {"mode": a.balance, **bal}
    else:
        stats["balance"] = {"mode": a.balance, "ratio": COMMIT_CAP_RATIO, "by_state": bal}
    stats["meta"] = {"attempt1": a.attempt1, "gens": a.gens, "protocol": a.protocol,
                     "variant": VARIANT, "seed": a.seed}
    if a.audit:
        stats["audit"] = audit_stats(problems, rows)
    df = pd.DataFrame(rows)
    df.to_parquet(out / "traces.parquet", index=False)
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"[out] {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
