#!/usr/bin/env python
r"""요약·평가 — 단일 패스 짝비교. mc/ 의 **모든** 지표가 여기서 나온다(rollout 도 부른다).

`--protocol single` 한 패스(`--max_tokens`, 기본 12,288) · `--regrade_only` 생성 없이 재채점.
지표(구 `scripts/local/math_revision_eval.py` 와 같은 자): revised = 박스 2개 이상 ∧ 첫 답과
마지막 답이 loose 동치 아님 · precision = w→r/revised · net_pp = 100·(w→r − r→w)/행 ·
`--ref` = 문제 짝 부트스트랩 CI(2,000회). 2턴(NOTICE) 요약·부분집합 표는 수정 41 에서 삭제
(생성 경로는 0924 에 이미 없음; 백업 /hdd_data/seungpil/tmp/eval.py.pre41).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mc.context import agreement_state
from mc.grade import grade_answer, grade_math, revision_zone

BOOTSTRAP_N = 2000


def _q(xs, p):
    if not xs:
        return float("nan")
    ys = sorted(xs)
    i = min(len(ys) - 1, max(0, int(round(p * (len(ys) - 1)))))
    return ys[i]


def paired_bootstrap(per_problem: list[tuple[float, float]], n: int = BOOTSTRAP_N,
                     seed: int = 20260921) -> dict:
    """문제 단위 짝 부트스트랩 — (a, b) 문제별 평균쌍에서 b − a 의 CI."""
    if not per_problem:
        return {"delta": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = random.Random(seed)
    m = len(per_problem)
    obs = sum(b - a for a, b in per_problem) / m
    ds = []
    for _ in range(n):
        s = 0.0
        for _ in range(m):
            a, b = per_problem[rng.randrange(m)]
            s += b - a
        ds.append(s / m)
    return {"delta": obs, "lo": _q(ds, 0.025), "hi": _q(ds, 0.975), "n": m,
            "excludes_zero": bool(_q(ds, 0.025) > 0 or _q(ds, 0.975) < 0)}


# ── 단일 패스 요약(자발 수정 계측) ────────────────────────────────────────────
def summarize_single(rows: list[dict]) -> dict:
    """rows: text/gold/n_tok/group_id(또는 problem_id). 채점은 mc.grade 로 **다시** 한다."""
    n = len(rows)
    groups: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        groups.setdefault(str(r.get("group_id") or r.get("problem_id") or i), []).append(i)
    g_state = {g: str(agreement_state([str(rows[i].get("final_answer") or "")
                                       for i in idx])["state"])
               for g, idx in groups.items()}
    acc_flags = [float(grade_math(str(r.get("text") or ""), str(r.get("gold") or "")))
                 for r in rows]
    revised = w2r = r2w = same = confirmed = 0
    state_tab: dict[str, dict] = {}
    for i, r in enumerate(rows):
        gid = str(r.get("group_id") or r.get("problem_id") or i)
        cell = state_tab.setdefault(g_state[gid], {"rows": 0, "revised": 0, "w2r": 0, "r2w": 0})
        cell["rows"] += 1
        z = revision_zone(str(r.get("text") or ""))
        if z is None:
            continue
        first, last = z["first_answer"], z["last_answer"]
        if not z["revised"]:
            confirmed += 1
            continue
        revised += 1
        cell["revised"] += 1
        gold = str(r.get("gold") or "")
        fc, lc = bool(grade_answer(first, gold)), bool(grade_answer(last, gold))
        if lc and not fc:
            w2r += 1
            cell["w2r"] += 1
        elif fc and not lc:
            r2w += 1
            cell["r2w"] += 1
        else:
            same += 1
    return {
        "rows": n, "n_problems": len(groups),
        "acc": sum(acc_flags) / max(1, n),
        "revised": revised, "revised_pct": 100.0 * revised / max(1, n),
        "w2r": w2r, "r2w": r2w, "unchanged_correctness": same,
        "precision": (w2r / revised) if revised else 0.0,
        "net_pp": 100.0 * (w2r - r2w) / max(1, n),
        "confirmed_same_answer": confirmed,
        "tok_per_row": sum(float(r.get("n_tok") or 0) for r in rows) / max(1, n),
        "by_state": state_tab,
        "per_problem": {g: [0.0, sum(acc_flags[i] for i in idx) / len(idx)]
                        for g, idx in groups.items()},
    }


def vs_ref(summary: dict, ref_path: str) -> dict:
    """문제 키로 짝지어 acc 차이의 부트스트랩 CI — `--ref` 는 다른 summary.json 이다."""
    ref = json.loads(Path(ref_path).read_text())
    a, b = ref.get("per_problem") or {}, summary.get("per_problem") or {}
    keys = sorted(set(a) & set(b))
    if not keys:
        return {"error": "per_problem 키 교집합이 0 — 같은 문제 집합이 아니다", "n": 0}
    out = paired_bootstrap([(float(a[k][1]), float(b[k][1])) for k in keys])
    out["ref"] = ref_path
    return out


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regrade_only", help="기존 texts.jsonl 경로 — 생성 없이 재채점·재요약")
    ap.add_argument("--protocol", default="single", choices=("single",))
    ap.add_argument("--ref", help="비교 기준 summary.json(문제 짝 부트스트랩)")
    ap.add_argument("--out_dir", required=True)
    # 생성이 필요한 경우는 mc/rollout.py 에 위임한다(엔진 설정 단일 원천).
    ap.add_argument("--model_path")
    ap.add_argument("--dataset")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--gpu_util", type=float, default=0.80)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_tokens", type=int, default=12288,
                    help="토큰 예산(기본 12,288 — L5 800). 어려운 시험지·전이 평가는 32768.")
    a = ap.parse_args(argv)

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if a.regrade_only:
        rows = load_jsonl(a.regrade_only)
        summ = summarize_single(rows)
        summ["source"] = a.regrade_only
    else:
        if not (a.model_path and a.dataset):
            raise SystemExit("[MC] 생성 평가는 --model_path/--dataset 이 필요하다(또는 --regrade_only).")
        from mc.rollout import run_eval  # noqa: PLC0415
        rows = run_eval(a)
        summ = summarize_single(rows)
        summ.update({"protocol": a.protocol, "model_path": a.model_path, "dataset": a.dataset,
                     "k": a.k, "seed": a.seed, "max_tokens": a.max_tokens})
        with (out / "texts.jsonl").open("w") as fh:
            fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    if a.ref:
        summ["vs_ref"] = vs_ref(summ, a.ref)
    (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2,
                                                 default=float))
    for k, v in summ.items():
        if k not in ("by_state", "per_problem"):
            print(f"  {k:36s} {v}")
    print(f"[out] {out}/summary.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
