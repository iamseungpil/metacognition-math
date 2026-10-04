#!/usr/bin/env python
r"""math_agree_probe — 저장된 math_end_probe 특징(features.npz) 위에서 «불일치
(disagree) 도 정오(r_corr) 만큼 끝 은닉상태에서 읽히는가»를 잰다. 새 forward pass
없음 — math_end_probe.py 가 이미 뽑아 둔 hidden_metaend/hidden_last 벡터를 그대로
재사용한다.

표적 두 개:
  target=r_corr     기존 math_end_probe 표(≈.855/.575, layer 18/metaend) 재현용.
  target=disagree   이 행의 final_answer 가 같은 문제(group_id) 안 다수 답과
                     다른가. 다수가 동률(plurality tie)이면 그 동률 집합 안에
                     있으면 disagree=0(동률 집합 자체가 "다수"이므로 어느 쪽도
                     불일치로 안 본다), 아니면 1.

행 매칭: features.npz 에는 행 id 가 없다(저장 시 안 붙였다) — 그래서
math_end_probe.select_rows 를 rollout 파일에 **같은 인자**(variant="math_opt")로
다시 돌려 같은 순서의 선택 리스트를 재구성하고, 길이가 npz 의 target 배열과
같은지 assert 한다(순수 리스트 컴프리헨션이라 결정적 — 같은 입력엔 같은 순서).
그 선택 리스트의 group_id 를 npz 의 group_id 와 원소별로 대조해 한 번 더 검증한다.

리포트(레이어 × 위치 × 표적):
  auc_overall        전체 위에서 그룹-OOF AUC
  auc_mixed          0<group pass rate<1 인 «흔들리는» 그룹만(math_end_probe.
                     nontrivial_auc 재사용)
  per_problem_auc     흔들리는 그룹마다 따로 잰 AUC 의 평균(math_ruler_pivot.
                     outcome_fixed_auc 를 strata=group_id 로 재사용 — 이게 바로
                     "문제 단위 평균 AUC"다; src.training.retry_metrics.
                     per_problem_auc 는 decision/first_correct 필드에 못박혀
                     있어 여기 못 그대로 쓴다 — import 시도 후 안 맞으면 이 재사용
                     경로로 대체한다).

추가로 disagree 자체(0/1, 학습 없음)가 «틀렸는가»의 예측자로서 흔들리는 그룹
안에서 갖는 AUC(사전 분석 대비 sanity, ≈.77 근방) 와 기준선(base rate)들.

사용:
  python scripts/local/math_agree_probe.py \
      --features <out>/features.npz --rollouts <dir>/texts.jsonl --out_dir <out>
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import math_ruler_pivot as P  # noqa: E402  (auc/outcome_fixed_auc/grouped_oof_probe 재사용)
import math_end_probe as E  # noqa: E402  (select_rows/group_pass_rates/nontrivial_auc 재사용)

_NAN = float("nan")

# retry_metrics.per_problem_auc 는 decision/first_correct 필드에 못박혀 있어 여기(연속
# 프로브 점수 vs disagree 라벨)엔 그대로 못 쓴다 — import 는 해 두되(문서화 목적) 실제
# 계산은 math_ruler_pivot.outcome_fixed_auc(strata=group_id) 로 대체한다(아래 참조).
try:
    from src.training.retry_metrics import per_problem_auc as _retry_per_problem_auc  # noqa: F401,E402
except Exception:  # pragma: no cover - optional, 필드가 안 맞아도 스크립트는 계속 돈다
    _retry_per_problem_auc = None


# ── disagree 표적 ────────────────────────────────────────────────────────────────
def _norm_answer(a) -> str:
    return "" if a is None else str(a).strip()


def compute_disagree(rows: list[dict], *, group_key: str = "group_id",
                     answer_key: str = "final_answer") -> list[int]:
    """행마다 disagree ∈ {0,1}: final_answer 가 같은 group_id 안 다수(plurality) 집합에
    없으면 1. 동률이면 그 동률 집합 전체가 "다수" 로 취급된다(그 안에 있으면 0)."""
    counts: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        counts[r[group_key]][_norm_answer(r.get(answer_key))] += 1
    out = []
    for r in rows:
        c = counts[r[group_key]]
        max_c = max(c.values())
        plurality = {a for a, ct in c.items() if ct == max_c}
        out.append(0 if _norm_answer(r.get(answer_key)) in plurality else 1)
    return out


def per_problem_auc(scores, labels, groups) -> dict:
    """흔들리는 그룹마다 따로 잰 AUC 의 평균 — math_ruler_pivot.outcome_fixed_auc 를
    strata=group_id 로 그대로 쓴 것(둘 다 클래스가 있는 그룹만, 그 그룹들 평균).
    호출자가 scores/labels/groups 를 이미 mixed 그룹으로 한정해 넘긴다."""
    return P.outcome_fixed_auc(scores, labels, groups)


# ── 특징-롤아웃 정합 ──────────────────────────────────────────────────────────────
def reconstruct_selected(rows: list[dict], variant: str, n_expected: int) -> list[dict]:
    """features.npz 를 낳은 select_rows 호출을 똑같이 재현하고 길이를 대조한다."""
    selected = E.select_rows(rows, variant)
    assert len(selected) == n_expected, (
        f"select_rows({variant!r}) 재구성 길이 {len(selected)} != npz target 길이 {n_expected} — "
        "npz 를 낸 variant 가 다르거나 rollout 파일이 바뀌었다.")
    return selected


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="math_end_probe.py --save_features 산출 npz")
    ap.add_argument("--rollouts", required=True, help="같은 실행에 쓰인 texts.jsonl")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--variant", default="math_opt",
                    help="features.npz 를 낼 때 쓴 variant(select_rows 재현용)")
    a = ap.parse_args()

    npz = np.load(a.features, allow_pickle=True)
    y_corr = np.asarray(npz["target"], dtype=float)
    groups = np.asarray(npz["group_id"])
    layers = [int(x) for x in npz["layers"]]
    n = len(y_corr)

    rows = [json.loads(l) for l in open(a.rollouts)]
    pass_rates = E.group_pass_rates(rows)          # 전체 4000행 기준(선별 전) — math_end_probe 와 동일
    disagree_all = compute_disagree(rows)           # 전체 4000행 기준(다수 판단은 그룹 전체로)

    selected = reconstruct_selected(rows, a.variant, n)
    sel_groups = np.asarray([s["group_id"] for s in selected])
    assert np.array_equal(sel_groups, groups), (
        "재구성한 select_rows 의 group_id 순서가 npz 의 group_id 와 다르다 — 정합 실패.")
    y_dis = np.asarray([disagree_all[s["idx"]] for s in selected], dtype=float)

    mixed_mask = np.array([0.0 < pass_rates.get(g, -1.0) < 1.0 for g in groups])
    n_mixed_rows = int(mixed_mask.sum())
    n_mixed_groups = len({g for g, ok in zip(groups.tolist(), mixed_mask.tolist()) if ok})

    report_rows = []
    for L in layers:
        for kind, key in (("metaend", f"hidden_metaend_L{L}"), ("last", f"hidden_last_L{L}")):
            X = np.asarray(npz[key])
            for tname, y in (("r_corr", y_corr), ("disagree", y_dis)):
                pr = P.grouped_oof_probe(X, y, groups.tolist())
                oof = pr["oof"]
                auc_overall = P.auc(y, oof)
                auc_mixed = E.nontrivial_auc(oof, y, groups.tolist(), pass_rates)
                ppa = per_problem_auc(oof[mixed_mask], y[mixed_mask], groups[mixed_mask])
                report_rows.append({
                    "layer": L, "position": kind, "target": tname,
                    "n": int(np.isfinite(oof).sum()),
                    "auc_overall": auc_overall, "auc_mixed": auc_mixed,
                    "per_problem_auc": ppa["auc"], "n_problems": ppa["n_strata"],
                })

    # sanity: disagree 자체(학습 없음)가 wrongness 를 얼마나 예측하는가, mixed 그룹 안에서.
    wrongness = 1.0 - y_corr
    sanity_auc = P.auc(wrongness[mixed_mask], y_dis[mixed_mask])

    base_rates = {
        "n_selected": n,
        "accuracy_overall(r_corr mean)": float(np.mean(y_corr)),
        "disagree_rate_overall": float(np.mean(y_dis)),
        "n_mixed_rows": n_mixed_rows,
        "n_mixed_groups": n_mixed_groups,
        "disagree_rate_mixed": float(np.mean(y_dis[mixed_mask])) if n_mixed_rows else _NAN,
        "wrongness_rate_mixed": float(np.mean(wrongness[mixed_mask])) if n_mixed_rows else _NAN,
        "disagree_predicts_wrongness_auc_mixed(sanity~.77)": sanity_auc,
    }

    md = to_markdown(report_rows, base_rates)
    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    (od / "agree_probe_table.md").write_text(md)
    (od / "agree_probe_table.json").write_text(json.dumps(
        {"rows": report_rows, "base_rates": base_rates}, ensure_ascii=False, indent=2, default=float))
    print(md)
    print(f"[out] {od}", flush=True)
    return 0


def _fmt(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return "nan" if not math.isfinite(v) else f"{v:.3f}"
    return str(v)


def to_markdown(rows: list[dict], base_rates: dict) -> str:
    cols = ["layer", "position", "target", "n", "auc_overall", "auc_mixed",
            "per_problem_auc", "n_problems"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(_fmt(r.get(c)) for c in cols) + " |")
    lines.append("")
    lines.append("### base rates")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    for k, v in base_rates.items():
        lines.append(f"| {k} | {_fmt(v)} |")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
