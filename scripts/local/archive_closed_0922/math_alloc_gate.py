#!/usr/bin/env python
r"""math_alloc_gate — M_DIFF(난이도 판단) 착수 **전** go/no-go: 매치드 계산 예산에서
    배분이 값을 사는가.

왜 (docstring, condensed). 이 4B 정책의 **내부 상태**는 문제 난이도를 읽는다(문제-홀드아웃
은닉 프로브 풀링 AUC .825, 자기 pass rate 와 Spearman +.51 — docs/HYPOTHESIS_LEDGER_cd9.md
§C1). 그런데 **말로 낸** confidence 는 그 축을 약하게만 담는다(+.13). 이 정책이 실제로 쓰는
메타인지 통제는 «필요한 곳에 더 많은 독립 표본을 뽑는 것»이다. 이 게이트는 학습 **전에**,
매치드 평균 예산에서 배분 정책이 얼마나 정확도를 사는지 세 판단원(oracle·hidden-probe·stated
confidence, + length·random 대조)에 대해 묻는다 — M_DIFF 팔의 go/no-go다.

PASS 규칙: B=2 에서 acc_probe − acc_uniform 의 95% CI 가 0 을 제외하고 평균 ≥ +0.02,
그리고 acc_oracle − acc_uniform ≥ +0.03. STATED-GAP = acc_probe − acc_stated (B=2) —
RL 이 말로 낸 confidence 를 내부 판독에 맞춰 가르쳐서 벌 수 있는 여지.

## 판단원(전부 «문제의 FIRST usable sample»에서 한 draw 로 결정 가능 — oracle 은 예외:
그 자체가 «다른 7개를 봤을 때의 천장»이라 정의상 한 draw 이상을 본다)
  oracle   그 그룹의 **첫 표본을 뺀** 나머지 7개의 실현 pass rate(LOO 천장 — gold 사용)
  probe    그 그룹의 **첫 EMITTED 표본**(parse_meta form="math") 의 OOF 은닉 프로브 점수
           (grouped_oof_probe, L36 metaend, 5-fold, l2=1.0 — math_ruler_pivot 과 c1.py 재현)
           emitted 표본이 없으면 모집단 중앙값으로 채우고 missing 으로 센다.
  stated   그 EMITTED 표본이 낸 confidence(parse_meta) — 같은 채움 규칙.
  length   첫 표본의 **음의** n_tok(길수록 어렵다고 본다) — emitted 여부 무관.
  random   균일 잡음(seed 11) — «아무 판단이나 배분에 쓰면 이긴다»는 착시를 막는 대조.

## 배분 시뮬레이션
판단원별로 hardest-first 정렬 후, k_max ∈ {2,4,8} 각각에 대해 평균이 예산 B 가 되도록
가장 어려운 n_hard = round(n·(B−1)/(k_max−1)) 문제에 k_max, 나머지에 1 을 준다. 표는 항상
**고정 롤아웃 순서**(그룹 안 표본이 기록된 순서)의 앞 k 개를 다수결한다(재표집 없음). 균일
self-consistency 는 `math_diff_eval.uniform_ks` 로 같은 평균 예산을 문제 순서대로 나눈다.
B 별로 세 k_max 후보 중 **최선**(최고 평균 정확도)을 그 판단원의 대표값으로 보고한다
(best k_max per B). 문제 단위 부트스트랩 CI(2,000회, seed 11).

이 파일은 순수 시뮬레이션 함수(order_hardest_first/alloc_ks/vote_correct/acc_for_ks)를
직접 갖는다 — `scripts/local/math_diff_eval.py` 의 `majority_vote`/`uniform_ks`/
`bootstrap_ci` 류는 이미 순수해서 **바로 import** 한다(중복 구현 금지). `alloc_ks` 는
math_diff_eval 의 것과 다르다 — 그쪽은 «말한 hard 비율」에서 k_max 를 유도하고, 이 게이트는
**k_max 를 먼저 고정**하고 그 평균이 B 가 되도록 hard 문제 수를 역산한다(다중 판단원·다중
k_max 그리드 탐색이 목적이라 방향이 반대다).

사용법:
  math_alloc_gate.py --population L5,math500 \
      --out_dir /hdd_data/seungpil/scratch/eval/alloc_gate_s1
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from math_cited_site_gate import bootstrap_ci, sign_test_p  # noqa: E402
from math_diff_eval import majority_vote, uniform_ks  # noqa: E402
from src.training.countdown_rewards import parse_meta  # noqa: E402

_NAN = float("nan")
BUDGETS = (1.0, 1.5, 2.0, 3.0, 4.0)
KMAXES = (2, 4, 8)
SOURCES = ("oracle", "probe", "stated", "length", "random")
SEED = 11
N_BOOT = 2000
# ★게이트 규칙(B=2 고정 — CLAUDE.md 지시대로 여기서 판정한다).
GATE_PROBE_DELTA_MIN = 0.02
GATE_ORACLE_DELTA_MIN = 0.03

POPULATIONS = {
    "L5": {
        "texts": "/hdd_data/seungpil/scratch/eval/mathL5_q3i2507_opt_b8k/texts.jsonl",
        "npz": "/hdd_data/seungpil/scratch/probes/end_q3i2507_L5/features.npz",
    },
    "math500": {
        "texts": "/hdd_data/seungpil/scratch/eval/math500_q3i2507_opt_b8k/texts.jsonl",
        "npz": "/hdd_data/seungpil/scratch/probes/end_q3i2507_math500/features.npz",
    },
}


# ── 순수 배분 시뮬레이션(테스트가 CPU 로 직접 두드린다) ─────────────────────────────
def order_hardest_first(judgment: Mapping[str, float]) -> list[str]:
    """group_id → 판단값(낮을수록 어렵다는 관례). 동률은 group_id 사전순(결정적)."""
    return sorted(judgment.keys(), key=lambda g: (float(judgment[g]), g))


def alloc_ks(order: Sequence[str], budget: float, k_max: int) -> dict[str, int]:
    r"""hardest-first `order` 의 앞 n_hard 개에 `k_max`, 나머지에 1.
        n_hard = round(n·(B−1)/(k_max−1))   (k_max=1 이거나 B≤1 이면 전부 1)
    평균이 정확히 B 가 안 될 수 있다(반올림) — 호출자가 realized_budget 으로 함께 보고한다."""
    n = len(order)
    if n == 0 or k_max <= 1 or budget <= 1.0:
        return {g: 1 for g in order}
    n_hard = int(round(n * (float(budget) - 1.0) / (float(k_max) - 1.0)))
    n_hard = max(0, min(n, n_hard))
    hard = set(order[:n_hard])
    return {g: (int(k_max) if g in hard else 1) for g in order}


def realized_budget(ks: Mapping[str, int]) -> float:
    return (sum(ks.values()) / len(ks)) if ks else _NAN


def _row_correct(r: Mapping) -> bool:
    """r_corr 를 0/1 로 읽는다 — NaN·None 은 **틀림**(FAIL)이다. `bool(float('nan'))`이
    True 라는 파이썬 함정을 피하려고 명시적으로 유한성을 확인한다."""
    try:
        v = float(r.get("r_corr"))
    except (TypeError, ValueError):
        return False
    return math.isfinite(v) and v >= 0.5


def vote_correct(rows: Sequence[Mapping], k: int) -> float:
    r"""고정 롤아웃 순서의 **앞 k 개**를 다수결(`math_diff_eval.majority_vote` — 동률은
    첫 표본). 그 답을 낸 표본 중 **하나라도** r_corr==1 이면 correct(스펙의 채점 규칙 —
    math_verify 재채점이 아니라 기록된 r_corr 를 그대로 읽는다)."""
    sub = list(rows)[:max(1, int(k))]
    ans = majority_vote(sub, k)
    if ans is None:
        return 0.0
    ans = str(ans).strip()
    return 1.0 if any(str(r.get("final_answer") or "").strip() == ans and _row_correct(r)
                      for r in sub) else 0.0


def acc_for_ks(groups: Mapping[str, Sequence[Mapping]], ks: Mapping[str, int]) -> dict[str, float]:
    return {g: vote_correct(rows, ks.get(g, 1)) for g, rows in groups.items()}


def uniform_ks_map(order: Sequence[str], budget: float, n_max: int) -> dict[str, int]:
    """균일 self-consistency — `math_diff_eval.uniform_ks`(순수)를 `order` 위에 그대로 편다."""
    return dict(zip(order, uniform_ks(len(order), budget, n_max)))


def best_alloc(judgment: Mapping[str, float], groups: Mapping[str, Sequence[Mapping]],
               budget: float, n_max: int) -> dict:
    """세 k_max 후보 중 평균 정확도가 최선인 배분(«best k_max per B»)."""
    order = order_hardest_first(judgment)
    best = None
    for kmax in KMAXES:
        if kmax > n_max:
            continue
        ks = alloc_ks(order, budget, kmax)
        accs = acc_for_ks(groups, ks)
        mean_acc = sum(accs.values()) / len(accs) if accs else _NAN
        cand = {"k_max": kmax, "accs": accs, "acc": mean_acc,
               "realized_budget": realized_budget(ks)}
        if best is None or (math.isfinite(mean_acc) and (not math.isfinite(best["acc"])
                                                          or mean_acc > best["acc"])):
            best = cand
    return best or {"k_max": 1, "accs": {g: 0.0 for g in groups}, "acc": _NAN,
                    "realized_budget": 1.0}


# ── 데이터 적재·정렬 ─────────────────────────────────────────────────────────────
def load_groups(texts_path: str) -> tuple[list[dict], dict[str, list[dict]], list[str]]:
    """texts.jsonl → (행 리스트, group_id → 행 리스트(파일 순서), 첫 등장 순서의 group_id 리스트)."""
    rows = [json.loads(l) for l in open(texts_path)]
    groups: dict[str, list[dict]] = defaultdict(list)
    order: list[str] = []
    for r in rows:
        g = r["group_id"]
        if g not in groups:
            order.append(g)
        groups[g].append(r)
    return rows, dict(groups), order


def emitted_indices(rows: Sequence[Mapping]) -> list[int]:
    """행 순서대로 `parse_meta(text, form="math")` 가 emitted=1 이고 start 가 있는 인덱스 —
    features.npz 를 만들 때 쓴 것과 같은 규약(scratchpad c1.py)."""
    out = []
    for i, r in enumerate(rows):
        m = parse_meta(r.get("text") or "", form="math")
        if m["emitted"] and m["start"] is not None:
            out.append(i)
    return out


def load_npz_signals(npz_path: str, rows: Sequence[Mapping]) -> tuple[dict[int, float] | None,
                                                                       dict[int, float] | None,
                                                                       str]:
    """features.npz → (row_idx→probe, row_idx→confidence, note). 정렬이 안 맞으면
    (None, None, 경고)를 돌려준다(probe/stated 을 그 모집단에서 건너뛴다)."""
    p = Path(npz_path)
    if not p.exists():
        return None, None, f"npz 없음({npz_path}) — probe/stated 생략"
    import numpy as np  # noqa: PLC0415

    import math_ruler_pivot as MRP  # noqa: PLC0415
    z = np.load(str(p), allow_pickle=True)
    sel = emitted_indices(rows)
    y = np.array([int(r.get("r_corr", 0)) for r in rows], dtype=float)
    gids = [r["group_id"] for r in rows]
    if len(sel) != len(z["target"]):
        return None, None, (f"npz 정렬 실패: emitted 행 {len(sel)} != npz {len(z['target'])} "
                            "— probe/stated 생략")
    if not (np.array_equal(z["target"], y[sel])
            and np.array_equal(np.asarray(z["group_id"]), np.array([gids[i] for i in sel]))):
        return None, None, "npz 정렬 실패: target/group_id 불일치 — probe/stated 생략"
    H = z["hidden_metaend_L36"].astype(np.float64)
    oof = MRP.grouped_oof_probe(H, z["target"], list(z["group_id"]), n_folds=5, l2=1.0,
                                seed=0)["oof"]
    probe = {int(sel[i]): float(oof[i]) for i in range(len(sel)) if np.isfinite(oof[i])}
    conf = {int(sel[i]): float(z["confidence"][i]) for i in range(len(sel))
           if np.isfinite(z["confidence"][i])}
    return probe, conf, f"npz 정렬 확인({len(sel)}/{len(rows)} 행 emitted)"


def _median(xs: Sequence[float]) -> float:
    v = sorted(float(x) for x in xs if x is not None and math.isfinite(float(x)))
    if not v:
        return 0.5
    n = len(v)
    m = n // 2
    return v[m] if n % 2 else (v[m - 1] + v[m]) / 2.0


def build_judgments(rows: list[dict], groups: dict[str, list[dict]], order: list[str],
                    npz_path: str, *, seed: int = SEED) -> tuple[dict[str, dict[str, float]],
                                                                 dict, str]:
    """다섯 판단원(oracle/probe/stated/length/random) → {group_id: value}. 값은 전부
    «낮을수록 어렵다»는 방향으로 정규화한다(order_hardest_first 와 짝)."""
    row_probe, row_conf, note = load_npz_signals(npz_path, rows)
    # 그룹 안 «첫 표본»(file order 0번째), «첫 EMITTED 표본»(있으면).
    first_row_idx: dict[str, int] = {}
    first_emit_idx: dict[str, int] = {}
    idx = 0
    for g in order:
        n = len(groups[g])
        first_row_idx[g] = idx
        if row_probe is not None:
            for k in range(n):
                if (idx + k) in row_probe:
                    first_emit_idx[g] = idx + k
                    break
        idx += n
    j_oracle, j_length, j_random = {}, {}, {}
    rng = random.Random(seed)
    for g in order:
        rows_g = groups[g]
        rest = rows_g[1:]
        j_oracle[g] = (sum(1 for r in rest if _row_correct(r)) / len(rest)) if rest else _NAN
        j_length[g] = -float(rows_g[0].get("n_tok", 0))
        j_random[g] = rng.random()
    n_missing_probe = n_missing_stated = 0
    j_probe: dict[str, float] = {}
    j_stated: dict[str, float] = {}
    if row_probe is not None:
        vals_p = [row_probe[first_emit_idx[g]] for g in order if g in first_emit_idx]
        vals_c = [row_conf[first_emit_idx[g]] for g in order
                 if g in first_emit_idx and first_emit_idx[g] in row_conf]
        med_p, med_c = _median(vals_p), _median(vals_c)
        for g in order:
            fi = first_emit_idx.get(g)
            if fi is not None and fi in row_probe:
                j_probe[g] = -row_probe[fi]      # ★probe = P(맞다) — 낮을수록 어렵다는
            else:                                 #   관례에 맞추려면 부호를 뒤집는다.
                j_probe[g] = -med_p
                n_missing_probe += 1
            if fi is not None and fi in row_conf:
                j_stated[g] = -row_conf[fi]
            else:
                j_stated[g] = -med_c
                n_missing_stated += 1
    else:
        # ★npz 정렬 실패 — 이 판단원들은 모집단 중앙값(무정보) 상수로 채워 «관측 없음»을
        #   정직하게 반영한다(전부 같은 값 → 배분이 효과 0 으로 보고된다).
        for g in order:
            j_probe[g] = 0.0
            j_stated[g] = 0.0
            n_missing_probe += 1
            n_missing_stated += 1
    judgments = {"oracle": j_oracle, "probe": j_probe, "stated": j_stated,
                "length": j_length, "random": j_random}
    meta = {"n_missing_probe": n_missing_probe, "n_missing_stated": n_missing_stated,
           "n_problems": len(order), "npz_note": note}
    return judgments, meta, note


# ── 요약 표 ────────────────────────────────────────────────────────────────────
def allocation_report(groups: dict[str, list[dict]], order: list[str],
                      judgments: dict[str, dict[str, float]], *,
                      budgets: Sequence[float] = BUDGETS, seed: int = SEED,
                      n_boot: int = N_BOOT) -> dict:
    n_max = min((len(v) for v in groups.values()), default=0)
    out: dict = {"n_problems": len(order), "n_max_samples": n_max, "budgets": {}}
    pass8 = {g: (1.0 if any(_row_correct(r) for r in rows) else 0.0)
            for g, rows in groups.items()}
    out["pass_at_8"] = sum(pass8.values()) / len(pass8) if pass8 else _NAN
    for b in budgets:
        row: dict = {"budget": float(b), "sources": {}}
        uni = uniform_ks_map(order, b, n_max)
        uni_accs = acc_for_ks(groups, uni)
        row["acc_uniform"] = sum(uni_accs.values()) / len(uni_accs) if uni_accs else _NAN
        row["realized_budget_uniform"] = realized_budget(uni)
        for src in SOURCES:
            best = best_alloc(judgments[src], groups, b, n_max)
            diffs = [best["accs"][g] - uni_accs[g] for g in order]
            ci = bootstrap_ci(diffs, n_boot=n_boot, seed=seed)
            row["sources"][src] = {
                "k_max": best["k_max"], "realized_budget": best["realized_budget"],
                "acc": best["acc"], "gain_vs_uniform": ci["mean"],
                "gain_ci_lo": ci["lo"], "gain_ci_hi": ci["hi"],
                "sign_p": sign_test_p(diffs),
            }
        out["budgets"][b] = row
    return out


def format_table(report: dict) -> str:
    lines = [f"n_problems={report['n_problems']} n_max_samples={report['n_max_samples']} "
            f"pass@8={report['pass_at_8']:.4f}", ""]
    header = (f"{'B':>5} {'uniform':>8}  " +
             "  ".join(f"{s:>28}" for s in SOURCES))
    lines.append(header)
    for b, row in sorted(report["budgets"].items()):
        cells = []
        for s in SOURCES:
            d = row["sources"][s]
            cells.append(f"acc={d['acc']:.3f} k={d['k_max']} "
                        f"Δ={d['gain_vs_uniform']:+.3f}[{d['gain_ci_lo']:+.3f},"
                        f"{d['gain_ci_hi']:+.3f}]")
        lines.append(f"{b:>5.1f} {row['acc_uniform']:>8.3f}  " + "  ".join(f"{c:>28}" for c in cells))
    return "\n".join(lines)


def gate_verdict(report: dict) -> dict:
    b2 = report["budgets"].get(2.0)
    if not b2:
        return {"pass": False, "reason": "B=2 없음"}
    probe = b2["sources"]["probe"]
    oracle = b2["sources"]["oracle"]
    stated = b2["sources"]["stated"]
    probe_ok = (probe["gain_ci_lo"] > 0.0 and probe["gain_vs_uniform"] >= GATE_PROBE_DELTA_MIN)
    oracle_ok = (oracle["gain_vs_uniform"] >= GATE_ORACLE_DELTA_MIN)
    stated_gap = probe["acc"] - stated["acc"]
    return {"pass": bool(probe_ok and oracle_ok), "probe_ok": probe_ok, "oracle_ok": oracle_ok,
           "stated_gap_b2": stated_gap, "probe_gain_b2": probe["gain_vs_uniform"],
           "probe_gain_ci_b2": [probe["gain_ci_lo"], probe["gain_ci_hi"]],
           "oracle_gain_b2": oracle["gain_vs_uniform"]}


# ── main ───────────────────────────────────────────────────────────────────────
def run_population(name: str, cfg: dict, *, seed: int = SEED, n_boot: int = N_BOOT) -> dict:
    rows, groups, order = load_groups(cfg["texts"])
    judgments, jmeta, note = build_judgments(rows, groups, order, cfg["npz"], seed=seed)
    print(f"[{name}] {jmeta['n_problems']} 문제 · {note} · "
         f"missing probe={jmeta['n_missing_probe']} stated={jmeta['n_missing_stated']}",
         flush=True)
    report = allocation_report(groups, order, judgments, seed=seed, n_boot=n_boot)
    verdict = gate_verdict(report)
    per_problem = [{"group_id": g, "oracle": judgments["oracle"][g], "probe": judgments["probe"][g],
                    "stated": judgments["stated"][g], "length": judgments["length"][g],
                    "random": judgments["random"][g], "n_samples": len(groups[g])}
                  for g in order]
    return {"population": name, "meta": jmeta, "report": report, "verdict": verdict,
           "per_problem": per_problem}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", default="L5,math500",
                    help="쉼표 목록(POPULATIONS 키) 또는 name=texts.jsonl[:npz] 오버라이드")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n_boot", type=int, default=N_BOOT)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args(argv)

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for spec in a.population.split(","):
        spec = spec.strip()
        if not spec:
            continue
        if "=" in spec:
            name, rest = spec.split("=", 1)
            parts = rest.split(":")
            cfg = {"texts": parts[0], "npz": parts[1] if len(parts) > 1 else ""}
        else:
            name, cfg = spec, POPULATIONS.get(spec)
        if not cfg:
            print(f"[skip] {spec}: 모집단 정의 없음(POPULATIONS 참고)", flush=True)
            continue
        if not Path(cfg["texts"]).exists():
            print(f"[skip] {name}: {cfg['texts']} 없음", flush=True)
            continue
        results.append(run_population(name, cfg, seed=a.seed, n_boot=a.n_boot))

    md = ["# math_alloc_gate — M_DIFF go/no-go", "",
         f"PASS 규칙(B=2): acc_probe−acc_uniform CI 가 0 제외 ∧ 평균 ≥ +{GATE_PROBE_DELTA_MIN} "
         f"∧ acc_oracle−acc_uniform ≥ +{GATE_ORACLE_DELTA_MIN}", ""]
    with (out / "per_problem.jsonl").open("w") as fh:
        for res in results:
            for r in res["per_problem"]:
                fh.write(json.dumps({"population": res["population"], **r},
                                    ensure_ascii=False) + "\n")
    for res in results:
        md.append(f"## {res['population']}")
        md.append("```")
        md.append(format_table(res["report"]))
        md.append("```")
        v = res["verdict"]
        md.append(f"- **{'PASS' if v['pass'] else 'FAIL'}** — probe_ok={v.get('probe_ok')} "
                  f"oracle_ok={v.get('oracle_ok')}")
        md.append(f"- STATED-GAP(B=2) = acc_probe − acc_stated = {v.get('stated_gap_b2'):+.4f}")
        md.append(f"- probe gain@B2 = {v.get('probe_gain_b2'):+.4f} "
                  f"CI=[{v['probe_gain_ci_b2'][0]:+.4f}, {v['probe_gain_ci_b2'][1]:+.4f}]  "
                  f"oracle gain@B2 = {v.get('oracle_gain_b2'):+.4f}")
        md.append(f"- missing probe={res['meta']['n_missing_probe']} "
                  f"stated={res['meta']['n_missing_stated']} / {res['meta']['n_problems']}")
        md.append("")
        print(f"\n=== {res['population']} ===")
        print(format_table(res["report"]))
        print(f"{'PASS' if v['pass'] else 'FAIL'} — STATED-GAP(B=2)={v.get('stated_gap_b2'):+.4f}")

    (out / "summary.md").write_text("\n".join(md))
    (out / "summary.json").write_text(json.dumps(
        [{"population": r["population"], "meta": r["meta"], "report": r["report"],
         "verdict": r["verdict"]} for r in results],
        ensure_ascii=False, indent=2, default=float))
    print(f"\n[out] {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
