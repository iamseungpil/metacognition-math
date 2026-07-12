#!/usr/bin/env python3
r"""SFT-quality GATE measurement for a metacognition SFT checkpoint.

Thin ORCHESTRATOR over existing tools. It does NOT reimplement vLLM, math_verify,
or the PMI-shift math — it drives:
  - scripts/eval_vllm_1030.py         (emission rate + accuracy; num_meta_blocks/row)
  - src/eval/pmi_shift_signal.py      (gold-vs-decoy PMI SHIFT: AUC, SAVE reversals,
                                       own!=gold answer-identity confound check)

It produces a single JSON with the four gate quantities the criteria doc scores:
  (a) emission_at_temp1  — meta emission rate at temperature 1.0 on a held-out
                           math slice.
  (b) wellformed_rate    — fraction of EMITTED metas that are properly closed
                           (<|meta|>...<|/meta|>).
  (c) accuracy_greedy    — accuracy at temperature 0 on the SAME slice.
  (d) pmi_signal         — key outputs from pmi_shift_signal.py (auc_shift,
                           n_save_reversal + rate, and the own!=gold confound
                           verdict).

The PASS/FAIL gate lives in docs/redesign/sft_gate_criteria.md; this script also
applies those thresholds (as constants below, kept in sync with the doc) and emits
a `gate` block for convenience, but the doc is the authoritative spec.

Pipeline (three model passes — all GPU, run only when SFT finishes):
  1. temp=1.0 eval  -> parquet; compute emission_at_temp1 + wellformed_rate.
  2. temp=0.0 eval  -> parquet; compute accuracy_greedy.
  3. build a rollouts parquet from the temp=1.0 completions that CARRY a closed
     meta block (columns text/answer/c_with), then run pmi_shift_signal on it.

Example (after SFT completes, on a GPU box):
  python scripts/measure_sft_gate.py \
    --model_path checkpoints/my_meta_sft_merged \
    --output_dir results/sft_gate_my_meta_sft/ \
    --benchmarks math500 --max_problems 300 --tp_size 4 \
    --base_accuracy_greedy 0.72
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# META tags — single source of truth (same constants the eval + signal use).
from src.metacot.prompt import META_END, META_START  # noqa: E402

# ── GATE THRESHOLDS (keep in sync with docs/redesign/sft_gate_criteria.md) ──────
GATE_EMISSION_MIN = 0.90        # emission_at_temp1 >= 0.90
GATE_WELLFORMED_MIN = 0.95      # wellformed_rate   >= 0.95
GATE_AUC_MIN = 0.55             # pmi auc_shift clearly > 0.5
GATE_ACC_DROP_MAX = 0.05        # accuracy_greedy not collapsed vs base (>= base - 0.05)
# own!=gold confound leg (leg-3 of the pmi signal) — FAIL-CLOSED constants:
CONFOUND_N_FLOOR = 30           # min own!=gold rows for the discrimination to be conclusive
GATE_AUC_NE_MIN = 0.50          # own!=gold AUC must be COMPUTABLE and clearly > this
GATE_PLACEBO_GAP_MIN = 0.0      # real-meta shift must clearly EXCEED placebo (content, not presence)


def _confound_genuine(conf: dict) -> dict:
    """Pure, FAIL-CLOSED verdict on whether the own!=gold SAVE signal is a genuine
    gold-belief update — not an A.6 answer-identity confound nor a
    presence-as-confidence (meta-CONTENT vs meta-PRESENCE) confound.

    `conf` is the `report.confound` dict emitted by src/eval/pmi_shift_signal.py.
    Any missing / inconclusive input yields genuine=False with a stated reason:
      (1) n-floor    : n_own_ne_gold < CONFOUND_N_FLOOR  -> inconclusive -> FAIL
      (2) direction  : mean_pmi_close_own_ne_gold must stay > 0 (toward gold)
      (3) AUC        : auc_shift_own_ne_gold must be COMPUTABLE (None=>FAIL) and > GATE_AUC_NE_MIN
      (4) placebo    : placebo_gap_own_ne_gold must be PRESENT and > GATE_PLACEBO_GAP_MIN
    Returns a dict with `genuine` (bool) and `reasons` (list[str])."""
    reasons: list[str] = []
    n_own_ne = conf.get("n_own_ne_gold", 0) or 0
    mean_close_ne = conf.get("mean_pmi_close_own_ne_gold")
    auc_ne = conf.get("auc_shift_own_ne_gold")
    placebo = conf.get("placebo") or {}
    placebo_present = "placebo_gap_own_ne_gold" in placebo
    placebo_gap = placebo.get("placebo_gap_own_ne_gold")

    # (1) n-floor: too few own!=gold rows => statistically inconclusive => FAIL.
    if n_own_ne < CONFOUND_N_FLOOR:
        reasons.append(
            f"n_own_ne_gold={n_own_ne} < floor {CONFOUND_N_FLOOR} (inconclusive)")
    # (2) gold-belief must stay POSITIVE toward gold on own!=gold rows.
    if mean_close_ne is None or mean_close_ne <= 0.0:
        reasons.append(f"mean_pmi_close_own_ne_gold={mean_close_ne} not > 0")
    # (3) own!=gold AUC must be COMPUTABLE (None=single correctness class=inconclusive)
    #     and clearly above 0.5 — do NOT pass leniently on an uncomputable AUC.
    if auc_ne is None:
        reasons.append(
            "auc_shift_own_ne_gold uncomputable (single correctness class) => inconclusive")
    elif auc_ne <= GATE_AUC_NE_MIN:
        reasons.append(f"auc_shift_own_ne_gold={auc_ne} <= {GATE_AUC_NE_MIN}")
    # (4) placebo (meta-CONTENT vs meta-PRESENCE): real-meta shift must clearly beat
    #     the shuffled-meta shift. Absent field => FAIL loudly (do not silently skip).
    if not placebo_present:
        reasons.append(
            "placebo_gap_own_ne_gold ABSENT from pmi_shift_signal output "
            "(cannot rule out presence-as-confidence confound)")
    elif placebo_gap is None or placebo_gap <= GATE_PLACEBO_GAP_MIN:
        reasons.append(f"placebo_gap_own_ne_gold={placebo_gap} not > {GATE_PLACEBO_GAP_MIN}")

    return {
        "genuine": not reasons,
        "reasons": reasons,
        "placebo_present": placebo_present,
        "placebo_gap_own_ne_gold": placebo_gap,
    }


def _run(cmd: list[str]) -> None:
    print("[measure_sft_gate] $ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _eval_vllm(args, temperature: float, tag: str) -> str:
    """Run scripts/eval_vllm_1030.py at a given temperature. Return parquet path."""
    out_dir = os.path.join(args.output_dir, tag)
    os.makedirs(out_dir, exist_ok=True)
    model_name = f"{args.model_name}_{tag}"
    parquet = os.path.join(out_dir, f"{model_name}.parquet")
    if args.reuse and os.path.exists(parquet):
        print(f"[measure_sft_gate] reuse existing {parquet}")
        return parquet
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "eval_vllm_1030.py"),
        "--model_path", args.model_path,
        "--model_name", model_name,
        "--output_dir", out_dir,
        "--benchmarks", *args.benchmarks,
        "--max_problems", str(args.max_problems),
        "--max_tokens", str(args.max_tokens),
        "--temperature", str(temperature),
        "--tp_size", str(args.tp_size),
        "--seed", str(args.seed),
    ]
    if temperature == 0.0:
        # deterministic: top_p=1 avoids nucleus interaction with a 0 temperature.
        cmd += ["--top_p", "1.0"]
    _run(cmd)
    return parquet


def _emission_and_wellformed(parquet: str) -> dict:
    """emission_at_temp1 = P(completion has a closed meta block); wellformed_rate =
    of completions that EMITTED a <|meta|> tag, the fraction with a matching close."""
    df = pd.read_parquet(parquet)
    n = len(df)
    # closed block: num_meta_blocks counts only <|meta|>...<|/meta|> pairs (the
    # eval's parse_meta_blocks regex requires both tags), i.e. "properly closed".
    n_closed = int((df["num_meta_blocks"] > 0).sum())
    emitted = df["completion"].str.contains(META_START, regex=False)
    n_emitted = int(emitted.sum())
    n_emitted_closed = int((emitted & (df["num_meta_blocks"] > 0)).sum())
    return {
        "n": n,
        "n_emitted_open_tag": n_emitted,
        "n_closed_block": n_closed,
        "emission_at_temp1": (n_closed / n) if n else None,
        "wellformed_rate": (n_emitted_closed / n_emitted) if n_emitted else None,
    }


def _accuracy(parquet: str) -> dict:
    df = pd.read_parquet(parquet)
    return {
        "n": int(len(df)),
        "accuracy_greedy": float(df["is_correct"].mean()) if len(df) else None,
    }


def _build_rollouts(temp1_parquet: str, out_parquet: str, max_rows: int) -> int:
    """From the temp=1.0 eval completions, keep rows that carry a CLOSED meta block
    and write the column shape pmi_shift_signal._load_rows expects:
        text   = full completion (parse_open_close finds the meta inside)
        answer = gold answer string  (mapped to `gt`)
        c_with = 1/0 correctness      (so the signal need not re-grade)
    Returns the number of rollout rows written."""
    df = pd.read_parquet(temp1_parquet)
    meta_df = df[df["num_meta_blocks"] > 0].copy()
    if max_rows and len(meta_df) > max_rows:
        meta_df = meta_df.head(max_rows)
    roll = pd.DataFrame({
        "text": meta_df["completion"].astype(str).values,
        "answer": meta_df["gold_answer"].astype(str).values,
        "c_with": meta_df["is_correct"].astype(float).values,
    })
    os.makedirs(os.path.dirname(out_parquet), exist_ok=True)
    roll.to_parquet(out_parquet, index=False)
    return len(roll)


def _run_pmi_signal(args, rollouts_parquet: str, n_rows: int) -> dict:
    out_json = os.path.join(args.output_dir, "pmi_shift", "pmi_signal.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    if not (args.reuse and os.path.exists(out_json)):
        cmd = [
            sys.executable, "-m", "src.eval.pmi_shift_signal",
            "--model_path", args.model_path,
            "--n", str(min(n_rows, args.n_pmi)),
            "--out", out_json,
            "--rollouts", rollouts_parquet,
            "--decoy_seed", str(args.decoy_seed),
        ]
        _run(cmd)
    with open(out_json) as f:
        full = json.load(f)
    rep = full.get("report", {})
    conf = rep.get("confound", {}) or {}
    n_scored = rep.get("n_scored", 0) or 0
    n_save = rep.get("n_save_reversal", 0) or 0
    auc = rep.get("auc_shift")
    # own!=gold confound verdict: a GENUINE gold-belief update stays POSITIVE (toward
    # gold) even when the model's own answer != gold, discriminates correct-vs-wrong
    # (computable AUC>0.5), survives on an adequately-sized subset (n-floor), and beats
    # a content-destroyed placebo. FAIL-CLOSED — see _confound_genuine.
    mean_close_ne = conf.get("mean_pmi_close_own_ne_gold")
    auc_ne = conf.get("auc_shift_own_ne_gold")
    n_own_ne = conf.get("n_own_ne_gold", 0) or 0
    verdict = _confound_genuine(conf)
    return {
        "json_path": out_json,
        "n_scored": n_scored,
        "auc_shift": auc,
        "auc_pmi_close": rep.get("auc_pmi_close"),
        "n_save_reversal": n_save,
        "n_derail_reversal": rep.get("n_derail_reversal"),
        "save_reversal_rate": (n_save / n_scored) if n_scored else None,
        "save_correct_rate": rep.get("save_correct_rate"),
        "confound": {
            "n_own_ne_gold": n_own_ne,
            "n_own_eq_gold": conf.get("n_own_eq_gold"),
            "mean_pmi_close_own_ne_gold": mean_close_ne,
            "auc_shift_own_ne_gold": auc_ne,
            "placebo_present": verdict["placebo_present"],
            "placebo_gap_own_ne_gold": verdict["placebo_gap_own_ne_gold"],
            "confound_n_floor": CONFOUND_N_FLOOR,
            "verdict_reasons": verdict["reasons"],
            "verdict_genuine_not_sole_own_identity": bool(verdict["genuine"]),
        },
        "dropped": rep.get("dropped"),
    }


def _apply_gate(emission, wellformed, acc_greedy, base_acc, pmi) -> dict:
    checks = {}
    checks["emission_at_temp1>=%.2f" % GATE_EMISSION_MIN] = (
        emission is not None and emission >= GATE_EMISSION_MIN)
    checks["wellformed_rate>=%.2f" % GATE_WELLFORMED_MIN] = (
        wellformed is not None and wellformed >= GATE_WELLFORMED_MIN)
    checks["pmi_auc_shift>%.2f" % GATE_AUC_MIN] = (
        pmi.get("auc_shift") is not None and pmi["auc_shift"] > GATE_AUC_MIN)
    checks["pmi_n_save_reversal>0"] = (pmi.get("n_save_reversal") or 0) > 0
    checks["pmi_own!=gold_not_sole_explanation"] = bool(
        pmi.get("confound", {}).get("verdict_genuine_not_sole_own_identity"))
    if base_acc is not None:
        checks["accuracy_greedy>=base-%.2f" % GATE_ACC_DROP_MAX] = (
            acc_greedy is not None and acc_greedy >= (base_acc - GATE_ACC_DROP_MAX))
    else:
        # FAIL-CLOSED: without the matched-base greedy accuracy we cannot detect an
        # accuracy collapse, so the accuracy leg FAILS. Pass --base_accuracy_greedy.
        checks["accuracy_greedy_vs_base(FAIL: pass --base_accuracy_greedy)"] = False
    return {"checks": checks, "PASS": all(checks.values())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_path", required=True, help="SFT checkpoint (merged HF dir)")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model_name", default=None,
                    help="short name for eval artifacts (default: basename of model_path)")
    ap.add_argument("--benchmarks", nargs="+", default=["math500"],
                    help="held-out math slice (default: math500)")
    ap.add_argument("--max_problems", type=int, default=300)
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--tp_size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_pmi", type=int, default=200,
                    help="max rollouts scored by pmi_shift_signal")
    ap.add_argument("--decoy_seed", type=int, default=42)
    ap.add_argument("--base_accuracy_greedy", type=float, default=None,
                    help="matched Base-SFT greedy accuracy on the SAME slice, for the "
                         "collapse check. REQUIRED in effect: if omitted the accuracy "
                         "leg FAILS the gate (fail-closed — no base = no collapse check).")
    ap.add_argument("--reuse", action="store_true",
                    help="skip a stage if its output artifact already exists")
    args = ap.parse_args()
    if args.model_name is None:
        args.model_name = Path(args.model_path.rstrip("/")).name
    os.makedirs(args.output_dir, exist_ok=True)

    # (a)+(b) temperature 1.0 pass.
    temp1_parquet = _eval_vllm(args, temperature=1.0, tag="temp1")
    em = _emission_and_wellformed(temp1_parquet)

    # (c) greedy pass.
    greedy_parquet = _eval_vllm(args, temperature=0.0, tag="greedy")
    ac = _accuracy(greedy_parquet)

    # (d) PMI-shift signal on temp1 rollouts that carry a closed meta block.
    rollouts_parquet = os.path.join(args.output_dir, "pmi_shift", "rollouts.parquet")
    n_roll = _build_rollouts(temp1_parquet, rollouts_parquet, args.n_pmi)
    if n_roll == 0:
        pmi = {"error": "no rollouts with a closed meta block at temp=1.0; "
                        "emission likely too low to measure a PMI signal.",
               "n_scored": 0, "auc_shift": None, "n_save_reversal": 0,
               "confound": {"verdict_genuine_not_sole_own_identity": False}}
    else:
        pmi = _run_pmi_signal(args, rollouts_parquet, n_roll)

    gate = _apply_gate(em["emission_at_temp1"], em["wellformed_rate"],
                       ac["accuracy_greedy"], args.base_accuracy_greedy, pmi)

    result = {
        "model_path": args.model_path,
        "model_name": args.model_name,
        "slice": {"benchmarks": args.benchmarks, "max_problems": args.max_problems,
                  "max_tokens": args.max_tokens},
        "emission_at_temp1": em["emission_at_temp1"],
        "wellformed_rate": em["wellformed_rate"],
        "accuracy_greedy": ac["accuracy_greedy"],
        "base_accuracy_greedy": args.base_accuracy_greedy,
        "pmi_signal": pmi,
        "counts": {"temp1": em, "greedy": ac, "n_rollouts_pmi": n_roll},
        "gate": gate,
        "artifacts": {"temp1_parquet": temp1_parquet,
                      "greedy_parquet": greedy_parquet,
                      "rollouts_parquet": rollouts_parquet},
    }
    out_json = os.path.join(args.output_dir, "sft_gate.json")
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n[measure_sft_gate] wrote {out_json}")
    print(f"[measure_sft_gate] GATE PASS = {gate['PASS']}")


if __name__ == "__main__":
    main()
