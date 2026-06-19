#!/usr/bin/env python3
"""PG0 — redirect-priming yield pilot (spec 2026-06-18 REV-6 §0 PG0 + §4.A).

Cheapest pre-gate: before the big harvest / any RL spend, project the
accepted-redirect yield and STOP if it cannot reach the pre-registered target
for a 15-30% SFT mix. This is the fail-fast that can declare the experiment
infeasible in minutes (spec §0 PG0, §6, §8).

Pipeline (GPU; runs on an H100 node via vLLM):
  1. Roll out the SFT-init model n=8 (temp 0.8) on a pilot pool from the RL
     train parquet. Grade each rollout with rewards._check_correctness vs gold.
  2. Keep problems whose per-problem pass-rate is in the FROZEN band
     [0.125, 0.5] AND have >=1 wrong rollout (the harvest source band, §4.A.1).
  3. For a sample of in-band WRONG rollouts: splice the wrong prefix at
     splice_index(len, frac) (frac ~ U[0.3,0.7]); regenerate 3 arms
     ANSWER-BLIND (gold hidden), k=8, temp 0.9:
       R  = append the redirect instruction then continue
       N' = append a null / confidence-restatement meta then continue
       Nc = plain continue (doubles as the B' plain-prose control here)
     Grade each arm vs gold; accept via accept_redirect(...).
  4. expected_yield(emission_rate, in_band_frac, accept_prob, pool=full_pool)
     vs --target -> GO/STOP verdict (JSON).

Reuses the ALREADY-BUILT + TESTED pure logic (does NOT reimplement):
  scripts/harvest_redirect_cf : well_formed_redirect, splice_index, arm_rate,
                                accept_redirect, expected_yield
  src/eval/redirect_behavior_detector : detect_redirect (regex_only ok here)
  src/training/rewards._check_correctness
vLLM LLM setup mirrors scripts/eval_vllm_1030.py. The redirect prompt text is
reused from src/metacot/prompt_behavior (BEHAVIOR_SYSTEM_PROMPT redirect rules).

The pure verdict helper pg0_verdict(...) is unit-tested in tests/test_pg0_yield.py
(CPU). main() wires the GPU rollout and is not unit-tested.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reused pure logic (already built + tested) — DO NOT reimplement.
from scripts.harvest_redirect_cf import (
    accept_redirect,
    arm_rate,
    expected_yield,
    gap_by_attribute,
    raw_yield_stats,
    splice_index,
    SPLICE_LO,
    SPLICE_HI,
)
from src.eval.redirect_behavior_detector import detect_redirect  # regex_only here
from src.training.rewards import _check_correctness

# Frozen source pass-rate band (spec §0 PG0 / §4.A.1).
BAND_LO, BAND_HI = 0.125, 0.5

# Redirect / null-meta tails appended to the wrong prefix before continuation.
# The redirect instruction reuses the §redirect scenario semantics from
# src/metacot/prompt_behavior (concrete trigger -> lower confidence -> switch).
_REDIRECT_TAIL = (
    "\n<|meta|>\ntrigger: current_path_is_weak\n"
    "diagnosis: the approach above is not leading to a consistent answer.\n"
    "confidence: 0.3\ndecision: switch_method\n<|/meta|>\n"
    "Let me reconsider and try a genuinely different method.\n"
)
_NULL_TAIL = (
    "\n<|meta|>\nconfidence: 0.7\n<|/meta|>\n"
    "Continuing with the same approach.\n"
)


def pg0_verdict(
    emission_rate: float,
    in_band_frac: float,
    accept_prob: float,
    full_pool: int,
    target: int,
) -> dict:
    """Pure, unit-testable PG0 verdict (spec §0 PG0).

    Projects accepted-redirect count = emission_rate * in_band_frac *
    accept_prob * full_pool (via the reused expected_yield) and compares to the
    pre-registered target. STOP if the projection cannot reach the target.
    """
    projected = expected_yield(emission_rate, in_band_frac, accept_prob, full_pool)
    verdict = "GO" if projected >= target else "STOP"
    return {
        "emission_rate": float(emission_rate),
        "in_band_frac": float(in_band_frac),
        "accept_prob": float(accept_prob),
        "full_pool": int(full_pool),
        "projected_accepted": int(projected),
        "target": int(target),
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# GPU wiring (vLLM) — not unit-tested.
# ---------------------------------------------------------------------------


def _load_pool(parquet_path: str, pool_size: int) -> list[dict]:  # pragma: no cover
    """Read (question, gold) from the RL train parquet (verl schema:
    prompt=[{role,content}], reward_model.ground_truth)."""
    import pandas as pd

    rows = pd.read_parquet(parquet_path).to_dict(orient="records")
    pool: list[dict] = []
    for row in rows:
        prompt = row.get("prompt")
        # pandas returns the verl prompt list-of-dict column as a numpy ndarray,
        # which `isinstance(..., (list, tuple))` misses -> question would be empty
        # and the whole pilot pool loads 0 problems. Normalise ndarray -> list.
        if hasattr(prompt, "tolist") and not isinstance(prompt, (list, tuple, str)):
            prompt = prompt.tolist()
        if isinstance(prompt, (list, tuple)) and len(prompt):
            question = str(prompt[0].get("content", "")).strip()
        else:
            question = str(row.get("question") or row.get("problem") or "").strip()
        rm = row.get("reward_model") or {}
        if not isinstance(rm, dict) and hasattr(rm, "item"):
            try:
                rm = rm.item()
            except Exception:
                rm = {}
        gold = str(
            (rm.get("ground_truth") if isinstance(rm, dict) else None)
            or row.get("gold_answer")
            or row.get("answer")
            or ""
        ).strip()
        # split_tags (difficulty/scenario/trigger) for the per-attribute gap
        # breakdown; same struct-column coercion as reward_model.
        st = row.get("split_tags") or {}
        if not isinstance(st, dict) and hasattr(st, "item"):
            try:
                st = st.item()
            except Exception:
                st = {}
        tags = {k: str(st.get(k)) for k in ("difficulty", "scenario", "trigger")} \
            if isinstance(st, dict) else {}
        if question and gold:
            pool.append({"question": question, "gold": gold, "tags": tags})
        if len(pool) >= pool_size:
            break
    return pool


def _render(tokenizer, question: str) -> str:  # pragma: no cover
    messages = [{"role": "user", "content": question}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"


def main() -> None:  # pragma: no cover - GPU wiring; pure logic is unit-tested
    parser = argparse.ArgumentParser()
    # SFT init = v8_meta_inside_strict_sft (the cold-start SFT for all RL,
    # CLAUDE.md). On the node it is staged to this local path from the HF dataset
    # repo iamseungpil/metacot (models/v8_meta_inside_strict_sft/checkpoint-254).
    parser.add_argument("--model_path", default="/scratch/models/v8_meta_inside_strict_sft")
    parser.add_argument(
        "--train_parquet",
        default="/scratch/metacognition/data/verl_train_meta_mix.parquet",
    )
    parser.add_argument("--pool_size", type=int, default=200)
    parser.add_argument("--full_pool", type=int, required=True,
                        help="full RL train pool size used in the yield projection")
    parser.add_argument("--target", type=int, default=1500)
    parser.add_argument("--rollout_k", type=int, default=8)
    parser.add_argument("--arm_k", type=int, default=8)
    parser.add_argument("--max_wrong_splices", type=int, default=200,
                        help="cap on in-band wrong rollouts spliced into 3-arm regen")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--tp_size", type=int, default=4)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="/scratch/eval_results/pg0_yield_pilot")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    pool = _load_pool(args.train_parquet, args.pool_size)
    if not pool:
        raise SystemExit(f"No problems loaded from {args.train_parquet}")
    print(f"[pg0] pilot pool = {len(pool)} problems from {args.train_parquet}")

    print(f"[pg0] loading vLLM: {args.model_path} (tp={args.tp_size})")
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tp_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        dtype="bfloat16",
        seed=args.seed,
    )
    tokenizer = llm.get_tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Step 1+2: roll out n=k @ temp 0.8, grade, keep in-band w/ >=1 wrong ──
    prompts = [_render(tokenizer, p["question"]) for p in pool]
    rollout_sp = SamplingParams(
        n=args.rollout_k, temperature=0.8, top_p=0.95,
        max_tokens=args.max_new_tokens, seed=args.seed, skip_special_tokens=False,
    )
    print(f"[pg0] rollout n={args.rollout_k} temp=0.8 on {len(prompts)} problems")
    outs = llm.generate(prompts, rollout_sp)

    in_band: list[dict] = []  # {question, gold, prompt, wrong_texts}
    for prob, prompt, out in zip(pool, prompts, outs):
        texts = [s.text for s in out.outputs]
        grades = [1 if _check_correctness(t, prob["gold"]) else 0 for t in texts]
        pass_rate = arm_rate(grades)
        wrong = [t for t, g in zip(texts, grades) if g == 0]
        if BAND_LO <= pass_rate <= BAND_HI and wrong:
            in_band.append({
                "question": prob["question"], "gold": prob["gold"],
                "prompt": prompt, "wrong_texts": wrong,
                "tags": prob.get("tags", {}),
            })
    in_band_frac = len(in_band) / len(pool)
    print(f"[pg0] in-band problems = {len(in_band)}/{len(pool)} "
          f"(frac={in_band_frac:.3f})")

    # ── Step 3: sample in-band wrong rollouts -> splice -> 3-arm regen ──
    splice_jobs: list[dict] = []  # one per spliced wrong rollout
    for prob in in_band:
        for wrong in prob["wrong_texts"]:
            n_tok = len(tokenizer(wrong, add_special_tokens=False)["input_ids"])
            if n_tok < 2:
                continue
            frac = rng.uniform(SPLICE_LO, SPLICE_HI)
            cut = splice_index(n_tok, frac)
            ids = tokenizer(wrong, add_special_tokens=False)["input_ids"][:cut]
            prefix = tokenizer.decode(ids, skip_special_tokens=False)
            splice_jobs.append({"gold": prob["gold"], "base": prob["prompt"] + prefix,
                                "tags": prob.get("tags", {})})
    rng.shuffle(splice_jobs)
    splice_jobs = splice_jobs[: args.max_wrong_splices]
    print(f"[pg0] splicing {len(splice_jobs)} in-band wrong rollouts -> 3 arms "
          f"k={args.arm_k} temp=0.9 (answer-blind)")

    # emission_rate (spec §0): fraction of attempted splices that produced a
    # well-formed redirect behavior in arm R (regex-only detector for the pilot).
    arm_sp = SamplingParams(
        n=args.arm_k, temperature=0.9, top_p=0.95,
        max_tokens=args.max_new_tokens, seed=args.seed, skip_special_tokens=False,
    )
    r_prompts = [j["base"] + _REDIRECT_TAIL for j in splice_jobs]
    n_prompts = [j["base"] + _NULL_TAIL for j in splice_jobs]
    c_prompts = [j["base"] for j in splice_jobs]
    r_out = llm.generate(r_prompts, arm_sp)
    n_out = llm.generate(n_prompts, arm_sp)
    c_out = llm.generate(c_prompts, arm_sp)

    attempted = 0
    accepted = 0
    emitted = 0
    grade_triples: list[tuple] = []  # (r,n',nc) grades per splice -> raw diagnostics
    tagged_triples: list[tuple] = []  # (tags, r,n',nc) -> per-attribute gap breakdown
    for job, ro, no, co in zip(splice_jobs, r_out, n_out, c_out):
        gold = job["gold"]
        r_texts = [_REDIRECT_TAIL + s.text for s in ro.outputs]
        n_texts = [_NULL_TAIL + s.text for s in no.outputs]
        c_texts = [s.text for s in co.outputs]
        # Emission = redirect BEHAVIOR present in any arm-R sample (regex-only).
        if any(detect_redirect(t, regex_only=True) for t in r_texts):
            emitted += 1
        r_grades = [1 if _check_correctness(t, gold) else 0 for t in r_texts]
        n_grades = [1 if _check_correctness(t, gold) else 0 for t in n_texts]
        c_grades = [1 if _check_correctness(t, gold) else 0 for t in c_texts]
        attempted += 1
        grade_triples.append((r_grades, n_grades, c_grades))
        tagged_triples.append((job.get("tags", {}), r_grades, n_grades, c_grades))
        # Nc doubles as the plain-prose B' control for the pilot (spec instruction).
        if accept_redirect(r_grades, n_grades, c_grades,
                           bprime_grades=c_grades, margin=0.5):
            accepted += 1

    emission_rate = (emitted / attempted) if attempted else 0.0
    accept_prob = (accepted / attempted) if attempted else 0.0
    print(f"[pg0] attempted={attempted} emitted={emitted} accepted={accepted}")

    # ── PRE-GATE raw diagnostics: separate 'model cannot redirect' (warmup) from
    #    'accept gate too strict' WITHOUT a new GPU run (pure arithmetic on the
    #    grades already computed above). A STOP with accepted=0 is ambiguous; this
    #    block makes it interpretable. ──
    raw = raw_yield_stats(grade_triples)
    print("[pg0-raw] " + json.dumps(raw))
    if raw.get("n"):
        print(f"[pg0-raw] mean r_rate={raw['mean_r_rate']:.3f} "
              f"nprime={raw['mean_nprime_rate']:.3f} nc={raw['mean_nc_rate']:.3f}")
        print(f"[pg0-raw] mean_gap(R-Nc)={raw['mean_gap_rc']:.3f} "
              f"mean_gap(R-N')={raw['mean_gap_rn']:.3f} "
              f"saves(R>Nc)={raw['saves_rc']}/{raw['n']} "
              f"strong(>=.25)={raw['saves_rc_strong']}")
        print(f"[pg0-raw] lci(R-Nc) p90={raw['lci_rc_p90']:.3f} "
              f"max={raw['lci_rc_max']:.3f} | accept@margin={raw['accept_at_margin']}")

    # ── PER-ATTRIBUTE breakdown: WHERE does forced redirect help? (localizes the
    #    redirect-SFT data + difficulty band to the problem types that respond). ──
    attr = gap_by_attribute(tagged_triples)
    print("[pg0-attr] " + json.dumps(attr))
    for a, groups in attr.items():
        for v, st in groups.items():
            print(f"[pg0-attr] {a}={v}: n={st['n']} mean_gap(R-Nc)={st['mean_gap_rc']:+.3f} "
                  f"saves={st['saves_frac']:.2f} strong={st['strong']}")

    # ── Step 4: verdict ──
    result = pg0_verdict(
        emission_rate=emission_rate,
        in_band_frac=in_band_frac,
        accept_prob=accept_prob,
        full_pool=args.full_pool,
        target=args.target,
    )
    result["pool"] = len(pool)
    result["attempted"] = attempted
    result["accepted"] = accepted
    result["raw"] = raw
    result["attr"] = attr

    print(json.dumps(result, indent=2))
    out_path = Path(args.output_dir) / "pg0_verdict.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[pg0] wrote {out_path}")
    if result["verdict"] == "STOP":
        raise SystemExit(
            f"[pg0] STOP: projected {result['projected_accepted']} < target {args.target}"
        )


if __name__ == "__main__":
    main()
