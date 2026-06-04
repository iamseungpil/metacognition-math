"""E.5 — GOLD-FREE pure-prompt contrastive STEERING test (CTSD Phase E, inference-only).

WHY this exists (the deployable complement):
  E.3 (the e2 steering probe) steered with a LOGIT-level contrast between two GOLD-CONDITIONED
  reveal contexts via a custom LogitsProcessor — a mechanism that cannot ship (it needs the gold
  answer in a reveal stream + KV-lockstep over a frozen HF model). E.4 v7 RL trained a teacher with
  the SAME contrastive signal, but the realized per-token factor was weak (factor_mean ≈ 1.000).

  E.5 asks the DEPLOYABLE question: with NO gold answer anywhere, does a plain stance/confidence
  INSTRUCTION — a STRING APPENDED TO THE USER PROMPT — steer (a) meta CONTENT (verify-rate, meta
  length) and (b) accuracy/calibration, contrastively and per-problem? Because the steer is just a
  prompt suffix, E.5 drops ALL of e2's heavy machinery: NO gold/decoy reveal, NO
  ContrastiveMetaSteerProcessor, NO HF model, NO LogitsProcessor, NO KV-lockstep, NO self-meta
  harvest. A condition = one batched vLLM generate + grade. The contrastive comparison is the PAIRED
  per-problem diff across conditions.

TWO AXES (both built):
  STANCE: cautious_instr vs confident_instr  -> accuracy, verify_rate, meta_token_frac.
  CONF:   "confidence: 0.15" vs "confidence: 0.95" -> ECE/calibration, overconfidence_rate,
          verbalized_conf (uptake check), accuracy.
  Plus a NEUTRAL (bare-question) anchor for both axes. Paired per-problem on GSM8K/MATH500/AIME.

KARPATHY MINIMAL-CHANGE / REUSE-BY-IMPORT:
  Generation path (VllmGen/safe_tokenizer_path), grading (robust_grade/is_gradeable/
  extract_last_boxed), stats (paired_perm_test via e2.contrast_stats), meta region mask
  (meta_region_mask), the representative stratified pool (b4.representative_pool), the e2 helpers
  (classify_stance/contrast_stats/parse_verbalized_conf/pass_at_k/headroom_band/split_around_meta),
  parse_meta_blocks (src/metacot/prompt), the rewards signal predicates (import-only), and the
  byte-identical CAUTIOUS_INSTR/CONFIDENT_INSTR (verl_sdc — the SINGLE SOURCE, the same bytes e2
  imports) are ALL imported. NO protected file is modified (a3*/a6/common/rewards/verl_sdc are
  import-only). New code = the SUFFIX/CONDITIONS registry + build_instr_prompt_ids, the inline
  calibration metrics (meta_token_frac/verbalized_conf/ece/overconf), the per-axis paired
  orchestration, and the smoke assertions / CLI / output.

PHASING (e2 hygiene): Phase 1 = DISCOVERY (n~45, seed A); Phase 2 = CONFIRMATION (n~110, FRESH seed
  B = STRATIFIED_SAMPLE_SEED + 7919 ⇒ a DIFFERENT problem pool). PASS is claimed ONLY in Phase 2 when
  a directional axis effect clears the realized-MDE power gate AND p<0.05 on the fresh seed.

Outputs reports/e5_goldfree_<model>_<tag>.jsonl (one row per (problem, condition)) +
  reports/e5_goldfree_<model>_<tag>.json (per-condition summary + per-axis contrast_stats + verdict).
"""
from __future__ import annotations
import argparse, json, time, gc
import random as _random
from pathlib import Path
import numpy as np
from huggingface_hub import hf_hub_download
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))         # experiments/
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # repo root (src/)
from common.env import (
    SFT_V8_STRICT, TEACHER_MODEL, EVAL_R10V2_V8, EVAL_R10V2_E20A, HF_DATASET,
    META_OPEN_ID, META_CLOSE_ID, REPORTS_DIR, STRATIFIED_SAMPLE_SEED,
)
from common.vllm_gen import VllmGen, safe_tokenizer_path
from common.grading import robust_grade, is_gradeable, extract_last_boxed  # noqa: F401 (extract used in e2 helpers)
from common.probe_utils import meta_region_mask
from probes.b4_teacher_steering import representative_pool
from probes.e2_contrastive_steering import (
    classify_stance, contrast_stats, parse_verbalized_conf, pass_at_k,
    headroom_band, split_around_meta,
)
from src.metacot.prompt import parse_meta_blocks
# SINGLE SOURCE for byte-identity: e2 imports these SAME bytes (its " " + CAUTIOUS_INSTR join is the
# cautious A-side; " confidence: 0.15"/"0.95" are its conf_down A/B sides). E.5 reuses both so each
# axis pole is byte-identical to the e2 contrast it complements.
from src.training.verl_sdc import CAUTIOUS_INSTR, CONFIDENT_INSTR
from src.training.rewards import (  # import-only signal predicates
    _has_verification_signal, _has_effective_verification_signal,
    _has_overconfidence_signal, _has_uncertainty_signal,
)
from transformers import AutoTokenizer

MODELS = {"v8_strict": SFT_V8_STRICT, "e20a": TEACHER_MODEL}
EVALS = {"v8_strict": EVAL_R10V2_V8, "e20a": EVAL_R10V2_E20A}

POWER_MDE_THR = 0.05       # realized_MDE hard-gate (same as e2)
GRADEABLE_THR = 0.5        # gradeable_rate power floor

# ── CONDITIONS registry: the gold-free steer is a SUFFIX appended to the user question ──────────
# CAUTIOUS_INSTR / CONFIDENT_INSTR are IMPORTED (never retyped). The leading-space join is the SAME
# byte form e2 uses (e2 CONTRASTS: " " + CAUTIOUS_INSTR / " confidence: 0.15"), so the stance/conf
# poles are byte-identical to the gold-conditioned e2 contrast they complement.
SUFFIX = {
    "neutral":         "",                          # bare question — anchor for BOTH axes
    "cautious_instr":  " " + CAUTIOUS_INSTR,         # STANCE pos (== e2 cautious A-side)
    "confident_instr": " " + CONFIDENT_INSTR,        # STANCE neg (== e2 cautious B-side)
    "conf_low":        " confidence: 0.15",          # CONF pos  (== e2 conf_down A-side)
    "conf_high":       " confidence: 0.95",          # CONF neg  (== e2 conf_down B-side)
}
ALL_CONDITIONS = ("neutral", "cautious_instr", "confident_instr", "conf_low", "conf_high")
STANCE_AXIS = ("cautious_instr", "confident_instr")   # paired diff = cautious − confident
CONF_AXIS = ("conf_low", "conf_high")                 # paired diff = low − high


# ── per-file helpers (copied verbatim from b4/e2 — per-probe by design) ─────────────────────────

def load_tokenizer(path: str):
    """Robust tokenizer load (copy of e2.load_tokenizer). The v8_strict checkpoint tokenizer FAILS
    under transformers 4.57; safe_tokenizer_path returns the E20a-substituted path (identical Qwen3
    vocab, <|meta|>=151669, <|/meta|>=151670) in that case. Always assert the meta-token IDs."""
    safe_path = safe_tokenizer_path(path)
    tok = AutoTokenizer.from_pretrained(safe_path)
    assert tok.convert_tokens_to_ids("<|meta|>") == META_OPEN_ID, "META_OPEN_ID mismatch"
    assert tok.convert_tokens_to_ids("<|/meta|>") == META_CLOSE_ID, "META_CLOSE_ID mismatch"
    return tok


def _json_safe(o):
    """Recursively cast numpy bool/float/int to python and NaN/Inf -> None for STRICT JSON.
    Copy of e2._json_safe / b4._json_safe."""
    import math
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    return o


# ── NEW: gold-free prompt builder (the only construction this probe adds) ───────────────────────

def build_instr_prompt_ids(tok, question: str, suffix: str) -> list[int]:
    """Gold-free prompt ids = chat_template(question + suffix). Identical to e2.build_prompt_ids on
    (question + suffix); NOT build_reveal_ids — there is NO gold reference block here. The suffix is
    the entire steer (empty for neutral, a stance instruction, or a confidence anchor)."""
    msgs = [{"role": "user", "content": question + suffix}]
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok.encode(s, add_special_tokens=False)[:1024]


# ── NEW: meta / calibration metrics (small, inline) ─────────────────────────────────────────────

def meta_token_frac(resp_ids: list[int]) -> float:
    """Fraction of RESPONSE tokens inside a meta span. vLLM token_ids are response-relative (the
    prompt is excluded), so meta_region_mask needs no prompt offset."""
    if not resp_ids:
        return 0.0
    m = meta_region_mask(resp_ids, len(resp_ids))
    return float(m.sum()) / max(1, len(resp_ids))


def roll_meta_text(tok, resp_ids: list[int]) -> str | None:
    """Inner meta CONTENT of a response (e2.split_around_meta[1]); None if no meta block."""
    return split_around_meta(tok, resp_ids)[1]


def roll_verify_flag(meta_text: str | None) -> bool:
    """Verify stance for one roll: e2.classify_stance == 'verify' OR a raw verification signal."""
    if not meta_text:
        return False
    if classify_stance(meta_text) == "verify":
        return True
    return bool(_has_verification_signal(meta_text)
                or _has_effective_verification_signal(meta_text)
                or _has_uncertainty_signal(meta_text))


def roll_verbalized_conf(text: str, meta_text: str | None) -> float | None:
    """Verbalized confidence for one roll. Primary = mean of parse_meta_blocks(text)['confidences']
    (the SAME parser eval_hf uses); fallback = e2.parse_verbalized_conf on the meta content."""
    try:
        confs = parse_meta_blocks(text).get("confidences") or []
    except Exception:
        confs = []
    if confs:
        return float(np.mean(confs))
    return parse_verbalized_conf(meta_text) if meta_text else None


def ece_problem(verbalized_conf: float | None, accuracy: float) -> float | None:
    """Per-problem calibration error = |confidence − accuracy| (eval_hf.py:193-201 mean|conf−correct|
    formula, inlined per-problem; the condition ECE is the mean of these over problems)."""
    if verbalized_conf is None:
        return None
    return abs(verbalized_conf - accuracy)


def roll_overconf_flag(conf: float | None, correct: bool, meta_text: str | None) -> bool:
    """Overconfidence for one roll: a high stated confidence on a WRONG answer, OR an explicit
    overconfidence signal in the meta."""
    if conf is not None and conf >= 0.9 and not correct:
        return True
    return bool(meta_text and _has_overconfidence_signal(meta_text))


# ── per-(problem, condition) metric aggregation over the k rolls ────────────────────────────────

def problem_cond_metrics(tok, roll_id_lists, gold: str):
    """Aggregate the k rolls of one (problem, condition) into a metrics dict + power counters.

    roll_id_lists = list[k] of response-relative token-id lists (vLLM o.token_ids). Returns
    (metrics_dict, gradeable_count, total_count, boxed_count)."""
    k = len(roll_id_lists)
    corrects, mtfs, verifies, confs, overconfs = [], [], [], [], []
    grade_n = box_n = 0
    for rids in roll_id_lists:
        text = tok.decode(rids, skip_special_tokens=False)
        correct = bool(robust_grade(text, gold))
        grade_n += int(is_gradeable(text))
        box_n += int(r"\boxed" in text)
        meta = roll_meta_text(tok, rids)
        conf = roll_verbalized_conf(text, meta)
        corrects.append(int(correct))
        mtfs.append(meta_token_frac(rids))
        verifies.append(int(roll_verify_flag(meta)))
        if conf is not None:
            confs.append(conf)
        overconfs.append(int(roll_overconf_flag(conf, correct, meta)))
    accuracy = float(np.mean(corrects)) if corrects else None
    verbalized_conf = float(np.mean(confs)) if confs else None
    metrics = {
        "accuracy": accuracy,
        "pass_at_k": pass_at_k([bool(c) for c in corrects]),
        "meta_token_frac": float(np.mean(mtfs)) if mtfs else None,
        "verify_rate": float(np.mean(verifies)) if verifies else None,
        "verbalized_conf": verbalized_conf,
        "ece": ece_problem(verbalized_conf, accuracy) if accuracy is not None else None,
        "overconf_rate": float(np.mean(overconfs)) if overconfs else None,
        "n_conf_parsed": len(confs),
    }
    return metrics, grade_n, k, box_n


# ── main (one model per invocation) ─────────────────────────────────────────────────────────────

def run_model(model_key: str, args, rng, rng_np, t0):
    model_path = MODELS[model_key]
    eval_path = EVALS[model_key]
    n = args.smoke or args.n
    k = args.k if not args.smoke else 2
    max_new = args.max_new if not args.smoke else min(args.max_new, 256)
    conditions = args.conditions_list
    out_jsonl = Path(args.out_dir) / f"e5_goldfree_{model_key}_{args.tag}.jsonl"
    out_json = Path(args.out_dir) / f"e5_goldfree_{model_key}_{args.tag}.json"
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if out_jsonl.exists():
        out_jsonl.unlink()

    # 1. eval json (the problem source).
    ev = hf_hub_download(repo_id=HF_DATASET, repo_type="dataset", filename=eval_path)
    results = json.load(open(ev))
    results = results if isinstance(results, list) else results.get("results") or list(results.values())[0]

    # 2. FIXED representative pool across ALL conditions => per-problem pairing. NO self-emit filter
    #    (E.5 measures meta UNDER INSTRUCTION regardless of whether the bare model self-triggers).
    n_pool = args.n_pool if args.n_pool is not None else (max(40, 20 * n) if args.smoke else max(2 * n, n + 30))
    pool = representative_pool(results, n_pool, rng)
    pool = pool[:n]                                # cap to the target N (fixed across conditions)
    print(f"[e5:{model_key}] model={model_path} phase={args.phase} n={len(pool)} k={k} "
          f"conds={conditions} max_new={max_new} pool_seed={args.pool_seed} "
          f"max_model_len={args.max_model_len}")

    tok = load_tokenizer(model_path)

    # 3. ONE vLLM (no co-resident HF, unlike e2's 0.45 split) — REUSED across all conditions.
    vgen = VllmGen(model_path, tokenizer_path=safe_tokenizer_path(model_path),
                   gpu_memory_utilization=0.85, max_model_len=args.max_model_len,
                   seed=args.pool_seed)

    # 4. per (problem, condition): one batched generate -> per-problem metrics row.
    #    grid[(pi, cond)] = metrics dict; rows = jsonl payload.
    grid, rows = {}, []
    grade_n = grade_d = box_n = 0
    for cond in conditions:
        pid_batch = [build_instr_prompt_ids(tok, r["question"], SUFFIX[cond]) for r in pool]
        rolls = vgen.generate(pid_batch, n=k, max_tokens=max_new,
                              temperature=0.7, top_p=0.95, seed=args.pool_seed)
        for pi, (r, roll_id_lists) in enumerate(zip(pool, rolls)):
            gold = str(r["gold_answer"]).strip()
            m, gn, gd, bn = problem_cond_metrics(tok, roll_id_lists, gold)
            grade_n += gn; grade_d += gd; box_n += bn
            grid[(pi, cond)] = m
            ex_text = tok.decode(roll_id_lists[0], skip_special_tokens=False) if roll_id_lists else ""
            pre0, meta0, _blk0, post0 = split_around_meta(
                tok, roll_id_lists[0]) if roll_id_lists else ("", None, None, "")
            rows.append(_json_safe({
                "model": model_key, "phase": args.phase,
                "problem_id": r.get("id", r.get("problem_id", f"{model_key}_{pi}")),
                "benchmark": r.get("benchmark"), "condition": cond,
                "suffix": SUFFIX[cond], "gold": gold,
                "headroom_band": headroom_band(m["accuracy"]) if m["accuracy"] is not None else None,
                "metrics": m, "meta_text": meta0,
                "reasoning": {"pre_meta": pre0[-2000:], "post_meta": post0[:2000]},
            }))
        print(f"  [{cond:16s}] done ({time.time()-t0:.0f}s)")
    vgen.free(); gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    with open(out_jsonl, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    # 5. PAIRED STATS PER AXIS (contrastive core). e2.contrast_stats(diffs, rng_np) ->
    #    (mean, sd, n, paired_perm_p, realized_MDE). Diff is computed only where BOTH poles exist.
    benches = sorted({r.get("benchmark") for r in pool if r.get("benchmark") is not None})

    def paired_diff(metric, pole_a, pole_b, pi_filter=None):
        ds = []
        for pi in range(len(pool)):
            if pi_filter is not None and not pi_filter(pi):
                continue
            ma, mb = grid.get((pi, pole_a)), grid.get((pi, pole_b))
            if ma is None or mb is None:
                continue
            va, vb = ma.get(metric), mb.get(metric)
            if va is None or vb is None:
                continue
            ds.append(va - vb)
        return ds

    def axis_block(metric, pole_a, pole_b):
        m, sd, nd, p, mde = contrast_stats(paired_diff(metric, pole_a, pole_b), rng_np)
        pb = {}
        for b in benches:
            bm, bsd, bn, bp, bmde = contrast_stats(
                paired_diff(metric, pole_a, pole_b,
                            pi_filter=lambda pi, b=b: pool[pi].get("benchmark") == b), rng_np)
            pb[b] = {"delta": bm, "sd": bsd, "n": bn, "p": bp, "realized_mde": bmde}
        return {"delta": m, "sd": sd, "n": nd, "p": p, "realized_mde": mde, "per_benchmark": pb}

    stance_present = all(c in conditions for c in STANCE_AXIS)
    conf_present = all(c in conditions for c in CONF_AXIS)
    neutral_present = "neutral" in conditions

    stance = None
    if stance_present:
        a, b = STANCE_AXIS
        stance = {m: axis_block(m, a, b) for m in ("accuracy", "verify_rate", "meta_token_frac")}
        if neutral_present:
            stance["vs_neutral"] = {
                "cautious_minus_neutral": {m: axis_block(m, a, "neutral")
                                           for m in ("accuracy", "verify_rate", "meta_token_frac")},
                "confident_minus_neutral": {m: axis_block(m, b, "neutral")
                                            for m in ("accuracy", "verify_rate", "meta_token_frac")},
            }
    conf = None
    if conf_present:
        a, b = CONF_AXIS
        conf = {m: axis_block(m, a, b)
                for m in ("ece", "overconf_rate", "verbalized_conf", "accuracy")}
        if neutral_present:
            conf["vs_neutral"] = {
                "low_minus_neutral": {m: axis_block(m, a, "neutral")
                                      for m in ("ece", "overconf_rate", "verbalized_conf", "accuracy")},
                "high_minus_neutral": {m: axis_block(m, b, "neutral")
                                       for m in ("ece", "overconf_rate", "verbalized_conf", "accuracy")},
            }

    # 6. POWER GATE (e2 logic): gradeable_rate >= 0.5; each TESTED axis realized_MDE <= 0.05.
    gradeable_rate = (grade_n / grade_d) if grade_d else None
    boxed_str_rate = (box_n / grade_d) if grade_d else None
    power_ok = (gradeable_rate is not None and gradeable_rate >= GRADEABLE_THR)

    def mde_ok(block):
        return block is not None and block.get("realized_mde") is not None \
            and block["realized_mde"] <= POWER_MDE_THR

    # 7. per-condition summary (mean over problems of each metric).
    def cond_summary(cond):
        out = {}
        for metric in ("accuracy", "pass_at_k", "verify_rate", "meta_token_frac",
                       "verbalized_conf", "ece", "overconf_rate"):
            vals = [grid[(pi, cond)].get(metric) for pi in range(len(pool))
                    if (pi, cond) in grid and grid[(pi, cond)].get(metric) is not None]
            out[metric] = float(np.mean(vals)) if vals else None
        return out
    per_cond = {cond: cond_summary(cond) for cond in conditions}

    # ── pre-registered verdict (per axis; phase-2 powered) ──────────────────────────────────────
    def stance_verdict():
        if stance is None:
            return "NOT-RUN", "stance axis not in --conditions"
        vr, mtf, acc = stance["verify_rate"], stance["meta_token_frac"], stance["accuracy"]
        if not power_ok or not (mde_ok(vr) and mde_ok(mtf)):
            return "INCONCLUSIVE", (f"power gate (gradeable_rate={gradeable_rate}, "
                                    f"verify_rate MDE={vr['realized_mde']}, meta_token_frac "
                                    f"MDE={mtf['realized_mde']})")
        vr_real = (vr["delta"] is not None and vr["delta"] >= 0.10 and vr["p"] is not None and vr["p"] < 0.05)
        mtf_real = (mtf["delta"] is not None and mtf["delta"] >= 0.05 and mtf["p"] is not None and mtf["p"] < 0.05)
        if vr_real or mtf_real:
            acc_tag = "null-within-power"
            if acc["delta"] is not None and acc["p"] is not None and acc["p"] < 0.05:
                if acc["delta"] >= 0.03:
                    acc_tag = "cautious-helps"
                elif acc["delta"] <= -0.03:
                    acc_tag = "cautious-hurts"
            return "DIRECTION-REAL", (f"stance steers meta process (verify_rate Δ={vr['delta']} "
                                      f"p={vr['p']}, meta_token_frac Δ={mtf['delta']} p={mtf['p']}); "
                                      f"accuracy={acc_tag} (Δ={acc['delta']})")
        return "DIRECTION-DEAD", ("instruction does not move meta process (verify_rate + "
                                  "meta_token_frac within ±MDE, p>0.05) under adequate power")

    def conf_verdict():
        if conf is None:
            return "NOT-RUN", "conf axis not in --conditions"
        ece, oc, vc = conf["ece"], conf["overconf_rate"], conf["verbalized_conf"]
        if not power_ok or not (mde_ok(ece) and mde_ok(oc)):
            return "INCONCLUSIVE", (f"power gate (gradeable_rate={gradeable_rate}, "
                                    f"ece MDE={ece['realized_mde']}, overconf MDE={oc['realized_mde']})")
        # uptake check FIRST: if the model did not echo the injected anchor, calibration is moot.
        uptake = (vc["delta"] is not None and abs(vc["delta"]) >= 0.05 and vc["delta"] < 0)
        if not uptake:
            return "INCONCLUSIVE-NO-UPTAKE", (f"verbalized_conf(low−high) Δ={vc['delta']} "
                                              f"(needs strongly negative |Δ|>=0.05); steer did not take")
        ece_real = (ece["delta"] is not None and ece["delta"] <= -0.03 and ece["p"] is not None and ece["p"] < 0.05)
        oc_real = (oc["delta"] is not None and oc["delta"] <= -0.05 and oc["p"] is not None and oc["p"] < 0.05)
        if ece_real or oc_real:
            return "DIRECTION-REAL", (f"conf_low improves calibration (ece Δ={ece['delta']} p={ece['p']}, "
                                      f"overconf Δ={oc['delta']} p={oc['p']}); uptake confirmed "
                                      f"(vconf Δ={vc['delta']})")
        return "DIRECTION-DEAD", ("uptake confirmed but ECE + overconf within ±MDE, p>0.05 under "
                                  "adequate power")

    stance_status, stance_why = stance_verdict()
    conf_status, conf_why = conf_verdict()

    if args.phase == 1:
        overall = "DISCOVERY"
        verdict = (f"DISCOVERY (Phase 1, seed={args.pool_seed}) — stance={stance_status} ({stance_why}); "
                   f"conf={conf_status} ({conf_why}). NO pass claim; re-run --phase 2 on the fresh seed "
                   f"to confirm any DIRECTION-REAL axis.")
    else:
        passed = [ax for ax, st in (("stance", stance_status), ("conf", conf_status))
                  if st == "DIRECTION-REAL"]
        if not power_ok:
            overall = "INCONCLUSIVE"
            verdict = (f"INCONCLUSIVE — gradeable_rate={gradeable_rate} < {GRADEABLE_THR}; never a "
                       f"substantive null under low power. stance={stance_status}; conf={conf_status}")
        elif passed:
            overall = "PASS"
            verdict = (f"PASS (Phase 2, fresh seed={args.pool_seed}) — axis/axes {passed} clear "
                       f"direction-real past MDE + p<0.05. stance={stance_status} ({stance_why}); "
                       f"conf={conf_status} ({conf_why})")
        else:
            overall = "FAIL"
            verdict = (f"FAIL — neither axis shows a powered directional steer. stance={stance_status} "
                       f"({stance_why}); conf={conf_status} ({conf_why})")

    summary = {
        "status": overall, "phase": args.phase, "model": model_key, "model_path": model_path,
        "pool_seed": args.pool_seed, "conditions": list(conditions), "n": len(pool), "k": k,
        "max_new": max_new, "max_model_len": args.max_model_len,
        "per_condition": per_cond,
        "stance_axis": stance, "stance_status": stance_status, "stance_why": stance_why,
        "conf_axis": conf, "conf_status": conf_status, "conf_why": conf_why,
        "gradeable_rate": gradeable_rate, "boxed_str_rate": boxed_str_rate, "power_ok": bool(power_ok),
        "n_graded": grade_d, "benchmarks": benches,
        "verdict": verdict, "wall_seconds": time.time() - t0,
    }

    # ── SMOKE assertions ────────────────────────────────────────────────────────────────────────
    if args.smoke:
        # (a) all requested conditions built per problem.
        for pi in range(len(pool)):
            for cond in conditions:
                assert (pi, cond) in grid, f"SMOKE FAIL: missing metrics for (pi={pi}, cond={cond})"
        # (b) suffix byte-identity at the prompt level (the steer == the imported bytes).
        q0 = pool[0]["question"]
        if "cautious_instr" in conditions:
            assert (q0 + SUFFIX["cautious_instr"]).endswith(CAUTIOUS_INSTR), "cautious suffix drift"
        if "confident_instr" in conditions:
            assert (q0 + SUFFIX["confident_instr"]).endswith(CONFIDENT_INSTR), "confident suffix drift"
        if "conf_low" in conditions:
            assert (q0 + SUFFIX["conf_low"]).endswith("confidence: 0.15"), "conf_low suffix drift"
        if "conf_high" in conditions:
            assert (q0 + SUFFIX["conf_high"]).endswith("confidence: 0.95"), "conf_high suffix drift"
        # (c) every meta_token_frac in [0,1].
        for (pi, cond), m in grid.items():
            mtf = m["meta_token_frac"]
            assert mtf is None or (0.0 <= mtf <= 1.0), f"SMOKE FAIL: meta_token_frac={mtf} out of range"
        # (d) gradeable_rate>0 — WARN not FAIL: at the smoke 256-tok budget a meta-CoT
        # model may not reach a boxed answer in the tiny sample even on a correct run.
        if (gradeable_rate or 0) == 0:
            print("[smoke] WARN: gradeable_rate==0 at the smoke token budget; "
                  "raise --max_new (e.g. 512) if this persists at larger N")
        any_conf = any(grid[(pi, c)]["n_conf_parsed"] > 0 for pi in range(len(pool)) for c in conditions)
        summary["smoke_assert"] = {
            "all_conditions_built": True, "suffix_byte_identity": True,
            "meta_token_frac_in_range": True, "gradeable_rate_pos": bool(gradeable_rate),
            "some_conf_parsed": bool(any_conf),
        }
        print(f"[smoke] conds_built=OK suffix_identity=OK mtf_in_range=OK "
              f"gradeable_rate={gradeable_rate} some_conf_parsed={any_conf}")

    payload = _json_safe({"summary": summary})
    json.dump(payload, open(out_json, "w"), indent=2)
    print(json.dumps(payload["summary"], indent=2)[:4000])
    print(f"[done:{model_key}] {out_jsonl}  +  {out_json}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), default=None,
                    help="single model key; default v8_strict (the meta self-emitter; e20a opt-in)")
    ap.add_argument("--models", default=None, help="comma-sep model keys (overrides --model)")
    ap.add_argument("--phase", type=int, choices=(1, 2), default=1,
                    help="1=DISCOVERY (n~45, seed A); 2=CONFIRMATION (n~110, FRESH seed B). Separation "
                         "is via the fresh problem pool (seed = STRATIFIED_SAMPLE_SEED + 7919 in P2).")
    ap.add_argument("--conditions", default=None,
                    help=f"comma-sep subset of {','.join(ALL_CONDITIONS)} (default = all 5)")
    ap.add_argument("--n", type=int, default=None, help="target N problems (default: phase1=45, phase2=110)")
    ap.add_argument("--n_pool", type=int, default=None, help="override the representative-pool size")
    ap.add_argument("--k", type=int, default=8, help="rolls per (problem,condition)")
    ap.add_argument("--max_new", type=int, default=4096, help="generation budget (vLLM); up to 16384")
    ap.add_argument("--max_model_len", type=int, default=20480, help="vLLM context window")
    ap.add_argument("--smoke", type=int, default=0, help="N problems, k=2, small max_new + assertions")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out_dir", default=str(REPORTS_DIR))
    args = ap.parse_args()

    if args.n is None:
        args.n = (args.smoke or (45 if args.phase == 1 else 110))
    # fresh-seed-per-phase (discovery/confirmation separation): seed A vs seed B (e2:1041 convention).
    args.pool_seed = STRATIFIED_SAMPLE_SEED + (0 if args.phase == 1 else 7919)
    if args.tag is None:
        args.tag = f"p{args.phase}"
    # conditions resolution: smoke = all 5; --conditions overrides; default = all 5.
    if args.smoke:
        args.conditions_list = list(ALL_CONDITIONS)
    elif args.conditions:
        sel = [c for c in args.conditions.split(",") if c in ALL_CONDITIONS]
        args.conditions_list = sel or list(ALL_CONDITIONS)
    else:
        args.conditions_list = list(ALL_CONDITIONS)

    if args.smoke:
        model_keys = ["v8_strict"]
    elif args.models:
        model_keys = [m for m in args.models.split(",") if m in MODELS]
    elif args.model:
        model_keys = [args.model]
    else:
        model_keys = ["v8_strict"]

    rng = _random.Random(args.pool_seed)
    rng_np = np.random.default_rng(args.pool_seed)
    t0 = time.time()
    summaries = {}
    for mk in model_keys:
        summaries[mk] = run_model(mk, args, rng, rng_np, t0)
    print("\n=== E.5 per-model status ===")
    for mk, s in summaries.items():
        print(f"  {mk:10s} {s['status']:12s} phase={s['phase']} "
              f"stance={s['stance_status']} conf={s['conf_status']}")


if __name__ == "__main__":
    main()
