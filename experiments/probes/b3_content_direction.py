"""B.3 — CONTENT-DIRECTION causal test (CTSD Phase B/C, plan_ctsd_B_probes_2026_06_01).

Intent (the CORE hypothesis, fast vLLM analog of A.3's content-inject gate):
  A.3 ran the 4-condition content gate (no-inject / marker / good-meta / bad-meta)
  on v8 with HF `model.generate` (slow). B.3 runs the SAME gate, vLLM-generated, so
  it is tractable on E20a (the substrate the user wants to train). It answers two
  pre-registered questions:
    helps     : does GOOD productive meta (re-derive + verify) beat NO-inject?
    direction : does GOOD meta beat BAD (over-confident, skip-checking) meta?
  PASS = helps AND direction → the model's continuation responds to meta CONTENT and
  DIRECTION, i.e. injected metacognitive guidance is causal. (Marker emission b-a is
  reported as descriptive context, as in A.3.)

Design (mirrors A.3 exactly; ONLY the generation MECHANISM is vLLM — every constant,
position rule, template, gate, and grading is REUSED from A.3, never reimplemented):
  Per hard-wrong problem:
    1. vLLM baseline rollout (n=1, seed=SEED+pi).
    2. HF forward → per-token RAW entropy (a3.raw_entropy).
    3. Inject position p = a3.body_argmax_entropy_pos (>=MIN_TOK, outside meta, BEFORE
       the first \\boxed) — the SAME position machinery as A.3/B.1.
    4. From the shared prefix [prompt + base[:p]], build 4 condition prefixes:
         a_noinject = base[:p]
         b_marker   = + MARKER_ONLY    ("\\n<|meta|>\\n")
         c_good     = + GOOD_META       (productive, answer-free)
         d_bad      = + BAD_META        (unproductive, answer-free)
       Batch ALL (problems × 4 conditions) into ONE vLLM generate (n=k).
    5. Grade each continuation vs gold via common.grading.robust_grade.
  Per-problem per-condition acc = fraction correct over k. Aggregate over problems
  with all 4 conditions present (paired intersection).

Pre-registered gates (A.3 / plan B.3):
  helps     : acc(c_good) - acc(a_noinject) >= +0.03 AND paired p<0.05
  direction : acc(c_good) - acc(d_bad)      >= +0.05 AND paired p<0.05
  PASS = helps AND direction.
  Power-metric fix (plan CHANGE 1): gate on gradeable_rate (math_verify can parse an
  answer from the continuation) >= 0.5, else INCONCLUSIVE — the real floor effect is
  "no parseable answer", not "no \\boxed".
  status ∈ {PASS, FAIL, INCONCLUSIVE}.

Karpathy minimal-change: imports A.3 (constants/entropy/inject/position/select),
common.grading (robust_grade/is_gradeable), common.vllm_gen (VllmGen/safe_tokenizer_path),
common.probe_utils (paired_perm_test). a3/a3b/a6/probe_utils/env are NOT modified.

Outputs reports/b3_content_direction_<tag>.json
"""
from __future__ import annotations
import argparse, json, time, gc
import random as _random
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))         # experiments/
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # repo root (src/)
from common.env import (
    SFT_V8_STRICT, EVAL_R10V2_V8, EVAL_R10V2_E20A, HF_DATASET,
    META_OPEN_ID, META_CLOSE_ID, REPORTS_DIR, STRATIFIED_SAMPLE_SEED,
)
from common.probe_utils import paired_perm_test
from common.vllm_gen import VllmGen, safe_tokenizer_path
# REUSE A.3's proven constants + position/entropy machinery (do NOT reimplement, do
# NOT modify a3). GOOD_META/BAD_META/MARKER_ONLY are the answer-free direction
# templates; body_argmax_entropy_pos is the SAME inject rule as A.3/B.1.
from probes.a3_inject_causal import (
    GOOD_META, BAD_META, MARKER_ONLY, body_argmax_entropy_pos, raw_entropy,
    first_boxed_token_idx, MIN_TOK, MAX_RESP_TOK,
)
from common.grading import robust_grade, is_gradeable
from collections import defaultdict as _defaultdict

COND = ("a_noinject", "b_marker", "c_good", "d_bad")

# E20a tokenizer fallback path (identical Qwen3 vocab) — same as B.1/B.2.
E20A_TOKENIZER_PATH = "/home/v-seungplee/sft_e20a_local"


def stratified_hard_pool(results, n_pool, rng):
    """FIX C pool sampler: n_pool hard-benchmark problems IGNORING stored is_correct.

    Sibling of a3.stratified_wrong_hard (written here so a3 stays unmodified) WITHOUT
    the `not is_correct` filter — the stored is_correct is unreliable (math_verify
    grading bug + stored/format mismatch: robust_grade gives E20a math500 ~21% vs
    stored 81%). True headroom = pool problems whose FRESH baseline robust_grade marks
    WRONG, selected below. Stratifies across the same hard benchmarks as a3."""
    hard = [r for r in results
            if r["benchmark"] in ("aime", "aime2024", "math500", "math")]
    by_b = _defaultdict(list)
    for r in hard:
        by_b[r["benchmark"]].append(r)
    picks, per = [], max(1, n_pool // max(1, len(by_b)))
    for b, lst in by_b.items():
        rng.shuffle(lst); picks.extend(lst[:per])
    rng.shuffle(picks)
    return picks[:n_pool]


def load_tokenizer(path: str):
    """Robust tokenizer load (same as B.1/B.2). The v8 checkpoint tokenizer FAILS
    under transformers 4.57.6; fall back to the E20a tokenizer (identical Qwen3 vocab,
    <|meta|>=151669, <|/meta|>=151670). Always assert the meta-token IDs so a silent
    vocab mismatch can never slip through."""
    try:
        tok = AutoTokenizer.from_pretrained(path)
    except Exception as e:
        print(f"[tok] {path} failed ({type(e).__name__}: {str(e)[:80]}); "
              f"falling back to E20a tokenizer (identical vocab)")
        tok = AutoTokenizer.from_pretrained(E20A_TOKENIZER_PATH)
    assert tok.convert_tokens_to_ids("<|meta|>") == META_OPEN_ID, "META_OPEN_ID mismatch"
    assert tok.convert_tokens_to_ids("<|/meta|>") == META_CLOSE_ID, "META_CLOSE_ID mismatch"
    return tok


def _json_safe(o):
    """Recursively cast numpy bool/float/int to python and NaN/Inf → None so the
    output is STRICT JSON (no NaN/Infinity literals)."""
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


def build_prompt_ids(tok, question: str):
    msgs = [{"role": "user", "content": question}]
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok.encode(s, add_special_tokens=False)[:1024]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=SFT_V8_STRICT, help="student that solves (v8 or E20a path)")
    ap.add_argument("--substrate", choices=["v8", "e20a"], default="v8",
                    help="selects eval json for headroom sampling (same as B.1)")
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--k", type=int, default=6, help="continuations per condition")
    ap.add_argument("--max_new", type=int, default=16384,
                    help="generation budget (real eval regime: max_tokens=16384)")
    ap.add_argument("--max_model_len", type=int, default=20480,
                    help="vLLM context window (real eval regime: 20480)")
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--tag", default="v8")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    n = args.smoke or args.n
    if args.smoke:
        args.k = 2
    out = args.out or str(REPORTS_DIR / f"b3_content_direction_{args.tag}.json")

    rng = _random.Random(STRATIFIED_SAMPLE_SEED)
    rng_np = np.random.default_rng(STRATIFIED_SAMPLE_SEED)
    t0 = time.time()

    # robust eval-record loader (handle dict/list — same as B.1)
    eval_path = EVAL_R10V2_E20A if args.substrate == "e20a" else EVAL_R10V2_V8
    ev = hf_hub_download(repo_id=HF_DATASET, repo_type="dataset", filename=eval_path)
    results = json.load(open(ev))
    results = results if isinstance(results, list) else results.get("results") or list(results.values())[0]
    # FIX C: POOL ignoring stored is_correct; headroom from fresh baselines below.
    n_pool = max(3 * n, n + 30)
    pool = stratified_hard_pool(results, n_pool, rng)
    print(f"[b3] substrate={args.substrate} model={args.model} n_target={n} "
          f"pool={len(pool)} k={args.k} max_new={args.max_new} max_model_len={args.max_model_len}")

    tok = load_tokenizer(args.model)
    dev = "cuda"
    # Generation MECHANISM = vLLM. Modest util (0.45 ≈ 36GB) so the HF entropy model
    # (~16GB, needed for raw_entropy → argmax position) coexists on the 80GB A100.
    # tokenizer_path: v8 ckpt tokenizer fails under transformers 4.57.6 → fall back
    # to the E20a tokenizer (identical vocab) via safe_tokenizer_path.
    vgen = VllmGen(args.model, tokenizer_path=safe_tokenizer_path(args.model),
                   gpu_memory_utilization=0.45, max_model_len=args.max_model_len,
                   seed=STRATIFIED_SAMPLE_SEED)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda").eval()

    # FIX C: FRESH 16k baselines for the pool → robust_grade → keep only WRONG (true
    # headroom). Each kept baseline is REUSED for entropy/argmax below (not regenerated).
    pool_prompt_ids = [build_prompt_ids(tok, r["question"]) for r in pool]
    pool_bases = vgen.generate(pool_prompt_ids, n=1, max_tokens=args.max_new,
                               seed=STRATIFIED_SAMPLE_SEED)
    probs, base_by_pi, dropped_correct = [], {}, 0
    for r, pids, outs in zip(pool, pool_prompt_ids, pool_bases):
        if len(probs) >= n:
            break
        base = outs[0][:MAX_RESP_TOK]
        full_resp = tok.decode(base, skip_special_tokens=False)
        if robust_grade(full_resp, str(r["gold_answer"]).strip()):
            dropped_correct += 1          # already-correct → NOT headroom, drop (logged)
            continue
        base_by_pi[len(probs)] = (pids, base)
        probs.append(r)
    headroom_drop_log = (f"[b3][headroom] pool={len(pool)} kept_wrong={len(probs)} "
                         f"dropped_already_correct={dropped_correct} target_n={n}")
    print(headroom_drop_log)
    if len(probs) < n:
        print(f"[b3][headroom] WARNING: only {len(probs)} robust-wrong found in pool "
              f"of {len(pool)} (< target {n}); proceeding with what we have")

    # Encode the (answer-free) A.3 templates ONCE with this tokenizer.
    marker_seg = tok.encode(MARKER_ONLY, add_special_tokens=False)
    good_seg = tok.encode(GOOD_META, add_special_tokens=False)
    bad_seg = tok.encode(BAD_META, add_special_tokens=False)
    seg_by = {"a_noinject": [], "b_marker": marker_seg, "c_good": good_seg, "d_bad": bad_seg}

    # Pass A: per-problem seeded vLLM baseline (n=1) + HF entropy → argmax p* → build
    # the 4 condition prefixes. Collect ALL (problem × condition) prefixes to batch the
    # k-sample generation in ONE vLLM call (the speed win).
    cond_prefixes, slots, info = [], [], []   # slots[i] = (pi, cond); info[pi] = meta
    for pi, r in enumerate(probs):
        q, gold = r["question"], str(r["gold_answer"]).strip()
        # FIX C: REUSE the fresh 16k baseline already generated for headroom selection.
        prompt_ids, base = base_by_pi[pi]
        # raw entropy → inject at the max-entropy PRE-ANSWER body position (a3 rule)
        H = raw_entropy(model, prompt_ids + base, len(prompt_ids), dev)
        cap = first_boxed_token_idx(tok, base)
        p_star, H_at, inject_frac = body_argmax_entropy_pos(base, H, cap)
        prefix_resp = base[:p_star]                       # response tokens before inject
        info.append({"pi": pi, "benchmark": r["benchmark"], "gold": gold,
                     "p_star": int(p_star), "H_at": float(H_at), "inject_frac": float(inject_frac),
                     "prefix_resp": prefix_resp})
        for c in COND:
            cond_prefixes.append(prompt_ids + prefix_resp + seg_by[c])
            slots.append((pi, c))
        print(f"  [pos {pi+1}/{len(probs)}] {r['benchmark']:8s} p*={p_star:4d} "
              f"H={H_at:.2f} ({time.time()-t0:.0f}s)")

    # Pass B: ONE batched vLLM generate (n=k) over ALL (problem × condition) prefixes.
    outs = vgen.generate(cond_prefixes, n=args.k, max_tokens=args.max_new,
                         seed=STRATIFIED_SAMPLE_SEED)
    # free vLLM after generation (the HF entropy model already did its work)
    vgen.free()
    del model; gc.collect(); torch.cuda.empty_cache()

    # ── grade every continuation; per-problem per-condition acc + power ────────────
    cont_by = {}
    for (pi, c), conts in zip(slots, outs):
        cont_by[(pi, c)] = conts

    grade_n, grade_d = 0, 0      # gradeable / total continuations (drives power)
    box_n = 0                    # \boxed STRING count (descriptive only)
    per_prob = []
    for meta in info:
        pi = meta["pi"]
        gold = meta["gold"]
        prefix_resp = meta["prefix_resp"]
        acc = {}
        for c in COND:
            conts = cont_by.get((pi, c), [])
            n_correct = 0
            for cnt in conts:
                full_resp = tok.decode(prefix_resp + seg_by[c] + cnt, skip_special_tokens=False)
                n_correct += 1 if robust_grade(full_resp, gold) else 0
                grade_d += 1
                if is_gradeable(full_resp):
                    grade_n += 1
                if r"\boxed" in full_resp:
                    box_n += 1
            acc[c] = (n_correct / len(conts)) if conts else None
        per_prob.append({
            "benchmark": meta["benchmark"], "gold": gold,
            "p_star": meta["p_star"], "H_at_inject": meta["H_at"], "inject_frac": meta["inject_frac"],
            "acc": acc,
            "all_conditions": all(acc.get(c) is not None for c in COND),
        })
        print(f"  [grade {pi+1}/{len(info)}] {meta['benchmark']:8s} "
              f"a={acc.get('a_noinject')} b={acc.get('b_marker')} "
              f"c={acc.get('c_good')} d={acc.get('d_bad')}")

    # ── aggregate over problems with all-4-conditions present (paired) ─────────────
    gated = [pp for pp in per_prob if pp["all_conditions"]]
    n_gated = len(gated)

    def mean_acc(c):
        v = [pp["acc"][c] for pp in gated]
        return float(np.mean(v)) if v else float("nan")

    def pdiff(c1, c2):
        return [pp["acc"][c1] - pp["acc"][c2] for pp in gated]

    A = mean_acc("a_noinject"); B = mean_acc("b_marker")
    C = mean_acc("c_good");     D = mean_acc("d_bad")
    helps_d = pdiff("c_good", "a_noinject")
    direction_d = pdiff("c_good", "d_bad")
    marker_d = pdiff("b_marker", "a_noinject")
    helps_p = paired_perm_test(helps_d, rng_np) if helps_d else None
    direction_p = paired_perm_test(direction_d, rng_np) if direction_d else None
    marker_p = paired_perm_test(marker_d, rng_np) if marker_d else None

    gate_helps = (C - A) >= 0.03 and helps_p is not None and helps_p < 0.05
    gate_direction = (C - D) >= 0.05 and direction_p is not None and direction_p < 0.05

    # Power-metric fix (plan CHANGE 1): gate on gradeable_rate (math_verify can parse
    # an answer), NOT the \boxed string. boxed_str_rate kept descriptive only.
    gradeable_rate = (grade_n / grade_d) if grade_d else None
    boxed_str_rate = (box_n / grade_d) if grade_d else None
    power_ok = (gradeable_rate is not None and gradeable_rate >= 0.5)

    # status ∈ {PASS, FAIL, INCONCLUSIVE}. Power-guard ENFORCED: a low-power run is
    # INCONCLUSIVE (truncation), never a terminal FAIL. PASS = helps AND direction.
    if not power_ok:
        status = "INCONCLUSIVE"
        verdict = (f"INCONCLUSIVE — power guard failed (gradeable_rate={gradeable_rate}); "
                   f"raise --max_new and rerun")
    elif gate_helps and gate_direction:
        status = "PASS"
        verdict = ("PASS — good meta beats both no-inject (helps) and bad meta (direction); "
                   "content/direction is causal")
    else:
        status = "FAIL"
        verdict = ("FAIL — content/direction gates miss with power OK; injected meta "
                   "content non-causal on this substrate")

    summary = {
        "status": status,
        "substrate": args.substrate, "model": args.model,
        "n_total": len(per_prob), "n_gated": n_gated, "k": args.k, "max_new": args.max_new,
        "inject_point": "body_argmax_entropy",
        "mean_acc_gated": {"a_noinject": A, "b_marker": B, "c_good": C, "d_bad": D},
        "gates": {
            "helps_c_minus_a": {"delta": C - A, "p": helps_p, "pass": bool(gate_helps)},
            "direction_c_minus_d": {"delta": C - D, "p": direction_p, "pass": bool(gate_direction)},
            "marker_b_minus_a_descriptive": {"delta": B - A, "p": marker_p},
        },
        "gradeable_rate": gradeable_rate, "boxed_str_rate": boxed_str_rate, "power_ok": bool(power_ok),
        # FIX C: fresh-baseline robust-grade headroom selection (no stored is_correct).
        "headroom_pool_size": len(pool), "headroom_kept_wrong": len(probs),
        "headroom_dropped_already_correct": dropped_correct, "headroom_target_n": n,
        "verdict": verdict, "wall_seconds": time.time() - t0,
    }
    payload = _json_safe({"summary": summary, "per_problem": per_prob})
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(payload, open(out, "w"), indent=2)
    print(json.dumps(payload["summary"], indent=2))
    print(f"[done] {out}")


if __name__ == "__main__":
    main()
