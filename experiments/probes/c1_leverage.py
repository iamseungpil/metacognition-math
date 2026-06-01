"""C.1 — Leverage gate, STEP 1 (CTSD Phase C, plan_ctsd_C1_leverage_2026_06_01 §STEP 1).

Intent (the locked Step-1 scope; see plan lines 91-121):
  B.4 was a decisive negative on teacher CONTENT-SELECTION, but it never first
  established LEVERAGE — does varying meta POSITION/CONTENT actually move the
  outcome at all? C.1 Step-1 measures that, PURELY CAUSAL, with a clean per-problem
  paired Δacc-vs-no_inject DV. It is the "good meta → correct answer?" gate:
    H-C1b-DIRECTION (line-decision): at body-frac 0.75, answer-free GOOD_META vs
      BAD_META, Δ(GOOD−BAD) ≥ +0.04 paired p<0.05 AND companion GOOD−no_inject ≥
      +0.05 paired p<0.05.
    H-C1a-POSITION: GOOD_META Δacc-vs-no_inject at 0.75 ≥ +0.05 paired p<0.05; full
      {0.25,0.5,0.75,0.9} curve reported (exploratory, Holm-adjusted, NO PASS).
  PASS → leverage exists → proceed to Step 2 (corrected outcome-conditioned teacher).
  FAIL (power_ok) → meta content has no leverage → Phase D.

Why this is CHEAPER than B.4 (the whole point):
  PURELY CAUSAL Δacc — NO teacher, NO HF forward, NO entropy, NO phase juggling
  between vLLM and HF. vLLM is the ONLY model loaded the entire run (P0 base + P0
  headroom + P2 arms all share one VllmGen; freed once at the end). Position classes
  come from token offsets via a3.first_boxed_token_idx — no entropy/HF needed.

Procedure (matches the locked scope exactly):
  P0a vLLM : ONE seeded base rollout per pool problem → response token seq `base`;
             a3.first_boxed_token_idx(tok, base) cap; position classes = body-frac
             {0.25,0.5,0.75,0.9} of [0, cap).
  P0b vLLM : per problem, k no_inject continuations from the PRIMARY prefix
             [prompt+base[:p0.75]] → robust_grade → keep problems whose no_inject
             acc ∈ (0,1) EXCLUSIVE (two-sided headroom). Log kept/ceiling/floor.
  P1  CPU  : per kept problem, per position p, build the 4 arm prefixes (differ ONLY
             in the injected segment): no_inject=[prompt+base[:p]];
             neutral=[...+MARKER_ONLY]; GOOD_META=[...+marker+GOOD_META_seg];
             BAD_META=[...+marker+BAD_META_seg]. All a3 templates are ANSWER-FREE.
  P2  vLLM : batch ALL (problem×position×arm) prefixes, k continuations each
             (seed=STRATIFIED_SAMPLE_SEED+pi), robust_grade. acc = fraction correct.
  Stats    : per-problem paired Δacc(arm,p) = acc(arm,p)−acc(no_inject,p); mean over
             problems gradeable in BOTH arms; paired_perm_test (two-sided). Power
             HARD-GATE: realized_MDE = 1.96·sd·√(2/n) per contrast; gradeable_rate≥0.5.

ANSWER-FREE VERIFICATION (locked control, plan lines 13/50-53):
  a3.GOOD_META / a3.BAD_META / a3.MARKER_ONLY are STATIC strings with NO format
  slots and NO numeric answer — they encode only metacognitive STANCE (re-check &
  verify vs over-confident & skip). They are NOT the a6 GOOD_REDIRECT_TEMPLATE,
  which embeds {gold} (answer leakage) and is DELIBERATELY NOT imported here. A
  runtime assertion (assert_answer_free) re-checks at startup that no '{' format
  slot is present, so a future edit to a3 that re-introduced a slot would fail loud.

Karpathy minimal-change: IMPORTS only — a3 (GOOD_META/BAD_META/MARKER_ONLY/
  first_boxed_token_idx), b4 (representative_pool), common.grading (robust_grade/
  is_gradeable), common.vllm_gen (VllmGen/safe_tokenizer_path), common.probe_utils
  (paired_perm_test). a3/a6/b3/b4/common are NOT modified — new file only. No
  AutoModelForCausalLM load; vLLM only.

Outputs reports/c1_leverage_<tag>.json
"""
from __future__ import annotations
import argparse, json, time, gc
import random as _random
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))         # experiments/
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # repo root (src/)
from common.env import (
    TEACHER_MODEL, EVAL_R10V2_E20A, HF_DATASET,
    META_OPEN_ID, META_CLOSE_ID, REPORTS_DIR, STRATIFIED_SAMPLE_SEED,
)
from common.vllm_gen import VllmGen, safe_tokenizer_path
from common.probe_utils import paired_perm_test
from common.grading import robust_grade, is_gradeable
from probes.a3_inject_causal import (
    GOOD_META, BAD_META, MARKER_ONLY, first_boxed_token_idx,
)
from probes.b4_teacher_steering import representative_pool

# E20a tokenizer fallback (identical Qwen3 vocab) — same as B.1/B.2/B.3/B.4.
E20A_TOKENIZER_PATH = "/home/v-seungplee/sft_e20a_local"
# Content arms (each injected after the IDENTICAL fixed-position prefix; differ ONLY
# in the injected segment). no_inject = raw prefix (the paired baseline).
ARMS = ("no_inject", "neutral", "GOOD_META", "BAD_META")
CONTENT_ARMS = ("neutral", "GOOD_META", "BAD_META")    # arms with a Δacc-vs-no_inject contrast
PRIMARY_POS = 0.75                                     # H-C1a/H-C1b primary class
FULL_POSITIONS = (0.25, 0.5, 0.75, 0.9)                # 0.75 primary; rest exploratory (Holm)
SMOKE_POSITIONS = (0.5, 0.75)
DIRECTION_THR = 0.04                                   # H-C1b-DIRECTION GOOD−BAD threshold
COMPANION_THR = 0.05                                   # H-C1b companion GOOD−no_inject threshold
POSITION_THR = 0.05                                    # H-C1a GOOD−no_inject @0.75 threshold


def assert_answer_free(*segments: str) -> None:
    """Re-check at startup that the imported a3 stance templates carry NO answer and
    NO format slot (plan leakage control). a3.GOOD_META/BAD_META/MARKER_ONLY are static
    metacognitive-STANCE strings (re-check vs over-confident) — NOT the a6
    GOOD_REDIRECT_TEMPLATE which embeds {gold}. If a future a3 edit re-introduced a
    '{...}' slot (an answer hook), fail LOUD here rather than silently leak."""
    for seg in segments:
        assert "{" not in seg and "}" not in seg, (
            f"answer-free violation: a3 template contains a format slot -> possible "
            f"answer leakage; do NOT use a6 GOOD_REDIRECT_TEMPLATE here: {seg!r}")


def load_tokenizer(path: str):
    """Robust tokenizer load (same as B.4). The v8 checkpoint tokenizer FAILS under
    transformers 4.57.6; fall back to the E20a tokenizer (identical Qwen3 vocab,
    <|meta|>=151669, <|/meta|>=151670). Always assert the meta-token IDs."""
    try:
        tok = AutoTokenizer.from_pretrained(path)
    except Exception as e:
        print(f"[tok] {path} failed ({type(e).__name__}: {str(e)[:80]}); "
              f"falling back to E20a tokenizer (identical vocab)")
        tok = AutoTokenizer.from_pretrained(E20A_TOKENIZER_PATH)
    assert tok.convert_tokens_to_ids("<|meta|>") == META_OPEN_ID, "META_OPEN_ID mismatch"
    assert tok.convert_tokens_to_ids("<|/meta|>") == META_CLOSE_ID, "META_CLOSE_ID mismatch"
    return tok


def build_prompt_ids(tok, question: str):
    msgs = [{"role": "user", "content": question}]
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok.encode(s, add_special_tokens=False)[:1024]


def _json_safe(o):
    """Recursively cast numpy bool/float/int to python and NaN/Inf -> None so the output
    is STRICT JSON. Same helper as b1/b2/b3/b4."""
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


def holm_adjust(pvals):
    """Holm-Bonferroni step-down adjusted p-values, returned IN INPUT ORDER. Used for
    the exploratory position family (plan: report Holm-adjusted p, no PASS claim).
    None entries pass through as None."""
    idx = [i for i, p in enumerate(pvals) if p is not None and not np.isnan(p)]
    m = len(idx)
    adj = [None] * len(pvals)
    order = sorted(idx, key=lambda i: pvals[i])
    running = 0.0
    for rank, i in enumerate(order):
        a = min(1.0, (m - rank) * pvals[i])
        running = max(running, a)            # enforce monotonic non-decreasing
        adj[i] = running
    return adj


def pos_token_idx(cap: int, frac: float) -> int:
    """Token index for a body-frac position class in the pre-\\boxed span [0, cap).
    The injection point is frac through the body; clamp to [0, cap-1] so a prefix
    base[:p] is always non-empty span-wise (p>=1 when cap>=1)."""
    if cap <= 0:
        return 0
    return int(max(0, min(cap - 1, round(frac * cap))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=TEACHER_MODEL, help="E20a base path (A.6 winner) — default substrate")
    ap.add_argument("--n", type=int, default=200, help="target headroom N (pool ~2.5x to net it)")
    ap.add_argument("--n_pool", type=int, default=None, help="override pool size (default ceil(2.5*n))")
    ap.add_argument("--k", type=int, default=24, help="continuations per (problem,position,arm)")
    ap.add_argument("--max_new", type=int, default=16384, help="generation budget (real eval regime)")
    ap.add_argument("--max_model_len", type=int, default=20480, help="vLLM context window")
    ap.add_argument("--smoke", type=int, default=0, help="clamp n->N, k->4, positions->{0.5,0.75}")
    ap.add_argument("--positions", default=None,
                    help="comma-sep body-fracs to run (e.g. '0.75' for the line-decision gate only); "
                         "default = full sweep. 0.75 is always force-included (gates pin to it).")
    ap.add_argument("--tag", default="e20a")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    n = args.smoke or args.n
    k = args.k if not args.smoke else 4
    if args.positions:
        positions = [float(x) for x in args.positions.split(",")]
    else:
        positions = list(SMOKE_POSITIONS if args.smoke else FULL_POSITIONS)
    # Power/companion gates pin to 0.75; guarantee it is always present (even in smoke).
    if PRIMARY_POS not in positions:
        positions.append(PRIMARY_POS)
    positions = sorted(set(positions))
    out = args.out or str(REPORTS_DIR / f"c1_leverage_{args.tag}.json")

    # ANSWER-FREE leakage guard (plan control): re-check the imported a3 stance templates
    # carry no format slot / answer. a6 GOOD_REDIRECT_TEMPLATE ({gold}) is NOT imported.
    assert_answer_free(GOOD_META, BAD_META, MARKER_ONLY)

    rng = _random.Random(STRATIFIED_SAMPLE_SEED)            # python RNG (sampling)
    rng_np = np.random.default_rng(STRATIFIED_SAMPLE_SEED)  # numpy RNG (perm tests)
    t0 = time.time()

    ev = hf_hub_download(repo_id=HF_DATASET, repo_type="dataset", filename=EVAL_R10V2_E20A)
    results = json.load(open(ev))
    results = results if isinstance(results, list) else results.get("results") or list(results.values())[0]
    n_pool = args.n_pool if args.n_pool is not None else int(np.ceil(2.5 * n))
    pool = representative_pool(results, n_pool, rng)        # reuse b4 sibling (import-only)
    print(f"[c1] model={args.model} n_target={n} pool={len(pool)} k={k} "
          f"positions={positions} max_new={args.max_new} max_model_len={args.max_model_len}")

    tok = load_tokenizer(args.model)
    marker_seg = tok.encode(MARKER_ONLY, add_special_tokens=False)
    good_seg = tok.encode(GOOD_META, add_special_tokens=False)
    bad_seg = tok.encode(BAD_META, add_special_tokens=False)
    # arm segment AFTER base[:p]: no_inject raw; neutral marker only; GOOD/BAD = marker+stance.
    # (a3 GOOD_META/BAD_META already begin with "\n<|meta|>\n", so they are self-marked;
    # neutral uses MARKER_ONLY = "\n<|meta|>\n" to isolate "a marker fired" from "stance".)
    arm_segs = {"no_inject": [], "neutral": marker_seg, "GOOD_META": good_seg, "BAD_META": bad_seg}

    # ── PHASE 0 (vLLM): ONE base rollout, then headroom no_inject baseline at 0.75 ────
    # The SAME VllmGen serves P0a base + P0b headroom + P2 arms (single model the whole run).
    vgen = VllmGen(args.model, tokenizer_path=safe_tokenizer_path(args.model),
                   gpu_memory_utilization=0.45, max_model_len=args.max_model_len,
                   seed=STRATIFIED_SAMPLE_SEED)
    pool_prompt_ids = [build_prompt_ids(tok, r["question"]) for r in pool]

    # P0a: one seeded base rollout per pool problem -> response token sequence `base`.
    pool_bases = vgen.generate(pool_prompt_ids, n=1, max_tokens=args.max_new,
                               seed=STRATIFIED_SAMPLE_SEED)
    cand = []                                              # candidate problems with a usable base
    for r, pids, outs in zip(pool, pool_prompt_ids, pool_bases):
        base = outs[0]
        cap = first_boxed_token_idx(tok, base)             # pre-\boxed span end (a3 rule)
        if cap < 2:                                        # no usable body span to inject into
            continue
        gold = str(r["gold_answer"]).strip()
        pos_idx = {p: pos_token_idx(cap, p) for p in positions}
        cand.append({"r": r, "prompt_ids": pids, "base": base, "gold": gold,
                     "cap": int(cap), "pos_idx": pos_idx,
                     "benchmark": r.get("benchmark")})
    print(f"[c1][P0a] pool={len(pool)} usable_base={len(cand)} ({time.time()-t0:.0f}s)")

    # P0b: per candidate, k no_inject continuations from the PRIMARY prefix [prompt+base[:p0.75]]
    # -> robust_grade -> two-sided headroom keep (acc in (0,1) exclusive).
    p075_prefixes = [c["prompt_ids"] + c["base"][:c["pos_idx"][PRIMARY_POS]] for c in cand]
    p075_outs = vgen.generate(p075_prefixes, n=k, max_tokens=args.max_new,
                              seed=STRATIFIED_SAMPLE_SEED)
    grade_n, grade_d, box_n = 0, 0, 0                      # power counters over ALL continuations
    kept, drop_ceiling, drop_floor = [], 0, 0
    for c, conts in zip(cand, p075_outs):
        pre_body = tok.decode(c["base"][:c["pos_idx"][PRIMARY_POS]], skip_special_tokens=False)
        nc = 0
        for cont in conts:
            full_resp = pre_body + tok.decode(cont, skip_special_tokens=False)
            nc += int(robust_grade(full_resp, c["gold"]))
            grade_d += 1
            grade_n += int(is_gradeable(full_resp))
            box_n += int(r"\boxed" in full_resp)
        base_acc = (nc / len(conts)) if conts else 0.0
        if base_acc <= 0.0:
            drop_floor += 1
            continue
        if base_acc >= 1.0:
            drop_ceiling += 1
            continue
        c["headroom_base_acc"] = base_acc                  # 0.75 no_inject acc, for context
        kept.append(c)
    # W1: keep the first n QUALIFIERS (pool is pre-shuffled by representative_pool, so this
    # is a random n of the headroom population — NOT front-biased toward the pool head). The
    # earlier per-iter `break` discarded already-generated headroom problems and only weakened
    # power; baselines are already paid for, so filter-then-cap is free.
    kept = kept[:n]
    print(f"[c1][P0b headroom] candidates={len(cand)} kept={len(kept)} "
          f"dropped_ceiling={drop_ceiling} dropped_floor={drop_floor} target_n={n} "
          f"({time.time()-t0:.0f}s)")

    # ── PHASE 1 (CPU): build per (problem, position, arm) prefixes ────────────────────
    # Each arm differs from no_inject ONLY in the injected segment (problem difficulty
    # cancels in the paired Δacc). assign each problem a stable pi index for seeding.
    prefix_by_slot = {}                                    # (pi, pos, arm) -> prompt+base[:p]+seg
    pre_body_text = {}                                     # (pi, pos) -> decoded base[:p] (for grading)
    for pi, c in enumerate(kept):
        c["pi"] = pi
        prompt_ids, base = c["prompt_ids"], c["base"]
        for p in positions:
            p_idx = c["pos_idx"][p]
            base_prefix = base[:p_idx]
            pre_body_text[(pi, p)] = tok.decode(base_prefix, skip_special_tokens=False)
            for arm in ARMS:
                seg = arm_segs[arm]
                prefix_by_slot[(pi, p, arm)] = prompt_ids + base_prefix + seg
    print(f"[c1][P1 CPU] kept={len(kept)} positions={len(positions)} arms={len(ARMS)} "
          f"-> {len(prefix_by_slot)} (problem x position x arm) prefixes")

    # ── PHASE 2 (vLLM, same model): k continuations per (problem, position, arm) ──────
    # Seed per problem = STRATIFIED_SAMPLE_SEED + pi (deterministic). One batched call.
    # vLLM SamplingParams takes a single seed; batch per pi so seeds vary by problem.
    cont_by = {}                                           # (pi, pos, arm) -> list[continuation ids]
    slots_by_pi = {}
    for slot in prefix_by_slot:
        slots_by_pi.setdefault(slot[0], []).append(slot)
    for pi in sorted(slots_by_pi):
        slots = slots_by_pi[pi]
        prefixes = [prefix_by_slot[s] for s in slots]
        outs = vgen.generate(prefixes, n=k, max_tokens=args.max_new,
                             seed=STRATIFIED_SAMPLE_SEED + pi)
        for s, o in zip(slots, outs):
            cont_by[s] = o
        print(f"  [P2 {pi+1}/{len(kept)}] {kept[pi]['benchmark']:8s} cap={kept[pi]['cap']} "
              f"({time.time()-t0:.0f}s)")
    vgen.free(); gc.collect(); torch.cuda.empty_cache()    # free the ONLY model at run end

    # grade every continuation -> acc[(pi,pos,arm)] = fraction robust_grade-correct.
    # W2: track gradeable counts PER POSITION so the power gate can use the 0.75
    # line-decision arms specifically (a bad EXPLORATORY position must not be able to
    # drag a global gradeable_rate under 0.5 and spuriously force the whole run INCONCLUSIVE).
    acc = {}                                               # (pi, pos, arm) -> acc in [0,1] or None
    grade_n_pos = {p: 0 for p in positions}                # gradeable continuations per position
    grade_d_pos = {p: 0 for p in positions}
    for (pi, p, arm), conts in cont_by.items():
        seg = arm_segs[arm]
        seg_text = tok.decode(seg, skip_special_tokens=False) if seg else ""
        pre_body = pre_body_text[(pi, p)]
        nc = 0
        for c in conts:
            full_resp = pre_body + seg_text + tok.decode(c, skip_special_tokens=False)
            nc += int(robust_grade(full_resp, kept[pi]["gold"]))
            grade_d += 1; grade_d_pos[p] += 1
            g = int(is_gradeable(full_resp)); grade_n += g; grade_n_pos[p] += g
            box_n += int(r"\boxed" in full_resp)
        acc[(pi, p, arm)] = (nc / len(conts)) if conts else None

    # ── AGGREGATE: per-position per-arm paired Δacc-vs-no_inject + perm p + MDE ────────
    def paired_diffs(p, arm):
        """Per-problem paired Δacc(arm,p) − Δacc(no_inject,p), over problems gradeable
        (acc not None) in BOTH arms at position p."""
        ds = []
        for pi in range(len(kept)):
            a_arm = acc.get((pi, p, arm))
            a_ni = acc.get((pi, p, "no_inject"))
            if a_arm is None or a_ni is None:
                continue
            ds.append(a_arm - a_ni)
        return ds

    def contrast(diffs):
        """(mean, sd, n, paired-perm p, realized_MDE) for a list of paired diffs."""
        d = np.asarray([x for x in diffs if x is not None and not np.isnan(x)])
        nd = len(d)
        if nd == 0:
            return None, None, 0, None, None
        mean = float(d.mean())
        sd = float(np.std(d, ddof=1)) if nd > 1 else None
        pval = float(paired_perm_test(d.tolist(), rng_np))
        mde = (1.96 * sd * np.sqrt(2.0 / nd)) if (sd is not None and nd > 0) else None
        return mean, sd, nd, pval, (float(mde) if mde is not None else None)

    def paired_pairwise(p, arm_a, arm_b):
        """Per-problem paired Δ = Δacc(arm_a,p) − Δacc(arm_b,p) = acc(arm_a)−acc(arm_b),
        over problems gradeable in BOTH arm_a and arm_b at p. (no_inject cancels.)"""
        ds = []
        for pi in range(len(kept)):
            a = acc.get((pi, p, arm_a))
            b = acc.get((pi, p, arm_b))
            if a is None or b is None:
                continue
            ds.append(a - b)
        return ds

    # per-position curve: each content arm's Δacc-vs-no_inject at every position.
    per_position_curve = {}
    holm_input_p, holm_input_keys = [], []                 # GOOD_META exploratory family for Holm
    for p in positions:
        per_position_curve[f"{p}"] = {}
        for arm in CONTENT_ARMS:
            mean, sd, nd, pval, mde = contrast(paired_diffs(p, arm))
            per_position_curve[f"{p}"][arm] = {
                "delta_vs_no_inject": mean, "sd": sd, "n_paired": nd,
                "p": pval, "realized_mde": mde}
            if arm == "GOOD_META":
                holm_input_p.append(pval)
                holm_input_keys.append(f"{p}")
    # Holm-adjusted p over the GOOD_META position family (exploratory; no PASS claim).
    holm_adj = holm_adjust(holm_input_p)
    for key, adj in zip(holm_input_keys, holm_adj):
        per_position_curve[key]["GOOD_META"]["p_holm"] = adj

    # power: gradeable_rate. Report the GLOBAL rate (P0b + all P2) for transparency, but
    # GATE on the PRIMARY-position (0.75) rate — the gates are all at 0.75, so a bad
    # EXPLORATORY position must not be able to drag the global rate under 0.5 and spuriously
    # flip the whole run INCONCLUSIVE (W2). Falls back to global if 0.75 wasn't run.
    gradeable_rate = (grade_n / grade_d) if grade_d else None
    boxed_str_rate = (box_n / grade_d) if grade_d else None
    gradeable_rate_primary = ((grade_n_pos[PRIMARY_POS] / grade_d_pos[PRIMARY_POS])
                              if grade_d_pos.get(PRIMARY_POS) else gradeable_rate)
    power_ok = (gradeable_rate_primary is not None and gradeable_rate_primary >= 0.5)

    # ── H-C1a-POSITION (primary class 0.75): GOOD_META Δacc-vs-no_inject >= +0.05 ─────
    c1a_mean, c1a_sd, c1a_n, c1a_p, c1a_mde = contrast(paired_diffs(PRIMARY_POS, "GOOD_META"))
    c1a_mde_ok = (c1a_mde is not None and c1a_mde <= POSITION_THR)
    c1a_pass = (power_ok and c1a_mde_ok and c1a_mean is not None
                and c1a_mean >= POSITION_THR and c1a_p is not None and c1a_p < 0.05)

    # ── H-C1b-DIRECTION (line-decision @0.75): GOOD−BAD >= +0.04 AND companion >= +0.05 ─
    dir_mean, dir_sd, dir_n, dir_p, dir_mde = contrast(paired_pairwise(PRIMARY_POS, "GOOD_META", "BAD_META"))
    comp_mean, comp_sd, comp_n, comp_p, comp_mde = contrast(paired_diffs(PRIMARY_POS, "GOOD_META"))
    dir_mde_ok = (dir_mde is not None and dir_mde <= DIRECTION_THR)
    comp_mde_ok = (comp_mde is not None and comp_mde <= COMPANION_THR)
    dir_effect_ok = (dir_mean is not None and dir_mean >= DIRECTION_THR
                     and dir_p is not None and dir_p < 0.05)
    comp_effect_ok = (comp_mean is not None and comp_mean >= COMPANION_THR
                      and comp_p is not None and comp_p < 0.05)
    direction_pass = (power_ok and dir_mde_ok and comp_mde_ok and dir_effect_ok and comp_effect_ok)
    # the DIRECTION contrast is under-powered if EITHER its own or the companion MDE > thr.
    direction_under_mde = (not dir_mde_ok) or (not comp_mde_ok)

    # ── STATUS (locked verdict logic) ─────────────────────────────────────────────────
    # INCONCLUSIVE if NOT power_ok OR DIRECTION under-MDE (KILL physically guarded behind
    # realized_MDE <= threshold; oracle deferred so DIRECTION is the line gate). PASS if
    # DIRECTION passes; else FAIL (power_ok, MDE<=thr, but null/negative).
    if (not power_ok) or direction_under_mde:
        status = "INCONCLUSIVE"
        why = (f"primary-position gradeable_rate {gradeable_rate_primary} < 0.5 (truncation/floor)" if not power_ok
               else f"DIRECTION under-powered (realized_MDE dir={dir_mde}/comp={comp_mde} "
                    f"> thr {DIRECTION_THR}/{COMPANION_THR})")
        verdict = (f"INCONCLUSIVE — {why}; leverage gate cannot resolve -> re-run with more "
                   f"k/N (NEVER read as a substantive null)")
    elif direction_pass:
        status = "PASS"
        verdict = ("PASS — answer-free GOOD_META beats BAD_META (DIRECTION) AND beats no_inject "
                   "(companion) @0.75; meta CONTENT has causal leverage -> proceed to Step 2 "
                   "(corrected outcome-conditioned teacher vs per-meta causal Δacc)")
    else:
        status = "FAIL_AT_0.75"
        verdict = ("FAIL_AT_0.75 — power OK & MDE<=thr but DIRECTION null/negative AT POSITION 0.75. "
                   "This does NOT license a global KILL: 0.75 is a plausible-but-UNVALIDATED position "
                   "(B.4 gave only a BETWEEN-problem observational corr ~0.39, NOT a causal position "
                   "sweep). A single-position null is ambiguous (no-content-leverage vs wrong-position) "
                   "-> the {0.25,0.5,0.9} position sweep is REQUIRED to disambiguate BEFORE any "
                   "KILL/Phase-D claim. (PASS@0.75 would be self-sufficient; FAIL@0.75 is not.)")

    summary = {
        "status": status,
        "model": args.model,
        "n_target": n, "n_pool": len(pool), "n_usable_base": len(cand), "n_kept": len(kept),
        "dropped_ceiling": drop_ceiling, "dropped_floor": drop_floor,
        "k": k, "positions": [float(p) for p in positions], "primary_position": PRIMARY_POS,
        "max_new": args.max_new, "max_model_len": args.max_model_len,
        "arms": list(ARMS), "answer_free_verified": True,
        # H-C1b-DIRECTION (line-decision)
        "direction_delta_good_minus_bad": dir_mean, "direction_sd": dir_sd,
        "direction_n_paired": dir_n, "direction_p": dir_p, "direction_realized_mde": dir_mde,
        "direction_thr": DIRECTION_THR,
        "companion_delta_good_minus_noinject": comp_mean, "companion_p": comp_p,
        "companion_realized_mde": comp_mde, "companion_thr": COMPANION_THR,
        "direction_pass": bool(direction_pass), "direction_under_mde": bool(direction_under_mde),
        # H-C1a-POSITION (primary class)
        "c1a_good_delta_vs_no_inject@0.75": c1a_mean, "c1a_sd": c1a_sd, "c1a_n_paired": c1a_n,
        "c1a_p": c1a_p, "c1a_realized_mde": c1a_mde, "c1a_thr": POSITION_THR,
        "c1a_pass": bool(c1a_pass),
        # power HARD-GATE
        "gradeable_rate": gradeable_rate, "gradeable_rate_primary": gradeable_rate_primary,
        "boxed_str_rate": boxed_str_rate,
        "power_ok": bool(power_ok), "n_graded_continuations": grade_d,
        "verdict": verdict, "wall_seconds": time.time() - t0,
    }

    # per-(problem,position,arm) rows for audit (acc + headroom base acc).
    rows = []
    for pi, c in enumerate(kept):
        rows.append({
            "pi": pi, "benchmark": c["benchmark"], "cap": c["cap"],
            "headroom_base_acc@0.75": c.get("headroom_base_acc"),
            "pos_idx": {f"{p}": c["pos_idx"][p] for p in positions},
            "acc": {f"{p}": {arm: acc.get((pi, p, arm)) for arm in ARMS} for p in positions},
        })

    payload = _json_safe({"summary": summary, "rows": rows,
                          "per_position_curve": per_position_curve})
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(payload, open(out, "w"), indent=2)
    print(json.dumps(payload["summary"], indent=2))
    print(f"[done] {out}")


if __name__ == "__main__":
    main()
