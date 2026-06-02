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
  P0b vLLM : per problem, PER POSITION p, k no_inject continuations from THAT
             position's prefix [prompt+base[:p]] → robust_grade → keep, FOR THAT
             position, problems whose no_inject acc ∈ (0,1) EXCLUSIVE (two-sided
             headroom). The per-position kept sets DIFFER (a problem the 0.75 prefix
             already solved can still have headroom at 0.25/0.5 — the earlier prefix
             has not committed the answer). Log kept/ceiling/floor PER position. These
             SAME no_inject continuations are REUSED as the no_inject ARM at P2 (same
             prefix, same seed) — no_inject is NEVER regenerated.
  P1  CPU  : per (problem, position-in-its-kept-set), build the 3 CONTENT arm prefixes
             (differ ONLY in the injected segment, no_inject already in hand):
             neutral=[...+MARKER_ONLY]; GOOD_META=[...+marker+GOOD_META_seg];
             BAD_META=[...+marker+BAD_META_seg]. All a3 templates are ANSWER-FREE.
  P2  vLLM : batch the (problem×position×CONTENT_arm) prefixes ONLY for that
             position's kept problems, k continuations each
             (seed=STRATIFIED_SAMPLE_SEED+pi), robust_grade. acc = fraction correct.
             no_inject acc comes straight from the P0b headroom gen (no double-gen).
  Stats    : per-problem paired Δacc(arm,p) = acc(arm,p)−acc(no_inject,p); mean over
             that position's kept set gradeable in BOTH arms; paired_perm_test
             (two-sided). Power HARD-GATE: realized_MDE = 1.96·sd·√(2/n) per contrast;
             gradeable_rate computed PER position; a position is INCONCLUSIVE if its
             own realized_MDE > thr.

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

    # P0b: PER POSITION p, k no_inject continuations from THAT position's prefix
    # [prompt+base[:p]] -> robust_grade -> per-position two-sided headroom keep (acc in
    # (0,1) exclusive). The per-position kept sets DIFFER on purpose: a problem the 0.75
    # prefix already solved (ceiling, dropped @0.75) may still have headroom at 0.25/0.5,
    # where the earlier prefix has NOT committed the answer — filtering every position by
    # the 0.75 set would bias the early-position test to a "hard-even-at-0.75" subset and
    # HIDE the early-position content leverage (C.1-core: content helps where the prefix
    # has not yet decided). Batch ALL (candidate x position) prefixes in ONE call.
    #
    # NO-DOUBLE-GEN: these no_inject continuations are the no_inject ARM at P2. We key them
    # by (pi, p) into cont_by below and SKIP no_inject in P2. Same prefix [prompt+base[:p]],
    # same seed (STRATIFIED_SAMPLE_SEED) — bit-identical reuse, not a re-roll.
    # Power counters (global + PER position). no_inject continuations are graded HERE
    # (once) for the headroom decision and counted into the power tallies; the P2 grading
    # loop must NOT re-count no_inject (it only re-reads its acc), so there is no
    # double-counting of the reused no_inject arm.
    grade_n, grade_d, box_n = 0, 0, 0                      # power counters over ALL continuations
    grade_n_pos = {p: 0 for p in positions}                # gradeable continuations per position
    grade_d_pos = {p: 0 for p in positions}
    ni_slots = [(ci, p) for ci, _ in enumerate(cand) for p in positions]
    ni_prefixes = [cand[ci]["prompt_ids"] + cand[ci]["base"][:cand[ci]["pos_idx"][p]]
                   for (ci, p) in ni_slots]
    ni_outs = vgen.generate(ni_prefixes, n=k, max_tokens=args.max_new,
                            seed=STRATIFIED_SAMPLE_SEED)
    ni_conts = {}                                          # (ci, p) -> no_inject continuations (REUSED at P2)
    ni_acc = {}                                            # (ci, p) -> no_inject acc (headroom decider)
    for (ci, p), conts in zip(ni_slots, ni_outs):
        c = cand[ci]
        pre_body = tok.decode(c["base"][:c["pos_idx"][p]], skip_special_tokens=False)
        nc = 0
        for cont in conts:
            full_resp = pre_body + tok.decode(cont, skip_special_tokens=False)
            nc += int(robust_grade(full_resp, c["gold"]))
            grade_d += 1; grade_d_pos[p] += 1
            g = int(is_gradeable(full_resp)); grade_n += g; grade_n_pos[p] += g
            box_n += int(r"\boxed" in full_resp)
        ni_conts[(ci, p)] = conts
        ni_acc[(ci, p)] = (nc / len(conts)) if conts else 0.0

    # Per-position headroom keep: FOR EACH position p, keep problems with no_inject acc in
    # (0,1) exclusive. kept_by_pos[p] = list of candidate indices kept AT THAT position.
    # W1: keep the first n QUALIFIERS at each position (pool pre-shuffled by representative_pool
    # -> random n of that position's headroom population, NOT front-biased). Baselines already
    # paid for, so filter-then-cap is free.
    kept_by_pos, drop_ceiling_pos, drop_floor_pos = {}, {}, {}
    for p in positions:
        keep_ci, ceil_c, floor_c = [], 0, 0
        for ci in range(len(cand)):
            a = ni_acc[(ci, p)]
            if a <= 0.0:
                floor_c += 1
            elif a >= 1.0:
                ceil_c += 1
            else:
                keep_ci.append(ci)
        kept_by_pos[p] = keep_ci[:n]
        drop_ceiling_pos[p], drop_floor_pos[p] = ceil_c, floor_c
        print(f"[c1][P0b headroom p={p}] candidates={len(cand)} kept={len(kept_by_pos[p])} "
              f"dropped_ceiling={ceil_c} dropped_floor={floor_c} target_n={n} "
              f"({time.time()-t0:.0f}s)")
    # Union of candidates kept at ANY position -> these get a stable pi index and per-(pi,p)
    # arm slots. A problem only contributes to positions where it is in that position's kept set.
    kept_ci = sorted({ci for p in positions for ci in kept_by_pos[p]})
    kept = [cand[ci] for ci in kept_ci]
    pi_of_ci = {ci: pi for pi, ci in enumerate(kept_ci)}
    # per-(pi,p) membership flag (is this problem in position p's headroom set?)
    in_headroom = {(pi_of_ci[ci], p): True
                   for p in positions for ci in kept_by_pos[p]}

    # ── PHASE 1 (CPU): build per (problem, position, CONTENT_arm) prefixes ────────────
    # ONLY for that position's headroom-kept problems (do NOT waste gen on ceiling/floor
    # problems at that position). no_inject is NOT rebuilt here — its continuations are
    # already in ni_conts (reused below). Each content arm differs from no_inject ONLY in
    # the injected segment (problem difficulty cancels in the paired Δacc). pi is the stable
    # union index assigned in P0b (pi_of_ci) and used for seeding.
    cont_by = {}                                           # (pi, pos, arm) -> list[continuation ids]
    prefix_by_slot = {}                                    # (pi, pos, CONTENT_arm) -> prompt+base[:p]+seg
    pre_body_text = {}                                     # (pi, pos) -> decoded base[:p] (for grading)
    for ci in kept_ci:
        c = cand[ci]
        pi = pi_of_ci[ci]
        c["pi"] = pi
        prompt_ids, base = c["prompt_ids"], c["base"]
        for p in positions:
            if (pi, p) not in in_headroom:                 # not in THIS position's headroom set
                continue
            p_idx = c["pos_idx"][p]
            base_prefix = base[:p_idx]
            pre_body_text[(pi, p)] = tok.decode(base_prefix, skip_special_tokens=False)
            cont_by[(pi, p, "no_inject")] = ni_conts[(ci, p)]   # REUSE P0b gen as no_inject arm
            for arm in CONTENT_ARMS:
                seg = arm_segs[arm]
                prefix_by_slot[(pi, p, arm)] = prompt_ids + base_prefix + seg
    print(f"[c1][P1 CPU] kept_union={len(kept)} positions={len(positions)} "
          f"content_arms={len(CONTENT_ARMS)} -> {len(prefix_by_slot)} "
          f"(problem x position x content_arm) prefixes (no_inject reused from P0b)")

    # ── PHASE 2 (vLLM, same model): k continuations per (problem, position, CONTENT_arm) ─
    # Seed per problem = STRATIFIED_SAMPLE_SEED + pi (deterministic). One batched call.
    # vLLM SamplingParams takes a single seed; batch per pi so seeds vary by problem.
    # cont_by already holds the no_inject arm (reused from P0b in P1); P2 only adds the
    # CONTENT arms (neutral/GOOD_META/BAD_META) -> no_inject is NEVER regenerated.
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
    # W2: per-position gradeable counts (grade_*_pos) drive the per-position power gate (a
    # bad EXPLORATORY position cannot drag a global rate under 0.5 and spuriously flip the
    # run INCONCLUSIVE). no_inject was already graded + power-counted in P0b — recompute its
    # acc here for the paired Δacc but DO NOT re-increment the power counters (no double-count).
    acc = {}                                               # (pi, pos, arm) -> acc in [0,1] or None
    for (pi, p, arm), conts in cont_by.items():
        seg = arm_segs[arm]
        seg_text = tok.decode(seg, skip_special_tokens=False) if seg else ""
        pre_body = pre_body_text[(pi, p)]
        nc = 0
        for c in conts:
            full_resp = pre_body + seg_text + tok.decode(c, skip_special_tokens=False)
            nc += int(robust_grade(full_resp, kept[pi]["gold"]))
            if arm != "no_inject":                         # no_inject power already tallied in P0b
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

    # headroom band for a problem at position p (bin by its no_inject acc). The C.1-core
    # result lives in the LOW band (prefix not yet decided -> content moves the answer), so
    # we split every position by band in the JSON. ni_acc[(ci,p)] is the per-position
    # no_inject acc; pi->ci recovered via kept_ci.
    HEADROOM_BANDS = (("0.0-0.4", 0.0, 0.4), ("0.4-0.7", 0.4, 0.7), ("0.7-1.0", 0.7, 1.0001))
    def band_of(pi, p):
        a = ni_acc.get((kept_ci[pi], p))
        if a is None:
            return None
        for name, lo, hi in HEADROOM_BANDS:
            if lo <= a < hi:
                return name
        return None

    def split_diffs(p, arm, predicate):
        """paired Δacc(arm,p)−Δacc(no_inject,p) over problems at p (gradeable in both arms)
        that ALSO satisfy predicate(pi) — used for per-band / per-benchmark splits."""
        ds = []
        for pi in range(len(kept)):
            a_arm = acc.get((pi, p, arm)); a_ni = acc.get((pi, p, "no_inject"))
            if a_arm is None or a_ni is None or not predicate(pi):
                continue
            ds.append(a_arm - a_ni)
        return ds

    def contrast_dict(mean, sd, nd, pval, mde):
        return {"delta": mean, "sd": sd, "n_paired": nd, "p": pval, "realized_mde": mde}

    # per-position curve: each content arm's Δacc-vs-no_inject at every position, PLUS the
    # per-band and per-benchmark split for GOOD_META (so the gsm8k/math500 sign inversion
    # and the headroom-band dependence are in the JSON, not recomputed from rows).
    benchmarks = sorted({c["benchmark"] for c in kept if c["benchmark"] is not None})
    per_position_curve = {}
    holm_input_p, holm_input_keys = [], []                 # GOOD_META exploratory family for Holm
    for p in positions:
        per_position_curve[f"{p}"] = {}
        for arm in CONTENT_ARMS:
            mean, sd, nd, pval, mde = contrast(paired_diffs(p, arm))
            per_position_curve[f"{p}"][arm] = contrast_dict(mean, sd, nd, pval, mde)
            if arm == "GOOD_META":
                holm_input_p.append(pval)
                holm_input_keys.append(f"{p}")
        # GOOD_META split by headroom band and by benchmark (paired Δ vs no_inject at p).
        per_position_curve[f"{p}"]["GOOD_META_by_band"] = {
            name: contrast_dict(*contrast(split_diffs(p, "GOOD_META",
                                lambda pi, _n=name: band_of(pi, p) == _n)))
            for name, _, _ in HEADROOM_BANDS}
        per_position_curve[f"{p}"]["GOOD_META_by_benchmark"] = {
            bench: contrast_dict(*contrast(split_diffs(p, "GOOD_META",
                                 lambda pi, _b=bench: kept[pi]["benchmark"] == _b)))
            for bench in benchmarks}
    # Holm-adjusted p over the GOOD_META position family (exploratory; no PASS claim).
    holm_adj = holm_adjust(holm_input_p)
    for key, adj in zip(holm_input_keys, holm_adj):
        per_position_curve[key]["GOOD_META"]["p_holm"] = adj

    # power: gradeable_rate. Report the GLOBAL rate (P0b + all P2) for transparency, and a
    # PER-POSITION rate (W2) so each position is gated on ITS OWN power — a bad exploratory
    # position cannot drag a global rate under 0.5 and spuriously flip another position.
    gradeable_rate = (grade_n / grade_d) if grade_d else None
    boxed_str_rate = (box_n / grade_d) if grade_d else None
    gradeable_rate_pos = {p: ((grade_n_pos[p] / grade_d_pos[p]) if grade_d_pos.get(p) else None)
                          for p in positions}
    power_ok_pos = {p: (gradeable_rate_pos[p] is not None and gradeable_rate_pos[p] >= 0.5)
                    for p in positions}
    _grp = gradeable_rate_pos.get(PRIMARY_POS)             # explicit None check: a real 0.0 is valid
    gradeable_rate_primary = _grp if _grp is not None else gradeable_rate
    power_ok = bool(power_ok_pos.get(PRIMARY_POS, False))

    # ── PER-POSITION result + PASS test (pre-registered) ───────────────────────────────
    # For EACH position p compute, over THAT position's headroom-kept set:
    #   companion Δ(GOOD−no_inject)  (>= COMPANION_THR, p<0.05)
    #   direction Δ(GOOD−BAD)        (>= DIRECTION_THR, p<0.05)
    #   c1a       Δ(GOOD−no_inject)  (== companion; reported for H-C1a continuity)
    # Position p PASSES iff power_ok_pos[p] AND companion>=+0.05 p<0.05 AND direction>=+0.04
    # p<0.05 AND realized_MDE(both) <= thr. A position is INCONCLUSIVE if its own MDE>thr.
    per_position_result = {}
    for p in positions:
        cm, csd, cn, cp, cmde = contrast(paired_diffs(p, "GOOD_META"))           # companion / c1a
        dm, dsd, dn, dp, dmde = contrast(paired_pairwise(p, "GOOD_META", "BAD_META"))  # direction
        comp_mde_ok_p = (cmde is not None and cmde <= COMPANION_THR)
        dir_mde_ok_p = (dmde is not None and dmde <= DIRECTION_THR)
        under_mde_p = (not comp_mde_ok_p) or (not dir_mde_ok_p)
        comp_ok_p = (cm is not None and cm >= COMPANION_THR and cp is not None and cp < 0.05)
        dir_ok_p = (dm is not None and dm >= DIRECTION_THR and dp is not None and dp < 0.05)
        pass_p = bool(power_ok_pos[p] and not under_mde_p and comp_ok_p and dir_ok_p)
        per_position_result[f"{p}"] = {
            "n_kept": len(kept_by_pos[p]),
            "ceiling": drop_ceiling_pos[p], "floor": drop_floor_pos[p],
            "gradeable_rate": gradeable_rate_pos[p], "power_ok": bool(power_ok_pos[p]),
            "companion_delta_good_minus_noinject": cm, "companion_sd": csd,
            "companion_n_paired": cn, "companion_p": cp, "companion_realized_mde": cmde,
            "direction_delta_good_minus_bad": dm, "direction_sd": dsd,
            "direction_n_paired": dn, "direction_p": dp, "direction_realized_mde": dmde,
            "c1a_good_delta_vs_no_inject": cm, "c1a_p": cp, "c1a_realized_mde": cmde,
            "under_mde": bool(under_mde_p), "pass": pass_p,
        }

    # primary-position scalars kept for backward-compatible top-level summary keys.
    pr = per_position_result[f"{PRIMARY_POS}"]
    c1a_mean, c1a_p, c1a_mde = pr["c1a_good_delta_vs_no_inject"], pr["c1a_p"], pr["c1a_realized_mde"]
    # H-C1a is companion-ONLY (GOOD−no_inject @0.75 >= POSITION_THR, p<0.05, power_ok, MDE<=thr).
    # It must NOT require the GOOD−BAD direction contrast (that is H-C1b, the v["pass"] gate).
    c1a_pass = (c1a_mean is not None and c1a_mean >= POSITION_THR
                and c1a_p is not None and c1a_p < 0.05
                and not pr["under_mde"] and pr["power_ok"])
    dir_mean, dir_p, dir_mde = (pr["direction_delta_good_minus_bad"],
                                pr["direction_p"], pr["direction_realized_mde"])
    comp_mean, comp_p, comp_mde = (pr["companion_delta_good_minus_noinject"],
                                   pr["companion_p"], pr["companion_realized_mde"])

    # ── STATUS (per-position verdict logic, pre-registered) ────────────────────────────
    # PASS  : ANY tested position passes (power_ok_pos[p] & companion>=+0.05 & direction>=+0.04,
    #         both p<0.05, both realized_MDE<=thr) -> meta CONTENT has causal leverage at >=1
    #         position -> proceed to Step 2.
    # INCONCLUSIVE: no position passes AND the best (most-powered / non-under-MDE) position is
    #         still under-MDE or under-power -> the gate cannot resolve (NEVER a substantive null).
    # FAIL_position: only if BOTH 0.25 AND 0.5 are powered (MDE<=thr) AND each has companion<+0.02
    #         & p>=0.05 -> the early (undecided-prefix) positions, where content SHOULD help,
    #         show no leverage -> licenses a KILL/Phase-D claim. A single-position null at 0.75
    #         alone is ambiguous (no-content-leverage vs wrong-position) and does NOT FAIL.
    passing_positions = [float(kk) for kk, v in per_position_result.items() if v["pass"]]
    EARLY = (0.25, 0.5)
    early_present = [p for p in EARLY if p in positions]
    def _early_null(p):
        v = per_position_result[f"{p}"]
        return (v["power_ok"] and not v["under_mde"]
                and v["companion_delta_good_minus_noinject"] is not None
                and v["companion_delta_good_minus_noinject"] < 0.02
                and v["companion_p"] is not None and v["companion_p"] >= 0.05)
    fail_positions = bool(early_present) and all(p in positions for p in EARLY) \
        and all(_early_null(p) for p in EARLY)

    if passing_positions:
        status = "PASS"
        verdict = (f"PASS — answer-free GOOD_META beats BAD_META (direction) AND beats no_inject "
                   f"(companion) at position(s) {sorted(passing_positions)} of the per-position "
                   f"headroom sweep; meta CONTENT has causal leverage -> proceed to Step 2 "
                   f"(corrected outcome-conditioned teacher vs per-meta causal Δacc)")
    elif fail_positions:
        status = "FAIL_position"
        verdict = ("FAIL_position — both early positions {0.25,0.5} are powered (MDE<=thr) yet "
                   "companion Δ(GOOD−no_inject)<+0.02 & p>=0.05. Content has NO leverage even where "
                   "the prefix has NOT yet decided (the band C.1-core predicted content should help) "
                   "-> licenses KILL / Phase-D.")
    else:
        status = "INCONCLUSIVE"
        # best position = the one with the smallest companion realized_MDE among tested.
        mdes = [(per_position_result[f"{p}"]["companion_realized_mde"], p) for p in positions
                if per_position_result[f"{p}"]["companion_realized_mde"] is not None]
        best = min(mdes)[1] if mdes else None
        verdict = (f"INCONCLUSIVE — no position passes and the best-powered position "
                   f"(p={best}, companion_MDE={per_position_result[f'{best}']['companion_realized_mde'] if best is not None else None}) "
                   f"is still under-MDE or under-power, or the early positions are not both powered "
                   f"for a FAIL; leverage gate cannot resolve -> re-run with more k/N "
                   f"(NEVER read as a substantive null)")

    # primary-position (0.75) scalars for the per-position result block, surfaced top-level.
    summary = {
        "status": status,
        "model": args.model,
        "n_target": n, "n_pool": len(pool), "n_usable_base": len(cand),
        "n_kept_union": len(kept), "n_kept_by_position": {f"{p}": len(kept_by_pos[p]) for p in positions},
        "dropped_ceiling_by_position": {f"{p}": drop_ceiling_pos[p] for p in positions},
        "dropped_floor_by_position": {f"{p}": drop_floor_pos[p] for p in positions},
        "k": k, "positions": [float(p) for p in positions], "primary_position": PRIMARY_POS,
        "max_new": args.max_new, "max_model_len": args.max_model_len,
        "arms": list(ARMS), "answer_free_verified": True,
        "passing_positions": sorted(passing_positions),
        # primary-position (0.75) summary (full per-position detail in per_position_result)
        "direction_delta_good_minus_bad@0.75": dir_mean, "direction_p@0.75": dir_p,
        "direction_realized_mde@0.75": dir_mde, "direction_thr": DIRECTION_THR,
        "companion_delta_good_minus_noinject@0.75": comp_mean, "companion_p@0.75": comp_p,
        "companion_realized_mde@0.75": comp_mde, "companion_thr": COMPANION_THR,
        "c1a_good_delta_vs_no_inject@0.75": c1a_mean, "c1a_p@0.75": c1a_p,
        "c1a_realized_mde@0.75": c1a_mde, "c1a_thr": POSITION_THR, "c1a_pass": bool(c1a_pass),
        # power HARD-GATE (per-position)
        "gradeable_rate": gradeable_rate, "gradeable_rate_by_position":
            {f"{p}": gradeable_rate_pos[p] for p in positions},
        "gradeable_rate_primary": gradeable_rate_primary, "boxed_str_rate": boxed_str_rate,
        "power_ok@0.75": bool(power_ok),
        "power_ok_by_position": {f"{p}": bool(power_ok_pos[p]) for p in positions},
        "n_graded_continuations": grade_d,
        "verdict": verdict, "wall_seconds": time.time() - t0,
    }

    # per-(problem,position,arm) rows for audit. acc only present where the problem is in
    # THAT position's headroom set; ni_acc gives the per-position no_inject acc + band.
    rows = []
    for pi, c in enumerate(kept):
        ci = kept_ci[pi]
        rows.append({
            "pi": pi, "benchmark": c["benchmark"], "cap": c["cap"],
            "pos_idx": {f"{p}": c["pos_idx"][p] for p in positions},
            "no_inject_acc_by_position": {f"{p}": ni_acc.get((ci, p)) for p in positions},
            "headroom_band_by_position": {f"{p}": band_of(pi, p) for p in positions},
            "in_headroom_by_position": {f"{p}": ((pi, p) in in_headroom) for p in positions},
            "acc": {f"{p}": {arm: acc.get((pi, p, arm)) for arm in ARMS} for p in positions},
        })

    payload = _json_safe({"summary": summary, "rows": rows,
                          "per_position_result": per_position_result,
                          "per_position_curve": per_position_curve})
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(payload, open(out, "w"), indent=2)
    print(json.dumps(payload["summary"], indent=2))
    print(f"[done] {out}")


if __name__ == "__main__":
    main()
