"""B.4 — Teacher-Steering causal gate (CTSD Phase B, plan_ctsd_B4_teacher_steering_2026_06_01).

Intent (the redesign that puts the TEACHER in an ACTIVE steering role; see plan):
  b1/b2 are read-only scorers; b3 injects FIXED hand-written GOOD/BAD templates with no
  teacher in the loop. B.4 asks the missing question: at a FIXED position (the deployable
  argmax body-entropy rule), does a teacher that SELECTS which self-generated meta to
  inject guide the student better than a RANDOM selection from the SAME pool? Random-from-
  same-pool removes the "a marker helps" and "any meta helps" confounds, isolating the
  teacher's CONTENT-SELECTION contribution. DV = per-problem continuation accuracy, which
  has signal across the WHOLE distribution (not only inject-rescue of wrong answers).

Pre-registered hypotheses (locked, see plan lines 32-47):
  H-B4-GUIDES (primary): mean_p[acc(teacher_top) - acc(random)] >= +0.04, paired p<0.05.
  H-B4-NOHARM (gating): on the already-CORRECT stratum, acc(teacher_top) - acc(no_inject)
      >= -0.02, ONE-SIDED paired p<0.05.
  H-B4-DISC (precondition): pooled AUC(contrastive_score -> meta-continuation-correct)
      >= 0.60; else INCONCLUSIVE (this teacher cannot tell good from bad meta).

Verdict (plan lines 49-55):
  INCONCLUSIVE if gradeable_rate < 0.5 OR AUC_disc < 0.60.
  PASS iff GUIDES AND NO-HARM AND power_ok.
  FAIL (power_ok, AUC>=0.60, GUIDES fails) -> contrastive teacher does NOT steer content.

Compute: phase-separated (vLLM gen -> free -> HF teacher), per the b1 16k OOM fix. vLLM and
  HF NEVER co-reside. Phases:
    P0 vLLM  : fresh 16k baseline per problem -> robust_grade -> tag is_correct. free().
    P1 HF    : body argmax-entropy p* (a3) -> marker-inject prefix. del model.
    P2 vLLM  : at p*, M self-meta continuations -> extract_first_meta_block -> meta pool;
               grade each continuation (the DISC label). Drop <2 distinct metas. free().
    P3 HF    : E20a teacher contrastive_score(meta) (a6); argsort -> top-1 + random-1. del.
    P4 vLLM  : per arm {teacher_top, random, no_inject} generate k continuations; grade. free().
  (P3 teacher scoring MUST precede P4 arm generation: the teacher_top arm depends on it.)

Karpathy minimal-change: IMPORTS only — b2 (extract_first_meta_block, _json_safe pattern),
  a3 (raw_entropy / first_boxed_token_idx / body_argmax_entropy_pos / MARKER_ONLY), a6
  (build_prompt_with_meta / score_meta_logp / find_answer_token_mask), common.grading
  (robust_grade / is_gradeable), common.vllm_gen (VllmGen / safe_tokenizer_path),
  common.probe_utils (paired_perm_test / mann_whitney_auc). a3/a6/b1/b2/b3/common are
  NOT modified — new file only. (NOHARM needs a ONE-SIDED test; paired_perm_test is
  two-sided, so a tiny local _one_sided_paired_perm is added, mirroring b2's local
  _label_perm_test deviation — a stats analog, not a reimplementation of probe machinery.)

Outputs results/reports/b4_teacher_steering_<tag>.json
"""
from __future__ import annotations
import argparse, json, time, gc
import random as _random
from pathlib import Path
from collections import defaultdict as _defaultdict
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))         # experiments/
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # repo root (src/)
from common.env import (
    TEACHER_MODEL, SFT_V8_STRICT, EVAL_R10V2_E20A, EVAL_R10V2_V8, HF_DATASET,
    META_OPEN_ID, META_CLOSE_ID, REPORTS_DIR, STRATIFIED_SAMPLE_SEED,
)
from common.vllm_gen import VllmGen, safe_tokenizer_path
from common.probe_utils import paired_perm_test, mann_whitney_auc
from probes.a3_inject_causal import (
    raw_entropy, first_boxed_token_idx, body_argmax_entropy_pos, MARKER_ONLY,
)
from probes.a6_six_cell_teacher_swap import (
    build_prompt_with_meta, score_meta_logp, find_answer_token_mask,
)
from probes.b2_e20a_content_variance import extract_first_meta_block, make_decoy
from common.grading import robust_grade, is_gradeable

# E20a tokenizer fallback (identical Qwen3 vocab) — same as B.1/B.2/B.3.
E20A_TOKENIZER_PATH = "/home/v-seungplee/sft_e20a_local"
ARMS = ("teacher_top", "random", "no_inject")
REPRESENTATIVE_BENCHES = {"aime": ("aime", "aime2024"),
                          "math500": ("math500", "math"),
                          "gsm8k": ("gsm8k",)}


def representative_pool(results, n_pool, rng):
    """Representative stratified pool IGNORING stored is_correct — the point of B.4 is a
    representative sample (NOT the hard-wrong headroom pool of b1/b2/b3). Strata =
    {aime, math500, gsm8k}, ~equal per stratum. Sibling of b2.stratified_hard_pool but
    (a) keeps gsm8k and (b) does NOT filter to hard benches — written here so a3/b2 stay
    unmodified."""
    by_strat = _defaultdict(list)
    for r in results:
        for strat, benches in REPRESENTATIVE_BENCHES.items():
            if r.get("benchmark") in benches:
                by_strat[strat].append(r)
                break
    picks, per = [], max(1, n_pool // max(1, len(by_strat)))
    for strat, lst in by_strat.items():
        rng.shuffle(lst); picks.extend(lst[:per])
    rng.shuffle(picks)
    return picks[:n_pool]


def load_tokenizer(path: str):
    """Robust tokenizer load (same as B.1/B.2/B.3). The v8 checkpoint tokenizer FAILS
    under transformers 4.57.6; fall back to the E20a tokenizer (identical Qwen3 vocab,
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
    is STRICT JSON (no NaN/Infinity literals). Same helper as b1/b2/b3."""
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


def _one_sided_paired_perm(diffs, rng, margin: float = -0.02, n_perm: int = 5000) -> float:
    """One-sided NON-INFERIORITY paired sign-flip test for H_alt: mean(diffs) > margin
    (NO-HARM: diffs = acc(teacher_top) - acc(no_inject), margin = -0.02). The margin MUST
    enter the p-value: a harmless inject with mean diff = -0.01 SATISFIES NO-HARM, so we
    shift by the margin and ask how often a symmetric (mean-0) null reproduces a shifted
    mean >= the observed shifted mean. p small => observed mean is significantly ABOVE the
    margin (i.e. non-inferior). probe_utils.paired_perm_test is two-sided and margin-free,
    so this is the one-sided non-inferiority analog (a stats variant, NOT a reimplementation
    of the probe's selection/grading machinery; same symmetry assumption as the paired test)."""
    diffs = np.asarray([d for d in diffs if d is not None and not np.isnan(d)])
    if len(diffs) == 0:
        return float("nan")
    shifted = diffs - margin                       # test non-inferiority to the margin
    obs = shifted.mean()
    hits = 0
    for _ in range(n_perm):
        signs = rng.choice([-1, 1], size=len(shifted))
        if (shifted * signs).mean() >= obs:
            hits += 1
    return (hits + 1) / (n_perm + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=TEACHER_MODEL, help="E20a base==teacher path (A.6 winner)")
    ap.add_argument("--substrate", choices=["e20a", "v8"], default="e20a",
                    help="selects eval json for the representative sample")
    ap.add_argument("--n", type=int, default=120, help="target sample (~40 each aime/math500/gsm8k)")
    ap.add_argument("--n_pool", type=int, default=None,
                    help="override the n_pool = max(2*n, n+30) representative-pool size")
    ap.add_argument("--m", type=int, default=12, help="self-meta pool size per problem (P2)")
    ap.add_argument("--k", type=int, default=8, help="continuations per arm (P4)")
    ap.add_argument("--max_new", type=int, default=16384,
                    help="generation budget (real eval regime: max_tokens=16384)")
    ap.add_argument("--max_model_len", type=int, default=20480,
                    help="vLLM context window (real eval regime: 20480)")
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--tag", default="e20a")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    n = args.smoke or args.n
    m = args.m if not args.smoke else max(2, min(args.m, 4))
    k = args.k if not args.smoke else max(2, min(args.k, 2))
    out = args.out or str(REPORTS_DIR / f"b4_teacher_steering_{args.tag}.json")

    rng = _random.Random(STRATIFIED_SAMPLE_SEED)            # python RNG (sampling, decoy)
    rng_np = np.random.default_rng(STRATIFIED_SAMPLE_SEED)  # numpy RNG (perm tests)
    t0 = time.time()

    eval_path = EVAL_R10V2_E20A if args.substrate == "e20a" else EVAL_R10V2_V8
    ev = hf_hub_download(repo_id=HF_DATASET, repo_type="dataset", filename=eval_path)
    results = json.load(open(ev))
    results = results if isinstance(results, list) else results.get("results") or list(results.values())[0]
    all_golds = [r.get("gold_answer") for r in results if r.get("gold_answer") is not None]
    n_pool = args.n_pool if args.n_pool is not None else max(2 * n, n + 30)
    pool = representative_pool(results, n_pool, rng)
    print(f"[b4] substrate={args.substrate} model={args.model} n_target={n} pool={len(pool)} "
          f"m={m} k={k} max_new={args.max_new} max_model_len={args.max_model_len}")

    tok = load_tokenizer(args.model)
    dev = "cuda"
    marker_seg = tok.encode(MARKER_ONLY, add_special_tokens=False)

    # ── PHASE 0 (vLLM only): fresh 16k baseline -> robust_grade -> tag is_correct ──────
    # KEEP ALL problems (correct AND wrong) — the representative sample is the point; the
    # is_correct tag only partitions strata for the NO-HARM (correct-stratum) gate.
    vgen = VllmGen(args.model, tokenizer_path=safe_tokenizer_path(args.model),
                   gpu_memory_utilization=0.45, max_model_len=args.max_model_len,
                   seed=STRATIFIED_SAMPLE_SEED)
    pool_prompt_ids = [build_prompt_ids(tok, r["question"]) for r in pool]
    pool_bases = vgen.generate(pool_prompt_ids, n=1, max_tokens=args.max_new,
                               seed=STRATIFIED_SAMPLE_SEED)
    probs, base_by_pi, n_correct_tag = [], {}, 0
    for r, pids, outs in zip(pool, pool_prompt_ids, pool_bases):
        if len(probs) >= n:
            break
        base = outs[0]                            # full 16k baseline (no cap) — matches b1
        is_correct = bool(robust_grade(tok.decode(base, skip_special_tokens=False),
                                       str(r["gold_answer"]).strip()))
        n_correct_tag += int(is_correct)
        base_by_pi[len(probs)] = (pids, base, is_correct)
        probs.append(r)
    print(f"[b4][sample] pool={len(pool)} kept={len(probs)} correct={n_correct_tag} "
          f"wrong={len(probs)-n_correct_tag} target_n={n}")
    vgen.free(); gc.collect(); torch.cuda.empty_cache()

    # ── PHASE 1 (HF student only): body argmax-entropy p* -> marker-inject prefix ──────
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    phase1 = []                                   # per problem: prefix + meta for later phases
    for pi, r in enumerate(probs):
        q, gold = r["question"], str(r["gold_answer"]).strip()
        prompt_ids, base, is_correct = base_by_pi[pi]
        H = raw_entropy(model, prompt_ids + base, len(prompt_ids), dev)
        cap = first_boxed_token_idx(tok, base)    # cap before first \boxed (a3 rule)
        p_star, H_at, frac = body_argmax_entropy_pos(base, H, cap)
        pre_meta_body = tok.decode(base[:p_star], skip_special_tokens=False)
        phase1.append({"pi": pi, "benchmark": r["benchmark"], "q": q, "gold": gold,
                       "decoy": make_decoy(gold, all_golds, rng), "is_correct": is_correct,
                       "prompt_ids": prompt_ids, "p_star": int(p_star), "H_at": float(H_at),
                       "inject_frac": float(frac),
                       "base_prefix": base[:p_star],        # response tokens before inject
                       "pre_meta_body": pre_meta_body,
                       "inject_prefix": prompt_ids + base[:p_star] + marker_seg})  # for P2 self-meta
    del model; gc.collect(); torch.cuda.empty_cache()

    # ── PHASE 2 (vLLM only, re-created): M self-meta continuations -> pool + DISC label ─
    # At p*, the model fills its OWN meta + tail. Each of the M continuations gives (a) a
    # meta block (extract_first_meta_block) for the pool and (b) its own graded outcome —
    # the DISC label (does this meta lead to a correct continuation?). Batch ALL problems.
    vgen = VllmGen(args.model, tokenizer_path=safe_tokenizer_path(args.model),
                   gpu_memory_utilization=0.45, max_model_len=args.max_model_len,
                   seed=STRATIFIED_SAMPLE_SEED)
    pool_conts = vgen.generate([ph["inject_prefix"] for ph in phase1], n=m,
                               max_tokens=args.max_new, seed=STRATIFIED_SAMPLE_SEED)
    grade_n, grade_d, box_n = 0, 0, 0             # power counters over ALL graded continuations
    n_dropped_pool = 0
    for ph, conts in zip(phase1, pool_conts):
        pool_metas = []                           # [{meta_block, disc_correct}] for this problem
        for c in conts:
            tail_text = tok.decode(marker_seg + c, skip_special_tokens=False)
            full_resp = ph["pre_meta_body"] + tail_text
            correct = int(robust_grade(full_resp, ph["gold"]))
            grade_d += 1
            grade_n += int(is_gradeable(full_resp))
            box_n += int(r"\boxed" in full_resp)
            mb = extract_first_meta_block(tail_text)
            if mb is None:                        # no scorable self-meta in this continuation
                continue
            pool_metas.append({"meta_block": mb, "disc_correct": correct})
        # drop problems with < 2 DISTINCT closed metas (a pool of one is not a selection)
        distinct = {mr["meta_block"] for mr in pool_metas}
        if len(distinct) < 2:
            n_dropped_pool += 1
            ph["pool_metas"] = None
            print(f"  [P2 {ph['pi']+1}/{len(phase1)}] {ph['benchmark']} DROP "
                  f"(distinct_metas={len(distinct)})")
            continue
        ph["pool_metas"] = pool_metas
        print(f"  [P2 {ph['pi']+1}/{len(phase1)}] {ph['benchmark']} p*={ph['p_star']} "
              f"metas={len(pool_metas)} distinct={len(distinct)} ({time.time()-t0:.0f}s)")
    vgen.free(); gc.collect(); torch.cuda.empty_cache()
    active = [ph for ph in phase1 if ph.get("pool_metas")]

    # ── PHASE 3 (HF teacher E20a only): contrastive_score per meta -> top-1 + random-1 ──
    # contrastive(meta) = mean_logp_{T+}(meta) - mean_logp_{T-}(meta), answer-token-masked
    # (UNION of gold+decoy masks, as in b2). E20a is base==teacher (A.6 winner) -> one load.
    teacher = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    disc_scores, disc_labels = [], []             # pooled DISC AUC over ALL (problem, meta)
    for ph in active:
        for mr in ph["pool_metas"]:
            scores = {}
            for ctx_label, ans in (("Tplus", ph["gold"]), ("Tminus", ph["decoy"])):
                input_ids, _, meta_start, meta_len = build_prompt_with_meta(
                    tok, ph["q"], ans, ph["pre_meta_body"], mr["meta_block"])
                logp = score_meta_logp(teacher, input_ids, meta_start, meta_len, dev)
                meta_token_ids = input_ids[meta_start:meta_start + meta_len]
                gmask = find_answer_token_mask(tok, mr["meta_block"], ph["gold"], meta_token_ids)
                dmask = find_answer_token_mask(tok, mr["meta_block"], ph["decoy"], meta_token_ids)
                non_ans = logp[~(gmask | dmask)]
                scores[ctx_label] = float(np.mean(non_ans)) if len(non_ans) else float(np.mean(logp))
            mr["contrastive"] = scores["Tplus"] - scores["Tminus"]
            disc_scores.append(mr["contrastive"]); disc_labels.append(mr["disc_correct"])
        # argsort -> teacher_top = argmax contrastive; random = seeded pick from SAME pool,
        # != top index where possible (a pool >= 2 distinct always allows this).
        order = sorted(range(len(ph["pool_metas"])),
                       key=lambda i: ph["pool_metas"][i]["contrastive"], reverse=True)
        top_i = order[0]
        top_text = ph["pool_metas"][top_i]["meta_block"]
        # exclude by TEXT, not just index: a different index whose meta string == top's
        # would collapse the GUIDES contrast to sampling noise (identical injected content).
        choices = [i for i in range(len(ph["pool_metas"]))
                   if ph["pool_metas"][i]["meta_block"] != top_text]
        choices = choices or [i for i in range(len(ph["pool_metas"])) if i != top_i] or [top_i]
        rand_i = rng.choice(choices)
        ph["meta_top"] = ph["pool_metas"][top_i]["meta_block"]
        ph["meta_random"] = ph["pool_metas"][rand_i]["meta_block"]
    del teacher; gc.collect(); torch.cuda.empty_cache()

    # ── PHASE 4 (vLLM only, re-created): 3 arms x k continuations -> grade ─────────────
    # teacher_top / random = re-inject chosen meta after [prompt+base[:p*]+marker];
    # no_inject = RAW [prompt+base[:p*]] (the NO-HARM baseline). Batch ALL (problem x arm).
    vgen = VllmGen(args.model, tokenizer_path=safe_tokenizer_path(args.model),
                   gpu_memory_utilization=0.45, max_model_len=args.max_model_len,
                   seed=STRATIFIED_SAMPLE_SEED)
    arm_prefixes, arm_slots = [], []              # arm_slots[i] = (pi, arm); aligns with prefixes
    arm_seg = {}                                  # (pi, arm) -> token segment injected after base_prefix
    for ph in active:
        prompt_ids, base_prefix = ph["prompt_ids"], ph["base_prefix"]
        seg_top = marker_seg + tok.encode(ph["meta_top"], add_special_tokens=False)
        seg_rand = marker_seg + tok.encode(ph["meta_random"], add_special_tokens=False)
        seg_none: list[int] = []
        for arm, seg in (("teacher_top", seg_top), ("random", seg_rand), ("no_inject", seg_none)):
            arm_prefixes.append(prompt_ids + base_prefix + seg)
            arm_slots.append((ph["pi"], arm)); arm_seg[(ph["pi"], arm)] = seg
    arm_outs = vgen.generate(arm_prefixes, n=k, max_tokens=args.max_new,
                             seed=STRATIFIED_SAMPLE_SEED)
    cont_by = {}
    for (pi, arm), conts in zip(arm_slots, arm_outs):
        cont_by[(pi, arm)] = conts
    rows = []                                     # per problem: arm accs + is_correct
    for ph in active:
        pi, gold, pre_meta_body = ph["pi"], ph["gold"], ph["pre_meta_body"]
        acc = {}
        for arm in ARMS:
            conts = cont_by.get((pi, arm), [])
            seg = arm_seg[(pi, arm)]
            seg_text = tok.decode(seg, skip_special_tokens=False) if seg else ""
            nc = 0
            for c in conts:
                full_resp = pre_meta_body + seg_text + tok.decode(c, skip_special_tokens=False)
                nc += int(robust_grade(full_resp, gold))
                grade_d += 1
                grade_n += int(is_gradeable(full_resp))
                box_n += int(r"\boxed" in full_resp)
            acc[arm] = (nc / len(conts)) if conts else None
        rows.append({"pi": pi, "benchmark": ph["benchmark"], "is_correct": ph["is_correct"],
                     "p_star": ph["p_star"], "inject_frac": ph["inject_frac"],
                     "n_pool_metas": len(ph["pool_metas"]),
                     "acc_teacher_top": acc["teacher_top"], "acc_random": acc["random"],
                     "acc_no_inject": acc["no_inject"]})
        print(f"  [P4 {pi+1}] {ph['benchmark']:8s} top={acc['teacher_top']} "
              f"rand={acc['random']} none={acc['no_inject']}")
    vgen.free(); gc.collect(); torch.cuda.empty_cache()

    # ── AGGREGATE + pre-registered verdict ────────────────────────────────────────────
    gated = [r for r in rows if r["acc_teacher_top"] is not None and r["acc_random"] is not None]
    # H-B4-GUIDES (primary): mean_p[acc(teacher_top) - acc(random)] >= +0.04, paired p<0.05
    guides_d = [r["acc_teacher_top"] - r["acc_random"] for r in gated]
    guides_delta = float(np.mean(guides_d)) if guides_d else None
    guides_p = float(paired_perm_test(guides_d, rng_np)) if guides_d else None
    guides_pass = (guides_delta is not None and guides_delta >= 0.04
                   and guides_p is not None and guides_p < 0.05)
    # H-B4-NOHARM (gating): correct stratum, acc(teacher_top)-acc(no_inject) >= -0.02, one-sided
    correct_rows = [r for r in gated if r["is_correct"] and r["acc_no_inject"] is not None]
    noharm_d = [r["acc_teacher_top"] - r["acc_no_inject"] for r in correct_rows]
    noharm_delta = float(np.mean(noharm_d)) if noharm_d else None
    noharm_p = float(_one_sided_paired_perm(noharm_d, rng_np)) if noharm_d else None
    noharm_pass = (noharm_delta is not None and noharm_delta >= -0.02
                   and noharm_p is not None and noharm_p < 0.05)
    # H-B4-DISC (precondition): pooled AUC(contrastive -> meta-continuation-correct) >= 0.60
    disc_pos = [s for s, lab in zip(disc_scores, disc_labels) if lab == 1]
    disc_neg = [s for s, lab in zip(disc_scores, disc_labels) if lab == 0]
    _a = float(mann_whitney_auc(disc_pos, disc_neg)) if (disc_pos and disc_neg) else float("nan")
    auc_disc = None if np.isnan(_a) else _a
    disc_ok = (auc_disc is not None and auc_disc >= 0.60)

    # power: gradeable_rate over ALL graded continuations (P2 + P4) >= 0.5, else INCONCLUSIVE.
    gradeable_rate = (grade_n / grade_d) if grade_d else None
    boxed_str_rate = (box_n / grade_d) if grade_d else None
    power_ok = (gradeable_rate is not None and gradeable_rate >= 0.5)
    # MDE (sd-based) for the +0.04 GUIDES effect at realized n: 1.96*sd*sqrt(2/n).
    sd_guides = float(np.std(guides_d, ddof=1)) if len(guides_d) > 1 else None
    mde_guides = (1.96 * sd_guides * np.sqrt(2.0 / len(guides_d))
                  if (sd_guides is not None and len(guides_d) > 0) else None)

    # NO-HARM is UNTESTABLE with no correct-stratum items → INCONCLUSIVE, never a FAIL
    # (a FAIL here would mislabel "untestable" as "teacher does not steer" and wrongly
    # trigger the Phase-D decision; likely in small/unlucky samples & the n=4 smoke).
    noharm_testable = len(noharm_d) > 0
    # status: INCONCLUSIVE if NOT power_ok OR AUC_disc<0.60 OR NO-HARM untestable;
    #         else PASS if GUIDES&NOHARM; else FAIL
    if (not power_ok) or (not disc_ok) or (not noharm_testable):
        status = "INCONCLUSIVE"
        why = ("power guard failed" if not power_ok
               else f"teacher cannot discriminate metas (AUC_disc={auc_disc})" if not disc_ok
               else "NO-HARM untestable (zero correct-stratum items)")
        verdict = (f"INCONCLUSIVE — {why}; selection is meaningless / re-run with more budget")
    elif guides_pass and noharm_pass:
        status = "PASS"
        verdict = ("PASS — teacher selection beats random (GUIDES) AND does not harm correct "
                   "items (NO-HARM); contrastive teacher steers content")
    else:
        status = "FAIL"
        verdict = ("FAIL — power OK & AUC>=0.60 but GUIDES/NO-HARM miss; contrastive teacher "
                   "does NOT steer content -> stop contrastive-content line -> Phase D")

    summary = {
        "status": status,
        "substrate": args.substrate, "model": args.model,
        "n_sampled": len(probs), "n_active": len(active), "n_gated": len(gated),
        "n_correct_stratum": len(correct_rows), "n_dropped_pool_lt2": n_dropped_pool,
        "m": m, "k": k, "max_new": args.max_new, "inject_point": "body_argmax_entropy",
        "guides_delta": guides_delta, "guides_p": guides_p, "guides_pass": bool(guides_pass),
        "noharm_delta": noharm_delta, "noharm_p": noharm_p, "noharm_pass": bool(noharm_pass),
        "auc_disc": auc_disc, "disc_ok": bool(disc_ok),
        "n_disc_metas": len(disc_scores), "n_disc_pos": len(disc_pos), "n_disc_neg": len(disc_neg),
        "gradeable_rate": gradeable_rate, "boxed_str_rate": boxed_str_rate, "power_ok": bool(power_ok),
        "mde_guides_at_n": (float(mde_guides) if mde_guides is not None else None),
        "sd_guides": sd_guides,
        "n_correct_tag": n_correct_tag, "n_wrong_tag": len(probs) - n_correct_tag,
        "verdict": verdict, "wall_seconds": time.time() - t0,
    }
    payload = _json_safe({"summary": summary, "rows": rows,
                          "disc": [{"contrastive": s, "correct": int(l)}
                                   for s, l in zip(disc_scores, disc_labels)]})
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(payload, open(out, "w"), indent=2)
    print(json.dumps(payload["summary"], indent=2))
    print(f"[done] {out}")


if __name__ == "__main__":
    main()
