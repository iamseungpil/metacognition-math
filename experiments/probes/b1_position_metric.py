"""B.1 — Position-benefit metric discovery (CTSD Phase B, plan_ctsd_B_probes_2026_06_01).

Intent:
  A.2 showed entropy predicts ROLLOUT wrongness (AUC 0.749); A.3b compared 3 fixed
  position RULES and found argmax neutral / onset harmful on v8. Neither asked:
  (i) do positions exist where a MARKER-only inject causally helps, beyond
      selection-bias chance (winner's curse)?  -> H-B1a (headroom)
  (ii) can a gold-free metric (entropy / onset / pause-propensity / teacher T+/T- KL /
       student-teacher gap) RANK positions by that benefit, beating a position-index
       baseline?  -> H-B1b (findability)
  (iii) is this cleaner on E20a (0% natural meta) than v8 (decorative meta)? -> H-B1c

Design (reuses proven machinery; a3/a3b/a6/probe_utils are NOT modified):
  Per hard-wrong problem: regenerate 1 baseline rollout -> per-token entropy (a3.raw_entropy).
  Candidate positions = {entropy p50,p75,p90,argmax,onset} U {R random body pos},
  each >=MIN_TOK, before first \\boxed, outside meta spans (a3b.body_candidates).
  Per position p: marker arm (prefix+<|meta|>) vs noinject arm, k continuations each,
  Delta_p = acc(marker)-acc(noinject) graded by common.grading.robust_grade.
  Winner's-curse control: k split k1(select best pos) / k2(estimate held-out), and
  best-REAL vs best-of-R-RANDOM under identical selection. Metrics ranked by AUC vs
  1[Delta_p>0] on a held-out CONFIRMATION half of problems (multiplicity control).

Gates (pre-registered, see plan):
  Stage-1 H-B1a: mean[Delta(best-real,k2) - Delta(best-random,k2)] >= +5pp, paired p<0.05.
  Stage-2 H-B1b: some metric AUC>=0.65 on confirmation half AND >= position-index AUC +0.05.
  Stage-1 FAIL on both substrates -> position-guiding dead -> STOP -> Phase D.

Outputs reports/b1_position_metric_<tag>.json
"""
from __future__ import annotations
import argparse, json, time, gc
import random as _random
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from huggingface_hub import hf_hub_download
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))         # experiments/
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # repo root (src/)
from common.env import (
    SFT_V8_STRICT, TEACHER_MODEL, EVAL_R10V2_V8, EVAL_R10V2_E20A, HF_DATASET,
    META_OPEN_ID, META_CLOSE_ID, REPORTS_DIR, STRATIFIED_SAMPLE_SEED,
)
from common.probe_utils import mann_whitney_auc, paired_perm_test, build_teacher_input
from common.vllm_gen import VllmGen, safe_tokenizer_path
from probes.a3_inject_causal import (
    raw_entropy, gen_batch, first_boxed_token_idx, find_meta_spans,
    MIN_TOK, MARKER_ONLY,
)
from probes.a3b_position_marker import body_candidates
from common.grading import robust_grade, is_gradeable
from collections import defaultdict as _defaultdict

# pause-propensity reference (metric ③): a meta-EMITTING SFT. E20a emits ~0% meta
# so its P(meta) is degenerate; the v8_strict SFT is the meta-emitting reference.
PAUSE_REF_MODEL = SFT_V8_STRICT
TEACHER_REF_MODEL = TEACHER_MODEL   # E20a teacher for metrics ④/⑤
ONSET_Q = 75           # onset = first body pos whose entropy >= this response's pXX
ONSET_SLOPE_W = 8      # window for onset-slope metric


def stratified_hard_pool(results, n_pool, rng):
    """FIX C pool sampler: n_pool hard-benchmark problems, IGNORING stored is_correct.

    Sibling of a3.stratified_wrong_hard (we write our own here rather than modify a3 —
    a3 is protected) but WITHOUT the `not r["is_correct"]` filter, because the stored
    is_correct is unreliable (same math_verify grading bug + a stored/format mismatch:
    robust_grade gives E20a math500 ~21% vs stored 81%). The probe then generates a
    FRESH baseline for each pool problem and keeps only those robust_grade marks WRONG
    (true headroom). Stratifies across the same hard benchmarks as a3."""
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
    """Robust tokenizer load. The v8 checkpoint tokenizer FAILS under transformers
    4.57.6 (AttributeError: 'list' object has no attribute 'keys' in
    _set_model_specific_special_tokens). Fall back to the E20a tokenizer, which has
    an IDENTICAL Qwen3 vocab (<|meta|>=151669, <|/meta|>=151670, verified). Then
    ASSERT the meta-token IDs so a silent vocab mismatch can never slip through."""
    try:
        tok = AutoTokenizer.from_pretrained(path)
    except Exception as e:
        print(f"[tok] {path} failed ({type(e).__name__}: {str(e)[:80]}); "
              f"falling back to E20a tokenizer (identical vocab)")
        tok = AutoTokenizer.from_pretrained("/home/v-seungplee/sft_e20a_local")
    assert tok.convert_tokens_to_ids("<|meta|>") == META_OPEN_ID, "META_OPEN_ID mismatch"
    assert tok.convert_tokens_to_ids("<|/meta|>") == META_CLOSE_ID, "META_CLOSE_ID mismatch"
    return tok


@torch.no_grad()
def next_token_dist(model, input_ids, resp_start, device):
    """ONE forward over the full prefix → per-response-position next-token softmax
    probabilities, shape (T, V) where T = len(input_ids) - resp_start. Position p of
    the return predicts the token that FOLLOWS response-token p (i.e. the dist the
    model would sample from at response position p). Used for metrics ③/④ in a SINGLE
    pass per (model, context) — NOT a per-position forward."""
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    out = model(ids, use_cache=False)
    pred = out.logits[0][resp_start - 1: -1].float()       # (T, V): pred[t] -> resp tok t+?
    return torch.nn.functional.log_softmax(pred, dim=-1)   # log-probs on device


@torch.no_grad()
def pause_propensity(model, input_ids, resp_start, device):
    """metric ③: P(next token == <|meta|>=151669 | prefix→position p) at each
    response position, under the meta-emitting reference SFT. ONE forward."""
    lp = next_token_dist(model, input_ids, resp_start, device)
    return lp[:, META_OPEN_ID].exp().cpu().numpy()


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
    return tok.encode(s, add_special_tokens=False)


@torch.no_grad()
def taken_token_logp(model, input_ids, resp_start, device):
    """Per-response-position logp of the token the response ACTUALLY took. Length T =
    len(input_ids) - resp_start. lp[p] is the dist predicting response token p, and the
    actually-taken token at position p is input_ids[resp_start+p], so taken[p] = lp[p]
    scored at that token. Used for metric ⑤ student−teacher gap. ONE forward."""
    lp = next_token_dist(model, input_ids, resp_start, device)   # (T, V), lp[p] -> resp tok p
    ids = torch.tensor(input_ids, device=device)
    targets = ids[resp_start:]                                   # resp tokens 0..T-1
    taken = lp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)     # logp of taken tok at each pos
    return taken.cpu().numpy()                                   # length T


def make_decoy(gold: str, all_golds: list, rng) -> str:
    """Per-problem decoy answer for the T- teacher context (metric ④). Reuse a6's
    decoy source — a DIFFERENT gold drawn from the eval pool — so T+/T- differ only
    in which answer the teacher is told is correct (matches a6.build_prompt_with_meta
    decoy semantics). Falls back to a numeric perturbation if the pool is degenerate."""
    g = str(gold).strip()
    for _ in range(20):
        d = str(rng.choice(all_golds)).strip()
        if d and d != g:
            return d
    return (g + "1") if not g.lstrip("-").isdigit() else str(int(g) + 1)


def candidate_positions(resp_ids, H, answer_cap, rng, n_random=4):
    """Named entropy positions + onset + n_random, all valid body candidates.
    Returns dict name->idx (idx into resp_ids). Reuses a3b.body_candidates for the
    valid-position set so exclusion logic (meta spans, MIN_TOK, answer_cap) matches."""
    cand = body_candidates(resp_ids, H, answer_cap)           # list of valid idx
    if not cand:
        return {}
    Hc = np.array([H[i] for i in cand])
    order = np.argsort(Hc)                                    # ascending entropy
    pos = {}
    pos["argmax"] = cand[int(order[-1])]
    for q, name in [(50, "p50"), (75, "p75"), (90, "p90")]:
        thr = np.percentile(Hc, q)
        pick = next((cand[int(i)] for i in order if Hc[int(i)] >= thr), cand[int(order[-1])])
        pos[name] = pick
    # onset = first (earliest index) candidate whose entropy >= this resp's pONSET_Q
    thr = np.percentile(Hc, ONSET_Q)
    onset = min((c for c in cand if H[c] >= thr), default=cand[0])
    pos["onset"] = onset
    for r in range(n_random):
        pos[f"rand{r}"] = rng.choice(cand)
    return pos


def grade(tok, prompt_ids, prefix_resp_ids, cont_ids, gold):
    """Grade a continuation: decode the full response (prefix + continuation), check vs gold."""
    full_resp = tok.decode(prefix_resp_ids + cont_ids, skip_special_tokens=False)
    return 1 if robust_grade(full_resp, gold) else 0


def boxed_rate(tok, prefix_resp_ids, conts):
    n = sum(1 for c in conts if r"\boxed" in tok.decode(prefix_resp_ids + c, skip_special_tokens=False))
    return n / max(1, len(conts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=SFT_V8_STRICT, help="student that solves (v8 or E20a path)")
    ap.add_argument("--substrate", choices=["v8", "e20a"], default="v8",
                    help="selects eval json for headroom sampling")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--k", type=int, default=8, help="continuations per arm (split k1/k2 for cross-fit)")
    ap.add_argument("--n_random", type=int, default=4)
    ap.add_argument("--max_new", type=int, default=16384,
                    help="generation budget (real eval regime: max_tokens=16384)")
    ap.add_argument("--max_model_len", type=int, default=20480,
                    help="vLLM context window (real eval regime: 20480)")
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--tag", default="v8")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    n = args.smoke or args.n
    k1 = args.k // 2
    out = args.out or str(REPORTS_DIR / f"b1_position_metric_{args.tag}.json")

    rng = _random.Random(STRATIFIED_SAMPLE_SEED)
    rng_np = np.random.default_rng(STRATIFIED_SAMPLE_SEED)
    eval_path = EVAL_R10V2_E20A if args.substrate == "e20a" else EVAL_R10V2_V8
    ev = hf_hub_download(repo_id=HF_DATASET, repo_type="dataset", filename=eval_path)
    results = json.load(open(ev))
    results = results if isinstance(results, list) else results.get("results") or list(results.values())[0]
    # FIX C: sample a POOL of hard problems IGNORING stored is_correct (unreliable), then
    # select headroom from FRESH robust-graded baselines below. Pool size ~max(3n, n+30).
    n_pool = max(3 * n, n + 30)
    pool = stratified_hard_pool(results, n_pool, rng)
    print(f"[b1] substrate={args.substrate} model={args.model} n_target={n} "
          f"pool={len(pool)} k={args.k} max_new={args.max_new} max_model_len={args.max_model_len}")

    all_golds = [r.get("gold_answer") for r in results if r.get("gold_answer") is not None]
    tok = load_tokenizer(args.model)
    # Generation MECHANISM = vLLM (replaces a3.gen_batch). Modest util (0.45 ≈ 36GB)
    # so the HF student (~16GB, needed for entropy/taken-token-logp forwards) coexists.
    # tokenizer_path: the v8 checkpoint tokenizer fails under transformers 4.57.6, so
    # safe_tokenizer_path falls back to the E20a tokenizer (identical vocab) when needed.
    vgen = VllmGen(args.model, tokenizer_path=safe_tokenizer_path(args.model),
                   gpu_memory_utilization=0.45, max_model_len=args.max_model_len,
                   seed=STRATIFIED_SAMPLE_SEED)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    dev = "cuda"

    # FIX C: FRESH 16k baselines for the pool → robust_grade → keep only WRONG (true
    # headroom). Reuse each kept baseline for entropy/positions (do NOT regenerate).
    pool_prompt_ids = [build_prompt_ids(tok, r["question"]) for r in pool]
    pool_bases = vgen.generate(pool_prompt_ids, n=1, max_tokens=args.max_new,
                               seed=STRATIFIED_SAMPLE_SEED)
    probs, base_by_pi, dropped_correct = [], {}, 0
    for r, pids, outs in zip(pool, pool_prompt_ids, pool_bases):
        if len(probs) >= n:
            break
        base = outs[0]
        full_resp = tok.decode(base, skip_special_tokens=False)
        if robust_grade(full_resp, r["gold_answer"]):
            dropped_correct += 1          # already-correct → NOT headroom, drop (logged)
            continue
        base_by_pi[len(probs)] = (pids, base)   # reuse this baseline downstream
        probs.append(r)
    headroom_drop_log = (f"[b1][headroom] pool={len(pool)} kept_wrong={len(probs)} "
                         f"dropped_already_correct={dropped_correct} target_n={n}")
    print(headroom_drop_log)
    if len(probs) < n:
        print(f"[b1][headroom] WARNING: only {len(probs)} robust-wrong found in pool "
              f"of {len(pool)} (< target {n}); proceeding with what we have")

    # discovery/confirmation split (multiplicity control for metric AUC, plan 88-91):
    # DISCOVERY half selects the single best metric; CONFIRMATION half gates it.
    idx = list(range(len(probs))); rng.shuffle(idx)
    confirm_set = set(idx[len(idx) // 2:])
    discovery_set = set(idx[:len(idx) // 2])

    rows = []                       # per (problem,position): metrics + Delta
    headroom = []                   # per problem: best-real vs best-random (held-out)
    # Power statistic (plan CHANGE 1): the floor effect that kills resolving power is
    # "no PARSEABLE answer", not "no \boxed". So power is now the fraction of the
    # GENERATED CONTINUATIONS (what actually gets graded) that are is_gradeable, NOT
    # the baseline-rollout \boxed presence. We keep the old \boxed string proxy as a
    # descriptive boxed_str_rate (renamed for clarity) but gate status on gradeable_rate.
    box_ok = []                     # per problem: 1 if baseline rollout has the \boxed STRING (descriptive only)
    grade_n, grade_d = 0, 0         # gradeable continuations / total continuations (drives power)
    # Saved per-problem prefixes for the SEQUENTIAL ref/teacher scoring passes
    # (student freed first → pause-ref → teacher). Each entry holds everything the
    # later passes need so we never reload the student.
    saved = []                      # list of {pi, prompt_ids, base, pos, gold, decoy, st_taken}
    marker_seg = tok.encode(MARKER_ONLY, add_special_tokens=False)
    for pi, r in enumerate(probs):
        q, gold = r["question"], r["gold_answer"]
        # FIX C: REUSE the fresh 16k baseline already generated for headroom selection
        # (do NOT regenerate) — same prompt_ids + base used for robust-wrong gating above.
        prompt_ids, base = base_by_pi[pi]
        # W4 power: did the BASELINE rollout itself reach a boxed answer (one value/problem)
        box_ok.append(1 if r"\boxed" in tok.decode(base, skip_special_tokens=False) else 0)
        cap = first_boxed_token_idx(tok, base)
        H = raw_entropy(model, prompt_ids + base, len(prompt_ids), dev)
        pos = candidate_positions(base, H, cap, rng, args.n_random)
        if not pos:
            continue
        # metric ⑤ part 1: student logp of the actually-taken next base token, ONE forward
        st_taken = taken_token_logp(model, prompt_ids + base, len(prompt_ids), dev)
        # Position arms: batch ALL (position × {marker, noinject}) prefixes into ONE
        # vLLM call (n=k each) — the big speed win vs the old per-(pos,arm) loop.
        names = list(pos.keys())
        prefixes, slots = [], []   # slots[i] = (name, "marker"|"noinj")
        for name in names:
            p = pos[name]
            prefixes.append(prompt_ids + base[:p] + marker_seg); slots.append((name, "marker"))
            prefixes.append(prompt_ids + base[:p]);              slots.append((name, "noinj"))
        outs = vgen.generate(prefixes, n=args.k, max_tokens=args.max_new,
                             seed=STRATIFIED_SAMPLE_SEED + pi)
        cont_by = {}
        for (name, arm), conts in zip(slots, outs):
            cont_by[(name, arm)] = conts
        per_pos = {}
        for name in names:
            p = pos[name]
            m = cont_by[(name, "marker")]
            a = cont_by[(name, "noinj")]
            gm, ga = [], []
            for arm_conts, store in ((m, gm), (a, ga)):
                for c in arm_conts:
                    full_resp = tok.decode(base[:p] + c, skip_special_tokens=False)
                    store.append(1 if robust_grade(full_resp, gold) else 0)
                    # power: did this graded continuation emit something answer-shaped?
                    grade_d += 1
                    if is_gradeable(full_resp):
                        grade_n += 1
            # cross-fit: k1 to select, k2 (rest) to estimate held-out Delta
            d_sel = np.mean(gm[:k1]) - np.mean(ga[:k1])
            d_est = np.mean(gm[k1:]) - np.mean(ga[k1:])
            per_pos[name] = {"p": int(p), "frac": p / max(1, len(base)),
                             "d_sel": float(d_sel), "d_est": float(d_est),
                             "H": float(H[p]) if p < len(H) else 0.0,
                             "onset_slope": float(H[p] - np.mean(H[max(0, p - ONSET_SLOPE_W):p])) if p < len(H) else 0.0,
                             # metric placeholders filled in the ref/teacher passes
                             "pause": None, "teacher_kl": None, "student_teacher_gap": None,
                             "st_logp": float(st_taken[p]) if p < len(st_taken) else None}
            rows.append({"prob": pi, "name": name, "confirm": pi in confirm_set,
                         "delta_est": float(d_est), **per_pos[name]})
        # winner's curse: pick best by d_sel among REAL (non-rand) vs RANDOM, score on d_est
        real = [v for nm, v in per_pos.items() if not nm.startswith("rand")]
        rand = [v for nm, v in per_pos.items() if nm.startswith("rand")]
        if real and rand:
            best_real = max(real, key=lambda v: v["d_sel"])["d_est"]
            best_rand = max(rand, key=lambda v: v["d_sel"])["d_est"]
            headroom.append(best_real - best_rand)
        saved.append({"pi": pi, "prompt_ids": prompt_ids, "base": base,
                      "pos": {nm: v["p"] for nm, v in per_pos.items()},
                      "question": q, "gold": gold,
                      "decoy": make_decoy(gold, all_golds, rng)})
        print(f"  [{pi+1}/{len(probs)}] {r['benchmark']} positions={len(pos)} "
              f"argmax d_est={per_pos.get('argmax',{}).get('d_est')}")

    # free the vLLM allocation + the student before loading the ref / teacher
    # (80GB safety: sequential). vgen.free() releases ~36GB so the HF ref/teacher
    # (~16GB each, one at a time) have room. Statistical/teacher phases below UNCHANGED.
    vgen.free()
    del model; gc.collect(); torch.cuda.empty_cache()

    # ---- metric ③ pause-propensity (meta-emitting reference SFT) ----------------
    # rows are keyed by (prob index pi, name); build a lookup so the ref/teacher
    # passes can write metric values back into the matching rows.
    row_by = {(rw["prob"], rw["name"]): rw for rw in rows}

    def _write(pi, posdict, vals_by_name):
        for nm, val in vals_by_name.items():
            rw = row_by.get((pi, nm))
            if rw is not None:
                rw.update(val)

    print(f"[b1] loading pause reference (meta-emitting SFT) {PAUSE_REF_MODEL}")
    ref_tok = load_tokenizer(PAUSE_REF_MODEL)
    ref = AutoModelForCausalLM.from_pretrained(PAUSE_REF_MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    for s in saved:
        ipt = build_prompt_ids(ref_tok, s["question"]) + s["base"]
        rstart = len(build_prompt_ids(ref_tok, s["question"]))
        pp = pause_propensity(ref, ipt, rstart, dev)            # per response position
        _write(s["pi"], s["pos"],
               {nm: {"pause": (float(pp[p]) if p < len(pp) else None)} for nm, p in s["pos"].items()})
    del ref; gc.collect(); torch.cuda.empty_cache()

    # ---- metrics ④ teacher T+/T- KL  &  ⑤ student−teacher gap (E20a teacher) -----
    print(f"[b1] loading teacher {TEACHER_REF_MODEL}")
    t_tok = load_tokenizer(TEACHER_REF_MODEL)
    teacher = AutoModelForCausalLM.from_pretrained(TEACHER_REF_MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    for s in saved:
        # Two teacher contexts that differ only in the revealed answer (a6 / probe_utils
        # build_teacher_input): T+ = gold-aware, T- = decoy-aware. ONE forward each.
        ids_p, rs_p = build_teacher_input(t_tok, s["question"], s["gold"], s["base"])
        ids_n, rs_n = build_teacher_input(t_tok, s["question"], s["decoy"], s["base"])
        lp_p = next_token_dist(teacher, ids_p, rs_p, dev)       # (T, V) log-probs
        lp_n = next_token_dist(teacher, ids_n, rs_n, dev)
        T = min(lp_p.shape[0], lp_n.shape[0])
        # KL(T+ || T-) per response position = sum_v P+ (logP+ - logP-)
        kl = (lp_p[:T].exp() * (lp_p[:T] - lp_n[:T])).sum(dim=-1).cpu().numpy()
        # metric ⑤: teacher T+ logp of the actually-taken next base token
        t_taken = taken_token_logp(teacher, ids_p, rs_p, dev)
        vals = {}
        for nm, p in s["pos"].items():
            kl_v = float(kl[p]) if p < len(kl) else None
            rw = row_by.get((s["pi"], nm))
            st_logp = rw.get("st_logp") if rw else None
            gap = (float(st_logp - t_taken[p])
                   if (st_logp is not None and p < len(t_taken)) else None)
            vals[nm] = {"teacher_kl": kl_v, "student_teacher_gap": gap}
        _write(s["pi"], s["pos"], vals)
    del teacher; gc.collect(); torch.cuda.empty_cache()

    # ---- Stage-1 gate: headroom (winner's-curse controlled) ----
    headroom = [h for h in headroom if h is not None]
    h_mean = float(np.mean(headroom)) if headroom else None
    h_p = paired_perm_test(headroom, rng_np) if headroom else None
    stage1_pass = (h_mean is not None and h_mean >= 0.05 and h_p is not None and h_p < 0.05)

    # ---- Stage-2 gate: two-stage findability (plan 88-91) ----
    # All 5 gold-free metrics + the position-index baseline a useful metric must beat.
    #   ① entropy ② onset_slope ③ pause ④ teacher_kl ⑤ student_teacher_gap ; baseline=frac
    # W1: a metric predictive in the NEGATIVE direction is still useful, so we orient
    #     EVERY metric (including position_index) by oriented_auc = max(auc, 1-auc),
    #     applied uniformly in BOTH discovery-selection and confirmation-gating. Raw AUC
    #     is kept for transparency.
    # W2: DISCOVERY half picks the single best metric (by oriented-AUC); CONFIRMATION half
    #     reports & gates that ONE pre-selected metric — no best-of-5 on the confirm half.
    metric_keys = {
        "entropy": "H", "onset_slope": "onset_slope", "pause_propensity": "pause",
        "teacher_tpm_kl": "teacher_kl", "student_teacher_gap": "student_teacher_gap",
        "position_index": "frac",
    }

    def _aucs_for(split_rows):
        """raw + oriented AUC per metric over split_rows (None-safe per metric)."""
        labels = [1 if r["delta_est"] > 0 else 0 for r in split_rows]
        raw, oriented = {}, {}
        if split_rows and 0 < sum(labels) < len(labels):
            lab = np.array(labels)
            for mname, key in metric_keys.items():
                keep = [i for i, r in enumerate(split_rows) if r.get(key) is not None]
                if not keep:
                    continue
                vals = np.array([split_rows[i][key] for i in keep], dtype=float)
                lk = lab[keep]
                if not (0 < lk.sum() < len(lk)):
                    continue
                a = float(mann_whitney_auc(vals[lk == 1], vals[lk == 0]))
                raw[mname] = a
                oriented[mname] = max(a, 1.0 - a)        # W1: orient
        return raw, oriented

    disc = [r for r in rows if not r["confirm"]]
    conf = [r for r in rows if r["confirm"]]
    disc_raw, disc_oriented = _aucs_for(disc)
    conf_raw, conf_oriented = _aucs_for(conf)

    # W2: pre-select the single best gold-free metric on the DISCOVERY half (oriented).
    best_metric = max(((m, a) for m, a in disc_oriented.items() if m != "position_index"),
                      key=lambda x: x[1], default=(None, None))[0]

    # W1+W2: gate the PRE-SELECTED metric's oriented-AUC on the CONFIRMATION half vs the
    # confirmation oriented position-index baseline. Stage-2 = oriented confirm-AUC >= 0.65
    # AND >= position-index oriented-AUC + 0.05.
    pos_idx_auc = conf_oriented.get("position_index", 0.5)
    confirm_auc = conf_oriented.get(best_metric) if best_metric is not None else None
    stage2_pass = (confirm_auc is not None and confirm_auc >= 0.65
                   and confirm_auc - pos_idx_auc >= 0.05)

    # Descriptive only (renamed from boxed_rate): fraction of baseline rollouts that
    # contain the literal \boxed string. Kept for transparency, NOT used to gate.
    boxed_str_rate = (float(np.mean(box_ok)) if box_ok else None)
    # Power-metric fix (plan CHANGE 1): gate on the fraction of GRADED CONTINUATIONS
    # that are is_gradeable (math_verify can parse an answer) — the real floor effect.
    gradeable_rate = (grade_n / grade_d) if grade_d else None
    power_ok = (gradeable_rate is not None and gradeable_rate >= 0.5)
    # W3: power guard ENFORCED in the verdict. If NOT power_ok the run is INCONCLUSIVE
    # (plan line 92) so a low-power run is never misread as the terminal STOP->Phase D
    # decision. Otherwise PASS iff both stages pass, else FAIL.
    if not power_ok:
        status = "INCONCLUSIVE"
    elif stage1_pass and stage2_pass:
        status = "PASS"
    else:
        status = "FAIL"
    verdict = {
        "status": status,
        "substrate": args.substrate, "model": args.model, "n": len(probs), "k": args.k,
        "stage1_headroom_mean_pp": (h_mean * 100 if h_mean is not None else None),
        "stage1_p": (float(h_p) if h_p is not None else None), "stage1_pass": bool(stage1_pass),
        "metric_aucs_confirm": conf_raw, "metric_oriented_aucs_confirm": conf_oriented,
        "metric_oriented_aucs_discovery": disc_oriented,
        "best_metric": best_metric, "best_metric_selected_on": "discovery_half",
        "best_metric_oriented_auc_confirm": (float(confirm_auc) if confirm_auc is not None else None),
        "best_metric_raw_auc_confirm": (conf_raw.get(best_metric) if best_metric is not None else None),
        "position_index_oriented_auc_confirm": float(pos_idx_auc), "stage2_pass": bool(stage2_pass),
        "gradeable_rate": gradeable_rate, "boxed_str_rate": boxed_str_rate, "power_ok": bool(power_ok),
        "metrics_scored": sorted(conf_oriented.keys()),
        # FIX C: fresh-baseline robust-grade headroom selection (no stored is_correct).
        "headroom_pool_size": len(pool), "headroom_kept_wrong": len(probs),
        "headroom_dropped_already_correct": dropped_correct, "headroom_target_n": n,
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    payload = _json_safe({"verdict": verdict, "rows": rows, "headroom": headroom})
    json.dump(payload, open(out, "w"), indent=2)
    print(json.dumps(payload["verdict"], indent=2))


if __name__ == "__main__":
    main()
