"""B.2 — E20a self-generated content variance (CTSD Phase B, plan_ctsd_B_probes_2026_06_01).

Intent (plan B.2; extends A.6):
  A.6 showed a teacher discriminates TEMPLATE good/bad meta content (E20a AUC 0.95)
  but never tested E20a's OWN self-generated meta. Here: on the E20a base, marker-inject
  at the argmax body-entropy position (B.1 deployable rule, reuse a3.body_argmax_entropy_pos),
  let E20a GENERATE its own meta + continuation (a3.gen_batch), grade the continuation
  (common.grading.robust_grade). For each generated continuation we extract
  the self-meta block and score it with the E20a-teacher contrastive logp (A.6 winner):
      contrastive = mean_logp_{T+}(meta) − mean_logp_{T-}(meta), answer-token-masked
  using a6.build_prompt_with_meta / score_meta_logp / find_answer_token_mask. We then test
  whether that contrastive score tracks continuation correctness.

Hypotheses (pre-registered, see plan):
  H-B2a (variance exists): correct-leading AND wrong-leading self-metas each number
    >= Nmin=15, else INCONCLUSIVE ("no variance" is itself the A.1 echo on E20a).
  H-B2b (discriminable): AUC(contrastive → continuation-correct) >= 0.65, perm p<0.05
    (reuse probe_utils.mann_whitney_auc, paired_perm_test). AUC>=0.65 → keep contrastive β;
    <0.65 with variance present → drop β (marker + correctness only).

Deviation from plan (documented): E20a was trained with ~0% meta, so when force-opened
it writes meta-style content but frequently never emits <|/meta|> within budget. To keep
the self-meta sample non-degenerate (and the contrastive path exercised), an OPEN-but-
unclosed <|meta|> is accepted: we take its content up to the first \\boxed / a char cap
and synthesize the close (extract_first_meta_block). A properly closed block is still
preferred when present. Also: correct/wrong groups are DISTINCT metas (unpaired), so the
H-B2b perm test is a label-shuffle two-sample test (_label_perm_test) rather than the
paired probe_utils.paired_perm_test, which would be statistically inappropriate here.

Karpathy minimal-change: imports a3 (entropy/inject/gen/grade), a6 (teacher scoring),
probe_utils (stats). a3/a6/probe_utils are NOT modified.

Outputs reports/b2_e20a_content_variance_<tag>.json
"""
from __future__ import annotations
import argparse, json, time, gc, re
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from huggingface_hub import hf_hub_download
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))         # experiments/
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # repo root (src/)
from common.env import (
    TEACHER_MODEL, EVAL_R10V2_E20A, HF_DATASET,
    META_OPEN_ID, META_CLOSE_ID, REPORTS_DIR, STRATIFIED_SAMPLE_SEED,
)
from common.vllm_gen import VllmGen, safe_tokenizer_path
from common.probe_utils import mann_whitney_auc  # noqa: F401 (paired_perm_test is paired;
# B.2 correct/wrong groups are DISTINCT metas → unpaired, so we use a local label-shuffle
# two-sample permutation (_label_perm_test) which is the correct analog. See deviation note.
from probes.a3_inject_causal import (
    raw_entropy, gen_batch, first_boxed_token_idx, body_argmax_entropy_pos,
    MARKER_ONLY, MAX_RESP_TOK,
)
from probes.a6_six_cell_teacher_swap import (
    build_prompt_with_meta, score_meta_logp, find_answer_token_mask,
)
from common.grading import robust_grade, is_gradeable
from collections import defaultdict as _defaultdict

NMIN = 15   # H-B2a: min metas per outcome class for "variance exists"


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
    """Robust tokenizer load (same as B.1). The v8 checkpoint tokenizer FAILS under
    transformers 4.57.6; fall back to the E20a tokenizer (identical Qwen3 vocab,
    <|meta|>=151669, <|/meta|>=151670). Always assert the meta-token IDs."""
    try:
        tok = AutoTokenizer.from_pretrained(path)
    except Exception as e:
        print(f"[tok] {path} failed ({type(e).__name__}: {str(e)[:80]}); "
              f"falling back to E20a tokenizer (identical vocab)")
        tok = AutoTokenizer.from_pretrained("/home/v-seungplee/sft_e20a_local")
    assert tok.convert_tokens_to_ids("<|meta|>") == META_OPEN_ID, "META_OPEN_ID mismatch"
    assert tok.convert_tokens_to_ids("<|/meta|>") == META_CLOSE_ID, "META_CLOSE_ID mismatch"
    return tok


def extract_first_meta_block(text: str, max_chars: int = 1200):
    """Return the first self-generated meta block (markers included), or None.

    Preferred: a properly CLOSED '<|meta|>...<|/meta|>' block. But E20a was trained
    with ~0% meta, so when force-opened it often writes meta-style content and never
    emits <|/meta|> within budget. Treating those as "no meta" would make the
    contrastive path almost never fire. So if an OPEN <|meta|> has no matching close,
    we take the content up to the first answer marker (\\boxed) / a char cap and
    synthesize the close, scoring the self-generated meta CONTENT (the A.6 structure)."""
    closed = re.search(r"<\|meta\|>.*?<\|/meta\|>", text, flags=re.DOTALL)
    if closed:
        return closed.group(0)
    o = text.find("<|meta|>")
    if o < 0:
        return None
    body = text[o + len("<|meta|>"):]
    cut = body.find(r"\boxed")            # stop before the answer is written
    if cut >= 0:
        body = body[:cut]
    body = body[:max_chars].rstrip()
    if not body.strip():
        return None
    return f"<|meta|>{body}\n<|/meta|>"


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


def make_decoy(gold: str, all_golds: list, rng) -> str:
    """T- decoy answer: a DIFFERENT gold from the eval pool (a6 decoy semantics)."""
    g = str(gold).strip()
    for _ in range(20):
        d = str(rng.choice(all_golds)).strip()
        if d and d != g:
            return d
    return (g + "1") if not g.lstrip("-").isdigit() else str(int(g) + 1)


def main():
    import random as _random
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=TEACHER_MODEL, help="E20a base/teacher path")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--k", type=int, default=8, help="self-meta+continuation samples per problem")
    ap.add_argument("--max_new", type=int, default=16384,
                    help="generation budget (real eval regime: max_tokens=16384)")
    ap.add_argument("--max_model_len", type=int, default=20480,
                    help="vLLM context window (real eval regime: 20480)")
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--tag", default="e20a")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    n = args.smoke or args.n
    out = args.out or str(REPORTS_DIR / f"b2_e20a_content_variance_{args.tag}.json")

    rng = _random.Random(STRATIFIED_SAMPLE_SEED)
    rng_np = np.random.default_rng(STRATIFIED_SAMPLE_SEED)
    t0 = time.time()

    ev = hf_hub_download(repo_id=HF_DATASET, repo_type="dataset", filename=EVAL_R10V2_E20A)
    results = json.load(open(ev))
    results = results if isinstance(results, list) else results.get("results") or list(results.values())[0]
    all_golds = [r.get("gold_answer") for r in results if r.get("gold_answer") is not None]
    # FIX C: POOL ignoring stored is_correct; headroom selected from fresh baselines below.
    n_pool = max(3 * n, n + 30)
    pool = stratified_hard_pool(results, n_pool, rng)
    print(f"[b2] model={args.model} n_target={n} pool={len(pool)} k={args.k} "
          f"max_new={args.max_new} max_model_len={args.max_model_len}")

    tok = load_tokenizer(args.model)
    dev = "cuda"

    # ── Phase 1: E20a base — marker-inject at argmax, generate self-meta + continuation
    # Generation MECHANISM = vLLM (replaces a3.gen_batch). E20a force-opened with
    # <|meta|> rarely emits EOS, so HF generate ran the full max_new every call
    # (intractable); vLLM respects EOS + batches. ONLY generation changes — the
    # extraction, grading, and contrastive/gate logic below are byte-for-byte the same.
    # vLLM at modest util (0.45 ≈ 36GB) coexists with the HF entropy forward (~16GB).
    vgen = VllmGen(args.model, tokenizer_path=safe_tokenizer_path(args.model),
                   gpu_memory_utilization=0.45, max_model_len=args.max_model_len,
                   seed=STRATIFIED_SAMPLE_SEED)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    marker_seg = tok.encode(MARKER_ONLY, add_special_tokens=False)

    # FIX C: FRESH 16k baselines for the pool → robust_grade → keep only WRONG (true
    # headroom). Each kept baseline is REUSED for entropy/argmax below (not regenerated).
    pool_prompt_ids = [tok.encode(
        tok.apply_chat_template([{"role": "user", "content": r["question"]}],
                                 tokenize=False, add_generation_prompt=True),
        add_special_tokens=False)[:1024] for r in pool]
    pool_bases = vgen.generate(pool_prompt_ids, n=1, max_tokens=args.max_new,
                               seed=STRATIFIED_SAMPLE_SEED)
    probs, base_by_pi, dropped_correct = [], {}, 0
    for r, pids, outs in zip(pool, pool_prompt_ids, pool_bases):
        if len(probs) >= n:
            break
        base = outs[0]            # FIX: full 16k baseline (no 4096 cap) — matches b1.
        full_resp = tok.decode(base, skip_special_tokens=False)
        if robust_grade(full_resp, str(r["gold_answer"]).strip()):
            dropped_correct += 1          # already-correct → NOT headroom, drop (logged)
            continue
        base_by_pi[len(probs)] = (pids, base)
        probs.append(r)
    headroom_drop_log = (f"[b2][headroom] pool={len(pool)} kept_wrong={len(probs)} "
                         f"dropped_already_correct={dropped_correct} target_n={n}")
    print(headroom_drop_log)
    if len(probs) < n:
        print(f"[b2][headroom] WARNING: only {len(probs)} robust-wrong found in pool "
              f"of {len(pool)} (< target {n}); proceeding with what we have")
    metas = []   # {question, gold, decoy, pre_meta_body, meta_block, correct}
    # Power-metric fix (plan CHANGE 1): gate on gradeable continuations (math_verify
    # can parse an answer), the real floor effect — not the \boxed STRING presence.
    box_n, box_d = 0, 0          # \boxed string count (descriptive only)
    grade_n = 0                  # is_gradeable continuations (drives power; denom = box_d)

    # Pass A: per-problem seeded baseline (vLLM n=1) + HF entropy → argmax p* →
    # build the marker-inject prefix. Collect prefixes to batch the k-sample inject
    # generation across ALL problems in ONE vLLM call (the speed win).
    inject_prefixes, meta_info = [], []   # meta_info[i] aligns with inject_prefixes[i]
    for pi, r in enumerate(probs):
        q, gold = r["question"], str(r["gold_answer"]).strip()
        # FIX C: REUSE the fresh 16k baseline already generated for headroom selection.
        prompt_ids, base = base_by_pi[pi]
        H = raw_entropy(model, prompt_ids + base, len(prompt_ids), dev)
        cap = first_boxed_token_idx(tok, base)
        p_star, _, _ = body_argmax_entropy_pos(base, H, cap)
        pre_meta_body = tok.decode(base[:p_star], skip_special_tokens=False)
        # marker-inject: model generates its OWN meta + continuation from the pinned prefix
        inject_prefixes.append(prompt_ids + base[:p_star] + marker_seg)
        meta_info.append({"pi": pi, "benchmark": r["benchmark"], "q": q, "gold": gold,
                          "p_star": p_star, "pre_meta_body": pre_meta_body})

    # Pass B: ONE batched vLLM call (n=k) for ALL problems' inject-prefixes, then the
    # extraction/grading loop is unchanged from the per-problem version.
    all_conts = vgen.generate(inject_prefixes, n=args.k, max_tokens=args.max_new,
                              seed=STRATIFIED_SAMPLE_SEED)
    for info, conts in zip(meta_info, all_conts):
        q, gold, pre_meta_body = info["q"], info["gold"], info["pre_meta_body"]
        for c in conts:
            # full forced response text: prefix-resp + injected marker + generated tail
            tail_text = tok.decode(marker_seg + c, skip_special_tokens=False)
            full_resp_text = pre_meta_body + tail_text
            correct = int(robust_grade(full_resp_text, gold))
            box_d += 1
            if r"\boxed" in full_resp_text:
                box_n += 1
            if is_gradeable(full_resp_text):
                grade_n += 1
            meta_block = extract_first_meta_block(tail_text)
            if meta_block is None:        # model never closed a meta → no scorable self-meta
                continue
            metas.append({"question": q, "gold": gold,
                          "decoy": make_decoy(gold, all_golds, rng),
                          "pre_meta_body": pre_meta_body[-3000:], "meta_block": meta_block,
                          # W5: instrument the unclosed-meta length confound — record the
                          # meta block token-length so a reviewer can check whether block
                          # length correlates with outcome (extraction logic unchanged).
                          "block_len": len(tok.encode(meta_block, add_special_tokens=False)),
                          "correct": correct})
        print(f"  [{info['pi']+1}/{len(probs)}] {info['benchmark']} p*={info['p_star']} "
              f"metas={len(metas)} ({time.time()-t0:.0f}s)")
    vgen.free()
    del model; gc.collect(); torch.cuda.empty_cache()

    # ── Phase 2: E20a-teacher contrastive scoring of each self-meta (T+ gold, T- decoy)
    # E20a is both base and A.6-winning teacher → same path, reload after the free.
    teacher = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    for mrow in metas:
        scores = {}
        for ctx_label, ans in (("Tplus", mrow["gold"]), ("Tminus", mrow["decoy"])):
            input_ids, _, meta_start, meta_len = build_prompt_with_meta(
                tok, mrow["question"], ans, mrow["pre_meta_body"], mrow["meta_block"])
            logp = score_meta_logp(teacher, input_ids, meta_start, meta_len, dev)
            meta_token_ids = input_ids[meta_start:meta_start + meta_len]
            # W6 answer-token leak fix: mask the UNION of {gold, decoy} strings in BOTH
            # the T+ and T- scorings. Masking only the routed answer per context let a
            # gold string quoted in a self-meta leak into the T- score (and vice-versa).
            # OR the two per-string masks (a6.find_answer_token_mask) so neither answer
            # token contributes to either contrastive arm.
            gold_mask = find_answer_token_mask(tok, mrow["meta_block"], mrow["gold"], meta_token_ids)
            decoy_mask = find_answer_token_mask(tok, mrow["meta_block"], mrow["decoy"], meta_token_ids)
            ans_mask = gold_mask | decoy_mask
            non_ans = logp[~ans_mask]
            scores[ctx_label] = float(np.mean(non_ans)) if len(non_ans) else float(np.mean(logp))
        mrow["contrastive"] = scores["Tplus"] - scores["Tminus"]
    del teacher; gc.collect(); torch.cuda.empty_cache()

    # ── H-B2a: variance exists (both outcome classes >= NMIN) ────────────────────
    scored = [m for m in metas if "contrastive" in m and not np.isnan(m["contrastive"])]
    correct_scores = [m["contrastive"] for m in scored if m["correct"] == 1]
    wrong_scores = [m["contrastive"] for m in scored if m["correct"] == 0]
    n_correct, n_wrong = len(correct_scores), len(wrong_scores)
    variance_exists = (n_correct >= NMIN and n_wrong >= NMIN)

    # ── H-B2b: AUC(contrastive → continuation correct) >= 0.65, perm p<0.05 ───────
    auc = perm_p = None
    if correct_scores and wrong_scores:
        _a = float(mann_whitney_auc(correct_scores, wrong_scores))
        auc = None if np.isnan(_a) else _a            # keep JSON strict (no NaN)
        # unpaired-by-construction (correct vs wrong are different metas): two-sample
        # sign-flip on the pooled per-meta deviation is not paired, so use the
        # permutation in probe_utils on the difference of group means via labels shuffle.
        perm_p = float(_label_perm_test(scored, rng_np))
    h_b2b_pass = (auc is not None and auc >= 0.65 and perm_p is not None and perm_p < 0.05)

    # Descriptive only (renamed from box_rate): fraction of continuations with the
    # literal \boxed string. Kept for transparency, NOT used to gate.
    boxed_str_rate = (box_n / box_d) if box_d else None
    # Power-metric fix (plan CHANGE 1): gate on the fraction of continuations that are
    # is_gradeable (math_verify can parse an answer) — the real floor effect.
    gradeable_rate = (grade_n / box_d) if box_d else None
    power_ok = (gradeable_rate is not None and gradeable_rate >= 0.5)

    # W5: meta block token-length per outcome class (length confound instrumentation).
    block_len_correct = [m["block_len"] for m in scored if m["correct"] == 1]
    block_len_wrong = [m["block_len"] for m in scored if m["correct"] == 0]
    mean_block_len_correct = (float(np.mean(block_len_correct)) if block_len_correct else None)
    mean_block_len_wrong = (float(np.mean(block_len_wrong)) if block_len_wrong else None)

    # W3 power-guard ENFORCED: status ∈ {PASS, FAIL, INCONCLUSIVE}.
    #   INCONCLUSIVE if NOT power_ok OR H-B2a fails (a class has < NMIN metas).
    #   PASS only if power_ok AND H-B2a AND H-B2b; else FAIL.
    if (not power_ok) or (not variance_exists):
        status = "INCONCLUSIVE"
    elif h_b2b_pass:
        status = "PASS"
    else:
        status = "FAIL"

    if not variance_exists:
        verdict = (f"INCONCLUSIVE — variance absent (correct={n_correct}, wrong={n_wrong}, "
                   f"need >= {NMIN} each); A.1 'no-variance' echo on E20a")
    elif not power_ok:
        verdict = (f"INCONCLUSIVE — power guard failed (gradeable_rate={gradeable_rate}); raise --max_new")
    elif h_b2b_pass:
        verdict = f"PASS — contrastive discriminates (AUC={auc:.3f}, p={perm_p:.3f}); keep contrastive β"
    else:
        verdict = (f"FAIL — variance present but not discriminable "
                   f"(AUC={auc}, p={perm_p}); drop β (marker + correctness only)")

    summary = {
        "status": status,
        "model": args.model, "n_problems": len(probs), "k": args.k,
        "n_metas_scored": len(scored), "n_correct_leading": n_correct, "n_wrong_leading": n_wrong,
        "nmin": NMIN, "variance_exists_HB2a": bool(variance_exists),
        "contrastive_auc": (float(auc) if auc is not None else None),
        "perm_p": (float(perm_p) if perm_p is not None else None),
        "h_b2b_pass": bool(h_b2b_pass),
        "gradeable_rate": (float(gradeable_rate) if gradeable_rate is not None else None),
        "boxed_str_rate": (float(boxed_str_rate) if boxed_str_rate is not None else None),
        "power_ok": bool(power_ok),
        "mean_contrastive_correct": (float(np.mean(correct_scores)) if correct_scores else None),
        "mean_contrastive_wrong": (float(np.mean(wrong_scores)) if wrong_scores else None),
        "mean_block_len_correct": mean_block_len_correct,
        "mean_block_len_wrong": mean_block_len_wrong,
        # FIX C: fresh-baseline robust-grade headroom selection (no stored is_correct).
        "headroom_pool_size": len(pool), "headroom_kept_wrong": len(probs),
        "headroom_dropped_already_correct": dropped_correct, "headroom_target_n": n,
        "verdict": verdict, "wall_seconds": time.time() - t0,
    }
    payload = {"summary": summary,
               "metas": [{"correct": int(m["correct"]), "contrastive": m.get("contrastive")} for m in scored]}
    payload = _json_safe(payload)        # cast numpy + NaN→None so output is strict JSON
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(payload, open(out, "w"), indent=2)
    print(json.dumps(payload["summary"], indent=2))
    print(f"[done] {out}")


def _label_perm_test(scored, rng, n_perm: int = 5000) -> float:
    """Two-sample permutation p for mean(contrastive|correct) - mean(contrastive|wrong)
    by shuffling correctness labels. Distinct metas → not paired, so we permute labels
    (matches the unpaired AUC test)."""
    vals = np.array([m["contrastive"] for m in scored], dtype=float)
    labs = np.array([m["correct"] for m in scored])
    if labs.sum() == 0 or labs.sum() == len(labs):
        return float("nan")
    obs = abs(vals[labs == 1].mean() - vals[labs == 0].mean())
    hits = 0
    for _ in range(n_perm):
        perm = rng.permutation(labs)
        if abs(vals[perm == 1].mean() - vals[perm == 0].mean()) >= obs:
            hits += 1
    return (hits + 1) / (n_perm + 1)


if __name__ == "__main__":
    main()
