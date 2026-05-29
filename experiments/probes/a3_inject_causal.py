"""A.3 — Force-inject CAUSAL test (THE gate for all CTSD training).

Intent (PLAN.md §Phase A.3, v5; user challenge 2026-05-28):
  Answer two questions in one inference-only probe:
    (i)  Does the model's NATURAL meta carry good/bad variance, and does that
         variance associate with correctness?  (descriptive natural-meta analysis)
    (ii) Does FORCE-INJECTING meta at high-entropy positions causally improve
         accuracy, and does the DIRECTION (productive vs unproductive) matter?

Design (shared-prefix isolates the injection effect; codex-reviewed 2026-05-29):
  Student = v8_strict SFT, solving on its own (NO reference answer in prompt).
  Sample = hard-benchmark problems whose PRIOR full rollout was wrong (headroom).
  We regenerate a fresh baseline rollout (needed for an entropy trace we can
  continue from), so the slice is "hard problems the model previously failed";
  a fresh baseline may occasionally differ. Selection rule fixed before scoring.
  For each problem:
    1. Generate one baseline rollout (sampled, per-problem seeded).
    2. One forward pass → per-token RAW entropy (same definition as A.2).
    3. Inject at the ARGMAX body-entropy position (>=MIN_TOK, outside meta, BEFORE
       the first \\boxed — post-answer injection is meaningless). Always fires, so
       no τ-coverage problem. Freeze prefix = [prompt + response[:p*]].
    4. From the SAME prefix, k continuations under 4 conditions:
         (a) no-inject     — continue normally
         (b) marker-only   — append "\\n<|meta|>\\n", model fills + closes
         (c) good-inject    — append PRODUCTIVE meta (low-conf, re-derive, verify)
         (d) bad-inject     — append UNPRODUCTIVE meta (over-conf, skip checking)
       Both (c) and (d) are ANSWER-FREE and ~length-matched (≈45-47 tok): neither
       names gold nor decoy, so (c) vs (d) is a clean test of meta DIRECTION, not
       answer priming (codex P0 fix).
    5. Grade each continuation vs gold via math_verify-backed _check_correctness.

Pre-registered gates (locked; over problems with all 4 conditions present —
paired intersection):
  helps     : acc(c_good) - acc(a_noinject) >= +3pp AND paired p < 0.05
  direction : acc(c_good) - acc(d_bad)      >= +5pp AND paired p < 0.05
  PASS for training = helps AND direction   (good inject must beat BOTH no-inject
                      and bad inject — rules out "d made worse" artifact, codex P0).
  power guard: baseline boxed_rate < 0.5 → INCONCLUSIVE (truncation), raise max_new.
  FAIL = gates miss AND power OK → meta inject non-causal → STOP + brainstorm.
  emission (b-a) and content (c-b) reported as descriptive context, not gates.

Outputs reports/a3_inject_causal.json
"""
from __future__ import annotations
import argparse, json, random, time, gc
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from huggingface_hub import hf_hub_download
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))         # experiments/
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # repo root (for src/)
from common.env import (
    SFT_V8_STRICT, EVAL_R10V2_V8, HF_DATASET,
    META_OPEN_ID, META_CLOSE_ID, REPORTS_DIR, STRATIFIED_SAMPLE_SEED,
)
from src.training.rewards import (
    _check_correctness, _meta_block_texts,
    _has_effective_verification_signal, _has_redirection_signal,
    _has_uncertainty_signal,
)

# Symmetric, ANSWER-FREE direction templates (codex P0 fix): the only difference
# is the metacognitive STANCE (reconsider+verify vs over-confident+skip), never
# any numeric answer. Length-matched.
GOOD_META = (
    "\n<|meta|>\n"
    "confidence: 0.3\n"
    "I am not fully sure this route is correct. Let me slow down, re-examine the "
    "setup, recompute the key step carefully, and verify the result before committing.\n"
    "<|/meta|>\n"
)
BAD_META = (
    "\n<|meta|>\n"
    "confidence: 0.95\n"
    "This route looks fine and I am confident. There is no need to re-check anything; "
    "I will just continue along the current line and commit to my result.\n"
    "<|/meta|>\n"
)
MARKER_ONLY = "\n<|meta|>\n"

MIN_TOK = 50         # don't inject in the first MIN_TOK response tokens
MAX_RESP_TOK = 4096
COND = ("a_noinject", "b_marker", "c_good", "d_bad")


# ── entropy + prefix finding ─────────────────────────────────────────────────

@torch.no_grad()
def raw_entropy(model, input_ids, resp_start, device):
    """Per-response-token RAW predictive entropy (same definition as A.2)."""
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    out = model(ids, use_cache=False)
    logits = out.logits[0]
    pred = logits[resp_start - 1: -1].float()
    lp = torch.nn.functional.log_softmax(pred, dim=-1)
    H = -(lp.exp() * lp).sum(dim=-1)
    return H.cpu().numpy()


def find_meta_spans(resp_ids):
    spans, in_meta, start = [], False, 0
    for i, t in enumerate(resp_ids):
        if t == META_OPEN_ID:
            in_meta, start = True, i + 1
        elif t == META_CLOSE_ID and in_meta:
            spans.append((start, i)); in_meta = False
    return spans


def first_boxed_token_idx(tok, resp_ids):
    """Token index just before the first ``\\boxed`` appears (else len). Used to
    cap the injection search: injecting meta AFTER the answer is written is
    meaningless and would inflate boxed_rate via the prefix (codex P1)."""
    decoded = ""
    for i, tid in enumerate(resp_ids):
        decoded += tok.decode([tid])
        if r"\boxed" in decoded:
            return i
    return len(resp_ids)


def body_argmax_entropy_pos(resp_ids, H, answer_cap):
    """Injection point = body position (>=MIN_TOK, outside meta, BEFORE the first
    answer) with MAX entropy.

    We inject at the model's single most-uncertain pre-answer body token (always
    fires) — exactly the H2 intent "intervene where the model is most uncertain"
    and consistent with A.2 (body H_max predicts wrongness, AUC 0.749). An
    absolute τ would fire on only ~10% of rollouts (codex P1) → empty gate;
    argmax has no coverage problem. answer_cap excludes post-answer positions.
    Returns (idx, H_at_idx, frac) where frac = idx / L."""
    L = min(len(H), len(resp_ids))
    hi = min(L, answer_cap)  # do not inject after the answer is already written
    in_meta = np.zeros(L, dtype=bool)
    for a, b in find_meta_spans(resp_ids):
        in_meta[min(a, L):min(b, L)] = True
    for pos in range(L):
        if resp_ids[pos] in (META_OPEN_ID, META_CLOSE_ID):
            in_meta[pos] = True
    cand = [(float(H[i]), i) for i in range(MIN_TOK, hi) if not in_meta[i]]
    if not cand:
        idx = min(hi, MIN_TOK) if hi > 0 else 0
        return idx, (float(H[idx - 1]) if idx > 0 else 0.0), (idx / L if L else 0.0)
    h, idx = max(cand)
    return idx, h, idx / L


# ── generation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def gen_batch(model, tok, prefix_ids, max_new, k, device, temperature=0.7):
    """k sampled continuations (token-id lists, response-relative to prefix)."""
    ids = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    out = model.generate(
        ids, max_new_tokens=max_new, do_sample=True, temperature=temperature,
        top_p=0.95, num_return_sequences=k,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
    )
    return [out[j, len(prefix_ids):].tolist() for j in range(k)]


# ── natural-meta variance (answers user Q-i) ─────────────────────────────────

def classify_natural_meta(full_resp_text):
    """Classify a no-inject continuation's natural meta:
      'none'        — no meta block emitted
      'substantive' — meta has verification/redirection/uncertainty signal
      'boilerplate' — meta emitted but no productive signal
    """
    blocks = _meta_block_texts(full_resp_text)
    if not blocks:
        return "none"
    joined = "\n".join(blocks)
    productive = (_has_effective_verification_signal(joined)
                  or _has_redirection_signal(joined)
                  or _has_uncertainty_signal(joined))
    return "substantive" if productive else "boilerplate"


# ── stats ────────────────────────────────────────────────────────────────────

def paired_perm_p(diffs, rng, n=5000):
    diffs = np.asarray([d for d in diffs if d is not None])
    if len(diffs) == 0:
        return float("nan")
    obs = abs(diffs.mean())
    hits = sum(abs((diffs * rng.choice([-1, 1], len(diffs))).mean()) >= obs for _ in range(n))
    return (hits + 1) / (n + 1)


def stratified_wrong_hard(results, n, rng):
    """Pick n baseline-WRONG rollouts from hard benchmarks (headroom). Fixed rule."""
    hard = [r for r in results
            if r["benchmark"] in ("aime", "math500", "math") and not r["is_correct"]]
    by_b = defaultdict(list)
    for r in hard:
        by_b[r["benchmark"]].append(r)
    picks, per = [], max(1, n // max(1, len(by_b)))
    for b, lst in by_b.items():
        rng.shuffle(lst); picks.extend(lst[:per])
    rng.shuffle(picks)
    return picks[:n]


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=SFT_V8_STRICT)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--max_new", type=int, default=2048)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--out", default=str(REPORTS_DIR / "a3_inject_causal.json"))
    args = ap.parse_args()
    if args.smoke:
        args.n = args.smoke; args.k = 2; args.max_new = 512
        args.out = args.out.replace(".json", f"_smoke{args.smoke}.json")

    rng = random.Random(STRATIFIED_SAMPLE_SEED)
    rng_np = np.random.default_rng(STRATIFIED_SAMPLE_SEED)
    t0 = time.time()

    p = hf_hub_download(repo_id=HF_DATASET, repo_type="dataset", filename=EVAL_R10V2_V8)
    data = json.load(open(p))
    picks = stratified_wrong_hard(data["results"], args.n, rng)
    print(f"[data] {len(picks)} baseline-wrong hard problems")

    tok = AutoTokenizer.from_pretrained(args.model)
    # P2: assert hard-coded meta token IDs match this tokenizer
    assert tok.convert_tokens_to_ids("<|meta|>") == META_OPEN_ID, "META_OPEN_ID mismatch"
    assert tok.convert_tokens_to_ids("<|/meta|>") == META_CLOSE_ID, "META_CLOSE_ID mismatch"
    print(f"[load] {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to("cuda:0").eval()

    good_seg = tok.encode(GOOD_META, add_special_tokens=False)
    bad_seg = tok.encode(BAD_META, add_special_tokens=False)
    marker_seg = tok.encode(MARKER_ONLY, add_special_tokens=False)

    per_prob = []
    # natural-meta accumulators (across all a_noinject continuations)
    nat = {"substantive": [], "boilerplate": [], "none": []}  # -> list of 0/1 correctness
    nat_emit, b_close = 0, 0  # counters: meta-emitted continuations, b closed-meta
    nat_total, b_total = 0, 0
    boxed = {c: [0, 0] for c in COND}  # per-condition [n_boxed, n_total] (power check)

    for pi, r in enumerate(picks):
        set_seed(STRATIFIED_SAMPLE_SEED + pi)  # reproducible per-problem sampling
        question = r["question"]
        gold = str(r["gold_answer"]).strip()

        msgs = [{"role": "user", "content": question}]
        prompt_str = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prompt_ids = tok.encode(prompt_str, add_special_tokens=False)[:1024]

        # 1. baseline rollout → locate injection point
        try:
            base_resp = gen_batch(model, tok, prompt_ids, args.max_new, 1, "cuda:0")[0]
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); continue
        base_resp = base_resp[:MAX_RESP_TOK]
        # 2. raw entropy; inject at the max-entropy PRE-ANSWER body position
        H = raw_entropy(model, prompt_ids + base_resp, len(prompt_ids), "cuda:0")
        answer_cap = first_boxed_token_idx(tok, base_resp)
        p_star, H_at, inject_frac = body_argmax_entropy_pos(base_resp, H, answer_cap)
        prefix_ids = prompt_ids + base_resp[:p_star]

        cond_prefix = {
            "a_noinject": prefix_ids,
            "b_marker": prefix_ids + marker_seg,
            "c_good": prefix_ids + good_seg,
            "d_bad": prefix_ids + bad_seg,
        }

        acc = {}
        for c in COND:
            pre = cond_prefix[c]
            mn = max(256, args.max_new - (len(pre) - len(prefix_ids)))
            try:
                conts = gen_batch(model, tok, pre, mn, args.k, "cuda:0")
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); acc[c] = None; continue
            n_correct = 0
            for cont in conts:
                full_resp_ids = pre[len(prompt_ids):] + cont
                text = tok.decode(full_resp_ids, skip_special_tokens=False)
                correct = _check_correctness(text, gold)
                n_correct += int(correct)
                boxed[c][1] += 1
                if r"\boxed" in text:
                    boxed[c][0] += 1
                # natural-meta variance bookkeeping (only condition a)
                if c == "a_noinject":
                    cls = classify_natural_meta(text)
                    nat[cls].append(int(correct))
                    nat_total += 1
                    if cls != "none":
                        nat_emit += 1
                # marker-only close-rate (did the model close the forced block?)
                if c == "b_marker":
                    b_total += 1
                    if "<|/meta|>" in text:
                        b_close += 1
            acc[c] = n_correct / len(conts) if conts else None

        per_prob.append({
            "benchmark": r["benchmark"], "gold": gold,
            "p_star": int(p_star), "H_at_inject": H_at, "inject_frac": inject_frac,
            "prefix_len": len(prefix_ids), "acc": acc,
            "all_conditions": all(acc.get(c) is not None for c in COND),
        })
        print(f"  [{pi+1}/{len(picks)}] {r['benchmark']:8s} p*={p_star:4d} "
              f"H={H_at:.2f} a={acc.get('a_noinject')} b={acc.get('b_marker')} "
              f"c={acc.get('c_good')} d={acc.get('d_bad')} ({time.time()-t0:.0f}s)")

    del model; gc.collect(); torch.cuda.empty_cache()

    # ── aggregate over problems with all-4-conditions present (paired) ─────────
    gated = [pp for pp in per_prob if pp["all_conditions"]]
    n_gated = len(gated)

    def mean_acc(pps, c):
        v = [pp["acc"][c] for pp in pps]
        return float(np.mean(v)) if v else float("nan")

    def pdiff(pps, c1, c2):
        return [pp["acc"][c1] - pp["acc"][c2] for pp in pps]

    A = mean_acc(gated, "a_noinject"); B = mean_acc(gated, "b_marker")
    C = mean_acc(gated, "c_good");     D = mean_acc(gated, "d_bad")
    helps_d, direction_d = pdiff(gated, "c_good", "a_noinject"), pdiff(gated, "c_good", "d_bad")
    helps_p = paired_perm_p(helps_d, rng_np)
    direction_p = paired_perm_p(direction_d, rng_np)

    gate_helps = (C - A) >= 0.03 and helps_p < 0.05
    gate_direction = (C - D) >= 0.05 and direction_p < 0.05
    # Power guard: if continuations rarely reach a \boxed answer, an all-low
    # result is truncation, not a real null. Flag instead of declaring FAIL.
    boxed_rate = {c: (boxed[c][0] / boxed[c][1] if boxed[c][1] else float("nan")) for c in COND}
    low_power = (not np.isnan(boxed_rate["a_noinject"])) and boxed_rate["a_noinject"] < 0.5
    # low_power takes precedence: a truncated run must NOT declare a training PASS.
    overall_pass = gate_helps and gate_direction and not low_power

    # natural-meta variance summary
    def acc_of(cls):
        return float(np.mean(nat[cls])) if nat[cls] else float("nan")
    nat_summary = {
        "emit_rate": (nat_emit / nat_total) if nat_total else float("nan"),
        "substantive_n": len(nat["substantive"]), "substantive_acc": acc_of("substantive"),
        "boilerplate_n": len(nat["boilerplate"]), "boilerplate_acc": acc_of("boilerplate"),
        "no_meta_n": len(nat["none"]), "no_meta_acc": acc_of("none"),
        "b_marker_close_rate": (b_close / b_total) if b_total else float("nan"),
    }

    print(f"\n=== A.3 inject causal (gated n={n_gated}/{len(per_prob)}) ===")
    print(f"  acc(a_noinject) = {A:.3f}")
    print(f"  acc(b_marker)   = {B:.3f}  | b-a={B-A:+.3f} (emission, descriptive)")
    print(f"  acc(c_good)     = {C:.3f}  | c-a={C-A:+.3f} p={helps_p:.3f} "
          f"{'GATE✓' if gate_helps else 'fail'} (HELPS)")
    print(f"  acc(d_bad)      = {D:.3f}  | c-d={C-D:+.3f} p={direction_p:.3f} "
          f"{'GATE✓' if gate_direction else 'fail'} (DIRECTION)")
    print(f"  natural meta: emit={nat_summary['emit_rate']:.2f} "
          f"subst_acc={nat_summary['substantive_acc']} boiler_acc={nat_summary['boilerplate_acc']} "
          f"none_acc={nat_summary['no_meta_acc']} | b_close={nat_summary['b_marker_close_rate']}")
    print(f"  boxed_rate: a={boxed_rate['a_noinject']:.2f} b={boxed_rate['b_marker']:.2f} "
          f"c={boxed_rate['c_good']:.2f} d={boxed_rate['d_bad']:.2f}"
          + ("  ⚠ LOW POWER (raise --max_new)" if low_power else ""))
    if low_power:
        verdict = "INCONCLUSIVE — low boxed_rate (truncation); raise --max_new and rerun"
    elif overall_pass:
        verdict = "PASS — inject is causal & helps, proceed to training"
    else:
        verdict = "FAIL — inject non-causal, STOP + brainstorm"
    print(f"  >>> VERDICT: {verdict}")

    out = {
        "config": {"model": args.model, "n_total": len(per_prob), "n_gated": n_gated,
                   "k": args.k, "max_new": args.max_new, "date": "2026-05-29",
                   "inject_point": "body_argmax_entropy"},
        "mean_acc_gated": {"a_noinject": A, "b_marker": B, "c_good": C, "d_bad": D},
        "boxed_rate": boxed_rate,
        "low_power": bool(low_power),
        "gates": {
            "helps_c_minus_a": {"delta": C - A, "p": helps_p, "pass": bool(gate_helps)},
            "direction_c_minus_d": {"delta": C - D, "p": direction_p, "pass": bool(gate_direction)},
            "emission_b_minus_a_descriptive": float(B - A),
            "content_c_minus_b_descriptive": float(C - B),
            "overall_pass": bool(overall_pass),
            "verdict": verdict,
            "pass_rule": "helps (c>a, p<.05) AND direction (c>d, p<.05); both answer-free templates",
        },
        "natural_meta_variance": nat_summary,
        "per_problem": per_prob,
        "wall_seconds": time.time() - t0,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[done] {args.out}")


if __name__ == "__main__":
    main()
