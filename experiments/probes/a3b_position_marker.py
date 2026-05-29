"""A.3b — WHERE to force the <|meta|> marker? (position-rule comparison, matched)

Intent (user, 2026-05-29): A.3 injected at the GLOBAL argmax of token entropy and
found a null. But "inject when uncertainty GROWS (onset)" != "inject at the single
peak (argmax)". This probe tests, with MARKER-ONLY inject, whether forcing a
<|meta|> marker at a given position rule causally helps — using PER-POSITION
MATCHED controls so the contrast is clean (codex P0 fix):

  For each rule R in {argmax, onset, random}, from the SAME baseline rollout:
    {R}_noinject = prompt + base[:j_R]              (continue, NO marker)
    {R}_marker   = prompt + base[:j_R] + <|meta|>   (continue, model fills meta)
  marker_delta[R] = acc({R}_marker) - acc({R}_noinject)   ← clean causal effect at R
  (both arms share the identical pinned prefix → no "fresh vs committed" confound)

Position rules:
  argmax — global max body-entropy point (what A.3 did)
  onset  — first body position whose entropy >= this response's own p75
           ("first time the model gets notably uncertain")
  random — random valid body position (CONTROL for position choice)

Two hypotheses, read apart:
  (A) position matters → marker_delta[argmax|onset] > 0 (p<0.05) AND
      > marker_delta[random] (forcing meta at a GOOD position helps).
  (B) capability null  → all marker_delta ~ 0 → meta non-causal regardless of where.

`natural` (prompt-only continuation) is reported for reference (deployment number).
Outputs reports/a3b_position_marker.json
"""
from __future__ import annotations
import argparse, json, time, gc, random as _random
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from huggingface_hub import hf_hub_download
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from common.env import SFT_V8_STRICT, EVAL_R10V2_V8, HF_DATASET, META_OPEN_ID, META_CLOSE_ID, REPORTS_DIR, STRATIFIED_SAMPLE_SEED
from src.training.rewards import _check_correctness
from probes.a3_inject_causal import (
    MARKER_ONLY, MIN_TOK, MAX_RESP_TOK,
    raw_entropy, find_meta_spans, first_boxed_token_idx,
    gen_batch, paired_perm_p, stratified_wrong_hard, stratified_mixed,
)

RULES = ("argmax", "onset", "random")


def body_candidates(resp_ids, H, answer_cap):
    """Valid body positions: >=MIN_TOK, <answer_cap, outside meta spans/markers."""
    L = min(len(H), len(resp_ids))
    hi = min(L, answer_cap)
    in_meta = np.zeros(L, dtype=bool)
    for a, b in find_meta_spans(resp_ids):
        in_meta[min(a, L):min(b, L)] = True
    for i in range(L):
        if resp_ids[i] in (META_OPEN_ID, META_CLOSE_ID):
            in_meta[i] = True
    return [i for i in range(MIN_TOK, hi) if not in_meta[i]]


def pick_position(rule, cand, H, rng):
    """Inject token index for a rule given candidate body positions, or -1."""
    if not cand:
        return -1
    if rule == "argmax":
        return max(cand, key=lambda i: float(H[i]))
    if rule == "onset":
        thr = np.percentile([float(H[i]) for i in cand], 75)  # response's own p75
        for i in cand:                                        # first crossing
            if float(H[i]) >= thr:
                return i
        return cand[-1]
    if rule == "random":
        return int(rng.choice(cand))
    raise ValueError(rule)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=SFT_V8_STRICT)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--max_new", type=int, default=1280)
    ap.add_argument("--select", choices=["wrong_hard", "mixed"], default="mixed")
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--out", default=str(REPORTS_DIR / "a3b_position_marker.json"))
    args = ap.parse_args()
    if args.smoke:
        args.n = args.smoke; args.k = 2; args.max_new = 512
        args.out = args.out.replace(".json", f"_smoke{args.smoke}.json")

    rng = _random.Random(STRATIFIED_SAMPLE_SEED)
    rng_np = np.random.default_rng(STRATIFIED_SAMPLE_SEED)
    t0 = time.time()

    p = hf_hub_download(repo_id=HF_DATASET, repo_type="dataset", filename=EVAL_R10V2_V8)
    data = json.load(open(p))
    select_fn = stratified_mixed if args.select == "mixed" else stratified_wrong_hard
    picks = select_fn(data["results"], args.n, rng)
    print(f"[data] {len(picks)} problems (select={args.select})")

    tok = AutoTokenizer.from_pretrained(args.model)
    assert tok.convert_tokens_to_ids("<|meta|>") == META_OPEN_ID
    assert tok.convert_tokens_to_ids("<|/meta|>") == META_CLOSE_ID
    print(f"[load] {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to("cuda:0").eval()
    marker_seg = tok.encode(MARKER_ONLY, add_special_tokens=False)

    # scored conditions: natural + per-rule {noinject, marker}
    COND = ["natural"] + [f"{r}_{arm}" for r in RULES for arm in ("noinject", "marker")]
    per_prob = []
    boxed = {c: [0, 0] for c in COND}

    for pi, r in enumerate(picks):
        set_seed(STRATIFIED_SAMPLE_SEED + pi)
        gold = str(r["gold_answer"]).strip()
        msgs = [{"role": "user", "content": r["question"]}]
        prompt_str = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prompt_ids = tok.encode(prompt_str, add_special_tokens=False)[:1024]

        try:
            base_resp = gen_batch(model, tok, prompt_ids, args.max_new, 1, "cuda:0")[0]
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); continue
        base_resp = base_resp[:MAX_RESP_TOK]
        H = raw_entropy(model, prompt_ids + base_resp, len(prompt_ids), "cuda:0")
        cap = first_boxed_token_idx(tok, base_resp)
        cand = body_candidates(base_resp, H, cap)

        prefixes = {"natural": prompt_ids}
        pos = {}
        for rule in RULES:
            j = pick_position(rule, cand, H, rng)
            pos[rule] = int(j)
            if j >= 0:
                prefixes[f"{rule}_noinject"] = prompt_ids + base_resp[:j]
                prefixes[f"{rule}_marker"] = prompt_ids + base_resp[:j] + marker_seg
            else:
                prefixes[f"{rule}_noinject"] = None
                prefixes[f"{rule}_marker"] = None

        acc = {}
        for c in COND:
            pre = prefixes.get(c)
            if pre is None:
                acc[c] = None; continue
            mn = max(256, args.max_new - (len(pre) - len(prompt_ids)))
            try:
                conts = gen_batch(model, tok, pre, mn, args.k, "cuda:0")
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); acc[c] = None; continue
            n_correct = 0
            for cont in conts:
                text = tok.decode(pre[len(prompt_ids):] + cont, skip_special_tokens=False)
                n_correct += int(_check_correctness(text, gold))
                boxed[c][1] += 1
                if r"\boxed" in text:
                    boxed[c][0] += 1
            acc[c] = n_correct / len(conts) if conts else None

        # gate inclusion depends ONLY on the per-rule arms, NOT natural (reference)
        gate_cond = [f"{ru}_{arm}" for ru in RULES for arm in ("noinject", "marker")]
        per_prob.append({"benchmark": r["benchmark"], "gold": gold, "pos": pos, "acc": acc,
                         "all_conditions": all(acc.get(c) is not None for c in gate_cond)})
        print(f"  [{pi+1}/{len(picks)}] {r['benchmark']:8s} "
              + " ".join(f"{r2[:3]}:{acc.get(r2+'_noinject')}->{acc.get(r2+'_marker')}" for r2 in RULES)
              + f" ({time.time()-t0:.0f}s)")

    del model; gc.collect(); torch.cuda.empty_cache()

    # ── aggregate: per-rule matched marker_delta = marker - noinject (paired) ───
    gated = [pp for pp in per_prob if pp["all_conditions"]]
    def mean_acc(c): return float(np.mean([pp["acc"][c] for pp in gated])) if gated else float("nan")
    def pdiff(c1, c2): return [pp["acc"][c1] - pp["acc"][c2] for pp in gated]
    boxed_rate = {c: (boxed[c][0] / boxed[c][1] if boxed[c][1] else float("nan")) for c in COND}
    low_power = (not np.isnan(boxed_rate["natural"])) and boxed_rate["natural"] < 0.5

    print(f"\n=== A.3b position-rule (marker-only, matched, gated n={len(gated)}/{len(per_prob)}) ===")
    print(f"  natural = {mean_acc('natural'):.3f} (reference)")
    delta = {}
    for rule in RULES:
        d = pdiff(f"{rule}_marker", f"{rule}_noinject")
        delta[rule] = {"marker_delta": float(np.mean(d)) if d else float("nan"),
                       "p": paired_perm_p(d, rng_np),
                       "acc_noinject": mean_acc(f"{rule}_noinject"),
                       "acc_marker": mean_acc(f"{rule}_marker")}
        print(f"  {rule:7s}: noinject={delta[rule]['acc_noinject']:.3f} "
              f"marker={delta[rule]['acc_marker']:.3f} "
              f"Δ(marker-noinject)={delta[rule]['marker_delta']:+.3f} p={delta[rule]['p']:.3f}")

    # position-choice: does the rule's marker_delta beat random's marker_delta?
    rand_d = pdiff("random_marker", "random_noinject")
    pos_choice = {}
    for rule in ("argmax", "onset"):
        dd = pdiff(f"{rule}_marker", f"{rule}_noinject")
        diff_of_delta = [a - b for a, b in zip(dd, rand_d)]
        pos_choice[rule] = {"delta_vs_random": float(np.mean(diff_of_delta)) if diff_of_delta else float("nan"),
                            "p": paired_perm_p(diff_of_delta, rng_np)}
        print(f"  {rule} marker_delta vs random: {pos_choice[rule]['delta_vs_random']:+.3f} p={pos_choice[rule]['p']:.3f}")

    # gates
    helps = {r: (delta[r]["marker_delta"] >= 0.03 and delta[r]["p"] < 0.05 and not low_power) for r in RULES}
    choice = {r: (pos_choice[r]["delta_vs_random"] >= 0.03 and pos_choice[r]["p"] < 0.05) for r in ("argmax", "onset")}
    # same-rule conjunction: a rule must BOTH help AND beat random (codex)
    position_matters = any(helps[r] and choice[r] for r in ("argmax", "onset"))
    any_helps = any(helps.values())

    print(f"  boxed_rate(natural)={boxed_rate['natural']:.2f}" + ("  ⚠ LOW POWER" if low_power else ""))
    if low_power:
        verdict = "INCONCLUSIVE — low boxed_rate (raise --max_new)"
    elif position_matters:
        verdict = "(A) POSITION MATTERS — marker at a good position helps AND beats random"
    elif any_helps:
        verdict = "(A-weak) marker helps at some position but not beyond random — position-agnostic emission effect"
    else:
        verdict = "(B) CAPABILITY NULL — no position rule's marker beats its matched no-inject; meta non-causal regardless of where"
    print(f"  >>> VERDICT: {verdict}")

    out = {"config": {"model": args.model, "n_total": len(per_prob), "n_gated": len(gated),
                      "k": args.k, "max_new": args.max_new, "select": args.select,
                      "date": "2026-05-29", "inject": "marker_only_matched"},
           "natural_acc": mean_acc("natural"), "marker_delta": delta, "position_choice": pos_choice,
           "boxed_rate": boxed_rate, "low_power": bool(low_power), "verdict": verdict,
           "per_problem": per_prob, "wall_seconds": time.time() - t0}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[done] {args.out}")


if __name__ == "__main__":
    main()
