"""GATE 2 -- does the context delta actually separate a template from a correction?

This is the whole redesign compressed into one offline measurement. It runs on
the pilot's own rollouts, on one GPU, with no generation, and it decides whether
a node gets requested at all.

FOUR CONDITIONS, one meta slot, everything else byte-identical:

  donor   the length-matched meta from a DIFFERENT problem   -> floor, must be 0.0
  real    the rollout's own meta
  boiler  the template rq3v2g_b4p2 converged on by step 295  -> the failure mode
  oracle  a meta that states the gold answer outright        -> ceiling

and, for real and boiler, the LEGACY position delta as the contrast arm.

PRE-REGISTERED PREDICTIONS (written before the first run; band edges are the
verdict, not a description of whatever comes out)

  P0  wiring     d_abl(donor)  == 0.000 exactly. A nonzero value is a code bug,
                 not a finding, and everything below is void.
  P1  ceiling    mean d_abl(oracle) > 2.0 nats. Below that the measurement has
                 no resolution on this model and the arm is not worth a node.
  P2  the claim  mean |d_abl(boiler)| < 0.3  AND
                 mean d_abl(real) - mean d_abl(boiler) > 0.3
  P3  contrast   mean d_pos(boiler) > 0.3 -- i.e. the CURRENT definition pays
                 for the template. rq3v2g_b4p2 step 295 measured +0.289 for
                 template rows against -0.145 for everything else, so a null
                 here means this harness disagrees with the live run and the
                 harness is what gets debugged first.

P2 is the one that can kill the redesign. If the template does not sit at zero,
the context delta is not the fix I claimed it was, and no arm launches.
"""
from __future__ import annotations

import argparse
import json
import statistics as st

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from abl_pmi import (body_before_answer, make_decoy, pick_donor, pmi_at,
                     split_meta, with_meta)

# Verbatim shape of the template rq3v2g_b4p2 converged on (67.4% of metas at
# step 295). Carries no quantity from any particular problem -- that is the
# property under test.
BOILER = (" confidence: 0.88 The answer looks plausible but should not be "
          "accepted without an independent check, since the result can be "
          "verified directly from the quantities given. decision: verify ")

ORACLE_TMPL = (" confidence: 0.99 Working it through once more, the value comes "
               "out to {gold}. decision: verify ")


def load(pilot_path: str, limit: int):
    with open(pilot_path) as f:
        recs = json.load(f)["records"]
    rows = [r for r in recs if r.get("wellformed")]
    return rows[:limit]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", default="pilot.json")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--out", default="gate2.json")
    args = ap.parse_args()

    rows = load(args.pilot, args.limit)
    if not rows:
        raise SystemExit("no wellformed rollouts in the pilot -- GATE 1 failed")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda").eval()

    pool = [(r["problem"], split_meta(r["text"])[1]) for r in rows]
    out, skipped = [], 0

    for r in rows:
        parts = split_meta(r["text"])
        if parts is None:
            skipped += 1
            continue
        before, own, after = parts
        donor = pick_donor(tok, own, pool, r["problem"])
        if donor is None:
            skipped += 1
            continue

        gold = str(r["gold"])
        decoy = make_decoy(gold)
        prompt = r["prompt"]

        def pmi_for(inner: str):
            body = body_before_answer(with_meta(before, inner, after))
            if body is None:
                return None
            return pmi_at(model, tok, prompt + body, gold, decoy)

        base = pmi_for(donor)                      # the control context
        if base is None:
            skipped += 1
            continue
        conds = {"donor": donor, "real": own, "boiler": BOILER,
                 "oracle": ORACLE_TMPL.format(gold=gold)}
        rec = {"problem": r["problem"], "correct": r["answer"] == gold}
        ok = True
        for name, inner in conds.items():
            v = pmi_for(inner)
            if v is None:
                ok = False
                break
            rec[f"d_abl_{name}"] = v - base
        if not ok:
            skipped += 1
            continue

        # legacy position delta, same rows, for real and boiler
        open_ctx = prompt + before
        p_open = pmi_at(model, tok, open_ctx, gold, decoy)
        for name, inner in (("real", own), ("boiler", BOILER)):
            close_ctx = prompt + with_meta(before, inner, "")
            rec[f"d_pos_{name}"] = pmi_at(model, tok, close_ctx, gold, decoy) - p_open

        out.append(rec)

    def m(k):
        v = [r[k] for r in out if k in r]
        return {"mean": st.mean(v), "sd": st.pstdev(v), "n": len(v)} if v else None

    keys = ["d_abl_donor", "d_abl_real", "d_abl_boiler", "d_abl_oracle",
            "d_pos_real", "d_pos_boiler"]
    stats = {k: m(k) for k in keys}

    verdict = {
        "P0_wiring_donor_is_zero": abs(stats["d_abl_donor"]["mean"]) < 1e-6,
        "P1_ceiling_oracle_gt_2": stats["d_abl_oracle"]["mean"] > 2.0,
        "P2_boiler_near_zero": abs(stats["d_abl_boiler"]["mean"]) < 0.3,
        "P2_real_beats_boiler": (stats["d_abl_real"]["mean"]
                                 - stats["d_abl_boiler"]["mean"]) > 0.3,
        "P3_legacy_pays_boiler": stats["d_pos_boiler"]["mean"] > 0.3,
    }
    verdict["GATE_2_PASS"] = all(
        verdict[k] for k in ("P0_wiring_donor_is_zero", "P1_ceiling_oracle_gt_2",
                             "P2_boiler_near_zero", "P2_real_beats_boiler"))

    res = {"model": args.model, "n_scored": len(out), "n_skipped": skipped,
           "stats": stats, "verdict": verdict}
    with open(args.out, "w") as f:
        json.dump({**res, "rows": out}, f)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
