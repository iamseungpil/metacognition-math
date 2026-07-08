#!/usr/bin/env python
"""Linear weight interpolation between two same-architecture HF causal LMs.

theta(alpha) = (1 - alpha) * theta_a + alpha * theta_b, saved as an HF model dir.

Used to test the "RL-induced degeneration is a weight-space perturbation" hypothesis:
interpolate the RL checkpoint (model_b, alpha=1) back toward its pre-RL SFT init
(model_a, alpha=0) and check whether intermediate alpha restores clean termination
while keeping accuracy / meta behavior.
"""
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_a", required=True, help="alpha=0 endpoint (e.g. pre-RL SFT init)")
    ap.add_argument("--model_b", required=True, help="alpha=1 endpoint (e.g. RL checkpoint)")
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    a = args.alpha

    print(f"[soup] loading A (alpha=0) = {args.model_a}", flush=True)
    ma = AutoModelForCausalLM.from_pretrained(args.model_a, torch_dtype=torch.bfloat16)
    print(f"[soup] loading B (alpha=1) = {args.model_b}", flush=True)
    mb = AutoModelForCausalLM.from_pretrained(args.model_b, torch_dtype=torch.bfloat16)

    sa = ma.state_dict()
    sb = mb.state_dict()
    if set(sa.keys()) != set(sb.keys()):
        only_a = set(sa) - set(sb)
        only_b = set(sb) - set(sa)
        raise ValueError(f"state-dict key mismatch; only_a={list(only_a)[:5]} only_b={list(only_b)[:5]}")
    for k in sa:
        if sa[k].shape != sb[k].shape:
            raise ValueError(f"shape mismatch at {k}: {sa[k].shape} vs {sb[k].shape}")
        sa[k] = ((1.0 - a) * sa[k].float() + a * sb[k].float()).to(torch.bfloat16)

    ma.load_state_dict(sa)
    ma.save_pretrained(args.out)
    AutoTokenizer.from_pretrained(args.model_b).save_pretrained(args.out)
    print(f"[soup] saved alpha={a} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
