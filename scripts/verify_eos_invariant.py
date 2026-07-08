#!/usr/bin/env python
"""Post-save EOS/PAD invariant verifier for Qwen3 SFT checkpoints.

The #1 engineering risk for the Qwen3-8B-Base pipeline is a terminator mismatch:
the token the model is trained to emit at end-of-turn must equal the id that
(a) tokenizer.eos_token_id reports (drives verl response-mask + meta_info),
(b) the checkpoint generation_config.json / config.json report (drives vLLM stop),
(c) eval stops on — and it must NOT equal pad. See docs/redesign/SPEC.md §2.

Canonical:  eos = <|im_end|> = 151645 ,  pad = <|endoftext|> = 151643 (eos != pad).

Usage:  python scripts/verify_eos_invariant.py <checkpoint_dir>
Exit 0 = GREEN (safe to launch RL); non-zero = broken checkpoint, DO NOT launch.
"""
import sys

EOS_ID = 151645  # <|im_end|>
PAD_ID = 151643  # <|endoftext|>


def main(ckpt: str) -> int:
    from transformers import AutoTokenizer, AutoConfig, GenerationConfig

    tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(ckpt, trust_remote_code=True)
    try:
        gen = GenerationConfig.from_pretrained(ckpt)
    except Exception:
        gen = None

    def _first(x):
        return x[0] if isinstance(x, (list, tuple)) and x else x

    checks = []
    checks.append(("tokenizer.eos_token_id == 151645", tok.eos_token_id == EOS_ID, tok.eos_token_id))
    checks.append(("tokenizer.pad_token_id == 151643", tok.pad_token_id == PAD_ID, tok.pad_token_id))
    checks.append(("pad != eos", tok.pad_token_id != tok.eos_token_id, (tok.pad_token_id, tok.eos_token_id)))
    checks.append(("config.eos_token_id == 151645", _first(cfg.eos_token_id) == EOS_ID, cfg.eos_token_id))
    if gen is not None and gen.eos_token_id is not None:
        checks.append(("generation_config.eos == 151645", _first(gen.eos_token_id) == EOS_ID, gen.eos_token_id))

    # the SFT template must actually terminate an assistant turn with 151645
    try:
        ids = tok.apply_chat_template(
            [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "bye"}],
            tokenize=True, add_generation_prompt=False,
        )
        # ChatML renders '...<|im_end|>\n', so the terminator (151645) is
        # second-to-last, not last. Assert it sits in the final tokens.
        checks.append(("chat template terminates with 151645", 151645 in ids[-3:], ids[-5:]))
    except Exception as e:
        checks.append(("chat template renders", False, str(e)[:80]))

    ok = True
    for name, passed, got in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}  (got={got})")
        ok = ok and passed

    print("EOS invariant:", "GREEN" if ok else "RED — DO NOT LAUNCH RL")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python scripts/verify_eos_invariant.py <checkpoint_dir>")
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
