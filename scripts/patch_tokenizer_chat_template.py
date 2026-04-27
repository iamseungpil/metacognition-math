#!/usr/bin/env python
"""Add a passthrough chat_template to a model's tokenizer_config.json.

Required when verl's rl_dataset.py applies chat_template but the model was
trained on raw text format (e.g. v8_meta_inside_strict_sft with <|meta|> tokens).
If tokenizer_config.json is missing entirely (some SFT runs upload only
tokenizer.json), this script creates a minimal config so the tokenizer can
load AND the chat_template hook works.
"""
import json
import sys
from pathlib import Path

PASSTHROUGH = "{% for m in messages %}{{ m['content'] }}{% endfor %}"


def main(model_dir: str) -> None:
    p = Path(model_dir) / "tokenizer_config.json"
    if not p.exists():
        # Minimal config that lets transformers load via tokenizer.json (fast).
        # Qwen3-8B uses Qwen2Tokenizer (Qwen3 reuses Qwen2's BPE).
        cfg = {
            "tokenizer_class": "PreTrainedTokenizerFast",
            "model_max_length": 32768,
            "padding_side": "left",
            "clean_up_tokenization_spaces": False,
            "bos_token": None,
            "eos_token": "<|endoftext|>",
            "pad_token": "<|endoftext|>",
            "chat_template": PASSTHROUGH,
        }
        with p.open("w") as f:
            json.dump(cfg, f, indent=2)
        print(f"[patch] created minimal tokenizer_config.json with chat_template at {p}")
        return
    with p.open() as f:
        cfg = json.load(f)
    existing = cfg.get("chat_template")
    changed = False
    if not existing:
        cfg["chat_template"] = PASSTHROUGH
        changed = True
    # Force fast tokenizer: model dir only ships tokenizer.json (no vocab.json/merges.txt),
    # so the slow Qwen2Tokenizer fallback fails with "vocab_file=None".
    if cfg.get("tokenizer_class") not in ("PreTrainedTokenizerFast", None):
        cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
        changed = True
    if changed:
        with p.open("w") as f:
            json.dump(cfg, f, indent=2)
        print(f"[patch] updated tokenizer_config (chat_template={'set' if not existing else 'kept'}, class=PreTrainedTokenizerFast)")
    else:
        print(f"[patch] tokenizer_config already correct (length {len(existing)} chars), skipping")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: patch_tokenizer_chat_template.py <model_dir>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
