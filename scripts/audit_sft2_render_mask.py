#!/usr/bin/env python3
"""Render-layer token-id audit of an SFT2 corpus, replicating sft.py exactly.

WHY THIS EXISTS (E-076 / E-093 / E-094)
---------------------------------------
Char-level prefix checks passed on data that was still broken at the template
render layer (E-076: 4 rows). And ``sft.py`` silently truncates at
``max_length`` (sft.py:102-103) without checking that the trained target — the
meta block and the recovery's final answer — survives the cut. Both failures are
invisible unless you look at the FINAL TOKEN IDS.

So this script rebuilds, per row, exactly what ``sft.py:tokenize_row`` builds:

    prompt_ids = apply_chat_template(messages[:-1], add_generation_prompt=True)
    full_ids   = apply_chat_template(messages,      add_generation_prompt=False)
    full_ids   = full_ids[:max_length]                       # silent truncation
    prefix_len = len(encode(wrong_prefix, add_special_tokens=False))
    spans      = redirect_train_spans(prompt_len, prefix_len, len(full_ids))
    keep       = build_segment_loss_mask(len(full_ids), spans)

and then asks four questions the char-level checks cannot answer:

  G1 TRUNCATION   — was the row cut at max_length at all?
  G2 ANSWER-KEPT  — does the final ``\\boxed{...}`` land inside the TRAINED
                    region of the surviving tokens? If not, the row teaches a
                    recovery that never reaches an answer.
  G3 TRACE-SHAPE  — DESCRIPTIVE, NOT A GATE. Does the loss mask start exactly
                    AT the ``<|meta|>`` open tag, i.e. does the corpus match the
                    ``[prompt][wrong_prefix]<|meta|>...`` shape that
                    ``segment_loss_mask`` was designed around? A corpus can
                    score 0% here and still be perfectly trainable when the meta
                    block merely starts a few tokens later (that intervening
                    text is recovery narration and SHOULD be trained). Use G5,
                    not G3, to decide whether a corpus is usable.
  G5 META-TRAINED — THE GATE THAT MATTERS. Is the whole ``<|meta|>...<|/meta|>``
                    block inside the TRAINED region? If it falls inside the
                    masked wrong_prefix the row teaches the model nothing about
                    emitting meta at all, which silently starves the very
                    behaviour the reward later scores.
  G4 EOS          — does the EOS the template appends survive in the tail? (The
                    template emits ``<|im_end|>\n``, so EOS is second-to-last.)

Read-only. Writes no data. Exit code is 0 always; the verdict is in the report.

Usage:
  python scripts/audit_sft2_render_mask.py \
      --data data/rv_redirect_verify_functional.parquet \
      --tokenizer /path/to/b2p2_rvseg_sft \
      --max-length 4096
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.segment_loss_mask import (  # noqa: E402
    build_segment_loss_mask,
    redirect_train_spans,
)


def _last_boxed_span(text: str) -> tuple[int, int] | None:
    """Char span [start, end) of the LAST brace-balanced ``\\boxed{...}``."""
    best, i, n = None, 0, len(text)
    while i < n:
        if text.startswith("\\boxed", i):
            j = i + 6
            while j < n and text[j] in " \t":
                j += 1
            if j < n and text[j] == "{":
                depth, k = 0, j
                while k < n:
                    if text[k] == "{":
                        depth += 1
                    elif text[k] == "}":
                        depth -= 1
                        if depth == 0:
                            best = (i, k + 1)
                            break
                    k += 1
                i = k + 1
                continue
        i += 1
    return best


def audit(data_path: Path, tokenizer_dir: str, max_length: int) -> dict:
    import pandas as pd
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    meta_open_id = tok.convert_tokens_to_ids("<|meta|>")
    if meta_open_id is None or meta_open_id == tok.unk_token_id:
        raise SystemExit(
            "tokenizer has no <|meta|> token — wrong tokenizer for a meta SFT corpus"
        )
    df = pd.read_parquet(data_path)

    rows = []
    for _, r in df.iterrows():
        messages = json.loads(r["messages"]) if isinstance(r["messages"], str) else r["messages"]
        prompt_ids = tok.apply_chat_template(
            messages[:-1], tokenize=True, add_generation_prompt=True
        )
        full_ids_untrunc = tok.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False
        )
        prompt_len = len(prompt_ids)
        truncated = len(full_ids_untrunc) > max_length
        full_ids = full_ids_untrunc[:max_length]

        asst_probe = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "assistant"), ""
        )
        wrong_prefix = str(r.get("wrong_prefix", "") or "")
        if wrong_prefix:
            prefix_len = len(tok.encode(wrong_prefix, add_special_tokens=False))
            spans = redirect_train_spans(prompt_len, prefix_len, len(full_ids))
            keep = build_segment_loss_mask(len(full_ids), spans)
        else:
            prefix_len = 0
            keep = [0] * min(prompt_len, len(full_ids)) + [1] * max(
                0, len(full_ids) - prompt_len
            )

        # --- G3: does the mask boundary land exactly where training must start? ---
        # DIRECT CONTRACT CHECK. An earlier version of this audit compared
        # decoded string LENGTHS to locate the prefix end; that metric is
        # unreliable (the tokenizer's decode normalizes leading spaces, so the
        # comparison reports a phantom 1-token drift) and it nearly produced a
        # false bug report on 0726. What actually matters is unambiguous:
        # training must begin AT the ``<|meta|>`` open tag — the prefix is
        # masked, the meta block is learned. So check the token sitting at the
        # boundary index directly.
        align_ok = None
        boundary_tok = None
        if wrong_prefix:
            b = prompt_len + prefix_len
            if 0 <= b < len(full_ids):
                boundary_tok = tok.convert_ids_to_tokens([full_ids[b]])[0]
                align_ok = full_ids[b] == meta_open_id
            else:
                align_ok = False

        # --- G2: is the final answer inside the TRAINED surviving region? ---
        asst = asst_probe
        box = _last_boxed_span(asst)
        answer_kept = None
        if box is not None:
            # Find the token index where the boxed answer ends, by growing the
            # assistant decode until it contains the boxed substring.
            needle = asst[box[0]:box[1]]
            answer_kept = False
            for t in range(prompt_len, len(full_ids) + 1):
                dec = tok.decode(full_ids[prompt_len:t], skip_special_tokens=False)
                if needle in dec:
                    # token t-1 is the last token of the answer; is it trained?
                    answer_kept = bool(keep[t - 1]) if 0 < t <= len(full_ids) else False
                    break

        # --- G5: is the meta block inside the trained region? ---
        meta_trained = None
        if "<|meta|>" in asst_probe:
            mo = asst_probe.find("<|meta|>")
            mc = asst_probe.find("<|/meta|>")
            cut = int(r.get("prefix_split_char", 0) or 0)
            meta_trained = bool(mo >= cut and mc >= cut)

        n_target = sum(1 for k in keep if k == 1)
        rows.append(
            {
                "scenario": r.get("scenario", "?"),
                "n_tokens_untrunc": len(full_ids_untrunc),
                "truncated": truncated,
                "n_target_tokens": n_target,
                "align_ok": align_ok,
                "boundary_tok": boundary_tok,
                "answer_kept": answer_kept,
                "meta_trained": meta_trained,
                "has_boxed": box is not None,
                # The chat template emits ``... <|im_end|>\n``, so EOS is the
                # SECOND-to-last token, not the last. Checking ``[-1] == eos``
                # reports 0% on a perfectly terminated corpus (measurement bug
                # found 0726) — look in the tail window instead.
                "ends_with_eos": bool(
                    full_ids and tok.eos_token_id in full_ids[-3:]
                ),
            }
        )

    out = defaultdict(dict)
    import pandas as pd

    a = pd.DataFrame(rows)
    for sc, g in a.groupby("scenario"):
        out[sc] = {
            "n": int(len(g)),
            "G1_truncated_pct": round(g["truncated"].mean() * 100, 1),
            "G1_tokens_p50": int(g["n_tokens_untrunc"].median()),
            "G1_tokens_p95": int(g["n_tokens_untrunc"].quantile(0.95)),
            "G1_tokens_max": int(g["n_tokens_untrunc"].max()),
            "G2_has_boxed_pct": round(g["has_boxed"].mean() * 100, 1),
            "G2_answer_kept_pct": round(
                g.loc[g["has_boxed"], "answer_kept"].mean() * 100, 1
            ),
            "G3_boundary_is_meta_pct": round(
                g.loc[g["align_ok"].notna(), "align_ok"].mean() * 100, 1
            ),
            "G3_boundary_tok_top": (
                g.loc[g["align_ok"] == False, "boundary_tok"].value_counts().head(4).to_dict()
                if (g["align_ok"] == False).any() else {}
            ),
            "G5_meta_trained_pct": round(
                g.loc[g["meta_trained"].notna(), "meta_trained"].mean() * 100, 1
            ),
            "G5_meta_masked_away_n": int(
                (g["meta_trained"] == False).sum()
            ),
            "G4_eos_pct": round(g["ends_with_eos"].mean() * 100, 1),
            "n_target_tokens_p50": int(g["n_target_tokens"].median()),
        }
    return dict(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    report = audit(Path(args.data), args.tokenizer, args.max_length)
    print(f"=== render-layer token-id audit: {args.data} (max_length={args.max_length}) ===")
    for sc, m in report.items():
        print(f"\n[{sc}]  n={m['n']}")
        print(
            f"  G1 truncation  {m['G1_truncated_pct']:5.1f}%   "
            f"tokens p50 {m['G1_tokens_p50']} / p95 {m['G1_tokens_p95']} / max {m['G1_tokens_max']}"
        )
        print(
            f"  G2 answer kept {m['G2_answer_kept_pct']:5.1f}%  (has_boxed {m['G2_has_boxed_pct']}%)"
        )
        print(
            f"  G3 mask boundary == <|meta|>  {m['G3_boundary_is_meta_pct']:5.1f}%"
        )
        if m["G3_boundary_tok_top"]:
            print(f"     when not: {m['G3_boundary_tok_top']}")
        print(
            f"  G5 meta block trained {m['G5_meta_trained_pct']:5.1f}%   "
            f"(masked away in {m['G5_meta_masked_away_n']} rows)  <-- THE GATE"
        )
        print(f"  G4 ends with EOS {m['G4_eos_pct']:5.1f}%   target tokens p50 {m['n_target_tokens_p50']}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=1))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
