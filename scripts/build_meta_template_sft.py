"""Rewrite v8_meta_inside_think assistant responses: replace each <|meta|>…<|/meta|>
body with the fixed short template (meta_template.rebuild_meta_block). Everything
outside the meta block is byte-identical. Output a new SFT parquet."""
import json, re, argparse, pandas as pd
from src.training.meta_template import rebuild_meta_block

_META = re.compile(r"<\|meta\|>(.*?)<\|/meta\|>", re.S)


_PROSE_LABELS = ("assessment", "action")


def _rewrite(content, max_chars):
    def repl(m):
        body = rebuild_meta_block(m.group(1), max_chars=max_chars)
        # Drop blocks that carry no metacognitive PROSE (assessment/action):
        #  - fully empty blocks -> degenerate <|meta|><|/meta|> (no form to prime)
        #  - confidence-only numeric stubs -> a different, near-zero-length form
        #    that bimodalizes the block-length distribution (18 vs ~148 chars) and
        #    blows up the close-position CV. The confidence number is handled by the
        #    PMI/calibration channel, not by the meta-FORM priming we do here.
        # Keeping only prose-bearing blocks makes the close position predictable.
        if not body.strip() or not any(
                re.search(rf"(?im)^{lab}:", body) for lab in _PROSE_LABELS):
            return ""
        return f"<|meta|>\n{body}\n<|/meta|>"
    return _META.sub(repl, content)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", default="data/v8_meta_inside_think.parquet")
    ap.add_argument("--out_path", default="data/v8_meta_template_sft.parquet")
    ap.add_argument("--max_chars", type=int, default=320)
    a = ap.parse_args()
    df = pd.read_parquet(a.in_path); rows = []
    for _, r in df.iterrows():
        msgs = json.loads(r["messages"]) if isinstance(r["messages"], str) else r["messages"]
        for x in msgs:
            if isinstance(x, dict) and x.get("role") == "assistant":
                x["content"] = _rewrite(x.get("content", ""), a.max_chars)
        rr = dict(r); rr["messages"] = json.dumps(msgs); rows.append(rr)
    out = pd.DataFrame(rows); out.to_parquet(a.out_path, index=False)
    print(f"wrote {len(out)} rows -> {a.out_path}")


if __name__ == "__main__":
    main()
