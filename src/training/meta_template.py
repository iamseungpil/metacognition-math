"""Rebuild a variable-length meta block into a short FIXED-ORDER template so the
closing position becomes predictable (spec 2026-06-15-s3b §3.1a). Keeps only the
three most common labels in a fixed order; each value is single-line, trimmed; the
whole block is char-capped. Labels absent in the source are skipped (never fabricated)."""
import re

_TEMPLATE_LABELS = ("confidence", "assessment", "action")  # observed coverage 0.89/0.65/0.47


def rebuild_meta_block(raw_body: str, max_chars: int = 320) -> str:
    lines = {}
    for lab in _TEMPLATE_LABELS:
        m = re.search(rf"(?im)^\s*{lab}\s*:\s*(.+)$", raw_body or "")
        if m:
            val = re.sub(r"\s+", " ", m.group(1)).strip()
            if val:
                lines[lab] = val
    out = "\n".join(f"{lab}: {lines[lab]}" for lab in _TEMPLATE_LABELS if lab in lines)
    return out[:max_chars].rstrip()
