"""Runtime tokenizer shims for local checkpoints with fragile tokenizer configs."""
from __future__ import annotations

import json
import shutil
from pathlib import Path


def prepare_runtime_tokenizer_dir(model_path: str, output_dir: str) -> tuple[str | None, str]:
    """Create a sanitized tokenizer shim for local checkpoints when needed.

    Some locally saved checkpoints carry `extra_special_tokens` as a list in
    `tokenizer_config.json`, which breaks newer tokenizer init paths in some
    environments. We do not mutate the original checkpoint; instead we write a
    runtime-only tokenizer directory under the provided output dir.
    """
    model_dir = Path(model_path)
    tokenizer_config = model_dir / "tokenizer_config.json"
    if not tokenizer_config.exists():
        return None, "auto"

    payload = json.loads(tokenizer_config.read_text(encoding="utf-8"))
    extra_special = payload.get("extra_special_tokens")
    if not isinstance(extra_special, list):
        return None, "auto"

    shim_dir = Path(output_dir) / "_runtime_tokenizer"
    shim_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "chat_template.jinja",
    ]:
        src = model_dir / name
        if src.exists():
            shutil.copy2(src, shim_dir / name)

    payload.pop("extra_special_tokens", None)
    (shim_dir / "tokenizer_config.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return str(shim_dir), "hf"


__all__ = ["prepare_runtime_tokenizer_dir"]
