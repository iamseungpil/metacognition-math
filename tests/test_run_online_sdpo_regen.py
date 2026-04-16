from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.self_distill.runtime_tokenizer import prepare_runtime_tokenizer_dir


def test_prepare_runtime_tokenizer_dir_sanitizes_list_extra_special_tokens(tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model_dir / "chat_template.jinja").write_text("{{ bos_token }}", encoding="utf-8")
    (model_dir / "tokenizer_config.json").write_text(
        json.dumps({
            "tokenizer_class": "Qwen2TokenizerFast",
            "extra_special_tokens": ["<|im_start|>", "<|im_end|>"],
        }),
        encoding="utf-8",
    )

    tokenizer_path, tokenizer_mode = prepare_runtime_tokenizer_dir(str(model_dir), str(tmp_path / "out"))
    assert tokenizer_mode == "hf"
    assert tokenizer_path is not None
    shim = Path(tokenizer_path)
    cfg = json.loads((shim / "tokenizer_config.json").read_text(encoding="utf-8"))
    assert "extra_special_tokens" not in cfg
    assert (shim / "tokenizer.json").exists()
    assert (shim / "chat_template.jinja").exists()


def test_prepare_runtime_tokenizer_dir_noop_when_not_needed(tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "Qwen2TokenizerFast", "extra_special_tokens": {"foo": "<foo>"}}),
        encoding="utf-8",
    )
    tokenizer_path, tokenizer_mode = prepare_runtime_tokenizer_dir(str(model_dir), str(tmp_path / "out"))
    assert tokenizer_path is None
    assert tokenizer_mode == "auto"
