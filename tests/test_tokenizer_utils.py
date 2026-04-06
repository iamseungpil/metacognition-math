"""Smoke tests for tokenizer compatibility helpers."""
import sys

sys.path.insert(0, ".")

from src.training.tokenizer_utils import ensure_meta_tokens_not_special


passed = 0
failed = 0


def check(name, condition):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        print(f"  FAIL: {name}")


class ModernTokenizer:
    def __init__(self):
        self.vocab = {"a": 0, "<|meta|>": 1, "<|/meta|>": 2}
        self.special_tokens_map = {
            "additional_special_tokens": ["<|meta|>", "<|/meta|>", "<extra>"]
        }
        self.additional_special_tokens = list(self.special_tokens_map["additional_special_tokens"])
        self.calls = []

    def get_vocab(self):
        return self.vocab

    def add_tokens(self, tokens):
        for token in tokens:
            self.vocab[token] = len(self.vocab)

    def add_special_tokens(self, payload, replace_additional_special_tokens=False):
        self.calls.append((payload, replace_additional_special_tokens))
        if replace_additional_special_tokens:
            new_tokens = list(payload.get("additional_special_tokens", []))
            self.special_tokens_map["additional_special_tokens"] = new_tokens
            self.additional_special_tokens = list(new_tokens)


class LegacyTokenizer:
    def __init__(self):
        self.vocab = {"a": 0, "<|meta|>": 1, "<extra>": 2}
        self.special_tokens_map = {
            "additional_special_tokens": ["<|meta|>", "<extra>"]
        }
        self.calls = []

    def get_vocab(self):
        return self.vocab

    def add_tokens(self, tokens):
        for token in tokens:
            self.vocab[token] = len(self.vocab)

    def add_special_tokens(self, payload):
        self.calls.append(payload)
        self.special_tokens_map["additional_special_tokens"] = list(
            payload.get("additional_special_tokens", [])
        )


tok = ModernTokenizer()
ensure_meta_tokens_not_special(tok, ["<|meta|>", "<|/meta|>"])
check("modern: meta start removed from special list", "<|meta|>" not in tok.special_tokens_map["additional_special_tokens"])
check("modern: meta end removed from special list", "<|/meta|>" not in tok.special_tokens_map["additional_special_tokens"])
check("modern: non-meta special token preserved", "<extra>" in tok.special_tokens_map["additional_special_tokens"])
check("modern: helper uses replacement kwarg when available", bool(tok.calls) and tok.calls[-1][1] is True)

legacy = LegacyTokenizer()
ensure_meta_tokens_not_special(legacy, ["<|meta|>", "<|/meta|>"])
check("legacy: meta start removed without replacement kwarg", "<|meta|>" not in legacy.special_tokens_map["additional_special_tokens"])
check("legacy: existing non-meta special token preserved", "<extra>" in legacy.special_tokens_map["additional_special_tokens"])
check("legacy: missing meta token added to vocab", "<|/meta|>" in legacy.get_vocab())
check("legacy: helper does not require additional_special_tokens attribute", len(legacy.calls) == 1)

print(f"\n=== SUMMARY: {passed} passed, {failed} failed ===")


def test_pytest_bridge():
    assert failed == 0


if __name__ == "__main__":
    if failed:
        sys.exit(1)
