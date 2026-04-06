"""Regression test for the TRL vLLM import stub."""
import ast
import importlib
import importlib.machinery
import importlib.util
import sys
import types


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


source = open("src/training/grpo_v2.py").read()
module = ast.parse(source)
fn_node = None
for node in module.body:
    if isinstance(node, ast.FunctionDef) and node.name == "_ensure_vllm_stub":
        fn_node = node
        break

if fn_node is None:
    raise RuntimeError("_ensure_vllm_stub not found")

fn_src = ast.get_source_segment(source, fn_node)
namespace = {
    "sys": sys,
    "types": types,
    "importlib": importlib,
}
exec(fn_src, namespace)

for name in list(sys.modules):
    if name == "vllm" or name.startswith("vllm.") or name == "vllm_ascend" or name.startswith("vllm_ascend."):
        sys.modules.pop(name, None)

namespace["_ensure_vllm_stub"]()
sampling_mod = importlib.import_module("vllm.sampling_params")
check("sampling params stub import works", hasattr(sampling_mod, "GuidedDecodingParams"))
check("vllm parent is package-like", hasattr(sys.modules["vllm"], "__path__"))
check("distributed utils stub import works", importlib.import_module("vllm.distributed.utils") is not None)

print(f"\n=== SUMMARY: {passed} passed, {failed} failed ===")


def test_pytest_bridge():
    assert failed == 0


if __name__ == "__main__":
    if failed:
        sys.exit(1)
