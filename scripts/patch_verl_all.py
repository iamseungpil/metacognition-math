"""Apply ALL verl 0.7.1 compatibility patches in one clean script.

Handles:
1. fsdp_utils.py — DTensorSpec fallback (torch<2.6)
2. vllm_rollout/utils.py — get_device_uuid (NvmlCudaPlatform missing method)
3. vllm_async_server.py — make vllm.entrypoints.cli optional
4. ppo_trainer.yaml — bump update_weights_bucket_megabytes

Must be run ONCE after fresh verl install. Idempotent check via marker string.
"""
import ast
import os
import sys

VENV_PREFIX = os.environ.get("VENV_PREFIX", "/scratch/simplerl_venv")
VERL = os.path.join(VENV_PREFIX, "lib/python3.10/site-packages/verl")
MARKER = "# META_COT_PATCHED"


def patch_fsdp():
    """Add module-level DTensorSpec fallback."""
    fp = os.path.join(VERL, "utils/fsdp_utils.py")
    with open(fp) as f:
        content = f.read()
    if MARKER in content:
        print("fsdp_utils.py: already patched")
        return
    # Add fallback at top of file (before any other import)
    patch = f"{MARKER}\nfrom typing import Any as _DTensorSpecFallback\nDTensorSpec = _DTensorSpecFallback\n\n"
    # Insert after the module docstring / first import
    lines = content.split("\n")
    insert_idx = 0
    for i, line in enumerate(lines):
        if line.startswith("import ") or line.startswith("from "):
            insert_idx = i
            break
    lines.insert(insert_idx, patch.rstrip())
    content = "\n".join(lines)
    with open(fp, "w") as f:
        f.write(content)
    ast.parse(content)
    print("fsdp_utils.py: PATCHED")


def patch_device_uuid():
    """Replace get_device_uuid with nvidia-smi fallback."""
    fp = os.path.join(VERL, "workers/rollout/vllm_rollout/utils.py")
    with open(fp) as f:
        lines = f.readlines()
    if any(MARKER in line for line in lines):
        print("utils.py: already patched")
        return
    new_lines = []
    skip = False
    for line in lines:
        if "def get_device_uuid" in line:
            skip = True
            new_lines.append(f"{MARKER}\n")
            new_lines.append("def get_device_uuid(device_id: int) -> str:\n")
            new_lines.append("    try:\n")
            new_lines.append("        from vllm.platforms import current_platform\n")
            new_lines.append("        return current_platform.get_device_uuid(device_id)\n")
            new_lines.append("    except (AttributeError, ImportError):\n")
            new_lines.append("        import subprocess as _sp\n")
            new_lines.append("        _r = _sp.run(\n")
            new_lines.append('            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],\n')
            new_lines.append("            capture_output=True, text=True,\n")
            new_lines.append("        )\n")
            new_lines.append("        _uuids = _r.stdout.strip().splitlines()\n")
            new_lines.append('        return _uuids[device_id] if device_id < len(_uuids) else "GPU-" + str(device_id)\n')
            new_lines.append("\n")
            continue
        if skip:
            if line.strip() and not line[0].isspace():
                skip = False
                new_lines.append(line)
            continue
        new_lines.append(line)
    with open(fp, "w") as f:
        f.writelines(new_lines)
    ast.parse("".join(new_lines))
    print("utils.py: PATCHED")


def patch_async_server():
    """Make vllm.entrypoints.cli import optional."""
    fp = os.path.join(VERL, "workers/rollout/vllm_rollout/vllm_async_server.py")
    with open(fp) as f:
        content = f.read()
    if MARKER in content:
        print("vllm_async_server.py: already patched")
        return
    # Replace the block of imports
    old = (
        "import vllm.entrypoints.cli.serve\n"
        "from packaging import version\n"
        "from ray.actor import ActorHandle\n"
        "from vllm import SamplingParams\n"
        "from vllm.engine.arg_utils import AsyncEngineArgs\n"
        "from vllm.entrypoints.cli.serve import run_headless\n"
        "from vllm.entrypoints.openai.api_server import build_app, init_app_state\n"
        "from vllm.inputs import TokensPrompt\n"
        "from vllm.lora.request import LoRARequest\n"
        "from vllm.outputs import RequestOutput\n"
        "from vllm.usage.usage_lib import UsageContext\n"
        "from vllm.v1.engine.async_llm import AsyncLLM\n"
    )
    new = (
        f"{MARKER}\n"
        "import vllm  # ensure top-level vllm is importable (for vllm.__version__)\n"
        "from packaging import version\n"
        "from ray.actor import ActorHandle\n"
        "from vllm import SamplingParams\n"
        "from vllm.engine.arg_utils import AsyncEngineArgs\n"
        "from vllm.inputs import TokensPrompt\n"
        "from vllm.lora.request import LoRARequest\n"
        "from vllm.outputs import RequestOutput\n"
        "from vllm.usage.usage_lib import UsageContext\n"
        "try:\n"
        "    import vllm.entrypoints.cli.serve  # noqa: F401\n"
        "    from vllm.entrypoints.cli.serve import run_headless\n"
        "    from vllm.entrypoints.openai.api_server import build_app, init_app_state\n"
        "    from vllm.v1.engine.async_llm import AsyncLLM\n"
        "except (ImportError, ModuleNotFoundError):\n"
        "    run_headless = None\n"
        "    build_app = None\n"
        "    init_app_state = None\n"
        "    AsyncLLM = None\n"
    )
    if old not in content:
        print("vllm_async_server.py: exact import block not found — file may differ from expected")
        return
    content = content.replace(old, new)
    with open(fp, "w") as f:
        f.write(content)
    ast.parse(content)
    print("vllm_async_server.py: PATCHED")


def patch_yaml():
    """Bump update_weights_bucket_megabytes."""
    fp = os.path.join(VERL, "trainer/config/ppo_trainer.yaml")
    if not os.path.exists(fp):
        print("ppo_trainer.yaml: not found")
        return
    with open(fp) as f:
        content = f.read()
    if "update_weights_bucket_megabytes: 2048" in content:
        content = content.replace(
            "update_weights_bucket_megabytes: 2048",
            "update_weights_bucket_megabytes: 4096",
        )
        with open(fp, "w") as f:
            f.write(content)
        print("ppo_trainer.yaml: bucket 2048->4096")
    else:
        print("ppo_trainer.yaml: bucket already ok")


def main():
    patch_fsdp()
    patch_device_uuid()
    patch_async_server()
    patch_yaml()
    # Final import test
    import subprocess
    result = subprocess.run(
        [f"{VENV_PREFIX}/bin/python", "-c",
         "from verl.trainer.ppo.ray_trainer import RayPPOTrainer; "
         "from verl.trainer.ppo.core_algos import AdvantageEstimator; "
         "print('IMPORT_OK', 'GDPO' in [e.name for e in AdvantageEstimator])"],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print("IMPORT_ERROR:", result.stderr[-1000:])
        sys.exit(1)
    print("ALL_PATCHES_APPLIED")


if __name__ == "__main__":
    main()
