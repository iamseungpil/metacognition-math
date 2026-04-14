"""Patch verl 0.7.1 for torch 2.5.1 + vllm 0.7.1 compatibility."""
import ast
import os

VENV = "/scratch/simplerl_venv/lib/python3.10/site-packages"

# Patch 1: fsdp_utils.py - DTensorSpec import
fsdp = os.path.join(VENV, "verl/utils/fsdp_utils.py")
if os.path.exists(fsdp):
    with open(fsdp) as f:
        content = f.read()
    # The original import is inside an if block (4-space indent)
    old = "    from torch.distributed.tensor._dtensor_spec import DTensorSpec"
    new = """    try:
        from torch.distributed.tensor._dtensor_spec import DTensorSpec
    except (ImportError, ModuleNotFoundError):
        from typing import Any as DTensorSpec"""
    if old in content:
        content = content.replace(old, new)
        with open(fsdp, "w") as f:
            f.write(content)
        print("Patched fsdp_utils.py: DTensorSpec")
    else:
        print("fsdp_utils.py: pattern not found, trying alternative")
        old2 = "from torch.distributed.tensor._dtensor_spec import DTensorSpec"
        new2 = """try:
    from torch.distributed.tensor._dtensor_spec import DTensorSpec
except (ImportError, ModuleNotFoundError):
    from typing import Any as DTensorSpec"""
        if old2 in content:
            content = content.replace(old2, new2, 1)
            with open(fsdp, "w") as f:
                f.write(content)
            print("Patched fsdp_utils.py (module-level): DTensorSpec")

# Patch 2: utils.py - get_device_uuid
utils = os.path.join(VENV, "verl/workers/rollout/vllm_rollout/utils.py")
if os.path.exists(utils):
    with open(utils) as f:
        lines = f.readlines()

    new_lines = []
    skip = False
    for line in lines:
        if "def get_device_uuid" in line:
            skip = True
            indent = line[:len(line) - len(line.lstrip())]
            new_lines.append(indent + "def get_device_uuid(device_id: int) -> str:\n")
            new_lines.append(indent + "    try:\n")
            new_lines.append(indent + "        return current_platform.get_device_uuid(device_id)\n")
            new_lines.append(indent + "    except AttributeError:\n")
            new_lines.append(indent + "        import subprocess as _sp\n")
            new_lines.append(indent + "        _r = _sp.run(\n")
            new_lines.append(indent + '            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],\n')
            new_lines.append(indent + "            capture_output=True, text=True,\n")
            new_lines.append(indent + "        )\n")
            new_lines.append(indent + "        _uuids = _r.stdout.strip().splitlines()\n")
            new_lines.append(indent + '        return _uuids[device_id] if device_id < len(_uuids) else "GPU-" + str(device_id)\n')
            new_lines.append("\n")
            continue
        if skip:
            if line.strip() and not line[0].isspace():
                skip = False
                new_lines.append(line)
            continue
        new_lines.append(line)

    with open(utils, "w") as f:
        f.writelines(new_lines)
    print("Patched utils.py: get_device_uuid")

# Patch 3: rollout yaml bucket size
yaml_path = os.path.join(VENV, "verl/trainer/config/ppo_trainer.yaml")
if os.path.exists(yaml_path):
    with open(yaml_path) as f:
        content = f.read()
    if "update_weights_bucket_megabytes: 2048" in content:
        content = content.replace(
            "update_weights_bucket_megabytes: 2048",
            "update_weights_bucket_megabytes: 4096",
        )
        with open(yaml_path, "w") as f:
            f.write(content)
        print("Patched ppo_trainer.yaml: bucket 4096")

# Verify syntax of patched files
for fpath in [fsdp, utils]:
    if os.path.exists(fpath):
        try:
            ast.parse(open(fpath).read())
            print(f"SYNTAX_OK: {os.path.basename(fpath)}")
        except SyntaxError as e:
            print(f"SYNTAX_ERROR in {fpath}: {e}")

print("ALL_PATCHES_DONE")
