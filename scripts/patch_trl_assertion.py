"""Patch gnosis_repo TRL: replace frozen-param assertion with auto-unfreeze.

PEFT wrapping freezes Gnosis head params. The assertion at line 303 of
grpo_trainer.py catches this and errors. This patch replaces the assert
with auto-unfreeze logic.
"""
import os

TARGET = "/scratch/metacognition/gnosis_repo/trl/trl/trainer/grpo_trainer.py"


def patch_trl_assertion():
    if not os.path.exists(TARGET):
        print(f"ERROR: {TARGET} not found")
        return False

    with open(TARGET) as f:
        lines = f.readlines()

    # Check if already patched
    for line in lines:
        if "Auto-unfreezing" in line or "auto-unfreeze" in line.lower():
            print("TRL assertion already patched")
            return True

    # Find: assert not bad, f"Correctness head accidentally frozen: ...
    patched = False
    for i, line in enumerate(lines):
        if "assert not bad" in line and "Correctness head accidentally frozen" in line:
            indent = "        "
            lines[i] = (
                f"{indent}if bad:\n"
                f"{indent}    print(f'[WARN] Auto-unfreezing {{len(bad)}} Gnosis params')\n"
                f"{indent}    for n_, p_ in model.named_parameters():\n"
                f"{indent}        if _trainable_correctness_param(n_): p_.requires_grad_(True)\n"
            )
            patched = True
            print(f"Patched line {i + 1}: assert -> auto-unfreeze")
            break

    if patched:
        with open(TARGET, "w") as f:
            f.writelines(lines)
        print("TRL assertion patched successfully")
        return True
    else:
        print("WARNING: Could not find TRL assertion to patch")
        for i, line in enumerate(lines):
            if "bad" in line and "frozen" in line.lower():
                print(f"  Line {i + 1}: {line.rstrip()}")
        return False


if __name__ == "__main__":
    patch_trl_assertion()
