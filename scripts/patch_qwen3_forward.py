"""Patch gnosis-modified Qwen3 forward() to skip Gnosis head during generate().

The gnosis-modified Qwen3ForCausalLM.forward() raises ValueError when
correctness_label is None during training mode. TRL's GRPOTrainer calls
model.generate() in training mode, which triggers this error.

This patch replaces the 'raise ValueError' with an early return of
normal CausalLMOutputWithPast when correctness_label is None.
"""
import sys
import os


def patch_qwen3_forward(transformers_path):
    """Patch the installed Qwen3 modeling file."""
    target = os.path.join(transformers_path, "models", "qwen3", "modeling_qwen3.py")

    if not os.path.exists(target):
        print(f"ERROR: {target} not found")
        return False

    with open(target) as f:
        lines = f.readlines()

    # Check if already patched
    for line in lines:
        if "Patched: skip Gnosis head during generate" in line:
            print("Qwen3 forward already patched")
            return True

    # Find the pattern:
    #   if correctness_label is None:
    #       raise ValueError(...)
    patched = False
    for i, line in enumerate(lines):
        if "if correctness_label is None:" in line.strip():
            # Check next line is raise ValueError
            if i + 1 < len(lines) and "raise ValueError" in lines[i + 1]:
                # Replace the raise with early return
                indent = "                "
                lines[i + 1] = (
                    f"{indent}# Patched: skip Gnosis head during generate()\n"
                    f"{indent}return CausalLMOutputWithPast(\n"
                    f"{indent}    loss=None, logits=logits,\n"
                    f"{indent}    past_key_values=outputs.past_key_values,\n"
                    f"{indent}    hidden_states=outputs.hidden_states,\n"
                    f"{indent}    attentions=outputs.attentions,\n"
                    f"{indent})\n"
                )
                patched = True
                print(f"Patched line {i + 2}: raise ValueError -> early return")
                break

    if patched:
        with open(target, "w") as f:
            f.writelines(lines)
        print(f"Qwen3 forward patched successfully: {target}")
        return True
    else:
        print("WARNING: Could not find patch target")
        # Print lines around 'correctness_label' for debugging
        for i, line in enumerate(lines):
            if "correctness_label" in line:
                print(f"  Line {i + 1}: {line.rstrip()}")
        return False


if __name__ == "__main__":
    import transformers
    path = os.path.dirname(transformers.__file__)
    print(f"Transformers at: {path}")
    patch_qwen3_forward(path)
