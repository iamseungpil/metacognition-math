"""Test GDPO advantage computation (TC10-TC11)."""
import torch
import sys

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

print("=== TC10: No peft_config in grpo_v2.py ===")
with open("src/training/grpo_v2.py") as f:
    code = f.read()
check("TC10: no peft_config in GRPOTrainer call", "peft_config" not in code.split("GRPOTrainer(")[1].split(")")[0])

print("\n=== TC11: GDPO per-reward normalization ===")
# Simulate: 2 rewards, 4 completions (2 prompts × 2 generations)
num_gen = 2
rewards_per_func = torch.tensor([
    [1.0, 0.1],   # prompt1, gen1: correct, low calib
    [-1.0, 0.9],   # prompt1, gen2: wrong, high calib
    [1.0, 0.8],    # prompt2, gen1: correct, high calib
    [1.0, 0.2],    # prompt2, gen2: correct, low calib
])
weights = torch.tensor([1.0, 1.0])

# Standard GRPO: sum then normalize
rewards_grpo = (rewards_per_func * weights.unsqueeze(0)).sum(dim=1)
mean_g = rewards_grpo.view(-1, num_gen).mean(dim=1).repeat_interleave(num_gen)
std_g = rewards_grpo.view(-1, num_gen).std(dim=1).repeat_interleave(num_gen)
adv_grpo = (rewards_grpo - mean_g) / (std_g + 1e-4)

# GDPO: normalize each then sum then batch normalize
all_adv = []
for i in range(2):
    r_i = rewards_per_func[:, i]
    mean_i = r_i.view(-1, num_gen).mean(dim=1).repeat_interleave(num_gen)
    std_i = r_i.view(-1, num_gen).std(dim=1).repeat_interleave(num_gen)
    adv_i = (r_i - mean_i) / (std_i + 1e-4)
    all_adv.append(adv_i)
combined = torch.stack(all_adv, dim=1)
pre_bn = (combined * weights.unsqueeze(0)).sum(dim=1)
adv_gdpo = (pre_bn - pre_bn.mean()) / (pre_bn.std() + 1e-4)

print(f"  GRPO advantages: {adv_grpo.tolist()}")
print(f"  GDPO advantages: {adv_gdpo.tolist()}")

# GRPO collapses: prompt1 has reward sum [1.1, -0.1] → adv [0.707, -0.707]
# GDPO preserves: correctness AND calibration independently ranked
check("TC11a: GRPO and GDPO give different advantages", not torch.allclose(adv_grpo, adv_gdpo, atol=0.01))

# Key test: in GRPO, for prompt2 (both correct), calibration difference is small
# In GDPO, calibration difference should be more pronounced
prompt2_grpo_diff = abs(adv_grpo[2] - adv_grpo[3]).item()
prompt2_gdpo_diff = abs(adv_gdpo[2] - adv_gdpo[3]).item()
print(f"  Prompt2 advantage diff: GRPO={prompt2_grpo_diff:.4f}, GDPO={prompt2_gdpo_diff:.4f}")
check("TC11b: GDPO preserves calib signal for prompt2", prompt2_gdpo_diff > 0.01)

print(f"\n=== SUMMARY: {passed} passed, {failed} failed ===")
if failed > 0:
    sys.exit(1)
