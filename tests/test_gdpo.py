"""Test GDPO advantage computation (TC10-TC11)."""
import math
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
trainer_call = code.split("trainer = GRPOTrainer(", 1)[1].split(")\n\n", 1)[0]
check("TC10: no peft_config in GRPOTrainer call", "peft_config=" not in trainer_call)
check("TC10b: E10 keeps calibration reward", '"E10": ([correctness_reward, format_reward, correct_meta_reward,\n                 calibration_reward,' in code)

print("\n=== TC11: GDPO per-reward normalization ===")
num_gen = 2
rewards_per_func = [
    [1.0, 0.1],   # prompt1, gen1: correct, low calib
    [-1.0, 0.9],  # prompt1, gen2: wrong, high calib
    [1.0, 0.8],    # prompt2, gen1: correct, high calib
    [1.0, 0.2],    # prompt2, gen2: correct, low calib
]
weights = [1.0, 1.0]


def mean(xs):
    return sum(xs) / len(xs)


def std(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def grouped(values, group_size):
    return [values[i:i + group_size] for i in range(0, len(values), group_size)]


def normalize_grouped(values, group_size):
    normalized = []
    for group in grouped(values, group_size):
        m = mean(group)
        s = std(group)
        normalized.extend((v - m) / (s + 1e-4) for v in group)
    return normalized


rewards_grpo = [sum(v * w for v, w in zip(row, weights)) for row in rewards_per_func]
adv_grpo = normalize_grouped(rewards_grpo, num_gen)

per_reward_adv = []
for reward_idx in range(len(weights)):
    reward_values = [row[reward_idx] for row in rewards_per_func]
    per_reward_adv.append(normalize_grouped(reward_values, num_gen))

pre_bn = [
    sum(per_reward_adv[j][i] * weights[j] for j in range(len(weights)))
    for i in range(len(rewards_per_func))
]
pre_bn_mean = mean(pre_bn)
pre_bn_std = std(pre_bn)
adv_gdpo = [(v - pre_bn_mean) / (pre_bn_std + 1e-4) for v in pre_bn]

print(f"  GRPO advantages: {adv_grpo}")
print(f"  GDPO advantages: {adv_gdpo}")

# GRPO collapses: prompt1 has reward sum [1.1, -0.1] → adv [0.707, -0.707]
# GDPO preserves: correctness AND calibration independently ranked
check(
    "TC11a: GRPO and GDPO give different advantages",
    any(abs(a - b) > 0.01 for a, b in zip(adv_grpo, adv_gdpo))
)

# Key test: in GRPO, for prompt2 (both correct), calibration difference is small
# In GDPO, calibration difference should be more pronounced
prompt2_grpo_diff = abs(adv_grpo[2] - adv_grpo[3])
prompt2_gdpo_diff = abs(adv_gdpo[2] - adv_gdpo[3])
print(f"  Prompt2 advantage diff: GRPO={prompt2_grpo_diff:.4f}, GDPO={prompt2_gdpo_diff:.4f}")
check("TC11b: GDPO preserves calib signal for prompt2", prompt2_gdpo_diff > 0.01)

print(f"\n=== SUMMARY: {passed} passed, {failed} failed ===")
if failed > 0:
    sys.exit(1)
