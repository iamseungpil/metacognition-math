"""Test GDPO advantage computation (TC10-TC11)."""
import ast
import math
import sys

sys.path.insert(0, ".")

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
check("TC10c: E5 keeps calibration reward", '"E5": ([correctness_reward, format_reward, meta_quality_reward,\n                calibration_reward, confidence_revision_reward],' in code)
check("TC10d: E8 keeps calibration reward but no explicit behavior rewards", '"E8": ([correctness_reward, format_reward, correct_meta_reward,\n                calibration_reward, confidence_revision_reward,' in code and "effective_verification_reward" not in code.split('"E8":', 1)[1].split('"E9":', 1)[0])


def extract_reward_config_lengths(source: str):
    module = ast.parse(source)
    for node in ast.walk(module):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "reward_configs":
                    out = {}
                    for key_node, value_node in zip(node.value.keys, node.value.values):
                        if not isinstance(key_node, ast.Constant):
                            continue
                        if not isinstance(value_node, ast.Tuple) or len(value_node.elts) != 2:
                            continue
                        rewards_node, weights_node = value_node.elts
                        reward_len = len(rewards_node.elts) if isinstance(rewards_node, ast.List) else None
                        weight_len = len(weights_node.elts) if isinstance(weights_node, ast.List) else None
                        out[key_node.value] = (reward_len, weight_len)
                    return out
    raise RuntimeError("reward_configs not found")


lengths = extract_reward_config_lengths(code)
check("TC10e: E10 reward/weight lengths match", lengths["E10"][0] == lengths["E10"][1])
check("TC10f: E5 reward/weight lengths match", lengths["E5"][0] == lengths["E5"][1])
check("TC10g: E9b reward/weight lengths match", lengths["E9b"][0] == lengths["E9b"][1])
check("TC10h: E9c reward/weight lengths match", lengths["E9c"][0] == lengths["E9c"][1])

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

print("\n=== TC12: veRL confidence-centered alignment ===")

with open("src/training/verl_gdpo.py") as f:
    verl_code = f.read()


def extract_uppercase_reward_configs(source: str, assign_name: str):
    module = ast.parse(source)
    for node in ast.walk(module):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == assign_name:
                    out = {}
                    for key_node, value_node in zip(node.value.keys, node.value.values):
                        if not isinstance(key_node, ast.Constant):
                            continue
                        if not isinstance(value_node, ast.Dict):
                            continue
                        reward_len = None
                        weight_len = None
                        for sub_key, sub_val in zip(value_node.keys, value_node.values):
                            if isinstance(sub_key, ast.Constant) and sub_key.value == "funcs":
                                reward_len = len(sub_val.elts) if isinstance(sub_val, ast.List) else None
                            if isinstance(sub_key, ast.Constant) and sub_key.value == "weights":
                                weight_len = len(sub_val.elts) if isinstance(sub_val, ast.List) else None
                        out[key_node.value] = (reward_len, weight_len)
                    return out
    raise RuntimeError(f"{assign_name} not found")


verl_lengths = extract_uppercase_reward_configs(verl_code, "REWARD_CONFIGS")
check("TC12a: veRL E21R reward/weight lengths match", verl_lengths["E21R"][0] == verl_lengths["E21R"][1])
check("TC12b: veRL E21 reward/weight lengths match", verl_lengths["E21"][0] == verl_lengths["E21"][1])

with open("configs/verl_gdpo_e21.yaml") as f:
    e21_yaml = f.read()
check("TC12b2: historical E21 config keeps adv_estimator=gdpo", "adv_estimator: gdpo" in e21_yaml)
check("TC12b3: historical E21 actor uses per_gpu micro batch only", "ppo_micro_batch_size: null" in e21_yaml and "ppo_micro_batch_size_per_gpu: 1" in e21_yaml)
check("TC12b4: historical E21 ref/rollout log prob uses per_gpu style", "log_prob_micro_batch_size: null" in e21_yaml and "log_prob_micro_batch_size_per_gpu: 32" in e21_yaml)

with open("scripts/relaunch_verl_e21_0410.sh") as f:
    e21_launch = f.read()
check("TC12b5: historical E21 launcher clears deprecated actor micro batch", "actor_rollout_ref.actor.ppo_micro_batch_size=null" in e21_launch and "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1" in e21_launch)

from src.training.verl_reward import compute_score_confidence_centered

score = compute_score_confidence_centered(
    solution_str="<|meta|>confidence: 0.9 I may be overcommitting.<|/meta|>I verify by substitution. \\boxed{4}",
    ground_truth="4",
)
expected_keys = {"score", "correctness", "confidence_revision", "redirect_execution", "verify_execution", "meta_floor", "meta_count_bonus"}
check("TC12c: confidence-centered veRL reward returns exact key set", set(score.keys()) == expected_keys)

print(f"\n=== SUMMARY: {passed} passed, {failed} failed ===")


def test_pytest_bridge():
    assert failed == 0


if __name__ == "__main__":
    if failed > 0:
        sys.exit(1)
