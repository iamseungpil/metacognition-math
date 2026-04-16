#!/usr/bin/env python3
"""Smoke tests for RQ2 reward families."""
import sys, importlib
sys.path.insert(0, ".")
import src.training.rewards as R
importlib.reload(R)
from src.training.rewards import (
    structural_switch_reward_v2 as SW,
    verify_outcome_v2 as VO,
    confidence_trajectory_reward as CT,
    confidence_revision_reward_v2 as CR,
    redirect_execution_reward_v2 as RR,
    verify_execution_reward_v2 as VR,
    stepwise_trajectory_reward as ST,
    confidence_omission_floor as FL,
    correctness_reward as CO,
    meta_count_bonus as MB,
)

cases = {
    "good_redirect": (
        "<think>\nCompleting the square. x^2+6x+5=(x+3)^2-4.\n"
        "<|meta|>\nConfidence: 0.7\nAssessment: Standard approach.\nAction: continue\n<|/meta|>\n"
        "Roots at x=-3+/-2.\n"
        "<|meta|>\nConfidence: 0.3\nAssessment: Something feels off. I may be forcing this.\n"
        "The issue is completing the square is error-prone.\n"
        "Action: redirect - try factoring instead\n<|/meta|>\n"
        "Let me try factoring instead. x^2+6x+5=(x+1)(x+5)=0. The smaller root is -5.\n"
        "</think>\n\\boxed{-5}"
    ),
    "no_meta": "<think>x^2+6x+5=(x+1)(x+5). The smaller root is -5.</think>\n\\boxed{-5}",
    "verify_only": (
        "<think>\n(x+1)(x+5)=x^2+6x+5. The smaller root is -5.\n"
        "<|meta|>\nConfidence: 0.95\nAssessment: The answer came too quickly. "
        "Committing without checking.\nAction: verify by substitution\n<|/meta|>\n"
        "Check: (-5)^2+6(-5)+5=25-30+5=0. Correct.\n"
        "</think>\n\\boxed{-5}"
    ),
    "trigger_no_exec": (
        "<think>\nSubstitution. u=x+3...\n"
        "<|meta|>\nConfidence: 0.2\nAssessment: I am stuck. "
        "The issue is substitution is not helping.\n"
        "Action: redirect - try different approach\n<|/meta|>\n"
        "Hmm, u=x+3 means u^2-4=0, so the smaller root is still -5.\n"
        "</think>\n\\boxed{-5}"
    ),
    "trigger_diag_only": (
        "<think>\nVietas: sum=-6, product=5.\n"
        "<|meta|>\nConfidence: 0.4\nAssessment: Something feels off. "
        "The issue is this ignores the sign.\nAction: continue\n<|/meta|>\n"
        "The smaller root is -5.\n</think>\n\\boxed{-5}"
    ),
    "wrong_redirect": (
        "<think>\nCompleting square.\n"
        "<|meta|>\nConfidence: 0.7\nAction: continue\n<|/meta|>\nHmm.\n"
        "<|meta|>\nConfidence: 0.3\nAssessment: Something feels off. I may be forcing this.\n"
        "The issue is error-prone.\nAction: redirect - try factoring instead\n<|/meta|>\n"
        "Factoring: (x+2)(x+3)=0. The smaller root is -3.\n</think>\n\\boxed{-3}"
    ),
    "good_recovery": (
        "<think>\nCompleting the square. x^2+6x+5=(x+3)^2-4.\n"
        "<|meta|>\nConfidence: 0.7\nAssessment: Standard approach.\nAction: continue\n<|/meta|>\n"
        "Roots at x=-3+/-2.\n"
        "<|meta|>\nConfidence: 0.3\nAssessment: Something feels off. I may be forcing this.\n"
        "The issue is completing the square is error-prone.\n"
        "Action: redirect - try factoring instead\n<|/meta|>\n"
        "Let me try factoring instead. x^2+6x+5=(x+1)(x+5)=0. The smaller root is -5.\n"
        "<|meta|>\nConfidence: 0.8\nAssessment: The redirected route is consistent.\n"
        "Action: verify by substitution\n<|/meta|>\n"
        "Check: (-5)^2+6(-5)+5=0.\n"
        "</think>\n\\boxed{-5}"
    ),
}

gt = ["-5"]
W = {"co": 1.0, "cr": 0.35, "rr": 0.30, "vr": 0.15, "fl": 0.50, "mb": 1.0}
W_E21 = {"co": 1.0, "sw": 0.15, "vo": 0.30, "ct": 0.15, "fl": 0.50, "mb": 1.0}

print("=== E21R Smoke ===")
header = f"{'Case':25s} {'co':>5s}  {'cr':>7s}  {'rr':>7s}  {'vr':>7s}  {'fl':>5s}  {'mb':>5s}  {'combined':>8s}"
print(header)
print("-" * len(header))

for name, text in cases.items():
    comp = [{"content": text}]
    co = CO(comp, gt)[0]
    cr = CR(comp, gt)[0]
    rr = RR(comp, gt)[0]
    vr = VR(comp, gt)[0]
    fl = FL(comp, gt)[0]
    mb = MB(comp, gt)[0]
    comb = co * W["co"] + cr * W["cr"] + rr * W["rr"] + vr * W["vr"] + fl * W["fl"] + mb * W["mb"]
    print(f"{name:25s} {co:+.1f}  {cr:+.3f}  {rr:+.3f}  {vr:+.3f}  {fl:+.2f}  {mb:+.2f}  {comb:+.3f}")

# Validation checks
print("\n=== Validation ===")
comp_gr = [{"content": cases["good_redirect"]}]
comp_nm = [{"content": cases["no_meta"]}]
comp_wr = [{"content": cases["wrong_redirect"]}]

scores = {}
for name, text in cases.items():
    comp = [{"content": text}]
    co = CO(comp, gt)[0]
    cr = CR(comp, gt)[0]
    rr = RR(comp, gt)[0]
    vr = VR(comp, gt)[0]
    fl = FL(comp, gt)[0]
    mb = MB(comp, gt)[0]
    scores[name] = co * W["co"] + cr * W["cr"] + rr * W["rr"] + vr * W["vr"] + fl * W["fl"] + mb * W["mb"]

checks = [
    ("good_redirect > no_meta", scores["good_redirect"] > scores["no_meta"]),
    ("good_redirect > verify_only", scores["good_redirect"] > scores["verify_only"]),
    ("good_redirect > trigger_no_exec", scores["good_redirect"] > scores["trigger_no_exec"]),
    ("verify_only > no_meta", scores["verify_only"] > scores["no_meta"]),
    ("trigger_no_exec < good_redirect", scores["trigger_no_exec"] < scores["good_redirect"]),
    ("wrong_redirect < good_redirect", scores["wrong_redirect"] < scores["good_redirect"]),
]

all_pass = True
for desc, result in checks:
    status = "PASS" if result else "FAIL"
    if not result:
        all_pass = False
    print(f"  [{status}] {desc}")

print(f"\n{'E21R CHECKS PASSED' if all_pass else 'E21R CHECKS FAILED'}")

print("\n=== E21 Anchor Smoke ===")
e21_scores = {}
header = f"{'Case':25s} {'co':>5s}  {'sw':>7s}  {'vo':>7s}  {'ct':>7s}  {'fl':>5s}  {'mb':>5s}  {'combined':>8s}"
print(header)
print("-" * len(header))
for name, text in cases.items():
    comp = [{"content": text}]
    co = CO(comp, gt)[0]
    sw = SW(comp, gt)[0]
    vo = VO(comp, gt)[0]
    ct = CT(comp, gt)[0]
    fl = FL(comp, gt)[0]
    mb = MB(comp, gt)[0]
    comb = co * W_E21["co"] + sw * W_E21["sw"] + vo * W_E21["vo"] + ct * W_E21["ct"] + fl * W_E21["fl"] + mb * W_E21["mb"]
    e21_scores[name] = comb
    print(f"{name:25s} {co:+.1f}  {sw:+.3f}  {vo:+.3f}  {ct:+.3f}  {fl:+.2f}  {mb:+.2f}  {comb:+.3f}")

for desc, result in [
    ("good_redirect > no_meta", e21_scores["good_redirect"] > e21_scores["no_meta"]),
    ("verify_only > no_meta", e21_scores["verify_only"] > e21_scores["no_meta"]),
    ("wrong_redirect < good_redirect", e21_scores["wrong_redirect"] < e21_scores["good_redirect"]),
]:
    print(f"  [{'PASS' if result else 'FAIL'}] {desc}")
    all_pass = all_pass and result

print("\n=== E21S Stepwise Smoke ===")
step_scores = {}
for name, text in cases.items():
    comp = [{"content": text}]
    step = ST(comp, gt)[0]
    step_scores[name] = step
    print(f"{name:25s} {step:+.3f}")

for desc, result in [
    ("good_recovery > no_meta", step_scores["good_recovery"] > step_scores["no_meta"]),
    ("good_recovery > wrong_redirect", step_scores["good_recovery"] > step_scores["wrong_redirect"]),
    ("trigger_no_exec < good_recovery", step_scores["trigger_no_exec"] < step_scores["good_recovery"]),
]:
    print(f"  [{'PASS' if result else 'FAIL'}] {desc}")
    all_pass = all_pass and result

print("\n=== Meta Count Bonus Smoke (correctness-gated) ===")
# Meta count bonus now requires correct answer. Use boxed{-5} to match gt.
meta_count_cases_correct = {
    "zero_correct": "<think>Direct. The answer is -5.</think>\n\\boxed{-5}",
    "one_correct": "<think><|meta|>Confidence: 0.6<|/meta|>Work. -5.</think>\n\\boxed{-5}",
    "two_correct": "<think><|meta|>Confidence: 0.6<|/meta|>A<|meta|>Confidence: 0.5<|/meta|>B. -5.</think>\n\\boxed{-5}",
    "three_correct": "<think><|meta|>Confidence: 0.7<|/meta|>A<|meta|>Confidence: 0.4<|/meta|>B<|meta|>Confidence: 0.8<|/meta|>C. -5.</think>\n\\boxed{-5}",
    "four_correct": "<think><|meta|>Confidence: 0.7<|/meta|>A<|meta|>Confidence: 0.4<|/meta|>B<|meta|>Confidence: 0.8<|/meta|>C<|meta|>Confidence: 0.9<|/meta|>D. -5.</think>\n\\boxed{-5}",
}
meta_count_cases_wrong = {
    "zero_wrong": "<think>Direct. The answer is 3.</think>\n\\boxed{3}",
    "one_wrong": "<think><|meta|>Confidence: 0.6<|/meta|>Work. 3.</think>\n\\boxed{3}",
    "three_wrong": "<think><|meta|>Confidence: 0.7<|/meta|>A<|meta|>Confidence: 0.4<|/meta|>B<|meta|>Confidence: 0.8<|/meta|>C. 3.</think>\n\\boxed{3}",
}
mb_scores = {}
print("--- Correct answer ---")
for name, text in meta_count_cases_correct.items():
    mb_scores[name] = MB([{"content": text}], gt)[0]
    print(f"{name:25s} {mb_scores[name]:+.2f}")
print("--- Wrong answer ---")
for name, text in meta_count_cases_wrong.items():
    mb_scores[name] = MB([{"content": text}], gt)[0]
    print(f"{name:25s} {mb_scores[name]:+.2f}")

for desc, result in [
    ("correct: 0 < 1 meta", mb_scores["zero_correct"] < mb_scores["one_correct"]),
    ("correct: 1 < 2 meta", mb_scores["one_correct"] < mb_scores["two_correct"]),
    ("correct: 2 < 3 meta", mb_scores["two_correct"] < mb_scores["three_correct"]),
    ("correct: 3 == 4 meta (cap)", abs(mb_scores["three_correct"] - mb_scores["four_correct"]) < 1e-9),
    ("wrong: all zero regardless of meta count", mb_scores["zero_wrong"] == 0.0 and mb_scores["one_wrong"] == 0.0 and mb_scores["three_wrong"] == 0.0),
    ("correct+3 > wrong+3 (anti-hacking)", mb_scores["three_correct"] > mb_scores["three_wrong"]),
]:
    print(f"  [{'PASS' if result else 'FAIL'}] {desc}")
    all_pass = all_pass and result

print(f"\n{'ALL SMOKE CHECKS PASSED' if all_pass else 'SOME SMOKE CHECKS FAILED'}")
