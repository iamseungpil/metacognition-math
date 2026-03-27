"""Unit tests for reward functions (TC1-TC9)."""
import math
import sys
sys.path.insert(0, ".")

from src.training.rewards import (
    correctness_reward, meta_quality_reward,
    calibration_reward, uncertainty_meta_reward,
    _check_correctness, _parse_meta_blocks,
)

passed = 0
failed = 0

def check(name, actual, expected, tol=0.01):
    global passed, failed
    if isinstance(expected, bool):
        ok = actual == expected
    elif isinstance(expected, (int, float)):
        ok = abs(actual - expected) < tol
    else:
        ok = actual == expected
    status = "PASS" if ok else "FAIL"
    if not ok:
        failed += 1
        print(f"  {status}: {name}: got {actual}, expected {expected}")
    else:
        passed += 1
        print(f"  {status}: {name}")

print("=== TC1-TC2: correctness_reward ===")
r = correctness_reward(["\\boxed{4}"], ground_truth=["4"])
check("TC1: correct answer", r[0], 1.0)

r = correctness_reward(["\\boxed{5}"], ground_truth=["4"])
check("TC2: wrong answer", r[0], -1.0)

print("\n=== TC3-TC4: meta_quality_reward ===")
meta_text = """<|meta|>
Q: Can I solve this problem?
A: This is a number theory problem about modular arithmetic. I think my probability of solving it correctly is about 0.40. The key risk is making errors in cycle detection.
<|/meta|>
Some solution...
<|meta|>
Q: Is my approach correct?
A: Let me verify the residues. Confidence: 0.70
<|/meta|>
More solution...
<|meta|>
Q: Final check?
A: Yes, the answer looks right. Confidence: 0.95
<|/meta|>
\\boxed{42}"""
r = meta_quality_reward([meta_text])
check("TC3: 3 meta blocks with Q&A", r[0] > 0.5, True)

r = meta_quality_reward(["Just solve: \\boxed{42}"])
check("TC4: no meta blocks", r[0], -0.5)

print("\n=== TC5-TC7: calibration_reward (Rewarding Doubt) ===")
correct_high = "<|meta|>My confidence: 0.9<|/meta|>\\boxed{4}"
r = calibration_reward([correct_high], ground_truth=["4"])
check("TC5: correct + conf 0.9", abs(r[0] - math.log(0.9)) < 0.1, True)

wrong_high = "<|meta|>My confidence: 0.9<|/meta|>\\boxed{5}"
r = calibration_reward([wrong_high], ground_truth=["4"])
check("TC6: wrong + conf 0.9 (overconfident)", r[0] < -1.5, True)

three_blocks = """<|meta|>probability 0.3<|/meta|>step1
<|meta|>confidence 0.6<|/meta|>step2
<|meta|>confidence 0.9<|/meta|>\\boxed{4}"""
r = calibration_reward([three_blocks], ground_truth=["4"])
check("TC7: 3 blocks summation (should be sum not single)", abs(r[0]) > 0.1, True)

# Verify summation: 3 blocks should give different result than 1
one_block = "<|meta|>confidence 0.6<|/meta|>\\boxed{4}"
r1 = calibration_reward([one_block], ground_truth=["4"])
r3 = calibration_reward([three_blocks], ground_truth=["4"])
check("TC7b: 3 blocks ≠ 1 block", abs(r3[0]) != abs(r1[0]), True)

print("\n=== TC8-TC9: uncertainty_meta_reward ===")
uncertain_long = """<|meta|>
Q: Can I solve this? My probability is 0.3.
A: This requires careful analysis of residues mod 7. I should check all cases systematically to avoid missing any.
<|/meta|>"""
r = uncertainty_meta_reward([uncertain_long])
check("TC8: uncertain + long meta", r[0] > 0.3, True)

confident_short = "<|meta|>confidence 0.9<|/meta|>"
r = uncertainty_meta_reward([confident_short])
check("TC9: confident + short meta", r[0] < 0.15, True)

# TC8 should be much higher than TC9
r_uncertain = uncertainty_meta_reward([uncertain_long])[0]
r_confident = uncertainty_meta_reward([confident_short])[0]
check("TC8>TC9: uncertain meta > confident meta", r_uncertain > r_confident, True)

print(f"\n=== SUMMARY: {passed} passed, {failed} failed ===")
if failed > 0:
    sys.exit(1)
