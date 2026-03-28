"""Unit tests for reward functions (TC1-TC9)."""
import math
import sys
sys.path.insert(0, ".")

from src.training.rewards import (
    correctness_reward, meta_quality_reward,
    calibration_reward, uncertainty_meta_reward,
    stepwise_probe_reward,
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

print("\n=== TC10-TC13: stepwise_probe_reward ===")

# TC10: Good trajectory (0.3 -> 0.6 -> 0.9 correct) should be positive
good_trajectory = """<|meta|>
Q: Can I solve this? Probability of solving: 0.3
<|/meta|>
Let me work through this step by step...
<|meta|>
Q: Is this right? Confidence: 0.6
<|/meta|>
Continuing the solution...
<|meta|>
Q: Final check? Confidence: 0.9
<|/meta|>
\\boxed{4}"""
r_good = stepwise_probe_reward([good_trajectory], ground_truth=["4"])
check("TC10: good trajectory (0.3->0.6->0.9 correct) positive", r_good[0] > 0, True)

# TC11: Always-0.99 and wrong should be very negative
always_high_wrong = """<|meta|>
Q: Can I solve this? Confidence: 0.99
<|/meta|>
Solution attempt...
<|meta|>
Q: Am I right? Confidence: 0.99
<|/meta|>
More work...
<|meta|>
Q: Final check? Confidence: 0.99
<|/meta|>
\\boxed{999}"""
r_bad = stepwise_probe_reward([always_high_wrong], ground_truth=["4"])
check("TC11: always-0.99 wrong is very negative", r_bad[0] < -1.0, True)

# Compare TC10 vs TC11: good trajectory should beat always-0.99 wrong
check("TC10 >> TC11: good trajectory beats overconfident wrong",
      r_good[0] - r_bad[0] > 1.0, True)

# TC12: Error-correction pattern gets bonus
# Compare mid block with and without error-correction keywords
with_correction = """<|meta|>
Q: Can I solve this? Probability: 0.4
<|/meta|>
First attempt...
<|meta|>
Wait, that's wrong. Let me fix this. Actually the answer is different. Confidence: 0.7
<|/meta|>
Corrected solution...
<|meta|>
Q: Final? Confidence: 0.85
<|/meta|>
\\boxed{4}"""
without_correction = """<|meta|>
Q: Can I solve this? Probability: 0.4
<|/meta|>
First attempt...
<|meta|>
Continuing the calculation normally. Confidence: 0.7
<|/meta|>
More solution...
<|meta|>
Q: Final? Confidence: 0.85
<|/meta|>
\\boxed{4}"""
r_with = stepwise_probe_reward([with_correction], ground_truth=["4"])
r_without = stepwise_probe_reward([without_correction], ground_truth=["4"])
check("TC12: error-correction gets bonus", r_with[0] > r_without[0], True)

# TC13: Starting low then increasing beats starting high
# Both correct, same final confidence, but different starting points
start_low = """<|meta|>
Q: Can I solve this? Probability: 0.3
<|/meta|>
Working through it...
<|meta|>
Q: Final check? Confidence: 0.85
<|/meta|>
\\boxed{4}"""
start_high = """<|meta|>
Q: Can I solve this? Probability: 0.95
<|/meta|>
Working through it...
<|meta|>
Q: Final check? Confidence: 0.85
<|/meta|>
\\boxed{4}"""
r_low_start = stepwise_probe_reward([start_low], ground_truth=["4"])
r_high_start = stepwise_probe_reward([start_high], ground_truth=["4"])
check("TC13: starting low beats starting high", r_low_start[0] > r_high_start[0], True)


print(f"\n=== SUMMARY: {passed} passed, {failed} failed ===")
if failed > 0:
    sys.exit(1)
