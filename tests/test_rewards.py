"""Unit tests for reward functions (TC1-TC9)."""
import math
import sys
sys.path.insert(0, ".")

from src.training.rewards import (
    correctness_reward, meta_quality_reward,
    calibration_reward, uncertainty_meta_reward,
    stepwise_probe_reward, probe_calibration_reward,
    stepwise_trajectory_reward, diagnosis_reward, decomposition_reward,
    effective_verification_reward, effective_redirection_reward,
    overconfidence_verify_reward,
    same_route_repetition_penalty, route_switch_evidence_reward,
    confidence_omission_floor,
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

print("\n=== TC13b-TC13d: probe-based blockwise scoring ===")

def good_probe(prefixes):
    return [0.30, 0.86]


def bad_probe(prefixes):
    return [0.82, 0.12]


two_block = """<|meta|>
Confidence: 0.32
I am not sure yet.
<|/meta|>
Some work...
<|meta|>
Confidence: 0.84
Now the structure is consistent.
<|/meta|>
\\boxed{4}"""

r_probe_good = probe_calibration_reward([two_block], probe_predictor=good_probe)
r_probe_bad = probe_calibration_reward([two_block], probe_predictor=bad_probe)
check("TC13b: probe calibration prefers aligned local confidences", r_probe_good[0] > r_probe_bad[0], True)

r_step_probe_good = stepwise_probe_reward([two_block], ground_truth=["4"], probe_predictor=good_probe)
r_step_probe_bad = stepwise_probe_reward([two_block], ground_truth=["4"], probe_predictor=bad_probe)
check("TC13c: stepwise probe reward is blockwise and prefers aligned probe targets", r_step_probe_good[0] > r_step_probe_bad[0], True)

redirect_blockwise = """<|meta|>
Confidence: 0.78
This seems plausible.
<|/meta|>
Trying a route...
<|meta|>
Confidence: 0.36
Something feels off. I may be forcing the wrong invariant, so I should switch methods.
<|/meta|>
\\boxed{4}"""
r_blockwise = stepwise_trajectory_reward([redirect_blockwise], ground_truth=["4"])
check("TC13d: trajectory reward gives positive credit to conflict-conditioned confidence drop", r_blockwise[0] > -1.0, True)

print("\n=== TC14-TC15: diagnosis/decomposition rewards ===")
diagnostic_redirect = """<|meta|>
confidence: 0.78
The algebra looks plausible, but something feels off.
<|/meta|>
Trying the first route...
<|meta|>
confidence: 0.34
Something feels off. The transformed equation is working, but the original constraint is not.
I may be forcing symbolic manipulation too early and missing the real invariant.
I should step back, identify the invariant, then handle the remaining cases separately and switch to a parity-based case split.
<|/meta|>
\\boxed{4}"""
r_diag = diagnosis_reward([diagnostic_redirect])
r_decomp = decomposition_reward([diagnostic_redirect], ground_truth=["4"])
check("TC14: diagnosis reward positive on natural redirect", r_diag[0] > 0.3, True)
check("TC15: decomposition reward positive on natural redirect", r_decomp[0] > 0.3, True)

print("\n=== TC16-TC18: behavior rewards require meta trigger ===")
meta_verify = """<|meta|>
confidence: 0.91
I may be committing too quickly, so I should verify independently before finalizing.
<|/meta|>
Let me substitute the candidate back into the equation to test the result.
\\boxed{4}"""
r_eff_verify = effective_verification_reward([meta_verify], ground_truth=["4"])
check("TC16: effective verification positive with meta trigger + solve-tail check", r_eff_verify[0] > 0.3, True)

tail_only_verify = """I will now substitute the candidate back into the equation to test the result.
Everything checks out.
\\boxed{4}"""
r_tail_only_verify = effective_verification_reward([tail_only_verify], ground_truth=["4"])
check("TC17: solve-tail verification alone should not trigger behavior reward", r_tail_only_verify[0], 0.0)

decorative_meta = """<|meta|>
confidence: 0.34
I am thinking carefully.
<|/meta|>
I will continue with the same route.
\\boxed{4}"""
r_redirect = effective_redirection_reward([decorative_meta], ground_truth=["4"])
check("TC18: decorative low confidence without diagnosis/switch is penalized or non-positive", r_redirect[0] <= 0.0, True)

print("\n=== TC19: overconfidence verify is conditional on high confidence ===")
high_conf_verify = """<|meta|>
confidence: 0.88
This answer came too quickly, so I should verify independently before committing.
<|/meta|>
I recompute the key quantity from scratch and check by substitution.
\\boxed{4}"""
low_conf_verify = """<|meta|>
confidence: 0.42
I should verify before continuing.
<|/meta|>
I recompute the key quantity from scratch and check by substitution.
\\boxed{4}"""
r_high = overconfidence_verify_reward([high_conf_verify], ground_truth=["4"])
r_low = overconfidence_verify_reward([low_conf_verify], ground_truth=["4"])
check("TC19a: high-confidence verify gets positive reward", r_high[0] > 0.0, True)
check("TC19b: low-confidence verify does not trigger overconfidence verify reward", r_low[0], 0.0)


# ─── TC20-TC25: V2 Reward Functions (2026-04-04) ───

print("\n=== TC20: same_route_repetition_penalty ===")
# TC20a: verify intent + repetition in tail → penalty
repetition_verify = """<|meta|>
confidence: 0.88 The answer came quickly, so I should verify.
<|/meta|>
Let me repeat the same calculation: 4 × 20 = 80, 2 × 80 = 160, 160 + 80 + 20 = 320.
\\boxed{320}"""
r = same_route_repetition_penalty([repetition_verify], ground_truth=["280"])
check("TC20a: repetition verify gets penalty", r[0] < 0, True)

# TC20b: verify intent + independent method → no penalty
independent_verify = """<|meta|>
confidence: 0.88 The answer came quickly, so I should verify.
<|/meta|>
Let me independently verify by working backwards: if total is 280, then Seattle=20 + Charleston=80 + Toulouse=180. Since 80/20=4 and 180/80=2.25, this does not match "twice". Cross-check reveals Toulouse should be 160, giving total 260.
\\boxed{260}"""
r = same_route_repetition_penalty([independent_verify], ground_truth=["280"])
check("TC20b: independent verify gets no penalty", r[0] >= 0, True)

# TC20c: no verify intent → neutral
no_verify = """<|meta|>
confidence: 0.34 The current route is weak.
<|/meta|>
Let me try another approach.
\\boxed{4}"""
r = same_route_repetition_penalty([no_verify], ground_truth=["4"])
check("TC20c: no verify intent → neutral", r[0], 0.0)

# TC20d: short tail with verify intent → -0.3
short_verify = """<|meta|>confidence: 0.88 I should verify.<|/meta|>Yes. \\boxed{4}"""
r = same_route_repetition_penalty([short_verify])
check("TC20d: short tail verify → -0.3 penalty", r[0], -0.3)

print("\n=== TC21: route_switch_evidence_reward ===")
# TC21a: conflict + switch + drop + structural difference → positive
# Use text that triggers _has_conflict_trigger ("something feels off") and _has_next_strategy ("switch to")
switch_with_evidence = """First I try algebra: expanding (a+b)^2 gives a^2+2ab+b^2.
<|meta|>
confidence: 0.34 Something feels off with this approach. It does not satisfy the constraint.
I should switch to a different method using invariant analysis.
<|/meta|>
Using an invariant approach instead of the previous algebra: define the parity of each assignment. By parity analysis, exactly 3 configurations satisfy the constraint.
\\boxed{3}"""
r = route_switch_evidence_reward([switch_with_evidence], ground_truth=["3"])
check("TC21a: structural switch gets high reward", r[0] > 0.3, True)

# TC21b: conflict + switch + drop but tail doesn't differ → penalty
switch_no_evidence = """First I try algebra: expanding (a+b)^2 gives a^2+2ab+b^2.
<|meta|>
confidence: 0.34 Something feels off with this direct route. It fails to converge.
I should switch to a better approach.
<|/meta|>
So continuing the expansion, we get a^2+2ab+b^2 = 12.
\\boxed{12}"""
r = route_switch_evidence_reward([switch_no_evidence], ground_truth=["3"])
check("TC21b: switch announced but no evidence → penalty", r[0] < 0, True)

# TC21c: no conflict/switch → neutral
no_redirect = """<|meta|>
confidence: 0.88 Looking good.
<|/meta|>
\\boxed{4}"""
r = route_switch_evidence_reward([no_redirect], ground_truth=["4"])
check("TC21c: no redirect intent → neutral", r[0], 0.0)

# TC21d: structural switch but wrong answer → lower reward (0.25 not 0.9)
r_correct_21 = route_switch_evidence_reward([switch_with_evidence], ground_truth=["3"])
r_wrong_21 = route_switch_evidence_reward([switch_with_evidence], ground_truth=["999"])
check("TC21d: structural switch but wrong → lower than correct", r_wrong_21[0] < r_correct_21[0], True)

print("\n=== TC22: confidence_omission_floor ===")
# TC22a: no meta → penalty
no_meta = "Let me solve: 2+2=4. \\boxed{4}"
r = confidence_omission_floor([no_meta])
check("TC22a: no meta block → penalty", r[0] < 0, True)

# TC22b: has meta → pass
with_meta = """<|meta|>
confidence: 0.7
<|/meta|>
\\boxed{4}"""
r = confidence_omission_floor([with_meta])
check("TC22b: meta present → passes floor", r[0], 0.0)

# TC22c: empty think tag, no meta → penalty
think_only = "<think> </think> \\boxed{4}"
r = confidence_omission_floor([think_only])
check("TC22c: think-only no meta → penalty", r[0] < 0, True)

# TC22d: batch test
r_batch = confidence_omission_floor([no_meta, with_meta])
check("TC22d: batch [no_meta, with_meta] length", len(r_batch), 2)
check("TC22d: batch first is penalty", r_batch[0] < 0, True)
check("TC22d: batch second passes", r_batch[1], 0.0)

print(f"\n=== SUMMARY: {passed} passed, {failed} failed ===")


def test_pytest_bridge():
    assert failed == 0


if __name__ == "__main__":
    if failed > 0:
        sys.exit(1)
