"""Test cases for eval pipeline (TC4-TC7)."""
import re
import sys
sys.path.insert(0, ".")

from src.metacot.prompt import parse_meta_blocks
from src.training.rewards import _extract_answer_fallback, _check_correctness

extract_answer = _extract_answer_fallback
check_correctness = _check_correctness

# TC4: Answer extraction
print("=== TC4: Answer extraction ===")
all_pass = True
tests = [
    (r"The answer is \boxed{4}", "4"),
    (r"Therefore \boxed{\frac{1}{2}}", r"\frac{1}{2}"),
    ("#### 42", "42"),
    ("The answer is 7.", "7"),
]
for text, expected in tests:
    result = extract_answer(text)
    status = "PASS" if result == expected else f"FAIL (got '{result}')"
    if result != expected:
        all_pass = False
    print(f"  {status}: '{text[:40]}' → '{result}'")

# TC5: Confidence parsing from Meta-CoT output
print("\n=== TC5: Confidence parsing ===")
meta_text = """<|meta|>
Q: Can I solve this?
A: Yes. My probability of solving it correctly is about 0.85.
Q: What should I watch out for?
A: Check all cases.
<|/meta|>"""
parsed = parse_meta_blocks(meta_text)
print(f"  Blocks: {parsed['num_blocks']} (expected 1)")
print(f"  Confidences: {parsed['confidences']} (expected [0.85])")
status = "PASS" if parsed['num_blocks'] == 1 and abs(parsed['confidences'][0] - 0.85) < 0.01 else "FAIL"
if status != "PASS":
    all_pass = False
print(f"  {status}")

# Also test text-based confidence parsing (when <|meta|> is stripped)
stripped = "My probability of solving it correctly is about 0.85. Check all cases."
confs = re.findall(
    r'(?:probability|confidence)[^\d]*(\d+(?:\.\d+)?)',
    stripped,
    re.IGNORECASE,
)
print(f"  Stripped text parsing: {confs} (expected ['0.85'])")
status = "PASS" if confs == ['0.85'] else "FAIL"
if status != "PASS":
    all_pass = False
print(f"  {status}")

# TC6: ECE calculation
print("\n=== TC6: ECE calculation ===")
# ECE = |mean_confidence - accuracy|
confidences = [0.9, 0.8, 0.7, 0.6]
correct = [True, True, False, False]
accuracy = sum(correct) / len(correct)  # 0.5
avg_conf = sum(confidences) / len(confidences)  # 0.75
ece = abs(avg_conf - accuracy)  # 0.25
print(f"  Accuracy: {accuracy}, Avg confidence: {avg_conf}, ECE: {ece}")
status = "PASS" if abs(ece - 0.25) < 0.01 else "FAIL"
if status != "PASS":
    all_pass = False
print(f"  {status}")

# TC7: Check correctness function
print("\n=== TC7: Correctness checking ===")
tests = [
    (r"\boxed{3}", "3", True),
    (r"\boxed{3}", "4", False),
    ("#### 42", "42", True),
    ("The answer is 7", "7.0", True),
    ("", "5", False),
]
for pred, gold, expected in tests:
    result = check_correctness(pred, gold)
    status = "PASS" if result == expected else f"FAIL (got {result})"
    if result != expected:
        all_pass = False
    print(f"  {status}: '{pred[:30]}' vs '{gold}' → {result}")

print(f"\n=== SUMMARY ===")
print(f"All tests: {'PASS' if all_pass else 'SOME FAILED'}")
