"""Unit checks for redirect-triggered retrieval helpers."""
import sys
sys.path.insert(0, ".")

from src.curriculum.control_rag import (
    ExampleRecord,
    TfidfExampleRetriever,
    analyze_completion_for_rag,
    build_incontext_user_prompt,
    build_retrieval_query,
)


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


example = ExampleRecord(
    question="Solve x + 4 = 9.",
    solution="Subtract 4 from both sides to get x = 5. \\boxed{5}",
    answer="5",
    source="synthetic",
    metadata={"topic": "linear equation", "difficulty": "easy"},
)
distractor = ExampleRecord(
    question="Compute 3 + 3.",
    solution="3 + 3 = 6. \\boxed{6}",
    answer="6",
    source="synthetic",
    metadata={"topic": "arithmetic", "difficulty": "easy"},
)

completion = """<|meta|>
confidence: 0.41
Something feels off. The current route is weak because I may be forcing the wrong algebra.
I should step back and switch to isolating the variable directly.
<|/meta|>
Trying again...
"""

analysis = analyze_completion_for_rag(completion)
check("redirect completion should trigger retrieval", analysis["should_retrieve"])
check("diagnosis text should be extracted", "forcing the wrong algebra" in analysis["diagnosis_text"])

query = build_retrieval_query("Solve x + 7 = 12.", analysis)
check("query should contain original question", "Solve x + 7 = 12." in query)
check("query should contain diagnosis", "forcing the wrong algebra" in query)

retriever = TfidfExampleRetriever([example, distractor])
hits = retriever.search(query, top_k=1)
check("retriever should return a hit", len(hits) == 1)
check("retriever should prefer similar linear equation", hits[0]["record"].question == example.question)

prompt = build_incontext_user_prompt("Solve x + 7 = 12.", analysis, hits)
check("prompt should include retrieved example", example.question in prompt)
check("prompt should include original problem", "Solve x + 7 = 12." in prompt)

print(f"\n=== SUMMARY: {passed} passed, {failed} failed ===")
if failed:
    sys.exit(1)
