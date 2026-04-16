"""Unit checks for redirect-triggered retrieval helpers."""
import sys
import tempfile
import json
sys.path.insert(0, ".")

from src.curriculum.control_rag import (
    ExampleRecord,
    RetrievalQuery,
    TfidfExampleRetriever,
    _classify_study_need_family,
    analyze_completion_for_rag,
    build_incontext_user_prompt,
    build_retrieval_query,
    build_retrieval_query_bundle,
    load_example_bank,
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

decorative_completion = """<|meta|>
confidence: 0.49
I am thinking carefully and will keep going.
<|/meta|>
Continuing the same route...
"""
decorative_analysis = analyze_completion_for_rag(decorative_completion)
check("decorative low confidence alone should not trigger retrieval", decorative_analysis["should_retrieve"] is False)

study_need_completion = """<|meta|>
confidence: 0.43
The current route is weak because I am missing the invariant that controls parity.
study_need: parity invariant for case-based reasoning
I should switch to a parity-based argument.
<|/meta|>
"""
study_analysis = analyze_completion_for_rag(study_need_completion)
study_query = build_retrieval_query("Find the parity pattern.", study_analysis)
check("study_need completion should trigger retrieval", study_analysis["should_retrieve"])
check("retrieval query should include study_need", "parity invariant for case-based reasoning" in study_query)

method_record = ExampleRecord(
    question="A sequence problem with alternating moves.",
    solution="Use a parity invariant and track odd/even state changes. \\boxed{1}",
    answer="1",
    source="synthetic",
    metadata={"topic": "invariants", "study_need": "parity invariant", "strategy_tags": ["parity", "invariant"]},
)
surface_record = ExampleRecord(
    question="Find the parity pattern in a table.",
    solution="Directly inspect the table entries. \\boxed{2}",
    answer="2",
    source="synthetic",
    metadata={"topic": "tables", "study_need": "table inspection", "strategy_tags": ["inspection"]},
)
method_retriever = TfidfExampleRetriever([surface_record, method_record])
bundle = build_retrieval_query_bundle("Color a graph under alternating flips.", study_analysis)
method_hits = method_retriever.search(bundle, top_k=1)
check("structured query should return a hit", len(method_hits) == 1)
check("solution-aware retrieval should prefer method-aligned example", method_hits[0]["record"].question == method_record.question)
check(
    "structured retrieval should expose score breakdown",
    "study_need_to_strategy" in method_hits[0]["score_breakdown"] and method_hits[0]["score_breakdown"]["total"] > 0,
)

manual_bundle = RetrievalQuery(
    problem="A geometry problem with a misleading equation.",
    diagnosis="The algebraic route is weak.",
    study_need="cross-section geometry",
    strategy_hint="needs a different solution method",
)
check("retrieval query bundle should serialize study_need", "cross-section geometry" in manual_bundle.to_text())

easy_record = ExampleRecord(
    question="A percent increase word problem.",
    solution="Use multiplicative growth rather than repeated addition. study_need: exponential growth / factor-power structure \\boxed{24}",
    answer="24",
    source="synthetic_easy",
    metadata={"topic": "percent", "difficulty": "easy", "study_need": "exponential growth / factor-power structure"},
)
hard_record = ExampleRecord(
    question="A hard olympiad functional equation.",
    solution="Use a structural invariant. study_need: invariant identification \\boxed{7}",
    answer="7",
    source="synthetic_hard",
    metadata={"topic": "olympiad", "difficulty": "hard", "study_need": "invariant identification"},
)
typed_retriever = TfidfExampleRetriever([hard_record, easy_record])
easy_bundle = build_retrieval_query_bundle(
    "A price grows by 20 percent every month.",
    {
        "diagnosis_text": "The current route is weak because repeated manual multiplication hides the multiplicative structure.",
        "study_need": "exponential growth / factor-power structure",
        "has_decomposition": False,
        "has_next_strategy": True,
        "has_switch": True,
    },
)
easy_hits = typed_retriever.search(easy_bundle, top_k=1)
check("easy-study query should prefer easy exemplar", easy_hits[0]["record"].question == easy_record.question)
check("easy-study query should expose easy bonus", easy_hits[0]["score_breakdown"]["easy_bonus"] > 0.0)
check("easy-study query should expose family match", easy_hits[0]["score_breakdown"]["study_need_family_match"] > 0.0)

generic_easy_record = ExampleRecord(
    question="A generic purchase word problem.",
    solution="Add the quantities and compute the total. \\boxed{18}",
    answer="18",
    source="synthetic_easy_generic",
    metadata={"topic": "word problem", "difficulty": "easy", "study_need": ""},
)
typed_growth_record = ExampleRecord(
    question="A bank account compounds by a fixed percent each year.",
    solution=(
        "Treat each year as multiplication by a constant factor rather than repeated addition. "
        "study_need: exponential growth / factor-power structure \\boxed{64}"
    ),
    answer="64",
    source="synthetic_easy_typed",
    metadata={"topic": "growth", "difficulty": "easy", "study_need": "exponential growth / factor-power structure"},
)
generic_vs_typed_retriever = TfidfExampleRetriever([generic_easy_record, typed_growth_record])
typed_query = build_retrieval_query_bundle(
    "A population increases by 25 percent every cycle.",
    {
        "diagnosis_text": "Repeated addition hides the multiplicative structure.",
        "study_need": "exponential growth / factor-power structure",
        "has_decomposition": False,
        "has_next_strategy": True,
        "has_switch": True,
    },
)
typed_hits = generic_vs_typed_retriever.search(typed_query, top_k=1)
check("typed study_need should beat generic easy fallback", typed_hits[0]["record"].question == typed_growth_record.question)
check("typed study_need should add typed bonus", typed_hits[0]["score_breakdown"]["typed_strategy_bonus"] > 0.0)
check("generic fallback should not win via easy bonus alone", typed_hits[0]["score_breakdown"]["generic_penalty"] == 0.0)
check("geometric sequence should map to exponential family", _classify_study_need_family("geometric sequence / repeated doubling") == "exponential_growth")
check("power of x should not map to geometry", _classify_study_need_family("binomial term identification by power of x") != "geometry")

with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as handle:
    json.dump([typed_growth_record.to_dict()], handle)
    handle.flush()
    roundtrip_records = load_example_bank([handle.name])
check("roundtrip loader should preserve one record", len(roundtrip_records) == 1)
check("roundtrip loader should preserve nested metadata", roundtrip_records[0].metadata["study_need_family"] == "exponential_growth")

print(f"\n=== SUMMARY: {passed} passed, {failed} failed ===")


def test_pytest_bridge():
    assert failed == 0


if __name__ == "__main__":
    if failed:
        sys.exit(1)
