"""Smoke tests for vLLM-based fixed_k_repair (with FakeLLM, no GPU)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class FakeSamplingParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


# Install fake vllm BEFORE importing online.py (which imports from vllm inside run fn).
_fake_vllm = type(sys)("vllm")
_fake_vllm.SamplingParams = FakeSamplingParams
sys.modules.setdefault("vllm", _fake_vllm)

from src.training.self_distill.online import (  # noqa: E402
    OnlineSdpoProblem,
    _load_completed_ids,
    _stable_problem_id,
    run_online_question_only_best_of_n_rollouts,
    run_online_fixed_k_repair_rollouts,
    write_online_sdpo_outputs,
)


class FakeCompletion:
    def __init__(self, text: str, token_ids: list[int]):
        self.text = text
        self.token_ids = token_ids


class FakeRequest:
    def __init__(self, prompt_token_ids: list[int], outputs: list[FakeCompletion]):
        self.prompt_token_ids = prompt_token_ids
        self.outputs = outputs


class FakeTokenizer:
    pad_token = "[PAD]"
    eos_token = "[EOS]"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        # Pass-through: return user message content
        return messages[0]["content"]


class FakeLLM:
    """Mimics vllm.LLM for unit tests.

    The canned_outputs_fn(prompt, n) returns list of (text, token_ids) tuples.
    """

    def __init__(self, canned_outputs_fn):
        self._fn = canned_outputs_fn
        self._tokenizer = FakeTokenizer()
        self.call_count = 0
        self.last_prompts = []

    def generate(self, prompts, sampling, **kwargs):
        self.call_count += 1
        self.last_prompts = list(prompts)
        n = getattr(sampling, "n", 1)
        outs = []
        for p in prompts:
            cands = self._fn(p, n)
            outs.append(
                FakeRequest(
                    prompt_token_ids=list(range(min(len(p.split()), 100))),
                    outputs=[FakeCompletion(text=t, token_ids=list(range(len(t.split())))) for (t, _) in cands],
                )
            )
        return outs

    def get_tokenizer(self):
        return self._tokenizer


@pytest.fixture(autouse=True)
def patch_sampling(monkeypatch):
    """Ensure a fake vllm module is importable inside run_online_fixed_k_repair_rollouts."""
    fake_module = type(sys)("vllm")
    fake_module.SamplingParams = FakeSamplingParams
    monkeypatch.setitem(sys.modules, "vllm", fake_module)


def make_problems(n: int) -> list[OnlineSdpoProblem]:
    return [
        OnlineSdpoProblem(
            question=f"Solve: 2x = {2*i}, find x.",
            gold_answer=str(i),
            benchmark="test",
            metadata={"idx": i},
        )
        for i in range(n)
    ]


def canned_root_then_repair(prompt: str, n: int) -> list[tuple[str, list[int]]]:
    """First call: simple wrong root. Second (repair): half correct via boxed{i}."""
    if "Previous attempt to repair" in prompt or "Previous attempt:" in prompt:
        # Repair phase: emit n candidates, half boxed correctly
        cands = []
        for i in range(n):
            if i % 2 == 0:
                cands.append((r"<think>let me retry</think>\boxed{42}", [1, 2, 3]))
            else:
                cands.append((r"<think>wrong route</think>\boxed{0}", [1, 2, 3]))
        return cands
    # Root phase: always wrong
    return [(r"<think>guessing</think>\boxed{99}", [1, 2, 3])]


def test_fixed_k_repair_smoke(tmp_path: Path):
    problems = make_problems(4)
    llm = FakeLLM(canned_root_then_repair)

    rows = run_online_fixed_k_repair_rollouts(
        llm=llm,
        tokenizer=llm.get_tokenizer(),
        problems=problems,
        output_dir=tmp_path,
        retriever=None,
        repair_candidates=4,
        chunk_size=4,
        resume=False,
    )

    assert len(rows) == 4

    traces_path = tmp_path / "online_sdpo_traces.jsonl"
    assert traces_path.exists()
    lines = traces_path.read_text().strip().split("\n")
    assert len(lines) == 4

    for line in lines:
        row = json.loads(line)
        # Schema check
        for key in [
            "problem_id",
            "question",
            "gold_answer",
            "benchmark",
            "generation_mode",
            "root_prompt",
            "root_completion",
            "root_judgment",
            "repair_prompt",
            "repair_candidates",
            "selector",
            "selected_completion",
            "selected_judgment",
        ]:
            assert key in row, f"Missing key: {key}"


def test_question_only_best_of_n_smoke(tmp_path: Path):
    def canned_question_only(prompt, n):
        if "2x = 0" in prompt:
            return [("\\boxed{99}", [1]), ("\\boxed{0}", [1]), ("\\boxed{1}", [1]), ("\\boxed{2}", [1])][:n]
        return [("\\boxed{99}", [1]), ("\\boxed{1}", [1]), ("\\boxed{0}", [1]), ("\\boxed{2}", [1])][:n]

    problems = make_problems(2)
    llm = FakeLLM(canned_question_only)
    rows = run_online_question_only_best_of_n_rollouts(
        llm=llm,
        tokenizer=llm.get_tokenizer(),
        problems=problems,
        output_dir=tmp_path,
        num_candidates=4,
        chunk_size=2,
        resume=False,
    )

    assert len(rows) == 2
    for row in rows:
        assert row["generation_mode"] == "question_only_best_of_n"
        assert row["selected_prompt_kind"] == "question_only"
        assert row["selector"]["selector_mode"] == "correctness_only"
        assert row["retrieval_enabled"] is False


def test_question_only_selector_prefers_correctness_first(tmp_path: Path):
    def canned(prompt, n):
        return [
            ("<|meta|> confidence: 0.95 <|/meta|> wrong \\boxed{99}", [1]),
            ("<|meta|> confidence: 0.40 <|/meta|> right \\boxed{0}", [1]),
        ][:n]

    problems = [OnlineSdpoProblem(question="Solve: 2x = 0, find x.", gold_answer="0", benchmark="test")]
    llm = FakeLLM(canned)
    rows = run_online_question_only_best_of_n_rollouts(
        llm=llm,
        tokenizer=llm.get_tokenizer(),
        problems=problems,
        output_dir=tmp_path,
        num_candidates=2,
        selector_mode="correctness_first",
        chunk_size=1,
        resume=False,
    )

    assert rows[0]["selector"]["selected_candidate_id"] == "sample_1"
    assert rows[0]["selected_judgment"]["is_correct"] is True


def test_question_only_selector_prefers_correctness_only(tmp_path: Path):
    def canned(prompt, n):
        return [
            ("<|meta|> confidence: 0.99 <|/meta|> wrong \\boxed{99}", [1]),
            ("plain but correct \\boxed{0}", [1]),
        ][:n]

    problems = [OnlineSdpoProblem(question="Solve: 2x = 0, find x.", gold_answer="0", benchmark="test")]
    llm = FakeLLM(canned)
    rows = run_online_question_only_best_of_n_rollouts(
        llm=llm,
        tokenizer=llm.get_tokenizer(),
        problems=problems,
        output_dir=tmp_path,
        num_candidates=2,
        selector_mode="correctness_only",
        chunk_size=1,
        resume=False,
    )

    assert rows[0]["selector"]["selected_candidate_id"] == "sample_1"


def test_resume_skips_processed(tmp_path: Path):
    problems = make_problems(4)
    traces_path = tmp_path / "online_sdpo_traces.jsonl"

    # Pre-seed jsonl with 2 rows
    pre_ids = [_stable_problem_id("test", problems[i].question) for i in range(2)]
    with traces_path.open("w") as f:
        for pid in pre_ids:
            f.write(json.dumps({"problem_id": pid, "question": "stub"}) + "\n")

    llm = FakeLLM(canned_root_then_repair)
    rows = run_online_fixed_k_repair_rollouts(
        llm=llm,
        tokenizer=llm.get_tokenizer(),
        problems=problems,
        output_dir=tmp_path,
        retriever=None,
        repair_candidates=2,
        chunk_size=4,
        resume=True,
    )

    # Should only call llm.generate twice (root + repair) on the 2 unprocessed
    assert llm.call_count == 2  # phase 1 + phase 2 over 1 chunk of 2
    assert len(llm.last_prompts) == 2
    # Returned rows = 2 existing + 2 new = 4
    assert len(rows) == 4


def test_streaming_writes_all_valid_json(tmp_path: Path):
    problems = make_problems(8)
    llm = FakeLLM(canned_root_then_repair)

    rows = run_online_fixed_k_repair_rollouts(
        llm=llm,
        tokenizer=llm.get_tokenizer(),
        problems=problems,
        output_dir=tmp_path,
        retriever=None,
        repair_candidates=2,
        chunk_size=4,
        resume=False,
    )

    assert len(rows) == 8
    traces_path = tmp_path / "online_sdpo_traces.jsonl"
    lines = traces_path.read_text().strip().split("\n")
    assert len(lines) == 8
    for line in lines:
        json.loads(line)  # must not raise


def test_load_completed_ids_tolerates_malformed(tmp_path: Path):
    traces_path = tmp_path / "x.jsonl"
    traces_path.write_text(
        '{"problem_id": "abc"}\n'
        '{"problem_id": "def"}\n'
        '{not valid json\n'  # corrupt line
        '{"problem_id": "ghi"}\n'
    )
    done = _load_completed_ids(traces_path)
    assert done == {"abc", "def", "ghi"}


def test_dataframe_projection_after_streaming(tmp_path: Path):
    """Verify write_online_sdpo_outputs(traces_already_written=True) builds parquet."""
    problems = make_problems(4)
    llm = FakeLLM(canned_root_then_repair)
    rows = run_online_fixed_k_repair_rollouts(
        llm=llm,
        tokenizer=llm.get_tokenizer(),
        problems=problems,
        output_dir=tmp_path,
        retriever=None,
        repair_candidates=2,
        chunk_size=4,
        resume=False,
    )

    payload = write_online_sdpo_outputs(
        rows=rows,
        output_dir=tmp_path,
        source_tag="test",
        mode="naive",
        claim_bearing=False,
        traces_already_written=True,
    )

    # jsonl is the streaming version; parquet should also exist
    assert (tmp_path / "online_sdpo_traces.jsonl").exists()
    assert payload["num_rollouts"] == 4
    # naive mode strips meta; rows that have boxed should pass require_boxed
    # Our canned outputs all have \boxed{} so all 4 should pass
    assert payload["num_dataset_rows"] >= 1


class FakeRetriever:
    """Returns a canned exemplar for any query."""
    def __init__(self):
        from src.curriculum.control_rag import ExampleRecord
        self.records = [ExampleRecord(
            question="Find x: 2x = 6", solution="x = 3", answer="3", source="seed", metadata={},
        )]
    def search(self, query, top_k=1):
        return [{
            "score": 0.42, "score_breakdown": {"problem_similarity": 0.42},
            "record": self.records[0],
        }] * min(top_k, 1)


def test_retrieval_attaches_evidence(tmp_path: Path):
    """Verify retrieval populates row['retrieved'] and feedback_kind."""
    problems = make_problems(2)
    llm = FakeLLM(canned_root_then_repair)
    rows = run_online_fixed_k_repair_rollouts(
        llm=llm, tokenizer=llm.get_tokenizer(),
        problems=problems, output_dir=tmp_path,
        retriever=FakeRetriever(), repair_candidates=2, chunk_size=2,
        rag_top_k=1, retrieval_query_mode="question_only", resume=False,
    )
    assert len(rows) == 2
    for row in rows:
        assert len(row["retrieved"]) == 1, "Retrieval should produce 1 evidence item per problem"
        assert row["selected_feedback_kind"] == "forced_rag"
        assert row["selected_feedback_context"]["lane"] == "fixed_k_repair"
        assert row["retrieval_nonempty"] is True
        assert row["retrieval_enabled"] is True


def test_selector_picks_best_candidate(tmp_path: Path):
    """Verify selector picks the highest-correctness candidate (not always idx 0)."""
    # Make a custom canned fn where candidate at idx 1 is correct (boxed{i}), others wrong.
    def canned(prompt, n):
        if "Previous attempt" in prompt:
            # Repair phase: extract problem index from prompt
            import re
            m = re.search(r"2x = (\d+)", prompt)
            target_x = int(m.group(1)) // 2 if m else 0
            cands = []
            for i in range(n):
                if i == 1:
                    # Only candidate 1 is correct
                    cands.append((f"<think>retry</think>\\boxed{{{target_x}}}", [1, 2, 3]))
                else:
                    cands.append((f"<think>wrong</think>\\boxed{{99}}", [1, 2, 3]))
            return cands
        return [("<think>guess</think>\\boxed{99}", [1, 2, 3])]

    problems = make_problems(3)
    llm = FakeLLM(canned)
    rows = run_online_fixed_k_repair_rollouts(
        llm=llm, tokenizer=llm.get_tokenizer(),
        problems=problems, output_dir=tmp_path,
        retriever=None, repair_candidates=4, chunk_size=3, resume=False,
    )
    for row in rows:
        # Selector should have chosen the correct candidate (id "repair_1")
        assert row["selector"]["selected_candidate_id"] == "repair_1", \
            f"Expected selector to pick repair_1 (the correct one), got {row['selector']['selected_candidate_id']}"
        assert row["selected_judgment"]["is_correct"] is True


def test_claim_bearing_zero_rows_raises(tmp_path: Path):
    """Verify claim_bearing=True with no meta blocks raises ValueError."""
    # Canned outputs have NO <|meta|> blocks
    def canned_no_meta(prompt, n):
        if "Previous attempt" in prompt:
            return [("<think>retry</think>\\boxed{42}", [1, 2, 3])] * n
        return [("<think>guess</think>\\boxed{99}", [1, 2, 3])]

    problems = make_problems(2)
    llm = FakeLLM(canned_no_meta)
    rows = run_online_fixed_k_repair_rollouts(
        llm=llm, tokenizer=llm.get_tokenizer(),
        problems=problems, output_dir=tmp_path,
        retriever=None, repair_candidates=2, chunk_size=2, resume=False,
    )
    # Streaming worked (rows in jsonl)
    assert len(rows) == 2
    # But claim_bearing=True with epistemic mode should reject all rows (no meta)
    import pytest
    with pytest.raises(ValueError, match="Claim-bearing"):
        write_online_sdpo_outputs(
            rows=rows, output_dir=tmp_path,
            source_tag="test", mode="epistemic", claim_bearing=True,
            traces_already_written=True,
        )


def test_multi_chunk_boundaries(tmp_path: Path):
    """Verify chunk boundaries don't break streaming/resume semantics."""
    problems = make_problems(10)
    llm = FakeLLM(canned_root_then_repair)
    rows = run_online_fixed_k_repair_rollouts(
        llm=llm, tokenizer=llm.get_tokenizer(),
        problems=problems, output_dir=tmp_path,
        retriever=None, repair_candidates=2, chunk_size=3, resume=False,
    )
    assert len(rows) == 10
    # ceil(10/3) = 4 chunks × 2 phases = 8 generate calls
    assert llm.call_count == 8, f"Expected 8 batched generate calls, got {llm.call_count}"
    lines = (tmp_path / "online_sdpo_traces.jsonl").read_text().strip().split("\n")
    assert len(lines) == 10
    pids = {json.loads(l)["problem_id"] for l in lines}
    assert len(pids) == 10, "All problem_ids should be distinct"


def test_truncate_preserves_boxed_in_long_text():
    """Verify middle-truncation keeps \\boxed{} at the end."""
    from src.training.self_distill.online import _truncate_for_repair
    text = "a" * 5000 + "\\boxed{42}"  # 5010 chars, boxed at end
    out = _truncate_for_repair(text, max_chars=2000)
    assert len(out) <= 2100  # slight slack for marker
    assert "\\boxed{42}" in out, "Middle-truncation must preserve tail with \\boxed{}"


def test_truncate_handles_small_max_chars():
    """Verify truncation doesn't go negative on tiny max_chars."""
    from src.training.self_distill.online import _truncate_for_repair
    text = "a" * 500 + "\\boxed{7}"
    out = _truncate_for_repair(text, max_chars=50)
    # Must not exceed max_chars by much
    assert len(out) <= 60, f"Output len {len(out)} exceeds safe bound for max_chars=50"
    # Should keep tail (where boxed lives) when budget is tiny
    assert "\\boxed{7}" in out, "Tiny max_chars should still preserve tail"


def test_root_fallback_failure_does_not_crash(tmp_path: Path):
    """Verify root-phase generation failure for one prompt doesn't crash the whole chunk.

    Regression for Bug 1: _safe_generate emits a _Stub placeholder (outputs=[])
    when a single prompt fails during per-prompt fallback. The root-phase loop
    must skip these rather than IndexError on outputs[0].
    """
    call_state = {"batch_call": 0, "per_prompt_call": 0}

    def raising_canned(prompt, n):
        # This canned is called once per prompt inside the underlying FakeLLM.generate.
        # We fail the very first per-prompt invocation (mimics a prompt-too-long error
        # for problem 0 during the per-prompt fallback), then serve normal outputs.
        call_state["per_prompt_call"] += 1
        if call_state["per_prompt_call"] == 1:
            raise RuntimeError("Simulated prompt-too-long")
        if "Previous attempt" in prompt:
            return [(r"<think>retry</think>\boxed{42}", [1, 2, 3])] * n
        return [(r"<think>guess</think>\boxed{99}", [1, 2, 3])]

    class RaisingFakeLLM(FakeLLM):
        def generate(self, prompts, sampling, **kwargs):
            call_state["batch_call"] += 1
            if call_state["batch_call"] == 1:
                # First call is the root batch; fail it so _safe_generate falls
                # back to per-prompt retries.
                raise RuntimeError("Simulated batch failure")
            return super().generate(prompts, sampling, **kwargs)

    problems = make_problems(3)
    llm = RaisingFakeLLM(raising_canned)

    # Should not crash; problem 0 skipped due to root-phase _Stub, rest succeed.
    rows = run_online_fixed_k_repair_rollouts(
        llm=llm, tokenizer=llm.get_tokenizer(),
        problems=problems, output_dir=tmp_path,
        retriever=None, repair_candidates=2, chunk_size=3, resume=False,
    )
    # Problem 0 skipped at root-phase guard -> at most 2 rows produced.
    assert len(rows) <= 2

    # Skipped log should exist and record the root-phase failure.
    skipped_path = tmp_path / "skipped_problems.jsonl"
    assert skipped_path.exists(), "skipped_problems.jsonl should be created"
    skipped = [
        json.loads(l) for l in skipped_path.read_text().strip().split("\n") if l.strip()
    ]
    assert len(skipped) >= 1, "At least one problem should be logged as skipped"
    assert any(s["phase"] == "root" for s in skipped), "Expected at least one root-phase skip"


def test_summary_includes_skipped_count(tmp_path: Path):
    """Verify write_online_sdpo_outputs summary includes skip counters.

    Regression for Bug 2: claim-bearing fairness requires explicit skip tracking
    so base/meta pairings can reconcile attempt vs success counts.
    """
    # Pre-create a skipped_problems.jsonl simulating mixed-phase failures.
    skipped_path = tmp_path / "skipped_problems.jsonl"
    with skipped_path.open("w") as f:
        f.write(json.dumps({"problem_id": "abc", "phase": "root", "reason": "no_outputs"}) + "\n")
        f.write(json.dumps({"problem_id": "def", "phase": "repair", "reason": "no_outputs"}) + "\n")
        f.write(json.dumps({"problem_id": "ghi", "phase": "selector", "reason": "selector_failed: no candidates"}) + "\n")

    problems = make_problems(2)
    llm = FakeLLM(canned_root_then_repair)
    rows = run_online_fixed_k_repair_rollouts(
        llm=llm, tokenizer=llm.get_tokenizer(),
        problems=problems, output_dir=tmp_path,
        retriever=None, repair_candidates=2, chunk_size=2, resume=False,
    )

    payload = write_online_sdpo_outputs(
        rows=rows, output_dir=tmp_path,
        source_tag="test", mode="naive", claim_bearing=False,
        traces_already_written=True,
    )

    assert payload["skipped_count"] == 3, f"Expected 3 skipped, got {payload.get('skipped_count')}"
    assert payload["skipped_by_phase"]["root"] == 1
    assert payload["skipped_by_phase"]["repair"] == 1
    assert payload["skipped_by_phase"]["selector"] == 1
