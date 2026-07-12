import json
import sys
from pathlib import Path
import pytest
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.self_distill.eval_metrics import load_eval_table
from src.training.sft import prepare_sft_dataset
from src.training.self_distill_data import (
    build_epistemic_teacher_completion,
    build_feedback_conditioned_messages,
    build_naive_teacher_completion,
    build_self_distill_dataframe,
    build_teacher_feedback_payload,
    normalize_teacher_row,
)


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        text = []
        for message in messages:
            text.append(f"{message['role']}:{message['content']}")
        if add_generation_prompt:
            text.append("assistant:")
        joined = "\n".join(text)
        if tokenize:
            return {"input_ids": [ord(ch) % 97 for ch in joined]}
        return joined


def test_normalize_messages_row():
    row = {
        "messages": json.dumps(
            [
                {"role": "user", "content": "Solve x+2=5."},
                {"role": "assistant", "content": "<|meta|>\nconfidence: 0.4\n<|/meta|>\nSo x=3. \\boxed{3}"},
            ]
        ),
        "source": "synthetic",
        "difficulty": "easy",
    }
    trace = normalize_teacher_row(row)
    assert trace is not None
    assert trace.question == "Solve x+2=5."
    assert "\\boxed{3}" in trace.teacher_completion
    assert trace.source == "synthetic"


def test_naive_builder_suppresses_meta():
    trace = normalize_teacher_row(
        {
            "messages": json.dumps(
                [
                    {"role": "user", "content": "Find x."},
                    {
                        "role": "assistant",
                        "content": "<|meta|>\nconfidence: 0.3\nI think the route is weak.\n<|/meta|>\nMaybe subtract 2, so x=4. \\boxed{4}",
                    },
                ]
            )
        }
    )
    assert trace is not None
    text = build_naive_teacher_completion(trace)
    assert "<|meta|>" not in text
    assert "Maybe" not in text
    assert "\\boxed{4}" in text


def test_epistemic_builder_preserves_or_synthesizes_structure():
    row = {
        "question": "Solve x+7=12.",
        "gold_answer": "5",
        "benchmark": "aime2024",
        "curriculum_retry": {
            "retry_completion": "Subtract 7 from both sides, so x=5. \\boxed{5}",
            "retry_judgment": {"is_correct": True},
            "meta_transition": {"confidence_gain": 0.35, "trigger_cleared": True},
            "retrieved": [{"question": "Solve x+4=9.", "source": "stable_seed_library", "score": 0.8}],
        },
        "root_completion": "I guessed \\boxed{4}",
        "root_analysis": {
            "diagnosis_text": "The earlier route guessed without controlling the equation.",
            "study_need": "direct isolation",
        },
    }
    trace = normalize_teacher_row(row)
    assert trace is not None
    text = build_epistemic_teacher_completion(trace)
    assert "<|meta|>" in text
    assert "study_need: direct isolation" in text
    assert "\\boxed{5}" in text
    assert trace.teacher_feedback_kind == "teacher_only_rag"
    payload = build_teacher_feedback_payload(trace)
    assert payload["feedback_kind"] == "teacher_only_rag"
    assert payload["teacher_feedback_context"]["evidence_items"][0]["question"] == "Solve x+4=9."


def test_dataframe_builder_tracks_mode_and_metrics():
    rows = [
        {
            "question": "Solve x+3=7.",
            "gold_answer": "4",
            "curriculum_retry": {
                "retry_completion": "Subtract 3 from both sides, so x=4. \\boxed{4}",
                "retry_judgment": {"is_correct": True},
                "meta_transition": {"confidence_gain": 0.2, "trigger_cleared": True},
            },
            "root_completion": "I guessed \\boxed{5}",
            "root_analysis": {"diagnosis_text": "Guessing is weak.", "study_need": "direct isolation"},
        }
    ]
    naive = build_self_distill_dataframe(rows, mode="naive")
    epistemic = build_self_distill_dataframe(rows, mode="epistemic")
    assert naive.iloc[0]["self_distill_mode"] == "naive"
    assert epistemic.iloc[0]["self_distill_mode"] == "epistemic"
    assert naive.iloc[0]["teacher_num_meta_blocks"] == 0
    assert epistemic.iloc[0]["teacher_num_meta_blocks"] >= 1
    assert bool(epistemic.iloc[0]["teacher_feedback_available"]) is False


def test_claim_bearing_epistemic_refuses_synthetic_meta():
    rows = [
        {
            "question": "Solve x+3=7.",
            "gold_answer": "4",
            "selected_completion": "Subtract 3 from both sides, so x=4. \\boxed{4}",
            "selected_judgment": {"is_correct": True},
            "root_completion": "I guessed \\boxed{5}",
            "root_analysis": {"diagnosis_text": "Guessing is weak.", "study_need": "direct isolation"},
            "repair_candidates": [{"candidate_id": "repair_0"}],
            "selector": {"selected_candidate_id": "repair_0", "selected_score": 1.0, "score_margin": 0.2},
        }
    ]
    built = build_self_distill_dataframe(rows, mode="epistemic", claim_bearing=True)
    assert built.empty


def test_claim_bearing_epistemic_keeps_real_meta_and_selector_provenance():
    rows = [
        {
            "question": "Solve x+3=7.",
            "gold_answer": "4",
            "selected_completion": "<|meta|>\nconfidence: 0.41\nThe earlier guess was unsupported.\nstudy_need: direct isolation\n<|/meta|>\nSubtract 3 from both sides, so x=4. \\boxed{4}",
            "selected_judgment": {"is_correct": True},
            "root_completion": "I guessed \\boxed{5}",
            "root_analysis": {"diagnosis_text": "Guessing is weak.", "study_need": "direct isolation"},
            "repair_candidates": [{"candidate_id": "repair_0"}, {"candidate_id": "repair_1"}],
            "selector": {
                "selected_candidate_id": "repair_1",
                "selected_score": 1.25,
                "selected_breakdown": {"correctness": 1.0, "total": 1.25},
                "score_margin": 0.15,
            },
        }
    ]
    built = build_self_distill_dataframe(rows, mode="epistemic", claim_bearing=True)
    assert len(built) == 1
    assert bool(built.iloc[0]["synthetic_meta_injected"]) is False
    assert built.iloc[0]["candidate_count"] == 2
    assert built.iloc[0]["selected_candidate_id"] == "repair_1"
    assert built.iloc[0]["selection_margin"] == pytest.approx(0.15)


def test_rq3_dataframe_preserves_feedback_context_and_benchmark():
    rows = [
        {
            "question": "Solve x+7=12.",
            "gold_answer": "5",
            "benchmark": "aime2024",
            "curriculum_retry": {
                "retry_completion": "Subtract 7 from both sides, so x=5. \\boxed{5}",
                "retry_judgment": {"is_correct": True},
                "meta_transition": {"confidence_gain": 0.3, "trigger_cleared": True},
                "retrieved": [{"question": "Solve x+4=9.", "source": "stable_seed_library", "score": 0.9}],
            },
            "root_completion": "I guessed \\boxed{4}",
            "root_analysis": {"diagnosis_text": "Guessing is weak.", "study_need": "direct isolation"},
        }
    ]
    built = build_self_distill_dataframe(rows, mode="epistemic")
    assert len(built) == 1
    assert built.iloc[0]["benchmark"] == "aime2024"
    assert bool(built.iloc[0]["teacher_feedback_available"]) is True
    payload = json.loads(built.iloc[0]["teacher_feedback_context_json"])
    assert payload["feedback_kind"] == "teacher_only_rag"
    assert payload["teacher_feedback_context"]["evidence_items"][0]["question"] == "Solve x+4=9."


def test_feedback_conditioned_mode_uses_feedback_in_prompt():
    rows = [
        {
            "question": "Solve x+7=12.",
            "gold_answer": "5",
            "benchmark": "aime2024",
            "curriculum_retry": {
                "retry_completion": "Subtract 7 from both sides, so x=5. \\boxed{5}",
                "retry_judgment": {"is_correct": True},
                "meta_transition": {"confidence_gain": 0.3, "trigger_cleared": True},
                "retrieved": [{"question": "Solve x+4=9.", "source": "stable_seed_library", "score": 0.9}],
            },
            "root_completion": "I guessed \\boxed{4}",
            "root_analysis": {"diagnosis_text": "Guessing is weak.", "study_need": "direct isolation"},
        }
    ]
    built = build_self_distill_dataframe(rows, mode="sdpo_regen")
    assert len(built) == 1
    messages = json.loads(built.iloc[0]["messages"])
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert "teacher-side recovery feedback and evidence" in messages[0]["content"]
    assert "unsuccessful earlier attempt" in messages[0]["content"]
    assert "Solve x+4=9." in messages[0]["content"]
    assert messages[1]["role"] == "assistant"
    assert "\\boxed{5}" in messages[1]["content"]
    assert built.iloc[0]["teacher_feedback_kind"] == "teacher_only_rag"
    assert built.iloc[0]["self_distill_mode"] == "sdpo_regen"
    assert built.iloc[0]["teacher_prompt_kind"] == "sdpo_regen"
    assert "Correctly solve the original question." in built.iloc[0]["teacher_prompt_text"]


def test_sdpo_regen_skips_incorrect_selected_teacher_rows():
    rows = [
        {
            "question": "Solve x+3=7.",
            "gold_answer": "4",
            "generation_mode": "sdpo_regen",
            "root_completion": "I guessed \\boxed{5}",
            "root_analysis": {"diagnosis_text": "Guessing is weak.", "study_need": "direct isolation"},
            "selected_completion": "<|meta|>\nconfidence: 0.62\n<|/meta|>\nWrong route. \\boxed{5}",
            "selected_judgment": {"is_correct": False},
            "selected_feedback_kind": "teacher_feedback_only",
            "selected_feedback_context": {"lane": "sdpo_regen", "evidence_items": []},
            "repair_candidates": [{"candidate_id": "regen_0"}],
            "selector": {"selected_candidate_id": "regen_0", "selected_score": 0.2},
        }
    ]
    built = build_self_distill_dataframe(rows, mode="sdpo_regen")
    assert built.empty


def test_claim_bearing_sdpo_regen_is_rejected():
    rows = [
        {
            "question": "Solve x+7=12.",
            "gold_answer": "5",
            "benchmark": "aime2024",
            "curriculum_retry": {
                "retry_completion": "Subtract 7 from both sides, so x=5. \\boxed{5}",
                "retry_judgment": {"is_correct": True},
                "retrieved": [{"question": "Solve x+4=9.", "source": "stable_seed_library", "score": 0.9}],
            },
            "root_completion": "I guessed \\boxed{4}",
            "root_analysis": {"diagnosis_text": "Guessing is weak.", "study_need": "direct isolation"},
        }
    ]
    built = build_self_distill_dataframe(rows, mode="sdpo_regen", claim_bearing=True)
    assert built.empty


def test_feedback_conditioned_messages_require_feedback():
    trace = normalize_teacher_row(
        {
            "question": "Solve x+3=7.",
            "gold_answer": "4",
            "curriculum_retry": {
                "retry_completion": "Subtract 3 from both sides, so x=4. \\boxed{4}",
                "retry_judgment": {"is_correct": True},
            },
            "root_completion": "I guessed \\boxed{5}",
            "root_analysis": {"diagnosis_text": "Guessing is weak.", "study_need": "direct isolation"},
        }
    )
    assert trace is not None
    with pytest.raises(ValueError):
        build_feedback_conditioned_messages(trace)


def test_feedback_conditioned_alias_maps_to_sdpo_regen():
    rows = [
        {
            "question": "Solve x+7=12.",
            "gold_answer": "5",
            "curriculum_retry": {
                "retry_completion": "Subtract 7 from both sides, so x=5. \\boxed{5}",
                "retry_judgment": {"is_correct": True},
                "retrieved": [{"question": "Solve x+4=9.", "source": "stable_seed_library", "score": 0.9}],
            },
            "root_completion": "I guessed \\boxed{4}",
            "root_analysis": {"diagnosis_text": "Guessing is weak.", "study_need": "direct isolation"},
        }
    ]
    built = build_self_distill_dataframe(rows, mode="feedback_conditioned")
    assert len(built) == 1
    assert built.iloc[0]["self_distill_mode"] == "sdpo_regen"


def test_incorrect_root_fallback_is_not_used():
    rows = [
        {
            "question": "Solve x+3=7.",
            "gold_answer": "4",
            "root_completion": "I guessed \\boxed{9}",
            "root_judgment": {"is_correct": False},
            "root_analysis": {"diagnosis_text": "Guessing is weak.", "study_need": "direct isolation"},
            "curriculum_retry": {
                "retry_completion": "I still guess \\boxed{9}",
                "retry_judgment": {"is_correct": False},
            },
            "selective_branching": {
                "best_branch_completion": "Also wrong \\boxed{8}",
                "best_branch_judgment": {"is_correct": False},
            },
        }
    ]
    built = build_self_distill_dataframe(rows, mode="naive")
    assert built.empty
    assert "messages" in built.columns


def test_empty_dataframe_keeps_expected_schema():
    built = build_self_distill_dataframe([], mode="epistemic")
    assert built.empty
    assert "messages" in built.columns
    assert "self_distill_mode" in built.columns


def test_eval_loader_rejects_invalid_json_contract(tmp_path):
    bad_path = tmp_path / "bad_eval.json"
    bad_path.write_text(json.dumps({"foo": "bar"}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_eval_table(str(bad_path))


def test_eval_loader_accepts_results_json(tmp_path):
    good_path = tmp_path / "good_eval.json"
    payload = {
        "results": [
            {
                "benchmark": "math500",
                "question": "q1",
                "full_question": "q1",
                "completion": "Plain answer \\boxed{5}",
                "is_correct": True,
                "num_meta_blocks": 0,
                "avg_confidence": None,
                "completion_length_tokens": 8,
            }
        ]
    }
    good_path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_eval_table(str(good_path))
    assert len(loaded) == 1
    assert bool(loaded.iloc[0]["is_correct"]) is True


def test_prepare_sft_dataset_drops_all_masked_rows(tmp_path):
    path = tmp_path / "tiny.parquet"
    rows = [
        {
            "messages": json.dumps(
                [
                    {"role": "user", "content": "A" * 200},
                    {"role": "assistant", "content": "\\boxed{1}"},
                ]
            )
        },
        {
            "messages": json.dumps(
                [
                    {"role": "user", "content": "2+2"},
                    {"role": "assistant", "content": "\\boxed{4}"},
                ]
            )
        },
    ]
    pd.DataFrame(rows).to_parquet(path, index=False)
    ds = prepare_sft_dataset(str(path), FakeTokenizer(), max_length=32)
    assert len(ds) == 1
