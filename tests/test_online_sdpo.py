import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.self_distill.online import (
    load_online_problems,
    write_online_sdpo_outputs,
)


def test_load_online_problems_single_question():
    problems = load_online_problems(question="2+2?", gold_answer="4")
    assert len(problems) == 1
    assert problems[0].question == "2+2?"
    assert problems[0].gold_answer == "4"


def test_write_online_sdpo_outputs_builds_sdpo_regen_artifact(tmp_path):
    rows = [
        {
            "question": "Solve x+7=12.",
            "gold_answer": "5",
            "benchmark": "aime2024",
            "root_completion": "I guessed \\boxed{4}",
            "root_analysis": {"diagnosis_text": "Guessing is weak.", "study_need": "direct isolation"},
            "root_judgment": {"is_correct": False},
            "trigger_fired": True,
            "curriculum_retry": {
                "retry_completion": "Subtract 7 from both sides, so x=5. \\boxed{5}",
                "retry_judgment": {"is_correct": True},
                "meta_transition": {"confidence_gain": 0.3, "trigger_cleared": True},
                "retrieved": [{"question": "Solve x+4=9.", "source": "stable_seed_library", "score": 0.9}],
            },
        }
    ]
    payload = write_online_sdpo_outputs(rows=rows, output_dir=tmp_path)
    assert payload["num_rollouts"] == 1
    assert payload["num_sdpo_regen_rows"] == 1
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["summary"]["feedback_available_rate"] == 1.0
    assert summary["summary"]["sdpo_prompt_rate"] == 1.0


def test_write_online_outputs_claim_bearing_tracks_selector_contract(tmp_path):
    rows = [
        {
            "question": "Solve x+3=7.",
            "gold_answer": "4",
            "benchmark": "math500",
            "generation_mode": "fixed_k_repair",
            "root_completion": "I guessed \\boxed{5}",
            "root_analysis": {"diagnosis_text": "Guessing is weak.", "study_need": "direct isolation"},
            "selected_completion": "<|meta|>\nconfidence: 0.42\nThe earlier guess was unsupported.\nstudy_need: direct isolation\n<|/meta|>\nSubtract 3 from both sides, so x=4. \\boxed{4}",
            "selected_judgment": {"is_correct": True},
            "repair_candidates": [{"candidate_id": "repair_0"}, {"candidate_id": "repair_1"}],
            "selector": {
                "selected_candidate_id": "repair_1",
                "selected_score": 1.2,
                "selected_breakdown": {"correctness": 1.0, "total": 1.2},
                "score_margin": 0.3,
            },
        }
    ]
    payload = write_online_sdpo_outputs(
        rows=rows,
        output_dir=tmp_path,
        source_tag="online_fixed_k_repair",
        mode="epistemic",
        claim_bearing=True,
    )
    assert payload["dataset_mode"] == "epistemic"
    assert payload["claim_bearing"] is True
    assert payload["num_dataset_rows"] == 1
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["summary"]["synthetic_meta_injected_rate"] == 0.0
    assert summary["retrieval"]["rows_with_retrieval_enabled"] == 0
    assert summary["retrieval"]["retrieval_nonempty_rate"] == 0.0


def test_write_online_outputs_reports_retrieval_contract(tmp_path):
    rows = [
        {
            "question": "Solve x+3=7.",
            "gold_answer": "4",
            "benchmark": "math500",
            "generation_mode": "fixed_k_repair",
            "root_completion": "I guessed \\boxed{5}",
            "root_analysis": {"diagnosis_text": "Guessing is weak.", "study_need": "direct isolation"},
            "selected_completion": "<|meta|>\nconfidence: 0.42\nThe earlier guess was unsupported.\nstudy_need: direct isolation\n<|/meta|>\nSubtract 3 from both sides, so x=4. \\boxed{4}",
            "selected_judgment": {"is_correct": True},
            "repair_candidates": [{"candidate_id": "repair_0"}],
            "selector": {"selected_candidate_id": "repair_0", "selected_score": 1.0},
            "retriever_active": True,
            "retrieval_enabled": True,
            "retrieval_nonempty": True,
            "retrieval_mode_requested": "question_only",
            "retrieval_mode_used": "question_only",
            "retrieved": [
                {
                    "score": 0.9,
                    "score_breakdown": {"question": 0.9},
                    "question": "Solve x+4=8.",
                    "source": "seed_bank",
                    "answer": "4",
                }
            ],
        }
    ]
    payload = write_online_sdpo_outputs(
        rows=rows,
        output_dir=tmp_path,
        source_tag="online_fixed_k_repair",
        mode="epistemic",
        claim_bearing=True,
    )
    assert payload["retrieval"]["rows_with_retriever"] == 1
    assert payload["retrieval"]["rows_with_retrieval_enabled"] == 1
    assert payload["retrieval"]["rows_with_nonempty_retrieval"] == 1
    assert payload["retrieval"]["requested_modes"] == ["question_only"]


def test_write_online_outputs_claim_bearing_refuses_empty_dataset(tmp_path):
    rows = [
        {
            "question": "Solve x+3=7.",
            "gold_answer": "4",
            "generation_mode": "fixed_k_repair",
            "root_completion": "I guessed \\boxed{5}",
            "root_analysis": {"diagnosis_text": "Guessing is weak.", "study_need": "direct isolation"},
            "selected_completion": "Subtract 3 from both sides, so x=4. \\boxed{4}",
            "repair_candidates": [{"candidate_id": "repair_0"}],
            "selector": {"selected_candidate_id": "repair_0", "selected_score": 1.0},
        }
    ]
    with pytest.raises(ValueError):
        write_online_sdpo_outputs(
            rows=rows,
            output_dir=tmp_path,
            source_tag="online_fixed_k_repair",
            mode="epistemic",
            claim_bearing=True,
        )
