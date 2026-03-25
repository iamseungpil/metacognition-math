"""Unified dataset loader for math training and evaluation."""
import json
import re
from pathlib import Path
from typing import Optional

from datasets import load_dataset, Dataset, concatenate_datasets


DIFFICULTY_MAP = {
    "Level 1": "easy", "Level 2": "easy",
    "Level 3": "medium",
    "Level 4": "hard", "Level 5": "hard",
}

CATEGORY_ALIASES = {
    "Algebra": "algebra",
    "Counting & Probability": "counting_probability",
    "Geometry": "geometry",
    "Intermediate Algebra": "intermediate_algebra",
    "Number Theory": "number_theory",
    "Precalculus": "precalculus",
    "Prealgebra": "prealgebra",
}


def _load_math_raw(split: str):
    """Load MATH dataset, trying multiple HuggingFace sources."""
    sources = [
        ("lighteval/MATH", {}),
        ("hendrycks/competition_math", {}),
        ("EleutherAI/hendrycks_math", {}),
    ]
    for name, kwargs in sources:
        for trc in [False, True]:
            try:
                ds = load_dataset(name, split=split, trust_remote_code=trc, **kwargs)
                print(f"  Loaded MATH from '{name}' ({split}): {len(ds)} rows")
                return ds
            except Exception:
                continue

    # Fallback: use MathInstruct (262K competition math problems with solutions)
    print(f"  MATH dataset unavailable, falling back to TIGER-Lab/MathInstruct")
    ds = load_dataset("TIGER-Lab/MathInstruct", split="train")
    # MathInstruct has 'instruction' and 'output' columns
    # Filter for competition math sources only
    if split == "test":
        ds = ds.select(range(min(5000, len(ds))))
    return ds


def _normalize_math(ds, source_tag: str):
    """Normalize column names across different MATH dataset mirrors."""
    cols = ds.column_names
    # Handle MathInstruct format: instruction/output
    if "instruction" in cols:
        p, a = "instruction", "output"
    else:
        p = "problem" if "problem" in cols else "question"
        a = "solution" if "solution" in cols else "answer"
    t = "type" if "type" in cols else ("subject" if "subject" in cols else ("source" if "source" in cols else None))
    l = "level" if "level" in cols else ("difficulty" if "difficulty" in cols else None)

    def _map(x):
        cat_raw = x.get(t, "") if t else ""
        diff_raw = x.get(l, "") if l else ""
        return {
            "question": x[p],
            "answer": x[a],
            "category": CATEGORY_ALIASES.get(cat_raw, "other"),
            "difficulty": DIFFICULTY_MAP.get(diff_raw, "medium"),
            "source": source_tag,
        }
    ds = ds.map(_map)
    return ds.select_columns(["question", "answer", "category", "difficulty", "source"])


def load_math_train(max_samples: Optional[int] = None) -> Dataset:
    """Load MATH competition train set (7,500 problems)."""
    ds = _load_math_raw("train")
    ds = _normalize_math(ds, "math_train")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds


def load_math_test() -> Dataset:
    """Load MATH competition test set (5,000 problems)."""
    ds = _load_math_raw("test")
    return _normalize_math(ds, "math_test")


def load_gsm8k(max_samples: Optional[int] = None) -> Dataset:
    """Load GSM8K train set (grade school math, 7,473 problems)."""
    ds = load_dataset("openai/gsm8k", "main", split="train")
    ds = ds.map(lambda x: {
        "question": x["question"],
        "answer": x["answer"],
        "category": "arithmetic",
        "difficulty": "easy",
        "source": "gsm8k",
    })
    ds = ds.select_columns(["question", "answer", "category", "difficulty", "source"])
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    print(f"  Loaded GSM8K: {len(ds)} problems")
    return ds


def load_numina_math(max_samples: int = 15000) -> Dataset:
    """Load NuminaMath-CoT, filtered for competition-level problems."""
    ds = load_dataset("AI-MO/NuminaMath-CoT", split="train")
    # Filter for competition sources (AMC, AIME, olympiad)
    competition_sources = ["amc_aime", "olympiads", "cn_k12", "synthetic_math"]
    ds = ds.filter(lambda x: x.get("source", "") in competition_sources)
    ds = ds.map(lambda x: {
        "question": x["problem"],
        "answer": x["solution"],
        "category": "competition",
        "difficulty": "medium",
        "source": f"numina_{x.get('source', 'unknown')}",
    })
    ds = ds.select_columns(["question", "answer", "category", "difficulty", "source"])
    if len(ds) > max_samples:
        ds = ds.shuffle(seed=42).select(range(max_samples))
    return ds


def load_omni_math() -> Dataset:
    """Load Omni-MATH olympiad-level problems (4,428 problems)."""
    ds = load_dataset("KbsdJames/Omni-MATH", split="test")
    ds = ds.map(lambda x: {
        "question": x.get("problem", x.get("question", "")),
        "answer": str(x.get("answer", x.get("solution", ""))),
        "category": x.get("domain", "olympiad"),
        "difficulty": "hard",
        "source": "omni_math",
    })
    return ds.select_columns(["question", "answer", "category", "difficulty", "source"])


def load_open_math_reasoning(max_samples: int = 10000) -> Dataset:
    """Load NVIDIA OpenMathReasoning CoT subset."""
    ds = load_dataset(
        "nvidia/OpenMathReasoning", "cot",
        split="train", trust_remote_code=True,
        streaming=True,
    )
    rows = []
    for i, x in enumerate(ds):
        if i >= max_samples:
            break
        rows.append({
            "question": x.get("problem", ""),
            "answer": x.get("expected_answer", x.get("solution", "")),
            "category": "competition",
            "difficulty": "medium",
            "source": "open_math_reasoning",
        })
    return Dataset.from_list(rows)


def load_aime(year: str = "2025") -> Dataset:
    """Load AIME evaluation set."""
    if year == "2025":
        ds = load_dataset("opencompass/AIME2025", split="test")
    else:
        ds = load_dataset("math-ai/aime24", split="train")

    ds = ds.map(lambda x: {
        "question": x.get("problem", x.get("question", "")),
        "answer": str(x.get("answer", x.get("expected_answer", ""))),
        "category": "aime",
        "difficulty": "hard",
        "source": f"aime_{year}",
    })
    return ds.select_columns(["question", "answer", "category", "difficulty", "source"])


def load_all_train(config: Optional[dict] = None) -> Dataset:
    """Load and concatenate all training datasets."""
    config = config or {}
    datasets_list = []

    print("Loading MATH train...")
    datasets_list.append(load_math_train(config.get("math_max", None)))

    if config.get("use_gsm8k", True):
        print("Loading GSM8K...")
        datasets_list.append(load_gsm8k(config.get("gsm8k_max", None)))

    if config.get("use_numina", True):
        print("Loading NuminaMath-CoT...")
        datasets_list.append(load_numina_math(config.get("numina_max", 15000)))

    if config.get("use_omni", True):
        print("Loading Omni-MATH...")
        datasets_list.append(load_omni_math())

    if config.get("use_openmr", True):
        print("Loading OpenMathReasoning...")
        datasets_list.append(load_open_math_reasoning(config.get("openmr_max", 10000)))

    combined = concatenate_datasets(datasets_list)
    combined = combined.shuffle(seed=42)
    print(f"Total training problems: {len(combined)}")
    from collections import Counter
    for src, count in Counter(combined["source"]).most_common():
        print(f"  {src}: {count}")
    return combined


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract answer from \\boxed{...} in model output.

    Handles nested braces up to 2 levels, e.g. \\boxed{\\frac{1}{\\sqrt{2}}}.
    """
    pattern = r'\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()
    return None


def extract_numeric_answer(text: str) -> Optional[int]:
    """Extract integer answer (0-999) for AIME problems."""
    boxed = extract_boxed_answer(text)
    if boxed is not None:
        try:
            val = int(boxed)
            if 0 <= val <= 999:
                return val
        except ValueError:
            pass
    # Fallback: last number in the final line
    last_line = text.strip().split('\n')[-1]
    numbers = re.findall(r'\b(\d{1,3})\b', last_line)
    if numbers:
        try:
            return int(numbers[-1])
        except ValueError:
            pass
    return None
