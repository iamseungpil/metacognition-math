"""Generate control-oriented Meta-CoT v4 SFT data via TRAPI."""
import argparse
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.metacot.generator import get_trapi_client
from src.metacot.prompt_control_v4 import (
    META_START,
    META_END,
    CONTROL_V4_SYSTEM_PROMPT,
    build_control_v4_prompt,
)


def _extract_boxed_answer(solution: str) -> str:
    matches = re.findall(r'\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}', solution or "")
    return matches[-1].strip() if matches else ""


def _load_gsm8k_rows(max_rows: int):
    ds = load_dataset("openai/gsm8k", "main", split="train")
    rows = list(ds)
    random.shuffle(rows)
    out = []
    for row in rows[:max_rows]:
        answer = row["answer"].split("####")[-1].strip() if "####" in row["answer"] else row["answer"]
        out.append(
            {
                "question": row["question"],
                "gold_answer": answer,
                "difficulty": "easy",
                "pass_rate": 0.85,
                "source": "gsm8k",
                "topic": "arithmetic_word_problem",
            }
        )
    return out


def _load_hendrycks_rows(per_subject: int, hard_only: bool):
    subjects = [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    ]
    rows = []
    for subject in subjects:
        ds = list(load_dataset("EleutherAI/hendrycks_math", subject, split="train"))
        random.shuffle(ds)
        picked = 0
        for row in ds:
            level = row.get("level", "")
            if hard_only and level not in {"Level 4", "Level 5"}:
                continue
            if not hard_only and level in {"Level 4", "Level 5"}:
                continue
            gold = _extract_boxed_answer(str(row.get("solution", "")))
            if not gold:
                continue
            difficulty = "hard" if hard_only else "medium"
            pass_rate = 0.28 if hard_only else 0.52
            rows.append(
                {
                    "question": row["problem"],
                    "gold_answer": gold,
                    "difficulty": difficulty,
                    "pass_rate": pass_rate,
                    "source": f"hendrycks_math/{subject}",
                    "topic": subject,
                }
            )
            picked += 1
            if picked >= per_subject:
                break
    random.shuffle(rows)
    return rows


def _load_omni_rows(max_rows: int):
    ds = list(load_dataset("KbsdJames/Omni-MATH", split="test"))
    random.shuffle(ds)
    rows = []
    for row in ds[:max_rows]:
        diff = row.get("difficulty", 8)
        domain = row.get("domain", ["olympiad"])
        if isinstance(domain, list):
            topic = domain[0] if domain else "olympiad"
        else:
            topic = str(domain)
        rows.append(
            {
                "question": row["problem"],
                "gold_answer": str(row.get("answer", "")),
                "difficulty": "hard",
                "pass_rate": max(0.05, min(0.3, 0.34 - 0.03 * float(diff))),
                "source": "omni-math",
                "topic": topic,
            }
        )
    return rows


def _disjoint_take(pool, n, used_questions):
    picked = []
    for row in pool:
        q = row["question"]
        if q in used_questions:
            continue
        picked.append(row)
        used_questions.add(q)
        if len(picked) >= n:
            break
    return picked


def _bucket_specs(args):
    return [
        ("straight", "easy", args.straight_easy),
        ("verify", "easy", args.verify_easy),
        ("straight", "medium", args.straight_medium),
        ("verify", "medium", args.verify_medium),
        ("redirect", "medium", args.redirect_medium),
        ("verify", "hard", args.verify_hard),
        ("redirect", "hard", args.redirect_hard),
    ]


def _load_question_pool(args):
    random.seed(args.seed)
    used = set()

    gsm_pool = _load_gsm8k_rows(max_rows=max((args.straight_easy + args.verify_easy) * args.oversample_factor, 50) * 4)
    med_pool = _load_hendrycks_rows(per_subject=max((args.straight_medium + args.verify_medium + args.redirect_medium) * args.oversample_factor, 40), hard_only=False)
    hard_math_pool = _load_hendrycks_rows(per_subject=max((args.verify_hard + args.redirect_hard) * args.oversample_factor, 30), hard_only=True)
    omni_pool = _load_omni_rows(max_rows=max((args.verify_hard + args.redirect_hard) * args.oversample_factor, 40) * 3)
    hard_pool = hard_math_pool + omni_pool
    random.shuffle(gsm_pool)
    random.shuffle(med_pool)
    random.shuffle(hard_pool)

    selected = []

    bucket_specs = [
        ("straight", gsm_pool, args.straight_easy, "easy"),
        ("verify", gsm_pool, args.verify_easy, "easy"),
        ("straight", med_pool, args.straight_medium, "medium"),
        ("verify", med_pool, args.verify_medium, "medium"),
        ("redirect", med_pool, args.redirect_medium, "medium"),
        ("verify", hard_pool, args.verify_hard, "hard"),
        ("redirect", hard_pool, args.redirect_hard, "hard"),
    ]

    for scenario, pool, n, difficulty in bucket_specs:
        rows = _disjoint_take(pool, max(n * args.oversample_factor, n), used)
        selected.extend([{**row, "scenario": scenario} for row in rows])

    random.shuffle(selected)
    return selected


def _extract_field(text: str, name: str) -> str:
    pattern = rf'{name}:\s*(.+?)(?:\n[A-Za-z_]+:|\n{META_END}|$)'
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def _has_effective_verify(text: str) -> bool:
    return bool(
        re.search(
            r'\b(substitute|plug(?:ging)? back|recomput|recalculat|sanity check|independent check|verify by|check by)\b',
            text,
            re.IGNORECASE,
        )
    )


def _has_switch_signal(text: str) -> bool:
    return bool(
        re.search(
            r'\b(switch(?:ing)?|different method|alternative approach|case split|reframe|instead use|next_strategy)\b',
            text,
            re.IGNORECASE,
        )
    )


def _meta_blocks(text: str):
    return re.findall(rf'{re.escape(META_START)}(.*?){re.escape(META_END)}', text, re.DOTALL)


def _parse_confidences(text: str):
    confs = re.findall(r'confidence[:\s]+(\d+\.\d+|\d+)', text, re.IGNORECASE)
    vals = []
    for conf in confs:
        value = float(conf)
        if value > 1.0:
            value /= 100.0
        vals.append(value)
    return vals


def _has_low_confidence(text: str, threshold: float = 0.55) -> bool:
    confs = _parse_confidences(text)
    return any(conf <= threshold for conf in confs)


def _has_conflict_signal(text: str) -> bool:
    return bool(
        re.search(
            r'\b(something feels off|this feels off|that seems off|this seems off|not consistent|inconsistent|contradiction|'
            r'doesn\'t satisfy|does not satisfy|fails|mismatch|unsupported|too early|forcing|cannot be right|can\'t be right|'
            r'I should not trust this|I don\'t trust this route|anomaly)\b',
            text,
            re.IGNORECASE,
        )
    )


def _has_diagnosis_signal(text: str) -> bool:
    return bool(
        re.search(
            r'\b(the issue is|the problem is|this route fails because|this route is weak because|'
            r'I may be forcing|I am forcing|I committed too early|I overcommitted|I am missing|'
            r'I\'m missing|this only checks|this does not control|this does not explain|'
            r'the current route|the real task is|not needed here|unnecessary|would be unnecessary|'
            r'would be weak|too indirect|too complicated|solve the wrong problem|'
            r'not the game structure|can hide a mismatch|does not match the structure)\b',
            text,
            re.IGNORECASE,
        )
    )


def _has_decomposition_signal(text: str) -> bool:
    return bool(
        re.search(
            r'\b(break this into|split the task into|first[, ]|second[, ]|then[, ]|finally[, ]|'
            r'step back and|I should check|I should first|subgoal|reduced condition|'
            r'identify the invariant|handle the cases)\b',
            text,
            re.IGNORECASE,
        )
    )


def _has_next_strategy_signal(text: str) -> bool:
    return bool(
        re.search(
            r'\b(switch to|instead I\'ll|instead I will|better to use|reframe this as|'
            r'case split|use a parity|use an invariant|use a counting argument|'
            r'use a direct check|different method|alternative approach)\b',
            text,
            re.IGNORECASE,
        )
    )


def _validate(text: str, scenario: str):
    if "\\boxed{" not in text:
        return False, {}

    if scenario == "straight":
        meta_count = text.count(META_START)
        has_conflict = _has_conflict_signal(text)
        has_switch = _has_switch_signal(text)
        valid = not has_conflict and not has_switch and meta_count <= 1
        return valid, {
            "has_verify": _has_effective_verify(text),
            "has_switch": has_switch,
            "has_conf_drop": False,
            "has_trigger": has_conflict,
            "has_diagnosis": _has_diagnosis_signal(text),
            "has_blocker": False,
            "has_decomposition": _has_decomposition_signal(text),
            "has_next_strategy": _has_next_strategy_signal(text),
            "meta_count": meta_count,
            "repeated_intervention": meta_count >= 2,
            "trigger": "anomaly" if has_conflict else "",
        }

    if META_START not in text or META_END not in text:
        return False, {"meta_count": 0}

    confs = _parse_confidences(text)
    has_drop = any(b <= a - 0.08 for a, b in zip(confs, confs[1:]))
    has_low_conf = _has_low_confidence(text)
    has_verify = _has_effective_verify(text)
    has_switch = _has_switch_signal(text) or _has_next_strategy_signal(text)
    has_trigger = _has_conflict_signal(text)
    has_diagnosis = _has_diagnosis_signal(text)
    has_decomposition = _has_decomposition_signal(text)
    has_next_strategy = _has_next_strategy_signal(text)
    meta_count = text.count(META_START)
    repeated_intervention = meta_count >= 2
    all_blocks_have_conf = all(
        re.search(r'confidence[:\s]+(\d+\.\d+|\d+)', block, re.IGNORECASE)
        for block in _meta_blocks(text)
    )

    if scenario == "verify":
        valid = has_verify and len(confs) >= 1 and all_blocks_have_conf and not has_switch
    else:
        valid = (
            has_trigger
            and (has_diagnosis or has_decomposition or has_next_strategy)
            and (has_switch or has_next_strategy)
            and (has_drop or has_low_conf)
            and len(confs) >= 1
            and all_blocks_have_conf
        )

    return valid, {
        "has_verify": has_verify,
        "has_switch": has_switch,
        "has_conf_drop": has_drop or has_low_conf,
        "has_trigger": has_trigger,
        "has_diagnosis": has_diagnosis,
        "has_blocker": False,
        "has_decomposition": has_decomposition,
        "has_next_strategy": has_next_strategy,
        "meta_count": meta_count,
        "repeated_intervention": repeated_intervention,
        "trigger": "anomaly" if has_trigger else "",
    }


def _generate_one(client, row, model_name, max_retries):
    prompt = build_control_v4_prompt(
        question=row["question"],
        scenario=row["scenario"],
        difficulty=row["difficulty"],
        pass_rate=row["pass_rate"],
        source=row["source"],
        topic=row["topic"],
    )
    for attempt in range(max_retries):
        try:
            resp = client.responses.create(
                model=model_name,
                input=[
                    {"role": "system", "content": CONTROL_V4_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_output_tokens=4096,
            )
            text = resp.output_text or ""
            valid, stats = _validate(text, row["scenario"])
            if valid:
                return text, stats
        except Exception as exc:
            if attempt == max_retries - 1:
                return "", {"error": str(exc)}
            wait = min(90, 5 * (2 ** attempt)) + random.uniform(0, 3)
            time.sleep(wait)
    return "", {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--straight-easy", type=int, default=30)
    parser.add_argument("--verify-easy", type=int, default=30)
    parser.add_argument("--straight-medium", type=int, default=40)
    parser.add_argument("--verify-medium", type=int, default=40)
    parser.add_argument("--redirect-medium", type=int, default=40)
    parser.add_argument("--verify-hard", type=int, default=60)
    parser.add_argument("--redirect-hard", type=int, default=60)
    parser.add_argument("--concurrent", type=int, default=12)
    parser.add_argument("--oversample-factor", type=int, default=3)
    parser.add_argument("--model", default="gpt-5.4_2026-03-05")
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--output", default="data/control_v4_trapi_round1.parquet")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = _load_question_pool(args)
    target_counts = {(scenario, difficulty): n for scenario, difficulty, n in _bucket_specs(args)}
    client = get_trapi_client()
    print(f"Generating {len(rows)} control-v4 chains with {args.model}")

    records = []
    failed = 0
    with ThreadPoolExecutor(max_workers=args.concurrent) as ex:
        futures = {ex.submit(_generate_one, client, row, args.model, args.max_retries): row for row in rows}
        for idx, future in enumerate(as_completed(futures), 1):
            row = futures[future]
            text, stats = future.result()
            if not text:
                failed += 1
            else:
                messages = json.dumps(
                    [
                        {"role": "user", "content": row["question"]},
                        {"role": "assistant", "content": text},
                    ],
                    ensure_ascii=False,
                )
                records.append(
                    {
                        "messages": messages,
                        "scenario": row["scenario"],
                        "difficulty": row["difficulty"],
                        "source": row["source"],
                        "topic": row["topic"],
                        "pass_rate": row["pass_rate"],
                        "has_verify": stats.get("has_verify", False),
                        "has_switch": stats.get("has_switch", False),
                        "has_conf_drop": stats.get("has_conf_drop", False),
                        "has_trigger": stats.get("has_trigger", False),
                        "has_diagnosis": stats.get("has_diagnosis", False),
                        "has_blocker": stats.get("has_blocker", False),
                        "has_decomposition": stats.get("has_decomposition", False),
                        "has_next_strategy": stats.get("has_next_strategy", False),
                        "repeated_intervention": stats.get("repeated_intervention", False),
                        "trigger": stats.get("trigger", ""),
                        "meta_count": stats.get("meta_count", 0),
                    }
                )
            if idx % 25 == 0:
                print(f"  {idx}/{len(rows)} processed, valid={len(records)}, failed={failed}", flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    if len(df):
        trimmed = []
        for (scenario, difficulty), limit in target_counts.items():
            bucket = df[(df["scenario"] == scenario) & (df["difficulty"] == difficulty)].copy()
            if len(bucket) > limit:
                bucket = bucket.sample(n=limit, random_state=args.seed)
            trimmed.append(bucket)
        df = pd.concat(trimmed, ignore_index=True) if trimmed else df
    df.to_parquet(out_path, index=False)
    print("=== Generation Complete ===")
    print(f"Saved {len(df)} rows to {out_path}")
    if len(df):
        print(df.groupby(["scenario", "difficulty"]).size().to_string())


if __name__ == "__main__":
    main()
