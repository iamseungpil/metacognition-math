"""Generate control-oriented Meta-CoT v5 SFT data via TRAPI."""
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
from src.metacot.prompt_control_v5 import (
    META_END,
    META_START,
    CONTROL_V5_SYSTEM_PROMPT,
    build_control_v5_prompt,
)


def _extract_boxed_answer(solution: str) -> str:
    matches = re.findall(r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}", solution or "")
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
        topic = domain[0] if isinstance(domain, list) and domain else str(domain)
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
    med_pool = _load_hendrycks_rows(
        per_subject=max((args.straight_medium + args.verify_medium + args.redirect_medium) * args.oversample_factor, 40),
        hard_only=False,
    )
    hard_math_pool = _load_hendrycks_rows(
        per_subject=max((args.verify_hard + args.redirect_hard) * args.oversample_factor, 30),
        hard_only=True,
    )
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


def _meta_blocks(text: str):
    return re.findall(rf"{re.escape(META_START)}(.*?){re.escape(META_END)}", text, re.DOTALL)


def _parse_confidences(text: str):
    confs = re.findall(r"confidence[:\s]+(\d+\.\d+|\d+)", text, re.IGNORECASE)
    vals = []
    for conf in confs:
        value = float(conf)
        if value > 1.0:
            value /= 100.0
        vals.append(value)
    return vals


def _parse_study_need(text: str) -> str:
    match = re.search(r"study_need:\s*(.+)", text, re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _has_mathy_content(text: str) -> bool:
    return bool(
        re.search(
            r"\\\[|\\\(|=|\\frac|\\boxed|x\^|y\^|z\^|\d+\s*[\+\-\*/]",
            text,
            re.IGNORECASE,
        )
    )


def _has_cot_planning(text: str) -> bool:
    return bool(
        re.search(
            r"\b(first,|second,|third,|finally,|let me compute|compute|calculate|substitute|plug back|recompute|"
            r"I'll solve|I will solve|expand|simplify|differentiate|integrate)\b",
            text,
            re.IGNORECASE,
        )
    )


def _has_verify_call(text: str) -> bool:
    return bool(
        re.search(
            r"\b(verify|independent check|double-check|I should check|before finalizing|before committing)\b",
            text,
            re.IGNORECASE,
        )
    )


def _text_after_last_meta(text: str) -> str:
    idx = text.rfind(META_END)
    if idx == -1:
        return text
    return text[idx + len(META_END):]


def _has_effective_verify(text: str) -> bool:
    verify_region = _text_after_last_meta(text)
    return bool(
        re.search(
            r"\b(substitute|plug(?:ging)? back|recomput|recalculat|sanity check|independent check|"
            r"check|verification|confirm|matches|consistent)\b",
            verify_region,
            re.IGNORECASE,
        )
    )


def _has_conflict_signal(text: str) -> bool:
    return bool(
        re.search(
            r"\b(something feels off|this feels off|that seems off|inconsistent|contradiction|mismatch|unsupported|"
            r"forcing|not exposing the constraint|route is weak|route fails|stalled|not making progress)\b",
            text,
            re.IGNORECASE,
        )
    )


def _has_overconfidence_signal(text: str) -> bool:
    return bool(
        re.search(
            r"\b(overconfiden|overcommit|committing too quickly|too quickly|too certain|too sure|"
            r"confidence is outrunning the support|support is thinner than the confidence|"
            r"about to commit without an independent check|answer came too quickly|"
            r"might be committing too quickly|risk of overcommitting|single route|"
            r"single straightforward route|single familiar route|recognition alone|over-trusting|"
            r"committing without checking|ready to commit)\b",
            text,
            re.IGNORECASE,
        )
    )


def _has_diagnosis_signal(text: str) -> bool:
    return bool(
        re.search(
            r"\b(the issue is|the problem is|this route is weak because|this route fails because|"
            r"what is missing is|I am missing|I'm missing|not lack of algebra|not lack of calculation|"
            r"wrong perspective|perspective is missing|constraint that actually controls|"
            r"does not expose|does not control|does not explain)\b",
            text,
            re.IGNORECASE,
        )
    )


def _has_failure_decomposition(text: str) -> bool:
    return bool(
        re.search(
            r"\b(missing skill|missing perspective|missing object|missing structure|"
            r"the failure is|the bottleneck is|the blocker is|"
            r"this is not a calculation problem|this is not an algebra problem|"
            r"need a structural view|need a constraint-based view|need an invariant)\b",
            text,
            re.IGNORECASE,
        )
    )


def _has_next_strategy_signal(text: str) -> bool:
    return bool(
        re.search(
            r"\b(switch strategy|switch to|change perspective|use a constraint-based analysis|"
            r"use parity|use modular|use an invariant|verify before finalizing|study before retrying|"
            r"redirect to|rewrite using|reframe around|reparameteriz|use the determinant rule|"
            r"abandon the raw trig route|stop expanding and switch)\b",
            text,
            re.IGNORECASE,
        )
    )


def _meta_is_pure(block: str) -> bool:
    return not _has_mathy_content(block) and not _has_cot_planning(block)


def _validate(text: str, scenario: str):
    reasons = []
    if "\\boxed{" not in text:
        return False, {"reasons": ["missing_boxed_answer"]}

    meta_blocks = _meta_blocks(text)
    meta_count = len(meta_blocks)
    confs = _parse_confidences(text)
    all_blocks_have_conf = meta_blocks and all(re.search(r"confidence[:\s]+(\d+\.\d+|\d+)", b, re.IGNORECASE) for b in meta_blocks)
    pure_meta = all(_meta_is_pure(block) for block in meta_blocks)
    has_study_need = any(_parse_study_need(block) for block in meta_blocks)
    has_verify_call = any(_has_verify_call(block) for block in meta_blocks)
    has_effective_verify = _has_effective_verify(text)
    has_trigger = any(_has_conflict_signal(block) for block in meta_blocks)
    has_overconfidence = any(_has_overconfidence_signal(block) for block in meta_blocks)
    has_diagnosis = any(_has_diagnosis_signal(block) for block in meta_blocks)
    has_failure_decomposition = any(_has_failure_decomposition(block) for block in meta_blocks)
    has_next_strategy = any(_has_next_strategy_signal(block) for block in meta_blocks)
    has_low_conf = any(conf <= 0.55 for conf in confs)
    has_high_conf = any(conf >= 0.75 for conf in confs)
    has_drop = any(b <= a - 0.08 for a, b in zip(confs, confs[1:]))
    repeated_intervention = meta_count >= 2

    if scenario == "straight":
        valid = meta_count == 0 or (
            meta_count == 1
            and pure_meta
            and has_high_conf
            and has_overconfidence
            and has_verify_call
            and has_effective_verify
            and not has_trigger
        )
        if meta_count > 1:
            reasons.append("straight_too_many_meta")
        if meta_count == 1 and not has_overconfidence:
            reasons.append("straight_meta_without_overconfidence")
    elif scenario == "verify":
        valid = (
            meta_count >= 1
            and pure_meta
            and all_blocks_have_conf
            and has_high_conf
            and has_overconfidence
            and has_verify_call
            and has_effective_verify
            and not has_trigger
            and not has_study_need
        )
        if meta_count == 0:
            reasons.append("verify_missing_meta")
        if meta_count >= 1 and not has_overconfidence:
            reasons.append("verify_missing_overconfidence_signal")
        if meta_count >= 1 and not has_effective_verify:
            reasons.append("verify_missing_effective_check")
    else:
        valid = (
            meta_count >= 1
            and pure_meta
            and all_blocks_have_conf
            and has_trigger
            and has_low_conf
            and (has_diagnosis or has_failure_decomposition)
            and has_next_strategy
        )
        if meta_count == 0:
            reasons.append("redirect_missing_meta")
        if meta_count >= 1 and not has_trigger:
            reasons.append("redirect_missing_trigger")
        if meta_count >= 1 and not (has_diagnosis or has_failure_decomposition):
            reasons.append("redirect_missing_diagnosis")
        if meta_count >= 1 and not has_next_strategy:
            reasons.append("redirect_missing_strategy_switch")

    if meta_blocks and not all_blocks_have_conf:
        reasons.append("missing_confidence_line")
    if meta_blocks and not pure_meta:
        reasons.append("meta_not_pure")

    return valid, {
        "has_verify": has_effective_verify,
        "has_switch": has_next_strategy,
        "has_conf_drop": has_drop or has_low_conf,
        "has_trigger": has_trigger,
        "has_overconfidence": has_overconfidence,
        "has_diagnosis": has_diagnosis,
        "has_blocker": has_failure_decomposition,
        "has_decomposition": has_failure_decomposition,
        "has_next_strategy": has_next_strategy,
        "repeated_intervention": repeated_intervention,
        "meta_count": meta_count,
        "pure_meta": pure_meta,
        "has_study_need": has_study_need,
        "study_need": _parse_study_need(meta_blocks[0]) if meta_blocks else "",
        "trigger": "anomaly" if has_trigger else ("overconfidence" if has_overconfidence else ""),
        "reasons": sorted(set(reasons)),
    }


def _generate_one(client, row, model_name, max_retries):
    prompt = build_control_v5_prompt(
        question=row["question"],
        scenario=row["scenario"],
        difficulty=row["difficulty"],
        pass_rate=row["pass_rate"],
        source=row["source"],
        topic=row["topic"],
    )
    last_invalid = {"reasons": ["no_attempt_made"]}
    for attempt in range(max_retries):
        try:
            resp = client.responses.create(
                model=model_name,
                input=[
                    {"role": "system", "content": CONTROL_V5_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_output_tokens=4096,
            )
            text = resp.output_text or ""
            valid, stats = _validate(text, row["scenario"])
            if valid:
                return text, stats, None
            last_invalid = {
                "text": text,
                "stats": stats,
                "attempt": attempt + 1,
            }
        except Exception as exc:
            if attempt == max_retries - 1:
                return "", {"error": str(exc), "reasons": ["exception"]}, {
                    "text": "",
                    "stats": {"error": str(exc), "reasons": ["exception"]},
                    "attempt": attempt + 1,
                }
            wait = min(90, 5 * (2 ** attempt)) + random.uniform(0, 3)
            time.sleep(wait)
    return "", last_invalid.get("stats", {}), last_invalid


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
    parser.add_argument("--output", default="data/control_v5_trapi_round1.parquet")
    parser.add_argument("--rejections-output", default="")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = _load_question_pool(args)
    target_counts = {(scenario, difficulty): n for scenario, difficulty, n in _bucket_specs(args)}
    client = get_trapi_client()
    print(f"Generating {len(rows)} control-v5 chains with {args.model}")

    records = []
    rejections = []
    failed = 0
    with ThreadPoolExecutor(max_workers=args.concurrent) as ex:
        futures = {ex.submit(_generate_one, client, row, args.model, args.max_retries): row for row in rows}
        for idx, future in enumerate(as_completed(futures), 1):
            row = futures[future]
            text, stats, rejection = future.result()
            if not text:
                failed += 1
                if rejection is not None:
                    rejections.append(
                        {
                            "scenario": row["scenario"],
                            "difficulty": row["difficulty"],
                            "source": row["source"],
                            "topic": row["topic"],
                            "attempt": rejection.get("attempt", 0),
                            "reasons": rejection.get("stats", {}).get("reasons", []),
                            "text": rejection.get("text", ""),
                        }
                    )
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
                        "has_overconfidence": stats.get("has_overconfidence", False),
                        "has_diagnosis": stats.get("has_diagnosis", False),
                        "has_blocker": stats.get("has_blocker", False),
                        "has_decomposition": stats.get("has_decomposition", False),
                        "has_next_strategy": stats.get("has_next_strategy", False),
                        "repeated_intervention": stats.get("repeated_intervention", False),
                        "trigger": stats.get("trigger", ""),
                        "meta_count": stats.get("meta_count", 0),
                        "pure_meta": stats.get("pure_meta", False),
                        "has_study_need": stats.get("has_study_need", False),
                        "study_need": stats.get("study_need", ""),
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
    if args.rejections_output:
        rej_path = Path(args.rejections_output)
        rej_path.parent.mkdir(parents=True, exist_ok=True)
        with rej_path.open("w", encoding="utf-8") as f:
            for row in rejections:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("=== Generation Complete ===")
    print(f"Saved {len(df)} rows to {out_path}")
    if len(df):
        print(df.groupby(["scenario", "difficulty"]).size().to_string())
    if rejections:
        reason_counts = {}
        for row in rejections:
            for reason in row.get("reasons", []):
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        if reason_counts:
            print("=== Rejection Reasons ===")
            for reason, count in sorted(reason_counts.items(), key=lambda x: (-x[1], x[0])):
                print(f"{reason}: {count}")


if __name__ == "__main__":
    main()
