"""Generate behavior-first Meta-CoT SFT data via TRAPI."""
import argparse
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.metacot.prompt_behavior import (
    META_START,
    META_END,
    BEHAVIOR_SYSTEM_PROMPT,
    build_behavior_prompt,
)
from src.metacot.generator import get_trapi_client


def _extract_math_answer(row):
    answer = row.get("answer")
    if answer:
        return str(answer)
    solution = str(row.get("solution", ""))
    boxed = re.findall(r'\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}', solution)
    if boxed:
        return boxed[-1].strip()
    return solution


def _load_question_pool(n_per_scenario):
    random.seed(42)
    pools = {"straight": [], "verify": [], "redirect": []}

    gsm = load_dataset("openai/gsm8k", "main", split="train")
    gsm_rows = list(gsm)
    random.shuffle(gsm_rows)
    for row in gsm_rows[: n_per_scenario * 2]:
        ans = row["answer"].split("####")[-1].strip() if "####" in row["answer"] else row["answer"]
        pools["straight"].append(
            {
                "question": row["question"],
                "gold_answer": ans,
                "difficulty": "easy",
                "pass_rate": 0.85,
                "source": "gsm8k",
            }
        )
        pools["verify"].append(
            {
                "question": row["question"],
                "gold_answer": ans,
                "difficulty": "easy",
                "pass_rate": 0.78,
                "source": "gsm8k",
            }
        )

    math_rows = []
    math_configs = [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    ]
    for cfg in math_configs:
        ds = load_dataset("EleutherAI/hendrycks_math", cfg, split="train")
        cfg_rows = list(ds)
        random.shuffle(cfg_rows)
        for row in cfg_rows[: max(n_per_scenario, 250)]:
            gt = _extract_math_answer(row)
            if not gt:
                continue
            math_rows.append(
                {
                    "question": row["problem"],
                    "gold_answer": gt,
                    "difficulty": "hard",
                    "pass_rate": 0.35,
                    "source": f"hendrycks_math/{cfg}",
                }
            )

    random.shuffle(math_rows)
    pools["redirect"].extend(math_rows[: n_per_scenario * 2])
    pools["verify"].extend(math_rows[n_per_scenario * 2 : n_per_scenario * 3])

    selected = []
    for scenario, rows in pools.items():
        random.shuffle(rows)
        selected.extend([{**row, "scenario": scenario} for row in rows[:n_per_scenario]])
    random.shuffle(selected)
    return selected


def _validate(text, scenario):
    if META_START not in text or META_END not in text or "\\boxed{" not in text:
        return False, {}

    has_verify = bool(
        re.search(
            r'\b(substitute|plug back|recomput|recalculat|sanity check|verify by|check by)\b',
            text,
            re.IGNORECASE,
        )
    )
    has_switch = bool(
        re.search(
            r'\b(switch_method|switch to|different method|alternative approach|instead use|another method)\b',
            text,
            re.IGNORECASE,
        )
    )
    confs = re.findall(r'confidence(?:_before|_after)?[:\s]+(\d+\.\d+|\d+)', text, re.IGNORECASE)
    confs = [float(c) if float(c) <= 1.0 else float(c) / 100.0 for c in confs]
    has_drop = any(b <= a - 0.08 for a, b in zip(confs, confs[1:]))

    ok = True
    if scenario == "verify":
        ok = has_verify and len(confs) >= 1
    elif scenario == "redirect":
        ok = has_switch and has_drop and len(confs) >= 2

    return ok, {
        "has_verify": has_verify,
        "has_switch": has_switch,
        "has_conf_drop": has_drop,
        "num_confidences": len(confs),
        "meta_count": text.count(META_START),
    }


def _generate_one(client, row, model_name, max_retries):
    prompt = build_behavior_prompt(
        question=row["question"],
        scenario=row["scenario"],
        difficulty=row["difficulty"],
        pass_rate=row["pass_rate"],
    )
    for attempt in range(max_retries):
        try:
            resp = client.responses.create(
                model=model_name,
                input=[
                    {"role": "system", "content": BEHAVIOR_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
            )
            text = resp.output_text or ""
            valid, stats = _validate(text, row["scenario"])
            if valid:
                return text, stats
        except Exception as e:
            wait = min(90, 5 * (2 ** attempt)) + random.uniform(0, 3)
            if attempt == max_retries - 1:
                return "", {"error": str(e)}
            time.sleep(wait)
    return "", {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-scenario", type=int, default=800)
    parser.add_argument("--concurrent", type=int, default=16)
    parser.add_argument("--model", default="gpt-5.4_2026-03-05")
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--output", default="data/metacot_behavior_trapi.parquet")
    args = parser.parse_args()

    rows = _load_question_pool(args.n_per_scenario)
    client = get_trapi_client()
    print(f"Generating {len(rows)} behavior-first chains with {args.model}")

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
                        "has_verify": stats.get("has_verify", False),
                        "has_switch": stats.get("has_switch", False),
                        "has_conf_drop": stats.get("has_conf_drop", False),
                        "meta_count": stats.get("meta_count", 0),
                    }
                )
            if idx % 50 == 0:
                print(f"  {idx}/{len(rows)} processed, valid={len(records)}, failed={failed}", flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_parquet(out_path, index=False)
    print("=== Generation Complete ===")
    print(f"Saved {len(df)} rows to {out_path}")
    if len(df):
        print(df.groupby('scenario').size().to_string())


if __name__ == "__main__":
    main()
