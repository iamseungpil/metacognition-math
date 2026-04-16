"""Quality-control summaries and stratified samples for control-v5 data."""
import argparse
import json
import random
import re

import pandas as pd


META_START = "<|meta|>"
META_END = "<|/meta|>"


def _meta_blocks(text: str) -> list[str]:
    return re.findall(rf"{re.escape(META_START)}(.*?){re.escape(META_END)}", text, re.DOTALL)


def _has_mathy_content(text: str) -> bool:
    return bool(re.search(r"\\\[|\\\(|=|\\frac|\\boxed|x\^|y\^|z\^|\d+\s*[\+\-\*/]", text, re.IGNORECASE))


def _has_planlike_content(text: str) -> bool:
    return bool(
        re.search(
            r"\b(first,|second,|third,|finally,|let me compute|compute|calculate|substitute|plug back|"
            r"I'll solve|I will solve|expand|simplify|differentiate|integrate|set up the equation|case 1|case 2)\b",
            text,
            re.IGNORECASE,
        )
    )


def _parse_confidences(text: str) -> list[float]:
    values = []
    for match in re.findall(r"confidence[:\s]+(\d+\.\d+|\d+)", text, re.IGNORECASE):
        value = float(match)
        if value > 1.0:
            value /= 100.0
        values.append(value)
    return values


def _parse_study_need(text: str) -> str:
    match = re.search(r"study_need:\s*(.+)", text, re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _load_answer(row) -> str:
    messages = json.loads(row["messages"])
    return messages[-1]["content"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--samples-per-bucket", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    df = pd.read_parquet(args.input)
    if df.empty:
        print("Dataset is empty.")
        return

    answers = df.apply(_load_answer, axis=1)
    meta_blocks = answers.apply(_meta_blocks)
    meta_text = meta_blocks.apply(lambda blocks: "\n".join(blocks))
    confs = answers.apply(_parse_confidences)

    qc = df.copy()
    qc["answer"] = answers
    qc["meta_text"] = meta_text
    qc["meta_has_math"] = meta_text.apply(_has_mathy_content)
    qc["meta_has_planlike"] = meta_text.apply(_has_planlike_content)
    qc["study_need_text"] = meta_text.apply(_parse_study_need)
    qc["min_conf"] = confs.apply(lambda xs: min(xs) if xs else None)
    qc["max_conf"] = confs.apply(lambda xs: max(xs) if xs else None)

    print("=== COUNTS ===")
    print(qc.groupby(["scenario", "difficulty"]).size().to_string())
    print()

    print("=== QC METRICS ===")
    print(f"rows={len(qc)}")
    print(f"pure_meta_rate={1.0 - qc['meta_has_math'].mean() - qc['meta_has_planlike'].mean() + (qc['meta_has_math'] & qc['meta_has_planlike']).mean():.3f}")
    print(f"meta_has_math_rate={qc['meta_has_math'].mean():.3f}")
    print(f"meta_has_planlike_rate={qc['meta_has_planlike'].mean():.3f}")
    if "pure_meta" in qc.columns:
        print(f"validator_pure_meta_rate={qc['pure_meta'].mean():.3f}")
    if "has_study_need" in qc.columns:
        print(f"has_study_need_rate={qc['has_study_need'].mean():.3f}")
    redirect = qc[qc["scenario"] == "redirect"]
    if len(redirect):
        print(f"redirect_study_need_rate={redirect['study_need_text'].astype(bool).mean():.3f}")
        print(f"redirect_diagnosis_rate={redirect.get('has_diagnosis', pd.Series(dtype=bool)).mean():.3f}")
        print(f"redirect_switch_rate={redirect.get('has_next_strategy', pd.Series(dtype=bool)).mean():.3f}")
        print(f"redirect_low_conf_rate={redirect['min_conf'].apply(lambda x: x is not None and x <= 0.55).mean():.3f}")
    verify = qc[qc["scenario"] == "verify"]
    if len(verify):
        print(f"verify_high_conf_rate={verify['max_conf'].apply(lambda x: x is not None and x >= 0.75).mean():.3f}")
        print(f"verify_overconfidence_rate={verify.get('has_overconfidence', pd.Series(dtype=bool)).mean():.3f}")
    print()

    for (scenario, difficulty), group in qc.groupby(["scenario", "difficulty"]):
        print(f"=== {scenario} / {difficulty} ===")
        sample_n = min(args.samples_per_bucket, len(group))
        sample_df = group.sample(n=sample_n, random_state=args.seed) if sample_n else group
        for _, row in sample_df.iterrows():
            print(
                {
                    "source": row.get("source"),
                    "topic": row.get("topic"),
                    "meta_count": row.get("meta_count"),
                    "pure_meta": row.get("pure_meta"),
                    "has_trigger": row.get("has_trigger"),
                    "has_overconfidence": row.get("has_overconfidence"),
                    "has_diagnosis": row.get("has_diagnosis"),
                    "has_next_strategy": row.get("has_next_strategy"),
                    "study_need": row.get("study_need"),
                }
            )
            print(row["answer"][:1800].replace("\n", " "))
            print()


if __name__ == "__main__":
    main()
