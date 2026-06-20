"""Stage-0b FULL build: mint the confidence redirect/verify SFT corpus from the FULL
student rollout dump (HF metacot-rv) via the teacher (TRAPI/Entra), CONCURRENTLY.

No GPU. The student rollouts were dumped once by scripts/rollout_dump.py; here we
replay them (dump-replay rollout_fn) and run the teacher distill + causal/structural
gate over EVERY easy/medium problem. The full pool is ~4171 problems and multi-anchor
redirect makes this ~1.6e4 teacher calls, so build_dataset runs a thread pool.

Result (redirect + verify SFT rows) is uploaded to the CLEAN metacot-rv repo (the
old metacot repo is hf_xet-tainted — see memory hf-xet-upload-pitfall) with the
verified uploader so a silent-404 cannot masquerade as success.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from huggingface_hub import hf_hub_download

from scripts.build_confidence_redirect_verify_sft import build_dataset, make_trapi_teacher_fn
from scripts.rollout_dump import _upload_verified

DUMP_REPO = "iamseungpil/metacot-rv"
DUMP_PATH = "data/rv_rollout_full.parquet"
OUT_LOCAL = "/tmp/rv_confidence_sft.parquet"
OUT_PATH = "data/rv_confidence_sft.parquet"
MAX_WORKERS = 12
N_ROLLOUTS = 8


def main() -> None:
    p = hf_hub_download(DUMP_REPO, DUMP_PATH, repo_type="dataset", token=os.environ["HF_TOKEN"])
    df = pd.read_parquet(p)
    print(f"[build] full dump: {len(df)} problems | mean pass_rate={df['pass_rate'].mean():.3f}")

    # dump-replay rollout_fn (read-only dict -> thread-safe).
    dump = {r["question"]: (list(r["rollouts"]), list(r["grades"]), list(r["answers"]))
            for _, r in df.iterrows()}

    def rollout_fn(question, gold, n):
        t, g, a = dump[question]
        k = min(n, len(t))
        return [(t[i], bool(g[i]), a[i]) for i in range(k)]

    problems = [{"question": r["question"], "gold": r["gold"],
                 "tags": {"difficulty": r["difficulty"]}} for _, r in df.iterrows()]

    teacher = make_trapi_teacher_fn()
    # pre-warm the lazy TRAPI client (single-threaded) so the worker threads don't
    # race to build it / fetch Entra tokens 12x at once.
    try:
        teacher({"question": "warmup: 1+1?", "gold": "2", "confidence": 0.9,
                 "arm": "verify", "wrong_prefix": "1+1=2"})
        print("[build] teacher client warmed")
    except Exception as e:  # pragma: no cover - network warm-up only
        print(f"[build] warmup call raised (continuing): {e}")

    print(f"[build] building {len(problems)} problems, max_workers={MAX_WORKERS} (TRAPI)...")
    summary = build_dataset(problems, rollout_fn, teacher, out_path=OUT_LOCAL,
                            n_rollouts=N_ROLLOUTS, max_workers=MAX_WORKERS)
    print("[build] SUMMARY:", json.dumps(summary, default=str))
    print(f"[build] kept: redirect={summary['kept_redirect']} verify={summary['kept_verify']} "
          f"(rows={summary['kept_rows']})")

    _upload_verified(OUT_LOCAL, DUMP_REPO, OUT_PATH)
    print(f"[build] uploaded -> hf://{DUMP_REPO}/{OUT_PATH}")


if __name__ == "__main__":
    main()
