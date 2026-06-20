"""Stage-0b FULL build: mint the confidence redirect/verify SFT corpus from the FULL
student rollout dump (HF metacot-rv) via the teacher (TRAPI/Entra), CONCURRENTLY.

No GPU. The student rollouts were dumped once by scripts/rollout_dump.py; here we
replay them and run the teacher distill + causal/structural gate over EVERY easy/
medium problem (~4171, ~1.6e4 teacher calls with multi-anchor redirect).

CHUNKED + RESUMABLE: a single in-process run grew to ~117GB RSS and was OOM-killed
(the data itself is only ~12MB, so the growth was per-client/concurrency state under
12 threads). So process the pool in CHUNKS, writing each chunk's parquet to disk and
SKIPPING chunks already on disk (resume after a kill). The teacher client is rebuilt
per chunk and gc runs between chunks so no per-client state accumulates across the
whole pool. Lower concurrency (6) caps peak memory. Final = concat of all chunks,
uploaded to the CLEAN metacot-rv repo (metacot is hf_xet-tainted) with the verified
uploader so a silent-404 cannot masquerade as success.
"""
from __future__ import annotations

import os
# sympy caches every expression it builds in an unbounded GLOBAL cache; over ~1.6e4
# grades that alone accumulates. Disable it BEFORE any import pulls sympy/math_verify.
os.environ.setdefault("SYMPY_USE_CACHE", "no")

import gc
import json
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
CHUNK_DIR = Path("/tmp/rv_chunks")
CHUNK_SIZE = 250
MAX_WORKERS = 6
N_ROLLOUTS = 8


def _flush(*a):
    print(*a, flush=True)


def main() -> None:
    p = hf_hub_download(DUMP_REPO, DUMP_PATH, repo_type="dataset", token=os.environ["HF_TOKEN"])
    df = pd.read_parquet(p)
    _flush(f"[build] full dump: {len(df)} problems | mean pass_rate={df['pass_rate'].mean():.3f}")
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)

    n_chunks = (len(df) + CHUNK_SIZE - 1) // CHUNK_SIZE
    agg: dict = {}
    for ci in range(n_chunks):
        cpath = CHUNK_DIR / f"chunk_{ci:03d}.parquet"
        spath = CHUNK_DIR / f"chunk_{ci:03d}.summary.json"
        if cpath.exists() and spath.exists():  # RESUME: skip a finished chunk
            s = json.loads(spath.read_text())
            for k, v in s.items():
                if isinstance(v, (int, float)):
                    agg[k] = agg.get(k, 0) + v
            _flush(f"[build] chunk {ci + 1}/{n_chunks}: SKIP (done) kept_r={s.get('kept_redirect')}")
            continue

        sub = df.iloc[ci * CHUNK_SIZE:(ci + 1) * CHUNK_SIZE]
        dump = {r["question"]: (list(r["rollouts"]), list(r["grades"]), list(r["answers"]))
                for _, r in sub.iterrows()}

        def rollout_fn(question, gold, n, _d=dump):
            t, g, a = _d[question]
            k = min(n, len(t))
            return [(t[i], bool(g[i]), a[i]) for i in range(k)]

        problems = [{"question": r["question"], "gold": r["gold"],
                     "tags": {"difficulty": r["difficulty"]}} for _, r in sub.iterrows()]

        teacher = make_trapi_teacher_fn()  # FRESH client per chunk (no cross-chunk state)
        s = build_dataset(problems, rollout_fn, teacher, out_path=str(cpath),
                          n_rollouts=N_ROLLOUTS, max_workers=MAX_WORKERS)
        spath.write_text(json.dumps(s, default=str))
        for k, v in s.items():
            if isinstance(v, (int, float)):
                agg[k] = agg.get(k, 0) + v
        _flush(f"[build] chunk {ci + 1}/{n_chunks}: kept_r={s['kept_redirect']} "
               f"kept_v={s['kept_verify']} teach_err={s['dropped_teacher_error']} "
               f"| cum_r={agg.get('kept_redirect', 0)} cum_v={agg.get('kept_verify', 0)}")

        del dump, problems, teacher, sub
        try:  # belt-and-suspenders: drop any sympy cache that slipped through
            from sympy.core.cache import clear_cache
            clear_cache()
        except Exception:
            pass
        gc.collect()

    # concat all chunks -> final corpus.
    parts = [pd.read_parquet(CHUNK_DIR / f"chunk_{ci:03d}.parquet") for ci in range(n_chunks)]
    final = pd.concat(parts, ignore_index=True)
    final.to_parquet(OUT_LOCAL)
    agg["kept_rows"] = len(final)
    _flush("[build] SUMMARY: " + json.dumps(agg, default=str))
    _flush(f"[build] kept: redirect={agg.get('kept_redirect', 0)} "
           f"verify={agg.get('kept_verify', 0)} rows={len(final)} "
           f"teacher_error={agg.get('dropped_teacher_error', 0)}")

    _upload_verified(OUT_LOCAL, DUMP_REPO, OUT_PATH)
    _flush(f"[build] uploaded -> hf://{DUMP_REPO}/{OUT_PATH}")


if __name__ == "__main__":
    main()
