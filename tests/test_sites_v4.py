"""E-133 수리(0907) 회귀 테스트.

1. `run_arm.sh --dry-run` 이 `SITES_DIR` env 를 세 mixed 분기(OPT_CF/mixed_cf,
   OPT_M 류/mixed opt, 그 외 mixed) 전부에서 존중하는지 — 기본값(sites_v1)은
   "지금까지"와 바이트 동일해야 하고, `SITES_DIR=sites_v4` 를 주면 그 디렉터리
   아래에서 읽어야 한다.
2. `check_no_val_overlap.py` 가 겹침이 있으면 exit!=0, 없으면 exit 0 인지 —
   합성 parquet 두 개(겹침 있음/없음)로 직접 검사한다(GPU/실 데이터 불필요).
3. `$WORK/data/sites_v4/` 산출물이 이미 있으면(실행됐으면) 그 위에서
   check_no_val_overlap 을 다시 돌려 0 겹침을 확인한다 — 없으면 스킵.

GPU 를 쓰지 않는다.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_ARM = ROOT / "scripts" / "local" / "run_arm.sh"
CHECK = ROOT / "scripts" / "local" / "check_no_val_overlap.py"

PY = os.environ.get("PY", "/hdd_data/seungpil/envs/simplerl/bin/python")


def _dry_run(env_extra: dict, args: list[str]) -> str:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", **env_extra)
    r = subprocess.run(
        ["bash", str(RUN_ARM), *args, "--dry-run"],
        cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=60,
    )
    assert r.returncode == 0, f"run_arm.sh --dry-run failed: rc={r.returncode}\n{r.stdout}\n{r.stderr}"
    return r.stdout + r.stderr


def test_sites_dir_default_matches_v1():
    out = _dry_run({"MIXED_DATA": "mixed_train_v3c"}, ["OPT_CF", "1", "100", "opt"])
    lines = [l for l in out.splitlines() if l.startswith("[run_arm] data.train_files=")]
    assert lines, out
    assert lines[0].endswith("sites_v1/mixed_train_v3c_cf_opt.parquet")


def test_sites_dir_v4_opt_cf():
    out = _dry_run({"SITES_DIR": "sites_v4", "MIXED_DATA": "mixed_train_v4"}, ["OPT_CF", "1", "100", "opt"])
    lines = [l for l in out.splitlines() if l.startswith("[run_arm] data.train_files=")]
    assert lines, out
    assert lines[0].endswith("sites_v4/mixed_train_v4_cf_opt.parquet")


def test_sites_dir_v4_opt_m():
    out = _dry_run({"SITES_DIR": "sites_v4", "MIXED_DATA": "mixed_train_v4"}, ["OPT_M", "1", "100", "opt"])
    lines = [l for l in out.splitlines() if l.startswith("[run_arm] data.train_files=")]
    assert lines, out
    assert lines[0].endswith("sites_v4/mixed_train_v4_opt.parquet")


def test_sites_dir_v4_m0():
    out = _dry_run({"SITES_DIR": "sites_v4", "MIXED_DATA": "mixed_train_v4"}, ["M0", "1", "100", "opt"])
    lines = [l for l in out.splitlines() if l.startswith("[run_arm] data.train_files=")]
    assert lines, out
    assert lines[0].endswith("sites_v4/mixed_train_v4.parquet")


def _have_pandas() -> bool:
    return Path(PY).exists()


def test_check_no_val_overlap_synthetic(tmp_path):
    if not _have_pandas():
        import pytest
        pytest.skip(f"python env not found: {PY}")
    import json

    script = f"""
import pandas as pd
val = pd.DataFrame({{"nums": [[1,2,3,4], [5,6,7,8]], "target": [10, 20]}})
val.to_parquet("{tmp_path}/val.parquet", index=False)
clean = pd.DataFrame({{"nums": [[9,9,9,9]], "target": [99]}})
clean.to_parquet("{tmp_path}/clean.parquet", index=False)
dirty = pd.DataFrame({{"nums": [[1,2,3,4]], "target": [10]}})
dirty.to_parquet("{tmp_path}/dirty.parquet", index=False)
"""
    r = subprocess.run([PY, "-c", script], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr

    r_clean = subprocess.run(
        [PY, str(CHECK), "--val", f"{tmp_path}/val.parquet", f"{tmp_path}/clean.parquet"],
        capture_output=True, text=True,
    )
    assert r_clean.returncode == 0, r_clean.stdout + r_clean.stderr

    r_dirty = subprocess.run(
        [PY, str(CHECK), "--val", f"{tmp_path}/val.parquet", f"{tmp_path}/dirty.parquet"],
        capture_output=True, text=True,
    )
    assert r_dirty.returncode != 0


def test_sites_v4_outputs_no_val_overlap_if_present():
    """실 산출물이 있으면(build_sites_v4.sh 가 이미 돌았으면) 0 겹침을 재확인한다.
    없으면(롤아웃 잡이 아직 안 끝났으면) 스킵 — 이 테스트는 산출물 존재를 요구하지 않는다.
    """
    import pytest

    work = Path(os.environ.get("WORK", "/hdd_data/seungpil/scratch"))
    out_dir = work / "data" / "sites_v4"
    files = [out_dir / n for n in (
        "sites_train.parquet", "sites_judge.parquet", "mixed_train_v4.parquet",
        "mixed_train_v4_opt.parquet", "mixed_train_v4_cf_opt.parquet",
    )]
    if not all(f.exists() for f in files):
        pytest.skip("sites_v4 outputs not built yet")
    if not _have_pandas():
        pytest.skip(f"python env not found: {PY}")

    val = work / "data" / "countdown_val_4num_opt.parquet"
    r = subprocess.run(
        [PY, str(CHECK), "--val", str(val), *[str(f) for f in files]],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr
