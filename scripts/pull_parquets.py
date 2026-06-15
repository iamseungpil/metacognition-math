#!/usr/bin/env python
"""Pull verl_*_redirect[_base].parquet from iamseungpil/metacot-sdc-data into data/."""
import os
import pathlib
from huggingface_hub import hf_hub_download

REPO = "iamseungpil/metacot-sdc-data"
TOKEN = os.environ.get("HF_TOKEN", "")
OUT = pathlib.Path("/scratch/metacognition/data")
OUT.mkdir(parents=True, exist_ok=True)

for fn in [
    "verl_train_redirect.parquet",
    "verl_val_redirect.parquet",
    "verl_train_redirect_base.parquet",
    "verl_val_redirect_base.parquet",
    "verl_train_meta_mix.parquet",
    "verl_val_meta_mix.parquet",
]:
    print(f"[pull] {fn}")
    src = hf_hub_download(repo_id=REPO, repo_type="dataset", filename=fn, token=TOKEN, local_dir=str(OUT))
    print(f"[pull] -> {src} ({pathlib.Path(src).stat().st_size/1e6:.1f} MB)")

print("[pull] done")
