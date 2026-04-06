#!/usr/bin/env python3
"""Check or repair the remote runtime used for control-v5 RL."""
from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from importlib import metadata


REQUIRED = {
    "trl": ("trl", "trl==0.19.1", "0.19.1"),
    "deepspeed": ("deepspeed", "deepspeed", None),
    "accelerate": ("accelerate", "accelerate", None),
    "sklearn": ("scikit-learn", "scikit-learn", None),
    "pandas": ("pandas", "pandas", None),
    "wandb": ("wandb", "wandb", None),
    "transformers": ("transformers", "transformers==4.52.3", "4.52.3"),
    "peft": ("peft", "peft", None),
    "datasets": ("datasets", "datasets", None),
    "pyarrow": ("pyarrow", "pyarrow", None),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--install-missing", action="store_true")
    args = parser.parse_args()

    missing = []
    for module, (dist_name, package, expected_version) in REQUIRED.items():
        try:
            importlib.import_module(module)
            installed_version = metadata.version(dist_name)
            if expected_version is not None and installed_version != expected_version:
                print(f"{module}:VERSION_MISMATCH:{installed_version}:expected={expected_version}")
                missing.append(package)
            else:
                print(f"{module}:OK:{installed_version}")
        except Exception as exc:  # pragma: no cover - runtime dependent
            print(f"{module}:MISSING:{type(exc).__name__}:{exc}")
            missing.append(package)

    if missing and args.install_missing:
        print("Installing missing packages:", " ".join(missing))
        subprocess.run(
            [sys.executable, "-m", "pip", "install", *missing],
            check=True,
        )
        return main()

    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
