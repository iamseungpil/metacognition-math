#!/usr/bin/env python
"""Patch math_verify's timeout wrapper to skip signal.SIGALRM in non-main threads.

ROOT CAUSE: math_verify's timeout decorator builds an inner ``wrapper`` that calls
``signal.signal(signal.SIGALRM, ...)``. ``signal.signal`` only works in the main
thread, so inside verl_sdc's Ray RewardLoopWorker THREADS it raises
``ValueError("signal only works in main thread")``. math_verify swallows that, so
``verify()`` silently returns False for CORRECT answers (and floods logs). Passing
``parsing_timeout=None`` / ``timeout_seconds=None`` at the verify()/parse() API does
NOT help on the deployed (older) math_verify version, because the per-extraction
``compare_single_extraction_wrapper`` (grader.py) still goes through this signal
wrapper regardless of the public timeout args.

FIX: insert a guard at the top of the timeout wrapper so that off the main thread it
just calls the wrapped function directly (no signal, no alarm) — symbolic comparison
then runs correctly in worker threads. Version-tolerant: locates math_verify from the
ACTIVE interpreter (not a hardcoded path) and inserts the guard via a line scan that
only touches the wrapper that actually uses signal.SIGALRM.

Idempotent. Exits 0 on success or already-patched; non-zero only if math_verify is
absent or no signal wrapper is found (so the bootstrap can surface it loudly).
"""
import pathlib
import sys


def _utils_path() -> pathlib.Path:
    import math_verify.utils as u

    return pathlib.Path(u.__file__)


def main() -> None:
    try:
        path = _utils_path()
    except Exception as exc:  # noqa: BLE001
        print(f"[patch] math_verify not importable: {exc}", file=sys.stderr)
        sys.exit(1)

    if not path.exists():
        print(f"[patch] {path} missing", file=sys.stderr)
        sys.exit(1)

    src = path.read_text()
    if "_mv_threading.main_thread()" in src:
        print(f"[patch] already patched at {path}")
        return

    lines = src.splitlines(keepends=True)
    out = []
    patched = 0
    for i, line in enumerate(lines):
        out.append(line)
        stripped = line.lstrip()
        # Identify the inner timeout wrapper: `def wrapper(*args, **kwargs):`
        if stripped.startswith("def wrapper(*args, **kwargs):"):
            # Only patch it if signal.SIGALRM is used within the next ~10 lines.
            window = "".join(lines[i + 1 : i + 11])
            if "signal.SIGALRM" in window:
                body_indent = line[: len(line) - len(stripped)] + "    "
                out.append(
                    f"{body_indent}import threading as _mv_threading\n"
                    f"{body_indent}if _mv_threading.current_thread() is not "
                    f"_mv_threading.main_thread():\n"
                    f"{body_indent}    return func(*args, **kwargs)\n"
                )
                patched += 1

    if patched == 0:
        print(
            "[patch] no signal-based timeout wrapper found — math_verify layout "
            f"may differ at {path}",
            file=sys.stderr,
        )
        sys.exit(1)

    path.write_text("".join(out))
    print(f"[patch] math_verify patched ({patched} wrapper(s)) at {path}")


if __name__ == "__main__":
    main()
