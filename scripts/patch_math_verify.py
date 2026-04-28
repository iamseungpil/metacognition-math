#!/usr/bin/env python
"""Patch math_verify timeout decorator for Ray non-main-thread workers.

The default decorator calls signal.signal(SIGALRM,...) which raises
ValueError("signal only works in main thread") inside Ray worker threads,
silently failing every math comparison and stuck-at-(-1) correctness reward.
"""
import pathlib
import sys

P = pathlib.Path(
    "/opt/conda/envs/simplerl/lib/python3.10/site-packages/math_verify/utils.py"
)

OLD = """            def wrapper(*args, **kwargs):
                old_handler = signal.getsignal(signal.SIGALRM)
                signal.signal(signal.SIGALRM, handler)
                signal.alarm(timeout_seconds)
                try:
                    return func(*args, **kwargs)
                finally:
                    # Cancel the alarm and restore previous handler
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)"""

NEW = """            def wrapper(*args, **kwargs):
                import threading
                if threading.current_thread() is not threading.main_thread():
                    return func(*args, **kwargs)
                old_handler = signal.getsignal(signal.SIGALRM)
                signal.signal(signal.SIGALRM, handler)
                signal.alarm(timeout_seconds)
                try:
                    return func(*args, **kwargs)
                finally:
                    # Cancel the alarm and restore previous handler
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)"""


def main() -> None:
    if not P.exists():
        print(f"[patch] {P} missing — math_verify not installed yet")
        sys.exit(1)
    s = P.read_text()
    if "threading.main_thread" in s:
        print("[patch] math_verify already patched")
        return
    if OLD not in s:
        print("[patch] PATTERN NOT FOUND — math_verify version may differ", file=sys.stderr)
        sys.exit(1)
    P.write_text(s.replace(OLD, NEW))
    print(f"[patch] math_verify patched at {P}")


if __name__ == "__main__":
    main()
