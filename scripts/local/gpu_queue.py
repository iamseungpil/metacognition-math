#!/usr/bin/env python3
"""scripts/local/gpu_queue.py — a tiny file-based job queue + one worker per GPU.

Layout (fixed): /hdd_data/seungpil/queue/{pending,running,done,failed}/
Job file: JSON, e.g. {"cmd": "...", "name": "...", "gpus": 1, "priority": 10}

Design notes:
  * Atomicity: a job moves pending -> running via os.rename (same filesystem,
    atomic), so two workers racing on the same file never both "win" it — one
    rename succeeds, the other raises FileNotFoundError/OSError and moves on.
  * Per-GPU exclusivity: `worker --gpu N` takes an flock() on
    <queue_dir>/gpu_N.lock for the duration of running ONE job, so two worker
    processes pinned to the same GPU index never run jobs concurrently on it
    (this is what --jobs-per-gpu 1, the default, relies on). Pass
    --jobs-per-gpu >1 to run multiple worker processes against the same GPU
    with NO lock (explicitly requested in the task spec for future
    higher-density eval jobs) — the caller is responsible for those jobs
    actually fitting in one GPU's memory concurrently.
  * Disk guard: before starting a job, check shutil.disk_usage on /hdd_data and
    on / ; refuse to start (log "[DISK] waiting", requeue via not consuming the
    job — i.e. leave it in pending, actually since we already renamed it into
    running we move it BACK to pending) if /hdd_data usage > 85% or / free < 15GB.
  * A file named <queue_dir>/STOP pauses ALL workers (they poll and sleep)
    without killing them, so `start-workers` doesn't need to be re-run after
    unpausing.

This is intentionally dependency-free (stdlib only) so it runs inside the
simplerl conda env or the system python equally.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

QUEUE_ROOT = Path("/hdd_data/seungpil/queue")
STATES = ("pending", "running", "done", "failed")
WORK = Path(os.environ.get("WORK", "/hdd_data/seungpil/scratch"))
LOG_DIR = WORK / "logs"
PIDS_DIR = QUEUE_ROOT / "pids"

DISK_USAGE_PCT_MAX = 85.0
ROOT_FREE_GB_MIN = 15.0


def _ensure_dirs() -> None:
    for s in STATES:
        (QUEUE_ROOT / s).mkdir(parents=True, exist_ok=True)
    PIDS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


# ── submit ────────────────────────────────────────────────────────────────
def cmd_submit(args: argparse.Namespace) -> int:
    _ensure_dirs()
    job_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
    job = {
        "id": job_id,
        "cmd": args.cmd,
        "name": args.name,
        "gpus": args.gpus,
        "priority": args.priority,
        "submitted_at": _now(),
    }
    path = QUEUE_ROOT / "pending" / f"{job_id}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(job, indent=2))
    tmp.rename(path)  # atomic within pending/
    print(f"[submit] {job_id} name={args.name!r} priority={args.priority} -> {path}")
    return 0


# ── status ────────────────────────────────────────────────────────────────
def _gpu_memory_table(gpu_indices) -> list[dict]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            text=True, timeout=10,
        )
    except Exception as exc:  # nvidia-smi missing, no GPU visible, etc.
        return [{"index": i, "memory": f"<nvidia-smi failed: {exc}>"} for i in gpu_indices]
    rows = []
    for line in out.strip().splitlines():
        idx, used, total = (p.strip() for p in line.split(","))
        if int(idx) in gpu_indices:
            rows.append({"index": int(idx), "used_mb": int(used), "total_mb": int(total)})
    return rows


def cmd_status(args: argparse.Namespace) -> int:
    _ensure_dirs()
    counts = {}
    for s in STATES:
        files = sorted((QUEUE_ROOT / s).glob("*.json"))
        counts[s] = files

    print(f"queue root: {QUEUE_ROOT}")
    stop_flag = QUEUE_ROOT / "STOP"
    print(f"STOP flag present: {stop_flag.exists()}")
    print()
    for s in STATES:
        files = counts[s]
        print(f"== {s} ({len(files)}) ==")
        for f in files[:20]:
            try:
                job = json.loads(f.read_text())
            except Exception as exc:
                print(f"  {f.name}: <unreadable: {exc}>")
                continue
            extra = ""
            if s == "done":
                extra = f" rc={job.get('exit_code')} finished_at={job.get('finished_at')}"
            elif s == "failed":
                extra = f" rc={job.get('exit_code')} finished_at={job.get('finished_at')}"
            elif s == "running":
                extra = f" gpu={job.get('assigned_gpu')} started_at={job.get('started_at')}"
            print(f"  {f.stem} name={job.get('name')!r} priority={job.get('priority')}{extra}")
        if len(files) > 20:
            print(f"  ... and {len(files) - 20} more")
        print()

    print("== GPUs 0-3 (nvidia-smi memory.used/memory.total MiB) ==")
    for row in _gpu_memory_table({0, 1, 2, 3}):
        print(f"  {row}")

    du_hdd = shutil.disk_usage("/hdd_data") if Path("/hdd_data").exists() else None
    du_root = shutil.disk_usage("/")
    if du_hdd:
        pct = 100.0 * du_hdd.used / du_hdd.total
        print(f"\n/hdd_data usage: {pct:.1f}% ({du_hdd.used/1e9:.0f}GB / {du_hdd.total/1e9:.0f}GB)")
    print(f"/ free: {du_root.free/1e9:.1f}GB")
    return 0


# ── disk guard ────────────────────────────────────────────────────────────
def _disk_ok() -> tuple[bool, str]:
    try:
        du_hdd = shutil.disk_usage("/hdd_data")
        pct = 100.0 * du_hdd.used / du_hdd.total
        if pct > DISK_USAGE_PCT_MAX:
            return False, f"/hdd_data at {pct:.1f}% > {DISK_USAGE_PCT_MAX}%"
    except FileNotFoundError:
        pass
    du_root = shutil.disk_usage("/")
    free_gb = du_root.free / 1e9
    if free_gb < ROOT_FREE_GB_MIN:
        return False, f"/ free {free_gb:.1f}GB < {ROOT_FREE_GB_MIN}GB"
    return True, "ok"


# ── worker ────────────────────────────────────────────────────────────────
def _pick_job() -> Path | None:
    """Return a pending job file sorted by (-priority, submitted order), or None."""
    pending = sorted((QUEUE_ROOT / "pending").glob("*.json"))
    if not pending:
        return None

    def sort_key(p: Path):
        try:
            job = json.loads(p.read_text())
            return (-int(job.get("priority", 0)), p.name)
        except Exception:
            return (0, p.name)

    pending.sort(key=sort_key)
    return pending[0]


def _claim(job_path: Path) -> Path | None:
    """Atomically move a pending job file to running/. Returns the new path, or
    None if another worker already claimed it (race lost)."""
    dest = QUEUE_ROOT / "running" / job_path.name
    try:
        job_path.rename(dest)
        return dest
    except OSError:
        return None


def _run_one_job(gpu: int, job_path: Path, lock_fd) -> None:
    job = json.loads(job_path.read_text())
    job["assigned_gpu"] = gpu
    job["started_at"] = _now()
    job_path.write_text(json.dumps(job, indent=2))

    name = job.get("name", job_path.stem)
    log_path = LOG_DIR / f"queue_{name}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    print(f"[worker gpu={gpu}] running {job_path.name} name={name!r} -> {log_path}")
    with open(log_path, "a") as logf:
        logf.write(f"\n=== [{_now()}] gpu={gpu} job={job_path.name} cmd={job['cmd']!r} ===\n")
        logf.flush()
        proc = subprocess.run(["bash", "-lc", job["cmd"]], stdout=logf, stderr=subprocess.STDOUT, env=env)
    rc = proc.returncode

    job["exit_code"] = rc
    job["finished_at"] = _now()
    final_state = "done" if rc == 0 else "failed"
    dest = QUEUE_ROOT / final_state / job_path.name
    job_path.write_text(json.dumps(job, indent=2))
    job_path.rename(dest)
    print(f"[worker gpu={gpu}] {job_path.name} -> {final_state} (rc={rc})")


def _worker_loop(gpu: int, use_lock: bool, poll_s: float) -> None:
    _ensure_dirs()
    lock_path = QUEUE_ROOT / f"gpu_{gpu}.lock"
    print(f"[worker gpu={gpu}] starting, pid={os.getpid()}, lock={'on' if use_lock else 'off'}")

    running = True

    def _sigterm(_signum, _frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT, _sigterm)

    while running:
        if (QUEUE_ROOT / "STOP").exists():
            print(f"[worker gpu={gpu}] STOP flag present, sleeping")
            time.sleep(poll_s)
            continue

        ok, reason = _disk_ok()
        if not ok:
            print(f"[DISK] waiting: {reason}")
            time.sleep(poll_s)
            continue

        job_path = _pick_job()
        if job_path is None:
            time.sleep(poll_s)
            continue

        claimed = _claim(job_path)
        if claimed is None:
            continue  # lost the race, try again immediately

        if use_lock:
            lock_fd = open(lock_path, "w")
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
                _run_one_job(gpu, claimed, lock_fd)
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                lock_fd.close()
        else:
            _run_one_job(gpu, claimed, None)


def cmd_worker(args: argparse.Namespace) -> int:
    _worker_loop(args.gpu, use_lock=(args.jobs_per_gpu <= 1), poll_s=args.poll_interval)
    return 0


# ── start-workers / stop-workers ─────────────────────────────────────────
def cmd_start_workers(args: argparse.Namespace) -> int:
    _ensure_dirs()
    this_file = os.path.abspath(__file__)
    for gpu in args.gpus:
        for slot in range(args.jobs_per_gpu):
            pid_file = PIDS_DIR / f"gpu_{gpu}_slot_{slot}.pid"
            if pid_file.exists():
                try:
                    old_pid = int(pid_file.read_text().strip())
                    os.kill(old_pid, 0)
                    print(f"[start-workers] gpu={gpu} slot={slot} already running (pid={old_pid}), skipping")
                    continue
                except (OSError, ValueError):
                    pass  # stale pid file
            log_path = LOG_DIR / f"worker_gpu{gpu}_slot{slot}.log"
            with open(log_path, "a") as logf:
                proc = subprocess.Popen(
                    [sys.executable, this_file, "worker", "--gpu", str(gpu),
                     "--jobs-per-gpu", str(args.jobs_per_gpu)],
                    stdout=logf, stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            pid_file.write_text(str(proc.pid))
            print(f"[start-workers] gpu={gpu} slot={slot} pid={proc.pid} log={log_path}")
    return 0


def cmd_stop_workers(args: argparse.Namespace) -> int:
    _ensure_dirs()
    any_found = False
    for pid_file in sorted(PIDS_DIR.glob("*.pid")):
        any_found = True
        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, signal.SIGTERM)
            print(f"[stop-workers] sent SIGTERM to {pid_file.name} pid={pid}")
        except (OSError, ValueError) as exc:
            print(f"[stop-workers] {pid_file.name}: {exc}")
        pid_file.unlink(missing_ok=True)
    if not any_found:
        print("[stop-workers] no pid files found (nothing to stop)")
    return 0


# ── argparse plumbing ────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("submit", help="add one job to pending/")
    sp.add_argument("--name", required=True)
    sp.add_argument("--cmd", required=True)
    sp.add_argument("--gpus", type=int, default=1)
    sp.add_argument("--priority", type=int, default=0)
    sp.set_defaults(func=cmd_submit)

    sp = sub.add_parser("status", help="table of pending/running/done/failed + GPU memory")
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("worker", help="run one worker loop pinned to --gpu (internal; used by start-workers)")
    sp.add_argument("--gpu", type=int, required=True)
    sp.add_argument("--jobs-per-gpu", type=int, default=1,
                     help="if >1, no flock is taken (multiple workers share the GPU with no exclusivity)")
    sp.add_argument("--poll-interval", type=float, default=5.0)
    sp.set_defaults(func=cmd_worker)

    sp = sub.add_parser("start-workers", help="nohup one worker process per GPU index given")
    sp.add_argument("gpus", type=int, nargs="+", help="GPU indices, e.g. 0 1 2 3")
    sp.add_argument("--jobs-per-gpu", type=int, default=1)
    sp.set_defaults(func=cmd_start_workers)

    sp = sub.add_parser("stop-workers", help="SIGTERM every worker started via start-workers")
    sp.set_defaults(func=cmd_stop_workers)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
