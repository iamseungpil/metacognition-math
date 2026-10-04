#!/usr/bin/env python
"""ckpt_keeper — 체크포인트 사이드카 (병합·가지치기·평가 제출).

왜 필요한가 (2026-09-04 실측):
  * `trainer.max_actor_ckpt_to_keep: 2` 가 verl 0.7.1 + SDC 트레이너 조합에서 **가지치기를 하지 않는다**
    (스텝 5~25 가 전부 남았다). FSDP 조각 하나가 47GB 라 4팔 × 100스텝이면 3.7TB — /hdd_data 가 터진다.
  * run_arm.sh 는 학습이 «끝난 뒤» 판정 지점(30/50/100)을 병합한다. 그러나 조각을 순환 삭제하면 그 전에
    30/50 이 사라진다. 그래서 판정 지점은 **나타나는 즉시** CPU 에서 bf16 으로 병합해 두고 조각을 지운다.

동작 (기본 5분마다):
  1. 각 계보(checkpoints/cd7_*)에서 «완결된» 스텝 = latest_checkpointed_iteration.txt 이하의 global_step_N.
  2. 판정 스텝(30·50, 그리고 잡이 끝난 뒤의 100) 중 병합본이 없으면 CPU 병합 → 평가 잡 제출(우선순위 90) → 조각 삭제.
  3. 나머지 완결 스텝 중 최신 2개를 제외하고 삭제. (실행 중인 저장은 latest 파일이 갱신되기 전이라 건드리지 않는다.)
  4. 로그: $WORK/logs/ckpt_keeper.log. `--once --dry-run` 으로 계획만 본다.

step 100 은 잡이 «running» 인 동안은 run_arm.sh 몫으로 남긴다(경쟁 방지). 잡이 끝났는데 병합본이 없으면 keeper 가 한다.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

WORK = Path(os.environ.get("WORK", "/hdd_data/seungpil/scratch"))
QUEUE = Path(os.environ.get("QUEUE_ROOT", "/hdd_data/seungpil/queue"))
REPO = Path(os.environ.get("REPO_ROOT", "/home/ubuntu/seungpil/metacognition-math"))
JUDGMENT = (30, 50, 100)
KEEP_LATEST = 2
LOG = WORK / "logs" / "ckpt_keeper.log"
_LIN = re.compile(
    r"^cd7_(?P<arm>[A-Z0-9_]+)_(?P<variant>[a-z0-9]+)_s(?P<seed>\d+)"
    # ★E-137(0909): 메모리·배선 접미사(_fs/_dn/_rg …)가 붙은 계보는 이 정규식에 **안 맞아**
    #   variant 가 "new"(메타 강제 프롬프트)로 떨어졌다 — VTR/g3dn/VTRW/EVC s30 이 학습 프롬프트와
    #   다른 프롬프트로 채점돼 .555/.582/.510 이 나왔다(전부 무효). 임의 소문자 접미사를 허용한다.
    r"(?P<mixed>_mixed)?(?:_r(?P<resp>\d+))?(?:_[a-z0-9]+)*$"
)


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as fh:
        fh.write(line + "\n")


def running_lineages() -> set[str]:
    out = set()
    for f in (QUEUE / "running").glob("*.json"):
        try:
            cmd = json.loads(f.read_text()).get("cmd", "")
        except Exception:
            continue
        m = re.search(r"run_arm\.sh\s+(\S+)\s+(\d+)\s+(\d+)\s+(\S+)", cmd)
        if m:
            arm, seed, _steps, variant = m.groups()
            mt = re.search(r"INIT_TAG=([A-Z0-9]+)", cmd)      # run_arm.sh: 계보 arm 자리에 _<INIT_TAG> 가 붙는다
            arm_name = f"{arm}_{mt.group(1)}" if mt else arm
            eff = "plain" if arm == "N0" else variant
            # run_arm.sh 의 data_hint=mixed 계보 접미사 — ARM_SPECS[arm]["data_hint"]=="mixed"
            # 인 팔이면 전부 이 접미사가 붙는다(run_arm.sh 의 DATA_HINT 분기와 동일 규약).
            # 여기서 매번 import 하지 않고 명시 목록으로 고정한다(ARM_SPECS 를 이 경량
            # 사이드카가 매 폴링마다 다시 import 하게 만들지 않기 위해서다 — OPT_MT2 는
            # 2026-09-06 추가, 나머지는 기존과 동일).
            suffix = "_mixed" if arm in ("M0", "MT", "OPT_M", "OPT_MT", "OPT_MTC", "OPT_MT2", "OPT_CF") else ""
            # ★RESP_LEN(0906, OPT_MT-L): run_arm.sh 는 큐 cmd 문자열 앞쪽에
            #   `RESP_LEN=NNNN bash scripts/local/run_arm.sh ...` 형태로 env var 를
            #   붙인다 — 값이 2048(기본)이 아니면 _r{RESP_LEN} 이 _mixed 뒤에 붙는다
            #   (run_arm.sh 의 LINEAGE 접미사 규약과 동일해야 이 계보가 "running" 으로
            #   잡혀 keeper 가 step 100 을 run_arm.sh 와 경합하지 않는다).
            resp_m = re.search(r"\bRESP_LEN=(\d+)\b", cmd)
            resp = resp_m.group(1) if resp_m else "2048"
            resp_suffix = f"_r{resp}" if resp != "2048" else ""
            out.add(f"cd7_{arm_name}_{eff}_s{seed}{suffix}{resp_suffix}")
    return out


def complete_steps(lineage_dir: Path) -> list[int]:
    latest_f = lineage_dir / "latest_checkpointed_iteration.txt"
    if not latest_f.exists():
        return []
    try:
        latest = int(latest_f.read_text().strip())
    except ValueError:
        return []
    steps = []
    for d in lineage_dir.glob("global_step_*"):
        try:
            n = int(d.name.split("_")[-1])
        except ValueError:
            continue
        if n <= latest and (d / "actor").is_dir():
            steps.append(n)
    return sorted(steps)


def merged_ok(target: Path) -> bool:
    return (target / "config.json").exists() and any(target.glob("*.safetensors"))


def merge(lineage: str, step: int, dry: bool) -> bool:
    src = WORK / "checkpoints" / lineage / f"global_step_{step}" / "actor"
    dst = WORK / "merged" / lineage / f"step_{step}"
    if merged_ok(dst):
        return True
    log(f"merge {lineage} step {step} -> {dst}")
    if dry:
        return False
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    r = subprocess.run([sys.executable, "-m", "verl.model_merger", "merge", "--backend", "fsdp",
                        "--local_dir", str(src), "--target_dir", str(dst)],
                       env=env, cwd=str(REPO), capture_output=True, text=True)
    if r.returncode != 0 or not merged_ok(dst):
        log(f"  merge FAILED rc={r.returncode}: {r.stderr[-400:]}")
        return False
    log("  merge ok")
    return True


def submit_eval(lineage: str, step: int, dry: bool) -> None:
    marker = WORK / "merged" / lineage / f"step_{step}" / ".eval_submitted"
    if marker.exists():
        return
    m = _LIN.match(lineage)
    if m is None:
        log(f"[E-137] lineage {lineage!r} 가 _LIN 에 안 맞는다 — variant 를 추정할 수 없어 평가를 **건너뛴다**(무음 'new' 금지)")
        return
    variant = m.group("variant")
    # ★RESP_LEN(0906, OPT_MT-L): 학습 예산이 커진 계보(_r{N} 접미사)는 eval/이어쓰기
    #   예산도 run_arm.sh 와 같은 비율로 늘린다 — resp 기본값(접미사 없음) 2048 에서는
    #   아래 두 숫자(2560/2048)가 예전 하드코딩과 바이트 동일하다.
    resp = int(m.group("resp")) if (m and m.group("resp")) else 2048
    merged = WORK / "merged" / lineage / f"step_{step}"
    out = WORK / "eval" / lineage / f"step_{step}"
    # ★0911: gpu_util 0.6 은 공유 카드에서 두 번 OOM 사고(09-10 새벽·저녁)를 냈다 — 카드
    #   여유가 외부 잡 때문에 흔들리는 게 정상이므로 0.4 로 낮추고, retry_cmd.sh 로 잡
    #   **안에서** 재시도한다(마커는 "큐 제출 성공"에만 찍혀 잡 자체가 죽어도 재제출 안 됨).
    inner = (f"source scripts/local/env.sh >/dev/null 2>&1; python scripts/countdown_gs0_eval.py "
             f"--model_path {merged} --data $WORK/data/countdown_val_4num_{variant}.parquet "
             f"--meta_format {variant} --num_samples 8 --seed 11 --max_tokens {resp + 512} --gpu_util 0.4 --out_dir {out}")
    # ★0911 수리(2차): 큰따옴표로 감싸면 gpu_queue 워커가 --cmd 를 셸에 넘기는 바깥
    #   시점에 $WORK 가 (env.sh 를 소싱하기도 전에, 빈 문자열로) 먼저 확장돼 버려
    #   `/data/...`(WORK 없이) 로 깨졌다(FIXED_CHK step100 평가 4연속 FileNotFoundError로
    #   발각). 작은따옴표는 바깥 셸이 안을 전혀 건드리지 않아 $WORK 확장이 안쪽
    #   `bash -c` 가 env.sh 를 소싱한 뒤로 미뤄진다. inner 자체엔 작은따옴표가 없다.
    cmd = f"bash scripts/local/retry_cmd.sh 4 90 -- bash -c '{inner}'"
    log(f"submit eval {lineage} step {step}")
    if dry:
        return
    r = subprocess.run([sys.executable, "scripts/local/gpu_queue.py", "submit", "--name",
                        f"eval_{lineage}_step{step}", "--priority", "90", "--need-mb", "40000", "--cmd", cmd],
                       cwd=str(REPO), capture_output=True, text=True)
    if r.returncode == 0:
        marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
    else:
        log(f"  submit FAILED: {r.stderr[-300:]}")
    # ★cd8 1차 지표: 판정 자리 1,000곳 × 16 이어쓰기(메타 허용) — 모든 팔이 같은 자리를 받는다.
    # SITES_DIR(0907, E-133 수리): sites_v1(및 v2/v3/v3c) 은 held-out val 에서 채굴돼
    # 오염됐다 — v4 라운드로 넘어가려면 이 프로세스를 SITES_DIR=sites_v4 로 재시작해야
    # 한다(env var 미설정이면 기존과 바이트 동일하게 sites_v1 을 계속 읽는다).
    sites_dir = os.environ.get("SITES_DIR", "sites_v1")
    jinner = (f"source scripts/local/env.sh >/dev/null 2>&1; python scripts/local/gen_continuations.py "
              f"--sites $WORK/data/{sites_dir}/sites_judge.parquet --model_path {merged} --policy_tag {lineage}_s{step} "
              f"--modes meta --k 16 --max_tokens {resp} --seed 11 --gpu_util 0.4 "
              f"--out $WORK/conts_v1/judge_{lineage}_step{step}.parquet")
    jcmd = f"bash scripts/local/retry_cmd.sh 4 90 -- bash -c '{jinner}'"
    log(f"submit judge-site continuations {lineage} step {step}")
    r2 = subprocess.run([sys.executable, "scripts/local/gpu_queue.py", "submit", "--name",
                         f"jsite_{lineage}_step{step}", "--priority", "90", "--need-mb", "40000", "--cmd", jcmd],
                        cwd=str(REPO), capture_output=True, text=True)
    if r2.returncode != 0:
        log(f"  judge-site submit FAILED: {r2.stderr[-300:]}")


def rm_step(lineage: str, step: int, dry: bool, why: str) -> None:
    d = WORK / "checkpoints" / lineage / f"global_step_{step}"
    log(f"delete {d} ({why})")
    if not dry and d.exists():
        shutil.rmtree(d, ignore_errors=True)


def pass_once(dry: bool) -> None:
    running = running_lineages()
    for ldir in sorted((WORK / "checkpoints").glob("cd7_*")):
        lineage = ldir.name
        steps = complete_steps(ldir)
        if not steps:
            continue
        newest = set(steps[-KEEP_LATEST:])
        for n in steps:
            if n in JUDGMENT:
                if n == 100 and lineage in running:
                    continue          # run_arm.sh 가 학습 종료 직후 처리한다
                if merge(lineage, n, dry):
                    submit_eval(lineage, n, dry)
                    if n not in newest:
                        rm_step(lineage, n, dry, "judgment merged")
            elif n not in newest:
                rm_step(lineage, n, dry, f"not in newest {KEEP_LATEST}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--interval", type=int, default=300)
    a = ap.parse_args()
    while True:
        try:
            pass_once(a.dry_run)
        except Exception as e:  # noqa: BLE001
            log(f"pass error: {e!r}")
        if a.once:
            break
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
