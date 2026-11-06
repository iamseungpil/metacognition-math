"""원격 서버 ↔ 이 서버 결과 주고받기 (huggingface_hub 만 쓴다). 래퍼: push_results.sh · pull_results.sh · get_data.sh.
  push RUN [--weights] [--failed] [--dry-run] | pull [--list] [--force] [RUN ...] | get-data [--no-models] [--no-prefetch]
  upload-data [--yes] (이 서버에서 한 번: data_manifest.txt 의 파일을 데이터 저장소로) | fetch-model REPO SUBDIR (R1: 모델 폴더 하나 받아 경로 출력)
토큰은 HF_TOKEN 환경변수로만 읽고 절대 출력하지 않는다. wandb 없음."""
import gzip, json, os, shutil, socket, sys, time
from pathlib import Path

ROOT = Path(os.environ.get("ROOT", "/hdd_data/seungpil"))
WORK = ROOT / "scratch"
RESULTS_REPO = os.environ.get("RESULTS_REPO", "iamseungpil/metacot-cd9-remote")
DATA_REPO = os.environ.get("DATA_REPO", "iamseungpil/metacot-cd9-data")
MODELS_REPO = os.environ.get("MODELS_REPO", "iamseungpil/metacot-cd9-models")
HERE = Path(__file__).resolve().parent
LIMIT_GB = 20.0
BASE_MODELS = {"Qwen3-4B-Instruct-2507": "Qwen/Qwen3-4B-Instruct-2507", "Qwen3-1.7B": "Qwen/Qwen3-1.7B"}
PREFETCH = [("HuggingFaceH4/aime_2024", "train"), ("math-ai/aime25", "test"), ("MathArena/hmmt_feb_2025", "train"),
            ("HuggingFaceH4/MATH-500", "test"), ("math-ai/amc23", "test"), ("math-ai/minervamath", "test")]


def api():
    from huggingface_hub import HfApi
    return HfApi(token=os.environ["HF_TOKEN"])


def retry(fn, tries=4):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if i == tries - 1:
                raise
            print(f"[hf_sync] 재시도 {i + 1}: {type(e).__name__}", flush=True)
            time.sleep(15 * (i + 1))


def gz(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "rb") as f, gzip.open(dst, "wb", compresslevel=6) as g:
        shutil.copyfileobj(f, g, 1 << 20)


def read_meta(run: str) -> dict:
    p = WORK / "remote_meta" / f"{run}.env"
    m = {}
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                m[k.strip()] = v.strip()
    return m


def du(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


# ── push ─────────────────────────────────────────────────────────────────────
def push(argv):
    run = argv[0]
    weights, failed, dry = "--weights" in argv, "--failed" in argv, "--dry-run" in argv
    meta = read_meta(run)
    lineage = meta.get("LINEAGE", "")
    stage = ROOT / "tmp" / "push_stage" / run
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    logs = WORK / "logs"
    note = {"run": run, "host": socket.gethostname(), "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "meta": meta, "evals": []}
    for lg, nm in ((logs / f"{lineage}.log" if lineage else None, "train_log.txt.gz"), (logs / f"queue_{run}.log", "queue_log.txt.gz")):
        if lg and lg.is_file():
            gz(lg, stage / nm)
    if not failed:
        globs = (meta.get("EVAL_GLOB") or f"rl_{run}_*").split()
        dirs = sorted({d for g in globs for d in (WORK / "eval").glob(g) if d.is_dir()})
        wait = float(os.environ.get("WAIT_SUMMARY_MIN", "30")) * 60
        t0 = time.time()
        for d in dirs:   # a79_gen 은 채점을 백그라운드로 떼므로 summary.json 이 몇 분 늦다
            while not (d / "summary.json").exists() and (d / "texts.jsonl").exists() and time.time() - t0 < wait:
                time.sleep(20)
        for d in dirs:
            o = stage / "eval" / d.name
            o.mkdir(parents=True)
            for fn in ("summary.json", "meta.json"):
                if (d / fn).is_file():
                    shutil.copy2(d / fn, o / fn)
            if (d / "texts.jsonl").is_file():
                gz(d / "texts.jsonl", o / "texts.jsonl.gz")
            elif (d / "texts.jsonl.gz").is_file():
                shutil.copy2(d / "texts.jsonl.gz", o / "texts.jsonl.gz")
            note["evals"].append({"dir": d.name, "summary": (d / "summary.json").is_file()})
        if not dirs:
            print(f"[push] 경고: 평가 폴더 없음 (EVAL_GLOB={globs})", file=sys.stderr)
        if lineage:
            ck = WORK / "checkpoints" / lineage
            for fn in ("RUN_ENV.txt", "TRAIN_DONE"):
                if (ck / fn).is_file():
                    (stage / "ckpt").mkdir(exist_ok=True)
                    shutil.copy2(ck / fn, stage / "ckpt" / fn)
            for fn in ("pfx_rollouts.jsonl.gz", "pfx_rollouts.jsonl"):
                if (ck / fn).is_file():
                    (stage / "ckpt").mkdir(exist_ok=True)
                    if fn.endswith(".gz"):
                        shutil.copy2(ck / fn, stage / "ckpt" / fn)
                    else:
                        gz(ck / fn, stage / "ckpt" / (fn + ".gz"))
                    break
        for rel in meta.get("EXTRA_FILES", "").split():
            f = WORK / rel
            if f.is_file():
                (stage / "extra" / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, stage / "extra" / rel)
    (stage / ("FAILED" if failed else "DONE")).write_text(json.dumps(note, indent=1))
    size = du(stage)
    print(f"[push] {run}: {len(note['evals'])} 평가 폴더, 스테이징 {size / 1e6:.1f} MB ({stage})")
    if dry:
        print("[push] --dry-run: 올리지 않음")
        return
    a = api()
    retry(lambda: a.create_repo(RESULTS_REPO, repo_type="dataset", private=True, exist_ok=True))
    if weights and not failed:
        cand = sorted((WORK / "models").glob(f"merged_{lineage}_step*"), key=lambda p: int(p.name.rsplit("step", 1)[1])) if lineage else []
        if cand:
            print(f"[push] 가중치 {cand[-1].name} ({du(cand[-1]) / 1e9:.1f} GB)")
            retry(lambda: a.upload_folder(repo_id=RESULTS_REPO, folder_path=str(cand[-1]), path_in_repo=f"runs/{run}/model", repo_type="dataset",
                                          commit_message=f"{run}: merged model"))
        else:
            print("[push] 경고: 병합 모델을 못 찾음", file=sys.stderr)
    retry(lambda: a.upload_folder(repo_id=RESULTS_REPO, folder_path=str(stage), path_in_repo=f"runs/{run}", repo_type="dataset",
                                  commit_message=f"{run}: {'FAILED' if failed else 'DONE'}"))   # DONE 은 같은 커밋 맨 끝(원자적)
    shutil.rmtree(stage, ignore_errors=True)
    print(f"[push] {run} → {RESULTS_REPO}/runs/{run} 완료")


# ── pull ─────────────────────────────────────────────────────────────────────
def pull(argv):
    force, only_list = "--force" in argv, "--list" in argv
    want = [x for x in argv if not x.startswith("--")]
    a = api()
    files = retry(lambda: a.list_repo_files(RESULTS_REPO, repo_type="dataset"))
    runs: dict = {}
    for f in files:
        p = f.split("/")
        if p[0] == "runs" and len(p) > 2:
            runs.setdefault(p[1], set()).add("/".join(p[2:]))
    from huggingface_hub import snapshot_download
    for run, fs in sorted(runs.items()):
        state = "DONE" if "DONE" in fs else ("FAILED" if "FAILED" in fs else "incomplete")
        pulled = (WORK / "remote_runs" / run / ".pulled").exists()
        print(f"[pull] {run}: {state}{' (이미 받음)' if pulled else ''} {len(fs)}개 파일")
        if only_list or (want and run not in want) or state != "DONE" or (pulled and not force):
            continue
        tmp = ROOT / "tmp" / "pull_stage"
        shutil.rmtree(tmp, ignore_errors=True)
        retry(lambda: snapshot_download(RESULTS_REPO, repo_type="dataset", allow_patterns=[f"runs/{run}/*"],
                                        local_dir=str(tmp), token=os.environ["HF_TOKEN"]))
        src = tmp / "runs" / run
        dst = WORK / "remote_runs" / run
        shutil.rmtree(dst, ignore_errors=True)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        shutil.rmtree(tmp, ignore_errors=True)
        for ed in sorted((dst / "eval").glob("*")) if (dst / "eval").is_dir() else []:
            out = WORK / "eval" / ed.name
            if out.exists() and not force:
                print(f"  [경고] {out} 이미 있음 — 덮지 않음 (remote_runs 에는 보관)")
                continue
            out.mkdir(parents=True, exist_ok=True)
            for fn in ("summary.json", "meta.json"):
                if (ed / fn).is_file():
                    shutil.copy2(ed / fn, out / fn)
            if (ed / "texts.jsonl.gz").is_file():
                with gzip.open(ed / "texts.jsonl.gz", "rb") as g, open(out / "texts.jsonl", "wb") as f:
                    shutil.copyfileobj(g, f, 1 << 20)
            print(f"  eval/{ed.name}")
        (dst / ".pulled").write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))


# ── get-data / models ────────────────────────────────────────────────────────
def get_data(argv):
    from huggingface_hub import snapshot_download
    tok = os.environ["HF_TOKEN"]
    WORK.mkdir(parents=True, exist_ok=True)
    retry(lambda: snapshot_download(DATA_REPO, repo_type="dataset", local_dir=str(WORK), allow_patterns=["data/*", "eval/*"], token=tok))
    print(f"[get-data] 데이터 → {WORK}/{{data,eval}}")
    if "--no-models" not in argv:
        for name, rid in BASE_MODELS.items():
            retry(lambda: snapshot_download(rid, local_dir=str(WORK / "models" / name), token=tok))
            print(f"[get-data] 모델 {rid} → {WORK / 'models' / name}")
    if "--no-prefetch" not in argv:
        from datasets import load_dataset   # HF_HOME 캐시에 받아 둔다 — a79 는 HF_DATASETS_OFFLINE=1
        for rid, split in PREFETCH:
            ds = retry(lambda: load_dataset(rid, split=split))
            print(f"[get-data] 시험 {rid}:{split} {len(ds)}문제 → HF_HOME={os.environ.get('HF_HOME')}")


def fetch_model(argv):
    from huggingface_hub import snapshot_download
    repo, sub = argv[0], argv[1]
    dest = WORK / "models" / "_hfdl"
    retry(lambda: snapshot_download(repo, allow_patterns=[f"{sub}/*"], local_dir=str(dest), token=os.environ["HF_TOKEN"]))
    print(dest / sub)


def upload_data(argv):
    items = [l.split("#")[0].strip() for l in (HERE / "data_manifest.txt").read_text().splitlines()]
    items = [x for x in items if x]
    rows = [(x, (WORK / x).stat().st_size) for x in items]
    tot = sum(s for _, s in rows)
    for x, s in rows:
        print(f"{s / 1e6:10.3f} MB  {x}")
    print(f"합계 {tot / 1e9:.4f} GB ({len(rows)}개)")
    if tot / 1e9 > LIMIT_GB:
        sys.exit(f"[upload-data] {LIMIT_GB} GB 초과 — 중단")
    if "--yes" not in argv:
        return
    from huggingface_hub import CommitOperationAdd
    a = api()
    retry(lambda: a.create_repo(DATA_REPO, repo_type="dataset", private=True, exist_ok=True))
    ops = [CommitOperationAdd(path_in_repo=x, path_or_fileobj=str(WORK / x)) for x, _ in rows]
    retry(lambda: a.create_commit(DATA_REPO, ops, repo_type="dataset", commit_message="cd9 remote data (R1+R2)"))
    print(f"[upload-data] {DATA_REPO}: {len(ops)}개 {tot / 1e6:.1f} MB 올림")


if __name__ == "__main__":
    cmd, rest = sys.argv[1], sys.argv[2:]
    {"push": push, "pull": pull, "get-data": get_data, "fetch-model": fetch_model, "upload-data": upload_data}[cmd](rest)
