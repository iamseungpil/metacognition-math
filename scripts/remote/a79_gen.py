"""a79 생성/채점 분리(10-14, 사용자 «GPU 쉬지 말고»): mc.eval 과 같은 생성을 하고 texts.jsonl·meta.json 만 쓴 뒤,
채점·요약(mc.eval.summarize_single, 16 프로세스)은 새 세션의 CPU 백그라운드로 떼어 GPU 잡이 바로 다음으로 넘어가게 한다.
  생성: python a79_gen.py --model_path M --dataset D --k K ... --out_dir O     채점: python a79_gen.py --grade O
결과 summary.json 은 mc.eval 과 같은 내용(같은 함수·같은 메타 키)."""
import argparse, json, os, subprocess, sys
from pathlib import Path
sys.path.insert(0, os.environ.get("REPO") or str(Path(__file__).resolve().parents[2]))   # scripts/remote 복사본


def put(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp"); tmp.write_text(text); os.replace(tmp, path)


def grade(out: Path) -> None:
    from mc.eval import load_jsonl, summarize_single
    summ = summarize_single(load_jsonl(str(out / "texts.jsonl")))
    summ.update(json.loads((out / "meta.json").read_text()))
    put(out / "summary.json", json.dumps(summ, ensure_ascii=False, indent=2, default=float))
    print(f"[a79_gen] graded {out} acc={summ['acc']:.4f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grade"); ap.add_argument("--protocol", default="single"); ap.add_argument("--out_dir")
    ap.add_argument("--model_path"); ap.add_argument("--dataset"); ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--seed", type=int, default=11); ap.add_argument("--gpu_util", type=float, default=0.80)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--max_tokens", type=int, default=12288)
    ap.add_argument("--temperature", type=float, default=1.0); ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=-1)
    a = ap.parse_args()
    if a.grade:
        return grade(Path(a.grade))
    from mc.rollout import run_eval
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    rows = run_eval(a)
    put(out / "meta.json", json.dumps({"protocol": a.protocol, "model_path": a.model_path, "dataset": a.dataset, "k": a.k,
                                       "seed": a.seed, "max_tokens": a.max_tokens, "temperature": a.temperature,
                                       "top_p": a.top_p, "top_k": a.top_k}))
    put(out / "texts.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    log = open(out / "grade.log", "a")
    subprocess.Popen([sys.executable, __file__, "--grade", str(out)], stdout=log, stderr=subprocess.STDOUT,
                     stdin=subprocess.DEVNULL, start_new_session=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    print(f"[a79_gen] generated {len(rows)} rows → {out}; grading detached", flush=True)


if __name__ == "__main__":
    main()
