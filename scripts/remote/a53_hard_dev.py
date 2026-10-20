"""수정 41 1단계 — 어려운 연습 시험지(CPU). DeepScaleR(4만, 옛 AIME·AMC·올림피아드; AIME24/25 미포함)에서
build: 학습·평가·최종 시험 문제와 겹치지 않고 채점 가능한 800문제를 고정 씨앗으로 뽑아 선별 parquet 을 쓴다.
select: base 선별 결과(k=4, 32k, 씨앗 7)에서 4번 중 1~3번 맞힌 문제(= 첫 답이 자주 틀리지만 풀 수 있는 문제)만 남겨
        최대 300문제 연습 시험지를 쓴다(평가는 다른 씨앗 11 로 다시 — 선별 운이 base 점수에 남지 않게).
겹침 판정 = 공백·$ 제거 소문자 앞 80자 일치. 제외: MATH 학습 풀(7,295 ⊃ L5 800·학습 188·probe 256)·math_val_plain·
MATH-500·AIME24·AIME25·HMMT25.
수정 42 확장: `build 2` = 다른 고정 씨앗, 이미 뽑은 선별 문제도 제외 → hdev_screen800_2 · `select <texts> 2` → hdev_b ·
`merge` → hdev_all(hdev300 153 + hdev_b; 평가는 한 번에 — per_problem 키 g<순번> 이 파일마다 다시 매겨지므로 합치지 않는다)."""
import json, random, re, sys
import os as _os
_ROOT = _os.environ.get("ROOT", "/hdd_data/seungpil")
sys.path.insert(0, _os.environ.get("REPO") or str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from mc.grade import grade_answer, grade_math  # noqa: E402

D = _ROOT + "/scratch/data/"
SCREEN = {"1": D + "hdev_screen800.parquet", "2": D + "hdev_screen800_2.parquet"}
DEV = {"1": D + "hdev300.parquet", "2": D + "hdev_b.parquet"}
SEED = {"1": (20260929, 20260930), "2": (20261001, 20261002)}
FINAL = [("HuggingFaceH4/aime_2024", "train"), ("math-ai/aime25", "test"),
         ("MathArena/hmmt_feb_2025", "train"), ("HuggingFaceH4/MATH-500", "test")]


def key(p: str) -> str:
    return re.sub(r"[\s$]", "", str(p)).lower()[:80]


def build(b: str):
    import pandas as pd
    from datasets import load_dataset
    seen = set()
    done = [SCREEN[k].split("/")[-1] for k in SCREEN if k < b]          # 앞 묶음 선별 문제 제외
    for f in ("math_train_math_opt.parquet", "math_val_plain.parquet", "mc_pfx_commit_b10_screen_plain_le8704.parquet", *done):
        d = pd.read_parquet(D + f)
        seen |= {key(p) for p in (d["problem"] if "problem" in d else [e["problem"] for e in d["extra_info"]])}
    for hid, sp in FINAL:
        seen |= {key(r["problem"]) for r in load_dataset(hid, split=sp)}
    src = pd.read_parquet(D + "math_train_math_opt_deepscaler.parquet")
    ok = [i for i, (p, g) in enumerate(zip(src["problem"], src["gold"]))
          if key(p) not in seen and 0 < len(str(g)) <= 40 and len(str(p)) <= 2500 and grade_answer(str(g), str(g))]
    pick = sorted(random.Random(SEED[b][0]).sample(ok, 800))
    out = src.iloc[pick].reset_index(drop=True)
    out.to_parquet(SCREEN[b])
    print(json.dumps({"pool": len(src), "clean": len(ok), "excluded_keys": len(seen), "screen": len(out)}))


def select(texts: str, b: str):
    import pandas as pd
    src = pd.read_parquet(SCREEN[b])
    hits: dict = {}
    for r in map(json.loads, open(texts)):
        hits.setdefault(int(r["problem_id"]), []).append(grade_math(r["text"], str(r["gold"])))   # eval 요약과 같은 채점
    mid = [i for i, h in sorted(hits.items()) if 1 <= sum(h) <= 3]
    keep = sorted(random.Random(SEED[b][1]).sample(mid, min(300, len(mid))))
    src.iloc[keep].reset_index(drop=True).to_parquet(DEV[b])
    dist = {n: sum(sum(h) == n for h in hits.values()) for n in range(5)}
    print(json.dumps({"screened": len(hits), "correct_of_4": dist, "mid": len(mid), "dev": len(keep)}))
    if len(keep) < 150:
        raise SystemExit("[hdev] 연습 시험지가 150문제 미만 — 선별 기준 재검토 필요")


def merge():
    import pandas as pd
    out = pd.concat([pd.read_parquet(DEV[k]) for k in sorted(DEV)], ignore_index=True)
    assert out["problem"].map(key).is_unique
    out.to_parquet(D + "hdev_all.parquet")
    print(json.dumps({"hdev_all": len(out)}))


if __name__ == "__main__":
    cmd, b = sys.argv[1], (sys.argv[-1] if sys.argv[-1] in SCREEN else "1")
    build(b) if cmd == "build" else select(sys.argv[2], b) if cmd == "select" else merge()
