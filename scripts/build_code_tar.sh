#!/usr/bin/env bash
# 노드용 코드 tar 를 만들고 **런처의 계약대로인지 실물로 검증**한다.
#
# 왜 스크립트인가. 2026-08-18 에 이 tar 를 손으로 세 번 만들었고 세 번 다 틀렸다.
#   ① `tar -czf ... src/ configs/` → 최상위 `metacognition/` 이 없어 노드의
#      `cd /scratch/metacognition` 이 실패했다(7잡 전멸).
#   ② `scripts/countdown_gs0_eval.py` 가 없는 채로 쌌다(가드가 잡음).
#   ③ `core/` 를 빼먹어 knob 레지스트리가 FileNotFoundError 로 죽었다(7잡 전멸).
# 셋 다 "무엇을 담아야 하는지"가 사람 머릿속에만 있어서 생겼다. 여기 적어 둔다.
#
# 사용:
#   set -a; source .env; set +a
#   scripts/build_code_tar.sh                 # 만들고 검증만
#   scripts/build_code_tar.sh --release TAG   # + GitHub 릴리스 자산으로 올리고 id 출력
#
# ⚠CODE_TAR_REVISION 에 넣는 값은 **GitHub 릴리스 자산 id(숫자)** 다.
#   HF 커밋 SHA 가 아니다. 런처의 curl 이 api.github.com/.../releases/assets/<id> 다.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
OUT="${OUT:-/tmp/metacognition.tar.gz}"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

# ── 담는 것. 노드가 /scratch/metacognition/<이 경로> 로 읽는 전부. ────────────────
PACK=(src configs scripts tests core)

# ── 런처의 가드가 실제로 `test -f` 하는 파일. 하나라도 없으면 창이 죽는다. ────────
REQUIRED=(
  configs/countdown_6arm.yaml
  src/training/countdown_rewards.py
  src/training/countdown_task.py
  src/training/countdown_pmi.py
  src/training/verl_sdc.py
  scripts/countdown_gs0_eval.py
  scripts/push_ckpts_to_hf.py
  scripts/pull_resume_ckpt.py
  core/KNOBS.yaml                 # ③ knob 레지스트리. mode 가 COUNTDOWN/TRIOBJ 면 읽는다.
)

mkdir -p "$STAGE/metacognition"                 # ① 최상위 이름이 계약이다
for d in "${PACK[@]}"; do
  [ -e "$d" ] || { echo "⛔ 담을 것이 없다: $d"; exit 1; }
  cp -r "$d" "$STAGE/metacognition/"
done
find "$STAGE" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
find "$STAGE" -name '*.pyc' -delete 2>/dev/null || true
tar -czf "$OUT" -C "$STAGE" metacognition

# ── 검증: 만든 것을 **다시 풀어서** 본다. 담았다고 믿지 않는다. ─────────────────
V="$(mktemp -d)"; trap 'rm -rf "$STAGE" "$V"' EXIT
tar -xzf "$OUT" -C "$V"
[ -d "$V/metacognition" ] || { echo "⛔ 최상위 metacognition/ 이 없다"; exit 1; }
fail=0
for f in "${REQUIRED[@]}"; do
  if [ -f "$V/metacognition/$f" ]; then printf '  ✅ %s\n' "$f"
  else printf '  ⛔ %s  (가드가 이걸 찾는다)\n' "$f"; fail=1; fi
done
grep -q 'COUNTDOWN\]\[WIRED\]' "$V/metacognition/src/training/verl_sdc.py" \
  && echo "  ✅ 배선 증거줄(WIRED)" || { echo "  ⛔ WIRED print 가 없다"; fail=1; }
[ "$fail" -eq 0 ] || { echo "⛔ 검증 실패 — 올리지 않는다"; exit 1; }
echo "  tar: $OUT  ($(stat -c%s "$OUT") bytes)"

# ── 릴리스 (선택) ────────────────────────────────────────────────────────────
if [ "${1:-}" = "--release" ]; then
  TAG="${2:?릴리스 태그를 줘라}"
  : "${GH_TOKEN:?.env 를 source 했나 — 셸 토큰은 401 이었다}"
  OUT="$OUT" TAG="$TAG" python3 - <<'PY'
import json, os, urllib.request
TOK=os.environ["GH_TOKEN"]; REPO="iamseungpil/metacognition-math"; TAG=os.environ["TAG"]
H={"Authorization":f"token {TOK}","Accept":"application/vnd.github+json"}
def api(u,d=None,m=None,c="application/json"):
    return json.load(urllib.request.urlopen(urllib.request.Request(u,data=d,method=m,headers={**H,"Content-Type":c})))
try:
    rel=api(f"https://api.github.com/repos/{REPO}/releases/tags/{TAG}")
except urllib.error.HTTPError:
    rel=api(f"https://api.github.com/repos/{REPO}/releases", method="POST",
            data=json.dumps({"tag_name":TAG,"name":TAG,"prerelease":True}).encode())
for a in rel.get("assets",[]):
    if a["name"]=="metacognition.tar.gz":
        urllib.request.urlopen(urllib.request.Request(
            f"https://api.github.com/repos/{REPO}/releases/assets/{a['id']}", method="DELETE", headers=H))
blob=open(os.environ["OUT"],"rb").read()
asset=api(rel["upload_url"].split("{")[0]+"?name=metacognition.tar.gz", blob, "POST", "application/gzip")
print(f"CODE_TAR_REVISION={asset['id']}")
PY
fi
