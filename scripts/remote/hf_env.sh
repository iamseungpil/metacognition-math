# source me: ROOT·REPO·HF 토큰 준비. 토큰은 환경변수 HF_TOKEN 이나 .env 파일에서만 읽는다(출력 금지).
ROOT=${ROOT:-/hdd_data/seungpil}
R_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); REPO=${REPO:-$(cd "$R_DIR/../.." && pwd)}
export ROOT REPO R_DIR
PY=${PY:-$ROOT/envs/verl09/bin/python}
if [ -z "${HF_TOKEN:-}" ]; then
  for _f in "${HF_ENV_FILE:-}" "$REPO/.env" "$REPO/../.env" "$HOME/.hf_env"; do
    [ -n "$_f" ] && [ -f "$_f" ] || continue
    set -a; . "$_f"; set +a
    [ -z "${HF_TOKEN:-}" ] || break
  done
fi
[ -n "${HF_TOKEN:-}" ] || { echo "[hf_env] HF_TOKEN 없음 — export HF_TOKEN=... 또는 HF_ENV_FILE=<.env 경로>" >&2; return 1 2>/dev/null || exit 1; }
export HF_TOKEN HF_HOME=${HF_HOME:-$ROOT/scratch/hf_home}
