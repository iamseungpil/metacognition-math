#!/usr/bin/env bash
# Source this to load secrets into env: `source experiments/common/load_secrets.sh`
# Fails fast if .env missing or token unset.
_ENV_FILE="$(git rev-parse --show-toplevel 2>/dev/null)/.env"
[ -f "$_ENV_FILE" ] || { echo "ERROR: .env not found at $_ENV_FILE (copy .env.example → .env)"; return 1 2>/dev/null || exit 1; }
set -a; source "$_ENV_FILE"; set +a
for v in HF_TOKEN GH_TOKEN WANDB_API_KEY; do
  val="${!v}"
  if [ -z "$val" ] || [[ "$val" == REPLACE_* ]] || [[ "$val" == *xxxx* ]]; then
    echo "WARNING: $v is unset or placeholder — update .env"
  fi
done
