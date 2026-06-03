#!/usr/bin/env bash
# package_e4_release.sh — package the repo into a GH release asset for the E.4
# amlt jobs (h200_e4_selfdistill_verl.yaml), then print the numeric asset id to
# paste into CODE_TAR_REVISION.  plan_ctsd_E4_selfdistill_rl_2026_06_03.
#
# WHAT IT DOES
#   1. git-archives the repo at HEAD into /tmp/metacognition.tar.gz with a
#      `metacognition/` top-level prefix (the amlt node does
#      `tar -xzf ... -C /scratch` → /scratch/metacognition, matching
#      PYTHONPATH=/scratch/metacognition). Falls back to a `tar --exclude` of
#      .git / envs / checkpoints if `git archive` is unavailable.
#   2. SCANS the tar for token leakage (hf_ / ghp_ / 40-hex wandb-shaped) and
#      ABORTS if any are found — CLAUDE.md carries live tokens in this repo, so
#      the archive must exclude/scrub them. The amlt YAMLs use ${...} env only.
#   3. Creates (or reuses) a GH release and uploads the asset via the `gh` CLI,
#      which consumes GH_TOKEN from the env (NEVER hardcoded here).
#   4. Resolves and prints the numeric asset id for the curl-by-asset-id pattern
#      (api.github.com/.../releases/assets/<id>) used by the amlt command.
#
# USAGE
#   GH_TOKEN=ghp_xxx ./scripts/package_e4_release.sh [TAG]
#   (TAG defaults to e4-selfdistill-<UTC timestamp>.)
#
# This script is WRITE-ONLY scaffolding: it is NOT run as part of E.4
# implementation. Run it manually once the code is ready to ship.

set -euo pipefail

# ── 1. preconditions ──────────────────────────────────────────────────────────
: "${GH_TOKEN:?set GH_TOKEN in env/.env (never hardcode a token in this script)}"

REPO="iamseungpil/metacognition-math"
TAG="${1:-e4-selfdistill-$(date -u +%Y%m%d-%H%M%S)}"
ASSET_NAME="metacognition.tar.gz"
TARBALL="/tmp/${ASSET_NAME}"

# Resolve the repo root from this script's location (scripts/ is a child of root).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

echo "[package_e4] repo_root=${REPO_ROOT} tag=${TAG} -> ${TARBALL}"

# ── 2. build the tarball (prefix metacognition/) ──────────────────────────────
rm -f "${TARBALL}"
if git -C "${REPO_ROOT}" rev-parse --git-dir >/dev/null 2>&1; then
  echo "[package_e4] git archive @ HEAD ..."
  # --worktree-attributes honors the working-tree .gitattributes (CLAUDE.md
  # export-ignore) even when .gitattributes is not yet committed, so the
  # token-bearing CLAUDE.md is dropped from the asset regardless of commit state.
  git -C "${REPO_ROOT}" archive --worktree-attributes --format=tar.gz \
      --prefix=metacognition/ -o "${TARBALL}" HEAD
else
  echo "[package_e4] not a git repo — falling back to tar --exclude"
  tar --exclude='./.git' \
      --exclude='./envs' \
      --exclude='./checkpoints' \
      --exclude='./code_snapshots' \
      --exclude='./reports' \
      --exclude='*.parquet' \
      --transform 's,^\./,metacognition/,' \
      -czf "${TARBALL}" -C "${REPO_ROOT}" .
fi
echo "[package_e4] built $(du -h "${TARBALL}" | cut -f1) tarball"

# ── 3. token-leak scan (ABORT on hit) ─────────────────────────────────────────
# Decompress to stdout and grep for token signatures. CLAUDE.md in this repo has
# live hf_/ghp_/wandb tokens — they MUST NOT ship. If found, abort and tell the
# user to gitignore/scrub CLAUDE.md (or add it to the tar --exclude list above).
# NOTE: never embed a literal secret here (it would self-trip this scan and ship
# inside the tarball). Match the WandB key by its 40-hex-char shape; if
# WANDB_API_KEY is exported, also match its exact value at runtime.
echo "[package_e4] scanning tar for token leakage ..."
# Decompress once to a temp file, then scan it twice (two independent patterns).
SCAN_TMP="$(mktemp)"
trap 'rm -f "${SCAN_TMP}"' EXIT
tar -xzOf "${TARBALL}" 2>/dev/null > "${SCAN_TMP}"

# (a) Prefix-typed tokens (HF / GH); plus the live WandB key by exact value if
#     it is exported — so no literal secret is ever written into this file.
LEAK_PATTERN='hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}'
if [ -n "${WANDB_API_KEY:-}" ]; then
  LEAK_PATTERN="${LEAK_PATTERN}|${WANDB_API_KEY}"
fi
# (b) WandB-shaped keys: exactly 40 lowercase-hex chars with >=1 a-f letter, so
#     pure-decimal 40-digit IDs in eval JSONs don't false-positive while the real
#     key still trips. grep -ow gives whole-word 40-hex; the [a-f] grep filters.
LEAKS="$( { grep -aoE "${LEAK_PATTERN}" "${SCAN_TMP}"; \
            grep -aowE '[0-9a-f]{40}' "${SCAN_TMP}" | grep -aE '[a-f]'; } \
          | sort -u || true)"
if [ -n "${LEAKS}" ]; then
  echo "[package_e4] ABORT: token-like strings found in the archive:" >&2
  echo "${LEAKS}" | sed 's/\(.\{8\}\).*/\1…(redacted)/' >&2
  echo "[package_e4] scrub these (e.g. gitignore CLAUDE.md or add to tar --exclude) and re-run." >&2
  rm -f "${TARBALL}"
  exit 2
fi
echo "[package_e4] token-leak scan clean"

# ── 4. create / reuse the GH release and upload the asset ─────────────────────
# gh reads GH_TOKEN from the env automatically.
if ! command -v gh >/dev/null 2>&1; then
  echo "[package_e4] ERROR: gh CLI not found (install: https://cli.github.com/)" >&2
  exit 3
fi

if gh release view "${TAG}" --repo "${REPO}" >/dev/null 2>&1; then
  echo "[package_e4] release ${TAG} exists — uploading asset (clobber) ..."
  gh release upload "${TAG}" "${TARBALL}" --repo "${REPO}" --clobber
else
  echo "[package_e4] creating release ${TAG} ..."
  gh release create "${TAG}" "${TARBALL}" --repo "${REPO}" \
      --title "E.4 self-distill code snapshot ${TAG}" \
      --notes "Code tarball for h200_e4_selfdistill_verl.yaml (CODE_TAR_REVISION asset)."
fi

# ── 5. resolve + print the numeric asset id ───────────────────────────────────
ASSET_ID="$(gh api "repos/${REPO}/releases/tags/${TAG}" \
  --jq ".assets[] | select(.name==\"${ASSET_NAME}\") | .id")"

if [ -z "${ASSET_ID}" ]; then
  echo "[package_e4] ERROR: could not resolve asset id for ${ASSET_NAME} on tag ${TAG}" >&2
  exit 4
fi

echo ""
echo "=================================================================="
echo "[package_e4] DONE. release tag : ${TAG}"
echo "[package_e4]       asset name  : ${ASSET_NAME}"
echo "[package_e4]       ASSET ID    : ${ASSET_ID}"
echo ""
echo ">>> paste this into CODE_TAR_REVISION in h200_e4_selfdistill_verl.yaml:"
echo ">>>   CODE_TAR_REVISION: ${ASSET_ID}"
echo "=================================================================="
