#!/bin/bash
# setup_env.sh — verl09 환경(이 서버의 학습·평가 환경)을 원격 서버에 재현한다.  결과: $ROOT/envs/verl09/bin/python
#   핀: requirements-freeze.txt (이 서버 `pip freeze` 244줄) · versions.txt (python 3.10.20 · torch 2.11.0+cu130 · CUDA 13.0 · vllm 0.20.2 · ray 2.58.0 · transformers 5.10.4 · 드라이버 580.105.08)
#   ★NVIDIA 드라이버 ≥ 580 (CUDA 13) 필요 — `nvidia-smi` 로 먼저 확인. 아니면 torch/vllm 의 cu12x 짝을 따로 정해야 한다(RUNBOOK "미확인").
#   ★verl 0.9.0 에는 로컬 수정 3곳 + 파일 1개가 있다(verl09_local.patch · verl_bert_padding_torch.py) — 아래에서 입힌다.
#   사용: ROOT=/data/me bash scripts/remote/env/setup_env.sh        (이미 있으면 건너뜀; 새로 만들려면 환경 폴더를 지우고 다시)
set -euo pipefail
ROOT=${ROOT:-/hdd_data/seungpil}; H=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); ENV=${ENV:-$ROOT/envs/verl09}
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)
echo "[setup] 드라이버: ${DRV:-없음} (580 이상이어야 cu130)"; [ "${DRV%%.*}" -ge 580 ] 2>/dev/null || [ "${SKIP_DRIVER_CHECK:-}" = 1 ] || { echo "[setup] 드라이버 < 580 — 중단 (SKIP_DRIVER_CHECK=1 로 무시)" >&2; exit 1; }
mkdir -p "$ROOT/envs"
if [ ! -x "$ENV/bin/python" ]; then
  if command -v conda >/dev/null; then conda create -y -p "$ENV" python=3.10.20 || conda create -y -p "$ENV" python=3.10
  elif command -v python3.10 >/dev/null; then python3.10 -m venv "$ENV"
  else echo "[setup] conda 도 python3.10 도 없음" >&2; exit 1; fi
fi
PIP="$ENV/bin/python -m pip"
$PIP install -U pip
$PIP install --extra-index-url https://download.pytorch.org/whl/cu130 -r "$H/requirements-freeze.txt"
SP=$("$ENV/bin/python" -c "import site;print(site.getsitepackages()[0])")
if ! grep -q "0921 metacognition-math" "$SP/verl/utils/attention_utils.py"; then patch -p1 -d "$SP" < "$H/verl09_local.patch"; fi
cp "$H/verl_bert_padding_torch.py" "$SP/verl/utils/_bert_padding_torch.py"
"$ENV/bin/python" - <<'PY'
import torch, vllm, ray, transformers, verl
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpus", torch.cuda.device_count(), "| vllm", vllm.__version__, "| ray", ray.__version__, "| transformers", transformers.__version__)
from verl.utils.attention_utils import index_first_axis  # 로컬 수정이 먹었는지
print("verl 로컬 수정 OK")
PY
echo "[setup] 완료: $ENV/bin/python"
