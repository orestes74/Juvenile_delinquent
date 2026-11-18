#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 /path/to/ComfyUI /path/to/venv [port]" >&2
  exit 2
fi
COMFY_ROOT=$(realpath "$1")
VENV_ROOT=$(realpath "$2")
PORT=${3:-8197}
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
[[ -f "$COMFY_ROOT/main.py" && -x "$VENV_ROOT/bin/python" ]] || exit 2
CACHE_ROOT="$REPO_ROOT/.runtime-cache"
mkdir -p "$CACHE_ROOT"/{triton,torch-extensions,inductor,cuda} "$REPO_ROOT/outputs"
export PYTHONDONTWRITEBYTECODE=1
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export TORCH_EXTENSIONS_DIR="$CACHE_ROOT/torch-extensions"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/inductor"
export CUDA_CACHE_PATH="$CACHE_ROOT/cuda"
if [[ -n ${PYTHON_DEV_INCLUDE_ROOT:-} ]]; then
  export CPATH="$PYTHON_DEV_INCLUDE_ROOT:$PYTHON_DEV_INCLUDE_ROOT/python3.12${CPATH:+:$CPATH}"
fi
exec "$VENV_ROOT/bin/python" "$COMFY_ROOT/main.py" \
  --listen 127.0.0.1 --port "$PORT" \
  --disable-all-custom-nodes --disable-api-nodes --disable-auto-launch \
  --disable-dynamic-vram --use-sage-attention \
  --output-directory "$REPO_ROOT/outputs"
