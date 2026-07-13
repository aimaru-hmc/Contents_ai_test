#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

if command -v nvcc >/dev/null 2>&1; then
  nvcc_path="$(command -v nvcc)"
  export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$nvcc_path")")}"
  export PATH="$CUDA_HOME/bin:$PATH"
elif [[ -x "${CONDA_PREFIX:-}/bin/nvcc" ]]; then
  export CUDA_HOME="${CUDA_HOME:-$CONDA_PREFIX}"
  export PATH="$CUDA_HOME/bin:$PATH"
else
  cat >&2 <<'EOF'
nvcc를 찾지 못했습니다.
현재 vLLM/FlashInfer가 시작 중 JIT 컴파일을 하면서 nvcc를 요구합니다.

먼저 CUDA toolkit/nvcc를 설치하거나 CUDA_HOME을 지정하세요.
예:
  conda install -c nvidia cuda-nvcc
또는:
  export CUDA_HOME=/path/to/cuda
  export PATH="$CUDA_HOME/bin:$PATH"

확인:
  which nvcc
  nvcc --version
EOF
  exit 1
fi

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CUDA_HOME=$CUDA_HOME"
echo "VLLM_USE_FLASHINFER_SAMPLER=$VLLM_USE_FLASHINFER_SAMPLER"
echo "nvcc=$(command -v nvcc)"

exec vllm serve google/gemma-4-31B-it \
  --served-model-name gemma4:31b \
  --host 127.0.0.1 \
  --port 8000 \
  --tensor-parallel-size 4
