#!/usr/bin/env bash
# Start the local Holo3.1-4B server (llama.cpp, OpenAI-compatible API on 127.0.0.1 only).
# The GPU is used ONLY by this process. Refuses to start if another job uses the GPU or < 5 GB VRAM is free.
set -euo pipefail
cd "$(dirname "$0")/../.."  # repo root
PORT=${PORT:-8090}
MODEL=${MODEL:-models/gguf/Holo-3.1-4B-Q4_K_M.gguf}
MMPROJ=${MMPROJ:-models/gguf/Holo-3.1-4B-mmproj-F16.gguf}
CTX=${CTX:-16384}

free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
others=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -v llama-server || true)
if [[ -n "$others" ]]; then echo "STOP: another job is using the GPU:"; echo "$others"; exit 2; fi
if (( free_mb < 5000 )); then echo "STOP: only ${free_mb} MiB VRAM free (< 5000)"; exit 2; fi
echo "GPU free: ${free_mb} MiB, no other compute jobs"

# Nix-built binary + the host NVIDIA driver library (only libcuda / ptxjit, nothing else from the host)
export LD_LIBRARY_PATH="$PWD/tools/hostcuda"
exec tools/llama-build/bin/llama-server -m "$MODEL" --mmproj "$MMPROJ" \
  --host 127.0.0.1 --port "$PORT" -ngl 999 -c "$CTX" --jinja --flash-attn on \
  --threads 4 --parallel 1 --no-webui "$@"
