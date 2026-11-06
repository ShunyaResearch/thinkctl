#!/usr/bin/env bash
# Get Holo3.1-4B (H Company, Apache-2.0) and make the files the demo uses:
#   models/gguf/Holo-3.1-4B-Q4_K_M.gguf  and  models/gguf/Holo-3.1-4B-mmproj-F16.gguf
# H Company publishes the 4B only as BF16 safetensors, so the GGUF is converted locally with llama.cpp's converter
# (needs a CPU-only PyTorch env, created here and deleted afterwards) and quantized with llama-quantize.
# Needs: src/scripts/build_llama.sh done, the `hf` CLI (huggingface_hub), `uv`, about 25 GB free disk temporarily.
set -euo pipefail
cd "$(dirname "$0")/../.."  # repo root
LLAMA_TAG=${LLAMA_TAG:-b12135}
SRC=tools/src/llama.cpp-$LLAMA_TAG
OUT=models/gguf
TMP=models/tmp
[ -x tools/llama-build/bin/llama-quantize ] || { echo "run src/scripts/build_llama.sh first"; exit 1; }
mkdir -p "$OUT" "$TMP"

hf download Hcompany/Holo-3.1-4B --local-dir "$TMP/hf"

uv venv "$TMP/venv" --python 3.10 -q
CUDA_VISIBLE_DEVICES="" uv pip install --python "$TMP/venv/bin/python" -q --index-strategy unsafe-best-match \
  -r "$SRC/requirements/requirements-convert_hf_to_gguf.txt"
PY="$TMP/venv/bin/python"
export CUDA_VISIBLE_DEVICES=""
# --no-mtp: the Holo checkpoint has no multi-token-prediction layers (the Qwen3.5 converter expects them)
"$PY" "$SRC/convert_hf_to_gguf.py" "$TMP/hf" --no-mtp --outtype bf16 --outfile "$TMP/Holo-3.1-4B-BF16.gguf"
"$PY" "$SRC/convert_hf_to_gguf.py" "$TMP/hf" --mmproj --outtype f16 --outfile "$OUT/Holo-3.1-4B-mmproj-F16.gguf"
LD_LIBRARY_PATH="$PWD/tools/hostcuda" tools/llama-build/bin/llama-quantize \
  "$TMP/Holo-3.1-4B-BF16.gguf" "$OUT/Holo-3.1-4B-Q4_K_M.gguf" Q4_K_M
rm -rf "$TMP"
ls -la "$OUT"
