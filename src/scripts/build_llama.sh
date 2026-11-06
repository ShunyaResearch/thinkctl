#!/usr/bin/env bash
# Build llama.cpp (pinned tag) with CUDA inside the Nix build shell; link the host NVIDIA driver library for runtime.
# Output: tools/llama-build/bin/{llama-server,llama-quantize,...} and tools/hostcuda/ (links to the host libcuda).
# Env: LLAMA_TAG (default b12135), CUDA_ARCH (default 86 = RTX 30xx; e.g. 89 for RTX 40xx).
set -euo pipefail
cd "$(dirname "$0")/../.."  # repo root
LLAMA_TAG=${LLAMA_TAG:-b12135}
CUDA_ARCH=${CUDA_ARCH:-86}
SRC=tools/src/llama.cpp-$LLAMA_TAG
if [ ! -d "$SRC" ]; then
  mkdir -p tools/src
  curl -sSL "https://github.com/ggml-org/llama.cpp/archive/refs/tags/$LLAMA_TAG.tar.gz" | tar xz -C tools/src
fi
JOBS=$(( $(nproc) > 1 ? $(nproc) - 1 : 1 ))
nix develop path:.#build --command bash -c "
  cmake -S $SRC -B tools/llama-build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
  && cmake --build tools/llama-build -j $JOBS --target llama-server llama-quantize"
# Nix-built binaries cannot see the host driver on non-NixOS systems: link only libcuda / ptxjit from the host.
mkdir -p tools/hostcuda
for lib in libcuda.so.1 libnvidia-ptxjitcompiler.so.1; do
  p=$(ldconfig -p | awk -v l="$lib" '$1 == l {print $NF; exit}')
  [ -n "$p" ] && ln -sf "$p" "tools/hostcuda/$lib" || echo "warning: $lib not found on the host"
done
echo "built: tools/llama-build/bin/llama-server"
