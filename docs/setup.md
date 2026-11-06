# Setup

Tested on Linux with an X11 session, Python 3.10, the Nix package manager and an NVIDIA GPU with at least 6 GB of
memory (the GPU is used only by the model server). Datasets, models and builds are not in the repository; the
scripts below create them in git-ignored folders (`data/`, `models/`, `tools/`, `logs/`). Each of these can also be a
symlink to a folder outside the repo.

## 1. Python environment
```bash
uv venv .venv --python 3.10            # or: python3 -m venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -I -m pytest tests -q  # offline tests (no data, no model, no screen)
```

## 2. System tools (no global installs)
```bash
nix develop path:.        # maim, scrot, xdotool, xbindkeys, wmctrl, ffmpeg, imagemagick
nix develop path:.#build  # CUDA 12.8 toolchain (only used by build_llama.sh)
```
Run the demo commands from inside `nix develop path:.`.

## 3. Datasets (about 10 GB)
```bash
bash src/scripts/download_data.sh all     # or: nakanishi | benchmark | beta | wearable
```
Downloads into `data/` from public mirrors and verifies checksums where the mirror provides them. Files are only
downloaded, never executed. The demo needs only `wearable`.

## 4. llama.cpp (CUDA)
```bash
bash src/scripts/build_llama.sh           # CUDA_ARCH=86 by default (RTX 30xx); set CUDA_ARCH for other GPUs
```
Builds a pinned llama.cpp tag in the Nix build shell into `tools/llama-build/` and links the host NVIDIA driver
library into `tools/hostcuda/`.

## 5. Model: Holo3.1-4B
```bash
bash src/scripts/get_model.sh             # needs the `hf` CLI and `uv`; about 25 GB free temporarily
```
Downloads the official weights from Hugging Face, converts them to GGUF (BF16 → Q4_K_M, plus the vision projector)
into `models/gguf/` and removes the intermediates. Result: about 3.7 GB, about 4.3 GB of GPU memory at a 16k context.

## 6. Decoder files (CPU)
```bash
.venv/bin/python -I src/band/build_models.py --set A     # templates + thresholds, menu 9.25 ... 14.25 Hz
.venv/bin/python -I src/band/build_models.py --set B     # alternate menu 9.75 ... 14.75 Hz
.venv/bin/python -I src/band/tune_swap.py                # combine / swap thresholds (2-fold)
```
Writes `models/decoder/`. Uses all CPU cores but one.

## 7. Configuration
```bash
cp config.example.json config.json        # config.json is git-ignored
```
Set `test_email_address` and `test_sender_name` to the account the agent may reply to. The other keys hold the
safety lists (words and keys that are never used without an explicit goal) and the agent limits.
