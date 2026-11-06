"""Repo paths and settings. Local, git-ignored items live next to the code (usually symlinks to a folder outside
the repo): data/ (datasets), models/ (Holo GGUF + decoder files), tools/ (llama.cpp build), logs/ (run logs).
config.json is personal (test address etc.) and git-ignored; config.example.json is the committed template."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def load_config():
    p = REPO / "config.json"
    return json.load(open(p if p.exists() else REPO / "config.example.json"))
