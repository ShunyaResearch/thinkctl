#!/usr/bin/env bash
# Download SSVEP datasets into data/ from public mirrors (see README references).
# Usage: bash src/scripts/download_data.sh [nakanishi|benchmark|wearable|beta|all]
# Files are only downloaded, never executed. Checksums are verified where the mirror provides them.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DATA="$ROOT/data"
WHAT="${1:-all}"
PY="python3 -I"
mkdir -p "$DATA"

get() {  # url dest
  [ -s "$2" ] && return 0
  curl -sSL --retry 5 --retry-delay 5 --max-time 3600 -o "$2.part" "$1" && mv "$2.part" "$2"
}
export -f get

if [[ $WHAT == nakanishi || $WHAT == all ]]; then
  [ -d "$DATA/nakanishi2015" ] || git clone --depth 1 https://github.com/mnakanishi/12JFPM_SSVEP "$DATA/nakanishi2015"
fi

if [[ $WHAT == benchmark || $WHAT == all ]]; then
  # D2: Wang et al. 2016 Benchmark. Original S*.mat files, mirrored in NEMAR nm000126 (sourcedata/). CC-BY-4.0.
  D="$DATA/benchmark"; mkdir -p "$D"
  B=https://data.nemar.org/nm000126/v1.0.3
  get "$B/manifest.json" "$D/nemar_manifest.json"
  for f in README.md dataset_description.json participants.tsv sourcedata/README.md sourcedata/sourcedata_provenance.json \
           sub-1/ses-0/eeg/sub-1_ses-0_task-ssvep_run-0_channels.tsv sub-1/ses-0/eeg/sub-1_ses-0_task-ssvep_run-0_events.tsv; do
    get "$B/$f" "$D/meta_$(echo "$f" | tr '/' '_')" || echo "warn: missing $f"
  done
  $PY -c "
import json,sys
m=json.load(open('$D/nemar_manifest.json'))
for f in m:
    if f['path'].startswith('sourcedata/S') and f['path'].endswith('.mat'): print(f['url'], '$D/'+f['path'].split('/')[-1])
" | xargs -P 4 -n 2 bash -c 'get "$0" "$1"'
  # verify git-blob sha1 checksums from the manifest
  $PY - "$D" <<'EOF'
import json, hashlib, sys, os
d = sys.argv[1]; bad = 0
for f in json.load(open(f"{d}/nemar_manifest.json")):
    p = f["path"]
    if not (p.startswith("sourcedata/S") and p.endswith(".mat")): continue
    fn = os.path.join(d, p.split("/")[-1]); h = hashlib.sha1()
    h.update(f"blob {os.path.getsize(fn)}\0".encode())
    with open(fn, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""): h.update(chunk)
    ok = (f.get("checksum_algorithm") != "git") or h.hexdigest() == f["checksum"]
    bad += not ok
    if not ok: print("CHECKSUM FAIL", fn)
print("benchmark checksum failures:", bad)
EOF
fi

if [[ $WHAT == wearable || $WHAT == all ]]; then
  # D4: Zhu et al. 2021 Wearable SSVEP. Figshare article 13560281 (authors' upload). CC-BY-4.0.
  D="$DATA/wearable"; mkdir -p "$D"
  get https://api.figshare.com/v2/articles/13560281 "$D/figshare_article.json"
  $PY -c "
import json
a=json.load(open('$D/figshare_article.json'))
for f in a['files']: print(f['download_url'], '$D/'+f['name'])
" | xargs -P 4 -n 2 bash -c 'get "$0" "$1"'
  $PY - "$D" <<'EOF'
import json, hashlib, sys, os
d = sys.argv[1]; bad = 0
for f in json.load(open(f"{d}/figshare_article.json"))["files"]:
    h = hashlib.md5(open(os.path.join(d, f["name"]), "rb").read()).hexdigest()
    if f.get("computed_md5") and h != f["computed_md5"]: bad += 1; print("CHECKSUM FAIL", f["name"])
print("wearable checksum failures:", bad)
EOF
fi

if [[ $WHAT == beta || $WHAT == all ]]; then
  # D3: Liu et al. 2020 BETA. Hugging Face dataset Bingchuan/BETA (first author). CC-BY-4.0.
  D="$DATA/beta"; mkdir -p "$D"
  get "https://huggingface.co/api/datasets/Bingchuan/BETA/tree/main" "$D/hf_tree.json"
  $PY -c "
import json
for f in json.load(open('$D/hf_tree.json')):
    if f['type']=='file' and not f['path'].startswith('.'):
        print('https://huggingface.co/datasets/Bingchuan/BETA/resolve/main/'+f['path'], '$D/'+f['path'])
" | xargs -P 4 -n 2 bash -c 'get "$0" "$1"'
  $PY - "$D" <<'EOF'
import json, hashlib, sys, os
d = sys.argv[1]; bad = 0
for f in json.load(open(f"{d}/hf_tree.json")):
    if f["type"] != "file" or f["path"].startswith("."): continue
    fn = os.path.join(d, f["path"])
    if "lfs" in f:
        h = hashlib.sha256()
        with open(fn, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""): h.update(chunk)
        if h.hexdigest() != f["lfs"]["oid"]: bad += 1; print("CHECKSUM FAIL", fn)
print("beta checksum failures:", bad)
EOF
fi
echo "done: $WHAT"; du -sh "$DATA"/* 2>/dev/null
