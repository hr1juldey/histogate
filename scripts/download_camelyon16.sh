#!/usr/bin/env bash
# P18-S1b: download 10 CAMELYON16 tumor WSIs from the AWS open-data mirror
# (s3://camelyon-dataset, CC0, anonymous). Resume-safe: curl -C -, each file
# tifffile-verified before .done marker.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw/camelyon16
BASE="https://camelyon-dataset.s3.amazonaws.com/CAMELYON16/images"
for i in 001 002 003 004 005 006 007 008 009 010; do
  name="tumor_${i}.tif"
  out="data/raw/camelyon16/$name"
  if [ -f "$out.done" ]; then echo "[skip] $name"; continue; fi
  echo "[get ] $name $(date +%H:%M:%S)"
  for attempt in 1 2 3; do
    curl -C - --retry 3 -sS -o "$out" "$BASE/$name" && break
    echo "[retry $attempt] $name"; sleep 5
  done
  uv run python -c "
import sys, tifffile
try:
    with tifffile.TiffFile('$out') as tf:
        print('OK $name pages', len(tf.pages), 'shape', tf.series[0].shape)
        open('$out.done','w').write('ok')
except Exception as e:
    print('BAD $name', e); sys.exit(1)
" || { echo "[FAILED] $name"; rm -f "$out"; }
done
echo "[done] $(ls data/raw/camelyon16/*.done 2>/dev/null | wc -l)/10 verified"
du -sh data/raw/camelyon16
