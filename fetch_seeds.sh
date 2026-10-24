#!/bin/bash
# Download the 3-seed ModernBERT results (logits + meta only, not the 600 MB models).
set -euo pipefail
cd "$(dirname "$0")"
set -a; source .env; set +a
for s in 0 1 2; do
  d=data/runs/modernbert-s$s
  [ -f "$d" ] && rm -f "$d"          # clean up the broken download from day3.sh
  mkdir -p "$d"
  for f in meta.json val_logits.npy test_logits.npy; do
    python3 -m modal volume get k8s-router /runs/modernbert-s$s/$f "$d/$f" --force
  done
done
python3 -m modal run shadow.py::report
echo "DONE"
