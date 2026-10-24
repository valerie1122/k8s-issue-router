#!/bin/bash
# Run GPU jobs on Modal from YOUR OWN Terminal (Modal's gRPC connection does not work through Claude's sandbox proxy).
# Usage:  cd ~/Documents/k8s-issue-router && bash run_modal.sh
set -euo pipefail
cd "$(dirname "$0")"
set -a; source .env; set +a                     # MODAL_TOKEN_ID / MODAL_TOKEN_SECRET
python3 -m pip install --quiet --user modal 2>/dev/null || pip3 install --quiet modal
MODAL="python3 -m modal"

$MODAL volume create k8s-router 2>/dev/null || true
$MODAL volume put k8s-router data/processed/dataset.jsonl /dataset.jsonl --force

echo "== 1/2 embeddings (about 2 min) =="
$MODAL run modal_app.py::embed

echo "== 2/2 ModernBERT fine-tune, runs on Modal in the background (about 20-30 min) =="
$MODAL run --detach modal_app.py::train
echo
echo "Done launching. Training continues on Modal even if you close this window."
echo "Watch it at https://modal.com/apps  — tell Claude when it shows 'best val top1'."
