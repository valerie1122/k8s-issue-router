#!/bin/bash
# Day 3 runbook. Run in YOUR Terminal:  cd ~/Documents/k8s-issue-router && bash day3.sh
# Takes ~15 min, mostly waiting on Modal. Safe to re-run: every step skips or overwrites cleanly.
set -euo pipefail
cd "$(dirname "$0")"
set -a; source .env; set +a
MODAL="python3 -m modal"

echo "== 1/5 re-send the 30 failed Haiku requests (runs in the background on Anthropic) =="
python3 b2_zeroshot.py submit --retry

echo "== 2/5 ModernBERT x3 seeds on Modal, saves the models (~8 min) =="
$MODAL volume put k8s-router data/processed/dataset.jsonl /dataset.jsonl --force
$MODAL run modal_app.py::seeds

echo "== 3/5 store keys on Modal for the shadow service =="
$MODAL secret create k8s-router --from-dotenv .env --force

echo "== 4/5 deploy the shadow service (polls every 10 min, backfills hourly) =="
$MODAL deploy shadow.py
$MODAL run shadow.py::poll        # first run now: records the start time

echo "== 5/5 collect the Haiku retry and download results =="
python3 b2_zeroshot.py collect --retry --wait
mkdir -p data/runs
for s in 0 1 2; do
  $MODAL volume get k8s-router /runs/modernbert-s$s/meta.json data/runs/modernbert-s$s/ --force
  $MODAL volume get k8s-router /runs/modernbert-s$s/val_logits.npy data/runs/modernbert-s$s/ --force
  $MODAL volume get k8s-router /runs/modernbert-s$s/test_logits.npy data/runs/modernbert-s$s/ --force
done
echo
echo "ALL DONE. Tell Claude: day3 done."
