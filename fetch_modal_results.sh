#!/bin/bash
# After training finishes: download embeddings + logits into data/  (run in your own Terminal)
set -euo pipefail
cd "$(dirname "$0")"
set -a; source .env; set +a
mkdir -p data/emb data/runs
python3 -m modal volume get k8s-router /emb data/ --force
python3 -m modal volume get k8s-router /runs data/ --force
ls -R data/emb data/runs
