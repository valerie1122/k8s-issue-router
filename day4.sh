#!/bin/bash
# Day 4 runbook. In YOUR Terminal:  cd ~/Documents/k8s-issue-router && bash day4.sh
# ~15-30 min (mostly waiting on the Sonnet batch). Cost: ~$5 Sonnet batch + ~$0.4 agent.
set -euo pipefail
cd "$(dirname "$0")"
set -a; source .env; set +a

echo "== 1/4 Sonnet 5.5 zero-shot on val+test (Batch API, runs on Anthropic's side) =="
python3 b2_zeroshot.py submit --model claude-sonnet-5-5

echo "== 2/4 agent prompt v2 on the issues the cascade hands to a human (~10 min) =="
python3 agent_eval.py run --prompt v2

echo "== 3/4 wait for Sonnet and collect =="
python3 b2_zeroshot.py collect --wait

echo "== 4/4 redeploy the shadow service (config v4: too-short rule) =="
python3 -m modal deploy shadow.py

echo
python3 agent_eval.py score --prompt v1
python3 agent_eval.py score --prompt v2
echo "ALL DONE. Paste everything from '== agent prompt v1 ==' down to Claude."
