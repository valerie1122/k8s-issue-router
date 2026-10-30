#!/bin/bash
# Re-run Haiku twice with the same prompt and measure how stable its answers are (~$0.5, 10-40 min).
set -euo pipefail
cd "$(dirname "$0")"
for r in r2 r3; do
  python3 b2_zeroshot.py submit --run $r
done
for r in r2 r3; do
  python3 b2_zeroshot.py collect --wait --run $r
done
python3 stability.py r2 r3
