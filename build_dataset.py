"""python build_dataset.py [--repo k8s|pytorch] -> dataset jsonl + printed summary"""
import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from router.data import build
from router.repos import REPOS

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="k8s", choices=list(REPOS))
repo = ap.parse_args().repo
SRC = REPOS[repo]["issues"]
OUT = Path(REPOS[repo]["dataset"])

now = max(datetime.fromisoformat(json.loads(l)["updated_at"].replace("Z", "+00:00"))
          for l in open(SRC))
rows = build(SRC, "data/raw/sigs.yaml", now, repo)
OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

print(f"[{repo}] wrote {len(rows)} rows to {OUT} (now={now:%Y-%m-%d})\n")
print(f"{'split':6s} {'all':>5s} {'gold':>5s} {'empty':>6s} {'too-new':>7s}   routed_by (labeled only)")
for sp in ("train", "val", "test"):
    r = [x for x in rows if x["split"] == sp]
    lab = [x for x in r if x["gold"]]
    print(f"{sp:6s} {len(r):5d} {len(lab):5d} {sum(1 for x in r if x['gold'] == []):6d} "
          f"{sum(1 for x in r if x['gold'] is None):7d}   {dict(Counter(x['routed_by'] for x in lab))}")
lab = [x for x in rows if x["gold"]]
print("\nmulti-label share:", f"{sum(len(x['gold']) > 1 for x in lab) / len(lab):.1%}")
print("not pre-assigned (someone had to route it):", sum(1 for x in lab if not x["preassigned"]))
print("label distribution:", Counter(s for x in lab for s in x["gold"]).most_common(8))
