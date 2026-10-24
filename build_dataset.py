"""python build_dataset.py -> data/processed/dataset.jsonl + printed summary"""
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from router.data import build

SRC = "data/interim/issues_3y_orig.jsonl"
OUT = Path("data/processed/dataset.jsonl")

now = max(datetime.fromisoformat(json.loads(l)["updated_at"].replace("Z", "+00:00"))
          for l in open(SRC))
rows = build(SRC, "data/raw/sigs.yaml", now)
OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

print(f"wrote {len(rows)} rows to {OUT} (now={now:%Y-%m-%d})\n")
print(f"{'split':6s} {'all':>5s} {'gold':>5s} {'no-sig':>6s} {'too-new':>7s}   routed_by (labeled only)")
for sp in ("train", "val", "test"):
    r = [x for x in rows if x["split"] == sp]
    lab = [x for x in r if x["gold"]]
    print(f"{sp:6s} {len(r):5d} {len(lab):5d} {sum(1 for x in r if x['gold'] == []):6d} "
          f"{sum(1 for x in r if x['gold'] is None):7d}   {dict(Counter(x['routed_by'] for x in lab))}")
lab = [x for x in rows if x["gold"]]
print("\nmulti-SIG share:", f"{sum(len(x['gold']) > 1 for x in lab) / len(lab):.1%}")
print("text length (chars) median:", sorted(len(x["text"]) for x in rows)[len(rows) // 2])
leak = [x for x in rows if any(f"/sig {s}" in x["text"] for s in (x["gold"] or []))]
print("rows where '/sig <gold>' survives cleaning:", len(leak))
