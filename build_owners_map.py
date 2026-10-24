"""Map code directories in kubernetes/kubernetes to SIGs using the sig/* labels in OWNERS files.

    git clone --depth 1 https://github.com/kubernetes/kubernetes /tmp/k8s
    python build_owners_map.py /tmp/k8s      # -> data/raw/owners_map.json
"""
import json
import os
import sys

import yaml

root = sys.argv[1]
out = {}
for d, _, files in os.walk(root):
    parts = d.split(os.sep)
    if ".git" in parts or "vendor" in parts or "OWNERS" not in files:
        continue
    try:
        y = yaml.safe_load(open(os.path.join(d, "OWNERS"))) or {}
    except yaml.YAMLError:
        continue
    labels = sorted({l[4:] for l in (y.get("labels") or []) if isinstance(l, str) and l.startswith("sig/")})
    if labels:
        out[os.path.relpath(d, root)] = labels
os.makedirs("data/raw", exist_ok=True)
json.dump(out, open("data/raw/owners_map.json", "w"), indent=0)
print(f"{len(out)} directories with SIG labels")
