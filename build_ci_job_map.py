"""Map Prow CI job names to SIGs from kubernetes/test-infra job configs.
Owner = the sig-* TestGrid dashboard the job reports to (annotation testgrid-dashboards), falling back
to the sig-* directory the config lives in (config/jobs/kubernetes/sig-node/...).

    git clone --depth 1 https://github.com/kubernetes/test-infra /tmp/test-infra
    python build_ci_job_map.py /tmp/test-infra     # -> data/raw/ci_job_map.json
"""
import json
import os
import re
import sys
from collections import Counter

import yaml

root = os.path.join(sys.argv[1], "config", "jobs")
SIG = re.compile(r"\bsig-([a-z0-9-]+?)(?:-(?:master|release|presubmits|periodics|blocking|informing|[0-9.]+)\b|$|[^a-z0-9-])")
valid = None
if os.path.exists("data/raw/sigs.yaml"):
    valid = {s["label"] for s in yaml.safe_load(open("data/raw/sigs.yaml"))["sigs"]}


def sigs_from(text):
    out = []
    for m in re.finditer(r"sig-([a-z0-9-]+)", text or ""):
        cand = m.group(1)
        # longest valid prefix: "node-containerd" -> "node", "cluster-lifecycle-kubeup" -> "cluster-lifecycle"
        parts = cand.split("-")
        for k in range(len(parts), 0, -1):
            s = "-".join(parts[:k])
            if valid is None or s in valid:
                out.append(s)
                break
    return out


def jobs_in(doc):
    for kind in ("periodics",):
        for j in doc.get(kind) or []:
            yield j
    for kind in ("presubmits", "postsubmits"):
        for _, js in (doc.get(kind) or {}).items():
            for j in js or []:
                yield j


jobs = {}
for d, _, files in os.walk(root):
    for f in files:
        if not f.endswith((".yaml", ".yml")):
            continue
        path = os.path.join(d, f)
        try:
            docs = list(yaml.safe_load_all(open(path)))
        except yaml.YAMLError:
            continue
        dir_sigs = sigs_from(os.path.relpath(path, root))
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            for j in jobs_in(doc):
                name = j.get("name")
                if not name:
                    continue
                ann = (j.get("annotations") or {})
                c = Counter(sigs_from(ann.get("testgrid-dashboards", "")))
                if not c:
                    c = Counter(dir_sigs)
                if c:
                    jobs[name] = [s for s, _ in c.most_common(3)]
os.makedirs("data/raw", exist_ok=True)
json.dump(jobs, open("data/raw/ci_job_map.json", "w"), indent=0)
print(f"{len(jobs)} jobs mapped; owners: {Counter(v[0] for v in jobs.values()).most_common(8)}")
