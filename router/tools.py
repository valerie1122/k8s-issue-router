"""The agent's three tools. Plain functions so they can be called in-process or served over MCP
(router_mcp.py serves exactly these). Everything is local data: no network calls at tool time.

Leakage guard: every tool that looks at history takes `before` (an ISO timestamp) and only sees
issues whose first SIG label already existed at that moment, i.e. what a triager could have seen.
"""
import json
import re
from datetime import datetime, timedelta
from functools import lru_cache

import yaml

DATASET = "data/processed/dataset.jsonl"
SIGS = "data/raw/sigs.yaml"
OWNERS_MAP = "data/raw/owners_map.json"


def _ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


@lru_cache(maxsize=1)
def _history():
    from sklearn.feature_extraction.text import TfidfVectorizer
    rows = [json.loads(l) for l in open(DATASET)]
    rows = [r for r in rows if r["gold"]]
    vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2, max_features=200_000,
                          token_pattern=r"(?u)\b[\w./-]{2,}\b")
    X = vec.fit_transform([r["text"] for r in rows])
    known_at = [_ts(r["created_at"]) + timedelta(hours=r["hours_to_sig"] or 0) for r in rows]
    return rows, vec, X, known_at


def search_similar_issues(query: str, k: int = 8, before: str = "") -> list:
    """Most similar past kubernetes/kubernetes issues and the SIGs humans gave them.
    High similarity to a recent issue can mean a duplicate."""
    rows, vec, X, known_at = _history()
    cutoff = _ts(before) if before else None
    sims = (X @ vec.transform([query]).T).toarray().ravel()
    out = []
    for i in sims.argsort()[::-1]:
        if sims[i] <= 0 or len(out) >= k:
            break
        if cutoff and (known_at[i] >= cutoff or rows[i]["number"] == 0):
            continue
        r = rows[i]
        out.append({"number": r["number"], "title": r["text"].split("\n")[0][:140],
                    "sigs": r["gold"], "created_at": r["created_at"][:10], "state": r["state"],
                    "similarity": round(float(sims[i]), 3)})
    return out


@lru_cache(maxsize=1)
def _sigs():
    return {s["label"]: s for s in yaml.safe_load(open(SIGS))["sigs"]}


def get_sig_charter(sig: str) -> dict:
    """What a SIG owns: mission statement and subproject names."""
    s = _sigs().get(sig)
    if not s:
        return {"error": f"unknown SIG '{sig}'", "valid": sorted(_sigs())}
    return {"sig": sig, "name": s["name"],
            "mission": " ".join((s.get("mission_statement") or "").split())[:1500],
            "subprojects": [p["name"] for p in (s.get("subprojects") or [])]}


@lru_cache(maxsize=1)
def _owners():
    return json.load(open(OWNERS_MAP))  # {"pkg/kubelet/cm": ["node"], ...}


def lookup_owners(term: str) -> dict:
    """Which SIGs own code paths matching a path or component name (e.g. 'pkg/kubelet', 'kube-proxy',
    'csi'). Built from the sig/* labels in kubernetes/kubernetes OWNERS files."""
    term = term.strip().strip("/").lower()
    if len(term) < 3:
        return {"error": "term too short"}
    tokens = re.split(r"[/\s]+", term)
    hits = {}
    for path, sigs in _owners().items():
        p = path.lower()
        if term in p or all(t in p for t in tokens):
            for s in sigs:
                hits.setdefault(s, []).append(path)
    if not hits and "/" in term:  # no labeled dir under this path: use the nearest labeled ancestor
        parts = term.split("/")
        for n in range(len(parts) - 1, 0, -1):
            anc = "/".join(parts[:n])
            if anc in _owners():
                for s in _owners()[anc]:
                    hits.setdefault(s, []).append(anc + " (ancestor)")
                break
    ranked = sorted(hits.items(), key=lambda kv: -len(kv[1]))[:5]
    return {"term": term, "matches": [{"sig": s, "n_paths": len(ps), "example_paths": ps[:4]} for s, ps in ranked]}


TOOLS = {
    "search_similar_issues": {
        "fn": search_similar_issues,
        "description": search_similar_issues.__doc__.strip(),
        "input_schema": {"type": "object", "additionalProperties": False,
                         "properties": {"query": {"type": "string", "description": "issue title plus key error text"},
                                        "k": {"type": "integer", "description": "how many results, default 8"}},
                         "required": ["query"]}},
    "get_sig_charter": {
        "fn": get_sig_charter,
        "description": get_sig_charter.__doc__.strip(),
        "input_schema": {"type": "object", "additionalProperties": False,
                         "properties": {"sig": {"type": "string", "description": "SIG label, e.g. node"}},
                         "required": ["sig"]}},
    "lookup_owners": {
        "fn": lookup_owners,
        "description": lookup_owners.__doc__.strip(),
        "input_schema": {"type": "object", "additionalProperties": False,
                         "properties": {"term": {"type": "string", "description": "code path or component name"}},
                         "required": ["term"]}},
}
