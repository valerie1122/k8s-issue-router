"""Turn raw issues into a routing dataset.

Each example is what a router could see at the moment the issue was created
(original title + original body, with routing commands removed), plus the
answer (SIG labels present 30 days later) and who did the routing.
"""
import json
import re
from datetime import datetime, timedelta

import yaml

WINDOW = timedelta(days=30)
# Prow commands that set or hint labels; remove so the answer is not in the input.
COMMAND_LINE = re.compile(
    r"^[ \t>]*/(sig|kind|area|priority|triage|wg|committee|label|remove-\S+|assign|cc|lifecycle)\b.*$",
    re.MULTILINE | re.IGNORECASE)
SIG_CMD = re.compile(r"^[ \t>]*/sig\s+([\w-]+)", re.MULTILINE | re.IGNORECASE)
SIG_TEAM = re.compile(r"@kubernetes/sig-([a-z0-9-]+?)(?:-(?:bugs|feature-requests|pr-reviews|misc|api-reviews|proposals|test-failures))?\b", re.I)
# Same commands written mid-line, in lists or pasted HTML (Prow ignores these, but they still leak).
INLINE_CMD = re.compile(r"(?<![\w/])/(sig|kind|area|wg|priority|triage|committee)\s+[\w/-]+", re.IGNORECASE)
HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)

SPLITS = [  # (name, start inclusive, end exclusive)
    ("train", "2000-01-01", "2026-04-01"),
    ("val", "2026-04-01", "2026-07-01"),
    ("test", "2026-07-01", "2100-01-01"),
]


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def load_sigs(path="data/raw/sigs.yaml"):
    d = yaml.safe_load(open(path))
    return {s["label"]: s for s in d["sigs"]}


def clean_text(title, body):
    body = HTML_COMMENT.sub("", body or "")          # issue templates hide instructions in comments
    body = COMMAND_LINE.sub("", body)
    body = INLINE_CMD.sub("", body)
    body = SIG_TEAM.sub("@kubernetes/[team]", body)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    return f"{title.strip()}\n\n{body}"


def sig_labels_at(issue, cutoff, prefix="sig/"):
    """Routing labels (prefix stripped) present at `cutoff`, replaying label/unlabel events."""
    cur = set()
    for e in sorted(issue["timeline"], key=lambda e: e["at"]):
        if ts(e["at"]) > cutoff:
            break
        lab = e.get("label") or ""
        if lab.startswith(prefix):
            (cur.add if e["type"] == "labeled" else cur.discard)(lab[len(prefix):])
    return cur


def first_label_by(issue, prefix, bots):
    """Who put the first routing label on: bot / author / other (a maintainer), or None."""
    for e in sorted(issue["timeline"], key=lambda e: e["at"]):
        if e["type"] == "labeled" and (e.get("label") or "").startswith(prefix):
            a = e.get("actor")
            return "bot" if a in bots else ("author" if a == issue["author"] else "other")
    return None


def routing_source(issue, body_for_routing):
    """Who chose the SIG: body / author_comment / other / unknown / none."""
    visible = HTML_COMMENT.sub("", body_for_routing)
    if SIG_CMD.search(visible) or SIG_TEAM.search(visible):
        return "body"
    for e in sorted(issue["timeline"], key=lambda e: e["at"]):
        if e["type"] == "comment" and (SIG_CMD.search(e["body"]) or SIG_TEAM.search(e["body"])):
            return "author_comment" if e["actor"] == issue["author"] else "other"
    return None


def first_sig_label_hours(issue, prefix="sig/"):
    for e in sorted(issue["timeline"], key=lambda e: e["at"]):
        if e["type"] == "labeled" and (e.get("label") or "").startswith(prefix):
            return (ts(e["at"]) - ts(issue["created_at"])).total_seconds() / 3600
    return None


def split_of(created_at, splits=SPLITS):
    day = created_at[:10]
    return next(n for n, a, b in splits if a <= day < b)


def build(issues_path, sigs_path, now, repo="k8s"):
    """Dataset rows for one repo. k8s keeps its original semantics exactly; other repos use the
    generic rules: routing label = cfg prefix, routed_by = who applied the first one."""
    from router.repos import BOTS, REPOS, teams
    cfg = REPOS[repo]
    prefix = cfg["label_prefix"]
    valid = set(load_sigs(sigs_path)) if repo == "k8s" else {t for t, _ in teams(repo)}
    rows = []
    for line in open(issues_path):
        i = json.loads(line)
        created = ts(i["created_at"])
        mature = created + WINDOW <= now
        gold = sorted(s for s in sig_labels_at(i, created + WINDOW, prefix) if s in valid) if mature else None
        if repo == "k8s":
            src = routing_source(i, i["body"])
            if gold == [] and src is None:
                src = "none"
            preassigned = src == "body"
        else:
            rp = cfg.get("route_prefixes", prefix)
            src = first_label_by(i, rp, BOTS) or ("none" if gold == [] else "unknown")
            # a bot or the author labeled it at creation time: nobody had to route it
            preassigned = src in ("bot", "author")
            if gold == [] and cfg["none_class"] and "triaged" in i["labels_now"]:
                gold = ["none"]   # triaged, deliberately left without an oncall queue
        rows.append({
            "number": i["number"],
            "created_at": i["created_at"],
            "split": split_of(i["created_at"], cfg["splits"]),
            "text": clean_text(i["title"], i["body"]),
            "gold": gold,                 # None = too new to judge; [] = no SIG after 30 days
            "routed_by": src or "unknown",
            "preassigned": preassigned,
            "hours_to_sig": first_sig_label_hours(i, cfg.get("route_prefixes", prefix)),
            "body_source": i.get("body_source"),
            "state": i["state"],
            "labels_now": i["labels_now"],
        })
    return rows
