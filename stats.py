"""Day-1 data stats: is there enough traffic, and how much of it really needs routing?

Usage:
    python stats.py data/raw/issues_6m.jsonl
"""

import json
import re
import statistics
import sys
from collections import Counter
from datetime import datetime, timedelta

SIG_CMD = re.compile(r"^\s*/sig\s+(\S+)", re.MULTILINE)
WINDOW = timedelta(days=30)


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def sig_labels_at(issue, cutoff):
    """sig/* labels present at `cutoff`, replaying label/unlabel events."""
    cur = set()
    for e in sorted(issue["timeline"], key=lambda e: e["at"]):
        if ts(e["at"]) > cutoff:
            break
        lab = e.get("label") or ""
        if not lab.startswith("sig/"):
            continue
        if e["type"] == "labeled":
            cur.add(lab)
        elif e["type"] == "unlabeled":
            cur.discard(lab)
    return cur


def first_sig_label(issue):
    for e in sorted(issue["timeline"], key=lambda e: e["at"]):
        if e["type"] == "labeled" and (e.get("label") or "").startswith("sig/"):
            return e
    return None


def who_decided(issue):
    """Who typed the first /sig command in a comment: 'author', 'other', or None."""
    for e in sorted(issue["timeline"], key=lambda e: e["at"]):
        if e["type"] == "comment" and SIG_CMD.search(e["body"]):
            return "author" if e["actor"] == issue["author"] else "other"
    return None


def pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "n/a"


def main(path):
    issues = [json.loads(l) for l in open(path)]
    if not issues:
        sys.exit("no issues")
    now = max(ts(i["updated_at"]) for i in issues)
    n = len(issues)

    # --- volume -----------------------------------------------------------
    weeks = Counter(ts(i["created_at"]).strftime("%G-W%V") for i in issues)
    full = sorted(weeks)[1:-1] or sorted(weeks)  # drop partial first/last week
    wk = [weeks[w] for w in full]
    print(f"Issues: {n}  ({min(i['created_at'] for i in issues)[:10]} .. "
          f"{max(i['created_at'] for i in issues)[:10]})")
    print(f"New issues/week: median {statistics.median(wk):.0f}, "
          f"min {min(wk)}, max {max(wk)}  over {len(wk)} full weeks")

    # --- who needs routing -------------------------------------------------
    body_sig = [i for i in issues if SIG_CMD.search(i["body"])]
    needs_sig = [i for i in issues
                 if any(e["type"] == "labeled" and e["label"] == "needs-sig"
                        for e in i["timeline"])]
    triage = [i for i in issues if not SIG_CMD.search(i["body"])]
    print(f"\nAuthor wrote /sig in body:     {len(body_sig):5d}  {pct(len(body_sig), n)}")
    print(f"Ever labeled needs-sig:         {len(needs_sig):5d}  {pct(len(needs_sig), n)}")
    print(f"No /sig in body (target set):   {len(triage):5d}  {pct(len(triage), n)}")

    # --- gold labels (only issues old enough to have a full 30-day window) --
    mature = [i for i in triage if ts(i["created_at"]) + WINDOW <= now]
    gold = {i["number"]: sig_labels_at(i, ts(i["created_at"]) + WINDOW) for i in mature}
    labeled = [i for i in mature if gold[i["number"]]]
    multi = [i for i in labeled if len(gold[i["number"]]) > 1]
    print(f"\nTarget set, >=30 days old:      {len(mature):5d}")
    print(f"  got a sig label within 30d:   {len(labeled):5d}  {pct(len(labeled), len(mature))}"
          "   <- usable as gold")
    print(f"  multi-SIG among those:        {len(multi):5d}  {pct(len(multi), len(labeled))}")
    print(f"  never got a sig in 30d:       {len(mature) - len(labeled):5d}")

    # --- human baseline ---------------------------------------------------
    delays, deciders = [], Counter()
    for i in labeled:
        e = first_sig_label(i)
        if e:
            delays.append((ts(e["at"]) - ts(i["created_at"])).total_seconds() / 3600)
        deciders[who_decided(i) or "no /sig comment"] += 1
    if delays:
        q = statistics.quantiles(delays, n=4)
        print(f"\nHours from creation to first sig label (target set): "
              f"p25 {q[0]:.1f}, median {statistics.median(delays):.1f}, p75 {q[2]:.1f}")
    print("Who typed the first /sig comment:", dict(deciders))

    # --- label distribution -----------------------------------------------
    dist = Counter(s for i in labeled for s in gold[i["number"]])
    print(f"\nSIGs seen: {len(dist)}. Top 15:")
    for s, c in dist.most_common(15):
        print(f"  {s:32s} {c:5d}  {pct(c, len(labeled))}")
    tail = [s for s, c in dist.items() if c < 10]
    print(f"SIGs with <10 examples: {len(tail)} -> {sorted(tail)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/raw/issues.jsonl")
