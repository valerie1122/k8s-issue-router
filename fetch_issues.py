"""Pull kubernetes/kubernetes issues + label/comment timeline via GitHub GraphQL.

Usage:
    python fetch_issues.py --months 6 --out data/raw/issues_6m.jsonl
    python fetch_issues.py --since 2023-10-01 --out data/raw/issues_3y.jsonl

Issues are walked newest-first by createdAt and the walk stops at the cutoff.
Each issue is written as one JSON line. Re-running resumes: issue numbers
already in the output file are skipped, so a crash or rate-limit stop is safe.

Auth: GITHUB_TOKEN env var, else `gh auth token`, else no header (works when
a proxy injects auth).
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

API = "https://api.github.com/graphql"

TIMELINE_FIELDS = """
  pageInfo { hasNextPage endCursor }
  nodes {
    __typename
    ... on LabeledEvent   { createdAt actor { login } label { name } }
    ... on UnlabeledEvent { createdAt actor { login } label { name } }
    ... on IssueComment   { createdAt author { login } body }
  }
"""

ISSUES_QUERY = """
query($owner: String!, $name: String!, $cursor: String, $pageSize: Int!) {
  rateLimit { cost remaining resetAt }
  repository(owner: $owner, name: $name) {
    issues(first: $pageSize, after: $cursor,
           orderBy: {field: CREATED_AT, direction: DESC}) {
      pageInfo { hasNextPage endCursor }
      nodes {
        id number title body createdAt updatedAt closedAt state
        lastEditedAt
        author { login }
        authorAssociation
        labels(first: 50) { nodes { name } }
        userContentEdits(first: 1) { totalCount }
        timelineItems(first: 100,
          itemTypes: [LABELED_EVENT, UNLABELED_EVENT, ISSUE_COMMENT]) {
          %s
        }
      }
    }
  }
}
""" % TIMELINE_FIELDS

TIMELINE_MORE_QUERY = """
query($id: ID!, $cursor: String) {
  node(id: $id) {
    ... on Issue {
      timelineItems(first: 100, after: $cursor,
        itemTypes: [LABELED_EVENT, UNLABELED_EVENT, ISSUE_COMMENT]) {
        %s
      }
    }
  }
}
""" % TIMELINE_FIELDS


def load_dotenv(path=".env"):
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def get_token():
    load_dotenv()
    tok = os.environ.get("GITHUB_TOKEN")
    if tok:
        return tok
    try:
        out = subprocess.run(["gh", "auth", "token"], capture_output=True,
                             text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


class GraphQL:
    def __init__(self, token):
        self.s = requests.Session()
        if token:
            self.s.headers["Authorization"] = f"bearer {token}"
        self.s.headers["User-Agent"] = "k8s-issue-router"

    def run(self, query, variables, max_tries=8):
        delay = 2.0
        for attempt in range(1, max_tries + 1):
            try:
                r = self.s.post(API, json={"query": query, "variables": variables},
                                timeout=60)
            except requests.RequestException as e:
                print(f"  network error ({e}); retry in {delay:.0f}s", file=sys.stderr)
                time.sleep(delay); delay = min(delay * 2, 300); continue

            # Secondary rate limit / abuse detection
            if r.status_code in (403, 429):
                wait = int(r.headers.get("Retry-After", 0)) or delay
                print(f"  HTTP {r.status_code}: {r.text[:200]}; wait {wait:.0f}s",
                      file=sys.stderr)
                time.sleep(wait); delay = min(delay * 2, 300); continue
            if r.status_code >= 500:
                # Big pages sometimes time out server-side; caller may shrink page
                print(f"  HTTP {r.status_code}; retry in {delay:.0f}s", file=sys.stderr)
                time.sleep(delay); delay = min(delay * 2, 300); continue
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")

            data = r.json()
            if "errors" in data:
                msg = json.dumps(data["errors"])[:500]
                if "RATE_LIMITED" in msg:
                    print(f"  rate limited; wait {delay:.0f}s", file=sys.stderr)
                    time.sleep(delay); delay = min(delay * 2, 300); continue
                raise RuntimeError(f"GraphQL errors: {msg}")

            rl = data["data"].get("rateLimit")
            if rl and rl["remaining"] < 100:
                reset = datetime.fromisoformat(rl["resetAt"].replace("Z", "+00:00"))
                sleep_s = max(0, (reset - datetime.now(timezone.utc)).total_seconds()) + 5
                print(f"  rate budget low ({rl['remaining']}); sleeping {sleep_s:.0f}s",
                      file=sys.stderr)
                time.sleep(sleep_s)
            return data["data"]
        raise RuntimeError("gave up after retries")


def flatten_timeline(nodes):
    out = []
    for n in nodes:
        t = n["__typename"]
        if t in ("LabeledEvent", "UnlabeledEvent"):
            out.append({
                "type": "labeled" if t == "LabeledEvent" else "unlabeled",
                "at": n["createdAt"],
                "actor": (n.get("actor") or {}).get("login"),
                "label": (n.get("label") or {}).get("name"),
            })
        elif t == "IssueComment":
            out.append({
                "type": "comment",
                "at": n["createdAt"],
                "actor": (n.get("author") or {}).get("login"),
                "body": n.get("body") or "",
            })
    return out


def fetch_rest_of_timeline(gql, issue_id, cursor):
    items = []
    while cursor:
        d = gql.run(TIMELINE_MORE_QUERY, {"id": issue_id, "cursor": cursor})
        tl = d["node"]["timelineItems"]
        items.extend(flatten_timeline(tl["nodes"]))
        cursor = tl["pageInfo"]["endCursor"] if tl["pageInfo"]["hasNextPage"] else None
    return items


def to_record(gql, n):
    tl = n["timelineItems"]
    timeline = flatten_timeline(tl["nodes"])
    if tl["pageInfo"]["hasNextPage"]:
        timeline += fetch_rest_of_timeline(gql, n["id"], tl["pageInfo"]["endCursor"])
    return {
        "id": n["id"],
        "number": n["number"],
        "title": n["title"],
        "body": n["body"] or "",
        "created_at": n["createdAt"],
        "updated_at": n["updatedAt"],
        "closed_at": n["closedAt"],
        "state": n["state"],
        "last_edited_at": n["lastEditedAt"],
        "edit_count": n["userContentEdits"]["totalCount"],
        "author": (n.get("author") or {}).get("login"),
        "author_association": n["authorAssociation"],
        "labels_now": [l["name"] for l in n["labels"]["nodes"]],
        "timeline": timeline,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="kubernetes/kubernetes")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--months", type=int, default=6)
    g.add_argument("--since", help="YYYY-MM-DD, createdAt cutoff")
    ap.add_argument("--out", default="data/raw/issues.jsonl")
    ap.add_argument("--page-size", type=int, default=25)
    args = ap.parse_args()

    owner, name = args.repo.split("/")
    if args.since:
        cutoff = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc)
    else:
        cutoff = datetime.now(timezone.utc) - timedelta(days=30 * args.months)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    if out.exists():
        with out.open() as f:
            for line in f:
                try:
                    seen.add(json.loads(line)["number"])
                except json.JSONDecodeError:
                    pass  # partial last line from an interrupted run
        print(f"resuming: {len(seen)} issues already in {out}")

    gql = GraphQL(get_token())
    cursor_file = out.with_suffix(".cursor")
    cursor = cursor_file.read_text().strip() if cursor_file.exists() else None
    if cursor:
        print("resuming from saved cursor")
    page_size, written, pages = args.page_size, 0, 0
    with out.open("a") as f:
        if out.stat().st_size and not out.read_bytes().endswith(b"\n"):
            f.write("\n")  # isolate a partial line left by an interrupted run
        while True:
            try:
                d = gql.run(ISSUES_QUERY, {"owner": owner, "name": name,
                                           "cursor": cursor, "pageSize": page_size})
            except RuntimeError as e:
                if page_size > 5:
                    page_size = max(5, page_size // 2)
                    print(f"  shrinking page to {page_size} after: {e}", file=sys.stderr)
                    continue
                raise
            conn = d["repository"]["issues"]
            pages += 1
            stop = False
            for n in conn["nodes"]:
                created = datetime.fromisoformat(n["createdAt"].replace("Z", "+00:00"))
                if created < cutoff:
                    stop = True
                    break
                if n["number"] in seen:
                    continue
                f.write(json.dumps(to_record(gql, n)) + "\n")
                seen.add(n["number"])
                written += 1
            f.flush()
            last = conn["nodes"][-1]["createdAt"] if conn["nodes"] else "-"
            rl = d.get("rateLimit") or {}
            print(f"page {pages}: +{written} total new, oldest on page {last}, "
                  f"rate remaining {rl.get('remaining')}")
            if stop or not conn["pageInfo"]["hasNextPage"]:
                break
            cursor = conn["pageInfo"]["endCursor"]
            cursor_file.write_text(cursor)

    if cursor_file.exists():
        cursor_file.write_text("")
    print(f"done: wrote {written} new issues to {out} (cutoff {cutoff.date()})")


if __name__ == "__main__":
    main()
