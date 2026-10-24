"""Recover each issue's title and body as they were at creation.

GitHub keeps body revisions in `userContentEdits` (newest first, so `last: 1`
is the original) and title changes as RenamedTitleEvent (`previousTitle` of the
first rename is the original title). Only issues with edits or renames need a
lookup; everything else already has its original text.

Usage:
    python fetch_originals.py data/raw/issues_3y.jsonl data/raw/originals_3y.jsonl
Output: one line per looked-up issue: {number, orig_title, orig_body, body_source}
Resumable: numbers already in the output are skipped.
"""
import json
import sys
from pathlib import Path

from fetch_issues import GraphQL, get_token

QUERY = """
query($ids: [ID!]!) {
  nodes(ids: $ids) {
    ... on Issue {
      number
      userContentEdits(last: 1) { totalCount nodes { diff createdAt } }
      timelineItems(first: 1, itemTypes: [RENAMED_TITLE_EVENT]) {
        nodes { ... on RenamedTitleEvent { previousTitle createdAt } }
      }
    }
  }
}
"""
BATCH = 40


def main(src, dst):
    issues = [json.loads(l) for l in open(src) if l.strip()]
    todo = issues  # all: titles can be renamed even when the body never was
    out = Path(dst)
    done = set()
    if out.exists():
        for l in out.open():
            try:
                done.add(json.loads(l)["number"])
            except json.JSONDecodeError:
                pass
    todo = [i for i in todo if i["number"] not in done]
    by_id = {i["id"]: i for i in issues}
    print(f"{len(todo)} issues to look up ({len(done)} already done)")

    gql = GraphQL(get_token())
    with out.open("a") as f:
        if out.stat().st_size and not out.read_bytes().endswith(b"\n"):
            f.write("\n")
        for k in range(0, len(todo), BATCH):
            ids = [i["id"] for i in todo[k:k + BATCH]]
            data = gql.run(QUERY, {"ids": ids})
            for node_id, n in zip(ids, data["nodes"]):
                cur = by_id[node_id]
                if not n:
                    continue
                edits = n["userContentEdits"]["nodes"]
                if edits and edits[0]["diff"] is not None:
                    body, src_tag = edits[0]["diff"], "edit_history"
                elif edits:
                    body, src_tag = cur["body"], "deleted_revision"  # oldest revision hidden
                else:
                    body, src_tag = cur["body"], "no_body_edits"
                ren = n["timelineItems"]["nodes"]
                title = ren[0]["previousTitle"] if ren and ren[0] else cur["title"]
                f.write(json.dumps({"number": n["number"], "orig_title": title,
                                    "orig_body": body, "body_source": src_tag}) + "\n")
            f.flush()
            print(f"  {min(k + BATCH, len(todo))}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
