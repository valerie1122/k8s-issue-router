"""Does a second "review your answer" pass make Haiku more accurate?

For every val/test issue, Haiku sees the issue, its own first answer and reason, and is asked to
check it and either keep it or change it. Then we compare first vs reviewed answers.

    python3 b2_review.py submit          # one Message Batch (~1,100 requests, well under $1)
    python3 b2_review.py collect --wait  # then prints the comparison
    python3 b2_review.py score           # re-print the comparison later
"""
import argparse
import json
import time
from pathlib import Path

import anthropic

from b2_zeroshot import schema, sig_catalog, system_prompt, MAX_ISSUE_CHARS
from fetch_issues import load_dotenv

MODEL = "claude-haiku-5-5"
STATE = Path("data/preds/b2_review_batch.json")
OUT = "data/preds/B2R-{m}_{sp}.json"

REVIEW = ("A first pass assigned this issue to SIG \"{sig}\" with confidence {conf:.2f}, reason: \"{reason}\"\n\n"
          "Review that decision. Check whether the issue is really about code that SIG owns, and whether another SIG "
          "or \"none\" fits better. Keep the answer if it holds up; change it only if you find a concrete reason. "
          "Give your final answer.")


def first_pass():
    return {p["number"]: p for sp in ("val", "test")
            for p in json.load(open(f"data/preds/B2-{MODEL}_{sp}.json"))}


def submit():
    labels, catalog = sig_catalog()
    sys_blocks = [{"type": "text", "text": system_prompt(catalog), "cache_control": {"type": "ephemeral"}}]
    rows = [json.loads(l) for l in open("data/processed/dataset.jsonl")]
    first = first_pass()
    reqs = []
    for r in rows:
        if r["split"] not in ("val", "test") or r["number"] not in first:
            continue
        f = first[r["number"]]
        msg = "Issue:\n\n" + r["text"][:MAX_ISSUE_CHARS] + "\n\n---\n" + REVIEW.format(
            sig=f["sig"], conf=f["confidence"], reason=f.get("reason", "").replace('"', "'"))
        reqs.append({"custom_id": str(r["number"]),
                     "params": {"model": MODEL, "max_tokens": 600, "system": sys_blocks,
                                "messages": [{"role": "user", "content": msg}],
                                "output_config": {"format": {"type": "json_schema", "schema": schema(labels)}}}})
    b = anthropic.Anthropic().messages.batches.create(requests=reqs)
    STATE.write_text(json.dumps({"id": b.id, "n": len(reqs), "submitted": time.time()}))
    print(f"submitted review batch {b.id} with {len(reqs)} requests")


def collect(wait):
    st = json.loads(STATE.read_text())
    client = anthropic.Anthropic()
    while True:
        b = client.messages.batches.retrieve(st["id"])
        print(b.processing_status, b.request_counts, flush=True)
        if b.processing_status == "ended":
            break
        if not wait:
            return
        time.sleep(30)
    rows = {json.loads(l)["number"]: json.loads(l) for l in open("data/processed/dataset.jsonl")}
    preds, usage, errors = {"val": [], "test": []}, {"input": 0, "cache_read": 0, "cache_write": 0, "output": 0}, 0
    for res in client.messages.batches.results(st["id"]):
        n = int(res.custom_id)
        if res.result.type != "succeeded":
            errors += 1; continue
        m = res.result.message
        for k, attr in (("input", "input_tokens"), ("cache_read", "cache_read_input_tokens"),
                        ("cache_write", "cache_creation_input_tokens"), ("output", "output_tokens")):
            usage[k] += getattr(m.usage, attr) or 0
        try:
            out = json.loads(next(c.text for c in m.content if c.type == "text"))
        except (StopIteration, json.JSONDecodeError):
            errors += 1; continue
        preds[rows[n]["split"]].append({"number": n, "sig": out["sig"], "reason": out["reason"],
                                        "confidence": min(max(float(out["confidence"]), 0.0), 1.0)})
    for sp, p in preds.items():
        json.dump(p, open(OUT.format(m=MODEL, sp=sp), "w"))
    print(f"wrote {sum(map(len, preds.values()))} reviewed answers, {errors} errors, usage {usage}")
    score()


def score():
    import numpy as np
    from router.eval import honest_report, report
    from shadow import decide
    rows = [json.loads(l) for l in open("data/processed/dataset.jsonl")]
    by = {r["number"]: r for r in rows}
    first = first_pass()
    rev = {p["number"]: p for sp in ("val", "test") for p in json.load(open(OUT.format(m=MODEL, sp=sp)))}
    test = [n for n in rev if by[n]["split"] == "test" and by[n]["gold"]]
    keep = sum(rev[n]["sig"] == first[n]["sig"] for n in test)
    w2r = sum(first[n]["sig"] not in by[n]["gold"] and rev[n]["sig"] in by[n]["gold"] for n in test)
    r2w = sum(first[n]["sig"] in by[n]["gold"] and rev[n]["sig"] not in by[n]["gold"] for n in test)
    print(f"\n== test, {len(test)} labeled issues ==")
    print(f"first pass top-1  {np.mean([first[n]['sig'] in by[n]['gold'] for n in test]):.3f}")
    print(f"after review      {np.mean([rev[n]['sig'] in by[n]['gold'] for n in test]):.3f}")
    print(f"kept the first answer on {keep/len(test):.0%}; changed {len(test)-keep}: wrong->right {w2r}, right->wrong {r2w}")
    same = [n for n in test if rev[n]["sig"] == first[n]["sig"]]
    diff = [n for n in test if rev[n]["sig"] != first[n]["sig"]]
    if same and diff:
        print(f"accuracy when review agrees with first pass: {np.mean([rev[n]['sig'] in by[n]['gold'] for n in same]):.3f}; "
              f"when it changes: {np.mean([rev[n]['sig'] in by[n]['gold'] for n in diff]):.3f}")
    preds = {sp: [p for p in rev.values() if by[p["number"]]["split"] == sp] for sp in ("val", "test")}
    print(report("Haiku reviewed", preds["test"], rows).split("\n")[0])
    print(honest_report("Haiku reviewed", preds["val"], preds["test"], rows))
    # cascade: swap the reviewed answer in for the first-pass one
    m2 = {p["number"]: p for sp in ("val", "test") for p in json.load(open(f"data/preds/M2s0_{sp}.json"))}
    for name, llm in (("first pass", first), ("reviewed", rev)):
        auto = right = 0
        for n in test:
            d, s = decide(m2[n]["sig"], m2[n]["confidence"], llm.get(n), by[n]["text"])
            if d == "auto":
                auto += 1; right += s in by[n]["gold"]
        print(f"cascade layers 1-2 with {name:10s}: auto-routes {auto/len(test):.1%}, precision {right/max(auto,1):.3f}")


if __name__ == "__main__":
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["submit", "collect", "score"])
    ap.add_argument("--wait", action="store_true")
    a = ap.parse_args()
    {"submit": submit, "collect": lambda: collect(a.wait), "score": score}[a.cmd]()
