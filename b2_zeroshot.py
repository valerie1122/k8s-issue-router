"""B2: LLM zero-shot router. Only the SIG charters, no examples, no tools.

    python b2_zeroshot.py submit [--model claude-haiku-5-5]   # sends one Message Batch (50% cheaper)
    python b2_zeroshot.py collect                             # waits, writes data/preds/B2-<model>_{val,test}.json

The SIG list is a fixed prefix marked for prompt caching, so after the first
request every call pays only for the issue text.
"""
import argparse
import json
import os
import time
from pathlib import Path

import anthropic
import yaml

from fetch_issues import load_dotenv

RUN = ""   # --run r2: a repeat run with the same prompt, saved separately (measures answer stability)
STATE = Path("data/preds/b2_batch.json")
RETRY_STATE = Path("data/preds/b2_batch_retry.json")


def state_path(retry):
    if RUN:
        return Path(f"data/preds/b2_batch_{RUN}{'_retry' if retry else ''}.json")
    return RETRY_STATE if retry else STATE
MAX_ISSUE_CHARS = 6000


def sig_catalog(path="data/raw/sigs.yaml"):
    sigs = yaml.safe_load(open(path))["sigs"]
    lines = []
    for s in sigs:
        subs = ", ".join(p["name"] for p in (s.get("subprojects") or []))[:400]
        mission = " ".join((s.get("mission_statement") or "").split())[:600]
        lines.append(f"- {s['label']}: {mission}\n  subprojects: {subs}")
    return [s["label"] for s in sigs], "\n".join(lines)


def system_prompt(catalog):
    return (
        "You route new GitHub issues in kubernetes/kubernetes to the Special Interest Group (SIG) "
        "that owns the affected code. Pick the single SIG most likely to own the issue. "
        "If the issue is a support question, a duplicate-looking report with no technical content, "
        "or not about Kubernetes code at all, answer \"none\".\n\n"
        "confidence = your probability (0 to 1) that a Kubernetes triager would assign this SIG.\n\n"
        f"SIGs:\n{catalog}")


def schema(labels):
    return {"type": "object", "additionalProperties": False,
            "properties": {"sig": {"type": "string", "enum": labels + ["none"]},
                           "confidence": {"type": "number"},
                           "reason": {"type": "string", "description": "one short sentence"}},
            "required": ["sig", "confidence", "reason"]}


def submit(model, retry=False):
    labels, catalog = sig_catalog()
    sys_blocks = [{"type": "text", "text": system_prompt(catalog), "cache_control": {"type": "ephemeral"}}]
    rows = [json.loads(l) for l in open("data/processed/dataset.jsonl")]
    rows = [r for r in rows if r["split"] in ("val", "test")]
    if retry:  # only the issues the first batch failed on, with more room for the answer
        done = {p["number"] for sp in ("val", "test")
                for p in json.load(open(f"data/preds/B2-{model}{'-' + RUN if RUN else ''}_{sp}.json"))}
        rows = [r for r in rows if r["number"] not in done]
        if not rows:
            print("nothing to retry: every issue already has an answer")
            return
    reqs = [{"custom_id": str(r["number"]),
             "params": {"model": model, "max_tokens": 600 if retry else 300, "system": sys_blocks,
                        "messages": [{"role": "user", "content": "Issue:\n\n" + r["text"][:MAX_ISSUE_CHARS]}],
                        "output_config": {"format": {"type": "json_schema", "schema": schema(labels)}}}}
            for r in rows]
    client = anthropic.Anthropic()
    b = client.messages.batches.create(requests=reqs)
    state_file = state_path(retry)
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps({"id": b.id, "model": model, "n": len(reqs), "submitted": time.time()}))
    print(f"submitted batch {b.id} with {len(reqs)} requests on {model}")


def collect(wait, retry=False):
    st = json.loads(state_path(retry).read_text())
    client = anthropic.Anthropic()
    while True:
        b = client.messages.batches.retrieve(st["id"])
        print(b.processing_status, b.request_counts)
        if b.processing_status == "ended":
            break
        if not wait:
            return
        time.sleep(30)
    rows = {json.loads(l)["number"]: json.loads(l) for l in open("data/processed/dataset.jsonl")}
    preds = {"val": [], "test": []}
    usage = {"input": 0, "cache_read": 0, "cache_write": 0, "output": 0}
    errors = 0
    for res in client.messages.batches.results(st["id"]):
        n = int(res.custom_id)
        if res.result.type != "succeeded":
            errors += 1
            continue
        m = res.result.message
        u = m.usage
        usage["input"] += u.input_tokens
        usage["cache_read"] += u.cache_read_input_tokens or 0
        usage["cache_write"] += u.cache_creation_input_tokens or 0
        usage["output"] += u.output_tokens
        try:
            out = json.loads(next(c.text for c in m.content if c.type == "text"))
        except (StopIteration, json.JSONDecodeError):
            errors += 1
            continue
        conf = min(max(float(out["confidence"]), 0.0), 1.0)
        preds[rows[n]["split"]].append({"number": n, "sig": out["sig"], "confidence": conf,
                                        "reason": out["reason"]})
    tag = "B2-" + st["model"] + ("-" + RUN if RUN else "")
    for sp, p in preds.items():
        path = f"data/preds/{tag}_{sp}.json"
        if retry:  # merge into the first batch's answers
            old = json.load(open(path))
            have = {x["number"] for x in old}
            p = old + [x for x in p if x["number"] not in have]
        json.dump(p, open(path, "w"))
    if retry:
        tag += "_retry"
    json.dump({"usage": usage, "errors": errors, "batch": st}, open(f"data/preds/{tag}_usage.json", "w"), indent=1)
    print(f"wrote {sum(map(len, preds.values()))} preds, {errors} errors, usage {usage}")


if __name__ == "__main__":
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["submit", "collect"])
    ap.add_argument("--model", default="claude-haiku-5-5")
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--retry", action="store_true", help="re-run only the issues that failed")
    ap.add_argument("--run", default="", help="tag for a repeat run, e.g. r2 (keeps the first run's files)")
    a = ap.parse_args()
    RUN = a.run
    submit(a.model, a.retry) if a.cmd == "submit" else collect(a.wait, a.retry)
