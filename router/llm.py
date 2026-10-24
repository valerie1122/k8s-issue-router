"""The LLM layer: Haiku picks one SIG (or "none") from the charter list, as structured output."""
import json
import time

import yaml

MAX_ISSUE_CHARS = 6000
# Haiku 5.5, prompts < 100K tokens, USD per million tokens (real-time API)
PRICE = {"input": 0.10, "cache_write": 0.125, "cache_read": 0.01, "output": 0.50}


def sig_catalog(path):
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


def request_params(model, labels, catalog, text, max_tokens=300, cache=True):
    block = {"type": "text", "text": system_prompt(catalog)}
    if cache:  # worth it for batches / bursts; not for one request an hour
        block["cache_control"] = {"type": "ephemeral"}
    return {"model": model, "max_tokens": max_tokens,
            "system": [block],
            "messages": [{"role": "user", "content": "Issue:\n\n" + text[:MAX_ISSUE_CHARS]}],
            "output_config": {"format": {"type": "json_schema", "schema": schema(labels)}}}


def cost_usd(usage):
    return (usage.input_tokens * PRICE["input"]
            + (usage.cache_creation_input_tokens or 0) * PRICE["cache_write"]
            + (usage.cache_read_input_tokens or 0) * PRICE["cache_read"]
            + usage.output_tokens * PRICE["output"]) / 1e6


def classify(client, model, labels, catalog, text, cache=True):
    """One real-time call. Returns dict with sig/confidence/reason + tokens, cost, latency."""
    t = time.time()
    m = client.messages.create(**request_params(model, labels, catalog, text, max_tokens=600, cache=cache))
    ms = (time.time() - t) * 1000
    out = json.loads(next(c.text for c in m.content if c.type == "text"))
    u = m.usage
    return {"sig": out["sig"], "confidence": min(max(float(out["confidence"]), 0.0), 1.0),
            "reason": out["reason"], "ms": ms, "in_tok": u.input_tokens, "out_tok": u.output_tokens,
            "cache_read": u.cache_read_input_tokens or 0, "cost": cost_usd(u)}
