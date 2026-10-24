"""M1: tool-using routing agent for the issues the cheap layers are unsure about.

It sees the issue, the fine-tuned classifier's top-3 guesses and the zero-shot LLM's guess, then
can search similar past issues, read SIG charters and look up code owners before it commits to
an answer through the `submit_decision` tool (strict schema, so the SIG is always a real one).

Tools run in-process by default; --via-mcp calls the same tools through router_mcp.py over stdio.
"""
import asyncio
import json
import time

import yaml

from router.llm import PRICE, MAX_ISSUE_CHARS
from router.tools import TOOLS

MAX_TURNS = 6

SYSTEM = """You triage new issues in kubernetes/kubernetes. Your job: name the one SIG that should own
the issue, or say it should not be routed.

Answer "none" when the issue is a support/usage question, is about a distribution or cloud
product rather than Kubernetes code, or duplicates an existing issue (then also give duplicate_of).

You get two hints from cheaper models; they are often right but disagree on hard cases. Check them:
search similar issues (how humans labeled near-identical reports), look up which SIG owns the code
paths or components the issue mentions, read a charter when two SIGs overlap. Use at most 4 tool
calls, then call submit_decision. confidence = probability a Kubernetes triager would agree."""


def submit_tool(labels):
    return {"name": "submit_decision", "strict": True,
            "description": "Final routing decision. Call exactly once.",
            "input_schema": {"type": "object", "additionalProperties": False,
                             "properties": {
                                 "sig": {"type": "string", "enum": labels + ["none"]},
                                 "duplicate_of": {"type": ["integer", "null"]},
                                 "confidence": {"type": "number"},
                                 "reason": {"type": "string"}},
                             "required": ["sig", "duplicate_of", "confidence", "reason"]}}


def tool_defs(labels):
    defs = [{"name": n, "description": t["description"], "input_schema": t["input_schema"], "strict": True}
            for n, t in TOOLS.items()]
    return defs + [submit_tool(labels)]


def user_message(text, m2_top3, llm_guess):
    hints = ", ".join(f"{s} ({p:.2f})" for s, p in m2_top3)
    return (f"Issue:\n\n{text[:MAX_ISSUE_CHARS]}\n\n---\nFine-tuned classifier top-3: {hints}\n"
            f"Zero-shot LLM guess: {llm_guess or 'unavailable'}")


class InProcess:
    def call(self, name, args, before):
        fn = TOOLS[name]["fn"]
        if name == "search_similar_issues":
            args = {**args, "before": before}   # agent never sees issues labeled after this one was filed
        return fn(**args)


class ViaMCP:
    """Same tools, called through the MCP server over stdio."""
    def __init__(self):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        self._params = StdioServerParameters(command="python3", args=["router_mcp.py"])
        self._stdio, self._Session = stdio_client, ClientSession
        self._loop = asyncio.new_event_loop()
        self._cm = self._stdio(self._params)
        r, w = self._loop.run_until_complete(self._cm.__aenter__())
        self._scm = self._Session(r, w)
        self.s = self._loop.run_until_complete(self._scm.__aenter__())
        self._loop.run_until_complete(self.s.initialize())

    def call(self, name, args, before):
        if name == "search_similar_issues":
            args = {**args, "before": before}
        res = self._loop.run_until_complete(self.s.call_tool(name, args))
        sc = getattr(res, "structured_content", None) or getattr(res, "structuredContent", None)
        if sc is not None:  # typed result; list returns come wrapped as {"result": [...]}
            return sc.get("result", sc) if isinstance(sc, dict) and set(sc) == {"result"} else sc
        items = [json.loads(c.text) for c in res.content if getattr(c, "text", None)]
        return items if name == "search_similar_issues" else (items[0] if items else None)


def cost(u):
    return (u.input_tokens * PRICE["input"] + (u.cache_creation_input_tokens or 0) * PRICE["cache_write"]
            + (u.cache_read_input_tokens or 0) * PRICE["cache_read"] + u.output_tokens * PRICE["output"]) / 1e6


def run(client, model, labels, text, created_at, m2_top3, llm_guess, tools_backend):
    t0 = time.time()
    msgs = [{"role": "user", "content": user_message(text, m2_top3, llm_guess)}]
    defs = tool_defs(labels)
    usage = {"in": 0, "out": 0, "cost": 0.0}
    trace = []
    for turn in range(MAX_TURNS + 1):
        force = turn == MAX_TURNS
        r = client.messages.create(model=model, max_tokens=800, system=SYSTEM, tools=defs, messages=msgs,
                                   tool_choice={"type": "tool", "name": "submit_decision"} if force else {"type": "auto"})
        usage["in"] += r.usage.input_tokens
        usage["out"] += r.usage.output_tokens
        usage["cost"] += cost(r.usage)
        msgs.append({"role": "assistant", "content": r.content})
        calls = [b for b in r.content if b.type == "tool_use"]
        final = next((b for b in calls if b.name == "submit_decision"), None)
        if final:
            d = dict(final.input)
            d["confidence"] = min(max(float(d["confidence"]), 0.0), 1.0)
            return {**d, "turns": turn + 1, "tools": trace, "ms": (time.time() - t0) * 1000, **usage}
        if not calls:  # talked without acting; nudge it to decide
            msgs.append({"role": "user", "content": "Call submit_decision now."})
            continue
        results = []
        for b in calls:
            try:
                out = tools_backend.call(b.name, dict(b.input), created_at)
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": json.dumps(out)[:6000]})
            except Exception as e:
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": str(e)[:300], "is_error": True})
            trace.append(b.name)
        msgs.append({"role": "user", "content": results})
    raise RuntimeError("agent did not submit")


def sig_labels(path="data/raw/sigs.yaml"):
    return [s["label"] for s in yaml.safe_load(open(path))["sigs"]]
