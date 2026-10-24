"""Run the agent on the issues the cascade would send to a human, then measure the new cascade.

    python3 agent_eval.py run  [--model claude-haiku-5-5] [--limit N] [--via-mcp]   # needs ANTHROPIC_API_KEY
    python3 agent_eval.py score                                                    # no API calls

Thresholds for "trust the agent" are picked on val and applied to test, as everywhere else.
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from router.eval import wilson

OUT = Path("data/preds/agent_runs.jsonl")


def m2_top3(run="modernbert-s0"):
    meta = json.load(open(f"data/runs/{run}/meta.json"))
    C, T = meta["classes"], 1.171
    top = {}
    for sp in ("val", "test"):
        L = np.load(f"data/runs/{run}/{sp}_logits.npy") / T
        P = np.exp(L - L.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
        for n, p in zip(meta[f"{sp}_numbers"], P):
            idx = p.argsort()[::-1][:3]
            top[n] = [(C[i], float(p[i])) for i in idx]
    return top


def load_cases():
    from shadow import decide
    rows = [json.loads(l) for l in open("data/processed/dataset.jsonl")]
    m2 = {p["number"]: p for sp in ("val", "test") for p in json.load(open(f"data/preds/M2s0_{sp}.json"))}
    llm = {p["number"]: p for sp in ("val", "test") for p in json.load(open(f"data/preds/B2-claude-haiku-5-5_{sp}.json"))}
    cases = []
    for r in rows:
        if r["split"] not in ("val", "test") or r["gold"] is None:
            continue
        a = m2[r["number"]]
        d, s = decide(a["sig"], a["confidence"], llm.get(r["number"]))
        cases.append({**r, "cascade": d, "cascade_sig": s, "llm_sig": (llm.get(r["number"]) or {}).get("sig")})
    return cases


def run(a):
    import anthropic
    from fetch_issues import load_dotenv
    from agent import InProcess, ViaMCP, run as run_agent, sig_labels
    load_dotenv()
    client = anthropic.Anthropic(max_retries=4)
    labels, top = sig_labels(), m2_top3()
    done = {json.loads(l)["number"] for l in OUT.open()} if OUT.exists() else set()
    todo = [c for c in load_cases() if c["cascade"] == "human" and c["number"] not in done]
    todo = todo[:a.limit] if a.limit else todo
    print(f"{len(todo)} human-bound issues to run ({len(done)} already done)")
    backend = ViaMCP() if a.via_mcp else InProcess()

    def one(c):
        try:
            r = run_agent(client, a.model, labels, c["text"], c["created_at"], top[c["number"]], c["llm_sig"], backend)
            return {"number": c["number"], "split": c["split"], "model": a.model, **r}
        except Exception as e:
            return {"number": c["number"], "split": c["split"], "model": a.model, "error": f"{type(e).__name__}: {e}"[:300]}

    workers = 1 if a.via_mcp else 6
    with OUT.open("a") as f, ThreadPoolExecutor(workers) as pool:
        for k, res in enumerate(pool.map(one, todo), 1):
            f.write(json.dumps(res) + "\n"); f.flush()
            if k % 20 == 0:
                print(f"  {k}/{len(todo)}", flush=True)
    print("done")


def score(_):
    cases = load_cases()
    ag = {}
    for l in OUT.open():
        x = json.loads(l)
        ag[x["number"]] = x
    def sim(sp, t):
        auto = right = fa = lab = 0
        for c in cases:
            if c["split"] != sp:
                continue
            lab += bool(c["gold"])
            d, s = c["cascade"], c["cascade_sig"]
            if d == "human":
                x = ag.get(c["number"])
                if x and "sig" in x and x["sig"] != "none" and not x.get("duplicate_of") and x["confidence"] >= t:
                    d, s = "auto", x["sig"]
            if d == "auto":
                auto += 1
                right += bool(c["gold"]) and s in c["gold"]
                fa += not c["gold"]
        return lab, auto, right, fa
    best = None
    for t in np.linspace(0.3, 1.0, 71):
        lab, auto, right, fa = sim("val", t)
        if right / max(auto - fa, 1) >= 0.95 and (best is None or auto > best[1]):
            best = (t, auto)
    t = best[0] if best else 1.01
    for sp in ("val", "test"):
        lab, auto, right, fa = sim(sp, t)
        lo, hi = wilson(right, auto - fa)
        print(f"{sp}: agent threshold {t:.2f} -> auto-routes {(auto-fa)/lab:.1%} of labeled issues, precision "
              f"{right/max(auto-fa,1):.3f} [{lo:.2f},{hi:.2f}]; strict precision (no-SIG counted wrong) {right/max(auto,1):.3f}")
    runs = [x for x in ag.values() if "sig" in x]
    errs = [x for x in ag.values() if "error" in x]
    # agent alone on the hard cases
    byn = {c["number"]: c for c in cases}
    hard = [x for x in runs if byn[x["number"]]["gold"]]
    acc = np.mean([x["sig"] in byn[x["number"]]["gold"] for x in hard]) if hard else float("nan")
    none_ok = [x["sig"] == "none" for x in runs if byn[x["number"]]["gold"] == []]
    print(f"agent on human-bound issues: n={len(runs)} (+{len(errs)} errors), top-1 {acc:.3f} on labeled, "
          f"said none on {np.mean(none_ok) if none_ok else float('nan'):.0%} of no-SIG ones")
    print(f"avg turns {np.mean([x['turns'] for x in runs]):.1f}, avg cost ${np.mean([x['cost'] for x in runs]):.4f}, "
          f"p50 latency {np.median([x['ms'] for x in runs])/1000:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "score"])
    ap.add_argument("--model", default="claude-haiku-5-5")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--via-mcp", action="store_true")
    a = ap.parse_args()
    run(a) if a.cmd == "run" else score(a)
