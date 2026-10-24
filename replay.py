"""Replay backtest: push the test period through the SAME decision code the shadow service runs
(shadow.decide + shadow.CONFIG), issue by issue in creation order, using the stored model outputs.

It answers the production questions offline, on hundreds of issues instead of a handful:
  - what share would be auto-routed, and how often right
  - how much sooner than a human (for issues a human actually had to route)
  - what the LLM would cost under the daily budget rule

python replay.py [--split test]
"""
import argparse
import json
import statistics
from collections import Counter

from router.eval import wilson
from shadow import CONFIG, decide

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="test")
ap.add_argument("--llm-cost-per-issue", type=float, default=0.00072, help="live measurement, config v1")
a = ap.parse_args()

rows = sorted((json.loads(l) for l in open("data/processed/dataset.jsonl")), key=lambda r: r["created_at"])
rows = [r for r in rows if r["split"] == a.split]
m2 = {p["number"]: p for p in json.load(open(f"data/preds/M2s0_{a.split}.json"))}
llm = {p["number"]: p for p in json.load(open(f"data/preds/B2-{CONFIG['llm_model']}_{a.split}.json"))}

spent, out = Counter(), []
for r in rows:
    day = r["created_at"][:10]
    p = m2[r["number"]]
    l = None
    if spent[day] < CONFIG["daily_llm_budget_usd"]:   # same budget rule as the service
        l = llm.get(r["number"])
        spent[day] += a.llm_cost_per_issue
    decision, sig = decide(p["sig"], p["confidence"], l)
    out.append((r, decision, sig))

labeled = [(r, d, s) for r, d, s in out if r["gold"]]
auto = [(r, d, s) for r, d, s in labeled if d == "auto"]
hits = sum(s in r["gold"] for r, d, s in auto)
lo, hi = wilson(hits, len(auto))
print(f"config {CONFIG['version']} on {a.split}: {len(out)} issues, {len(labeled)} with a 30-day SIG answer")
print(f"auto-routed {len(auto)}/{len(labeled)} = {len(auto)/len(labeled):.1%}, "
      f"precision {hits}/{len(auto)} = {hits/len(auto):.3f} [{lo:.2f},{hi:.2f}]")

# issues with no SIG after 30 days: auto-routing them is a false alarm for some SIG
nosig = [(r, d, s) for r, d, s in out if r["gold"] == []]
print(f"no-SIG issues auto-routed anyway: {sum(d == 'auto' for r, d, s in nosig)}/{len(nosig)}")

# speed: only where a person other than the author had to route it (author self-routing takes ~2 min)
for name, f in (("routed by someone else", lambda r: r["routed_by"] == "other"),
                ("author did not name a SIG", lambda r: r["routed_by"] != "body")):
    sub = [(r, d, s) for r, d, s in auto if f(r) and r["hours_to_sig"] is not None]
    if sub:
        h = [r["hours_to_sig"] for r, d, s in sub]
        ok = sum(s in r["gold"] for r, d, s in sub)
        print(f"{name}: router would auto-route {len(sub)} of them ({ok} correct); "
              f"human took median {statistics.median(h):.1f} h (p75 {sorted(h)[int(0.75*(len(h)-1))]:.1f} h)")

days = len(spent)
print(f"LLM spend: ${sum(spent.values()):.2f} over {days} days, max/day ${max(spent.values()):.3f} "
      f"(budget ${CONFIG['daily_llm_budget_usd']:.2f})")
