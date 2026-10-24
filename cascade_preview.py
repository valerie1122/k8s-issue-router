"""Quick look: M2 first, Haiku only when M2 is unsure, human for the rest.
Both thresholds chosen on val (grid search for max coverage at the precision target), applied to test."""
import json
import numpy as np
from router.eval import wilson

rows = {json.loads(l)["number"]: json.loads(l) for l in open("data/processed/dataset.jsonl")}
def load(name, sp): return {p["number"]: p for p in json.load(open(f"data/preds/{name}_{sp}.json"))}

def run(sp, t1, t2):
    m2, llm = load("M2", sp), load("B2-claude-haiku-5-5", sp)
    nums = [n for n in m2 if rows[n]["gold"]]
    routed = hit = llm_calls = 0
    for n in nums:
        a = m2[n]
        if a["confidence"] >= t1:
            p = a
        else:
            llm_calls += 1
            b = llm.get(n)
            if not b or b["sig"] == "none" or b["confidence"] < t2:
                continue
            # agreement bonus: if both models agree, trust it even below t2? keep simple for now
            p = b
        routed += 1
        hit += p["sig"] in rows[n]["gold"]
    return len(nums), routed, hit, llm_calls

for target in (0.90, 0.95):
    best = None
    for t1 in np.linspace(0.3, 0.99, 70):
        for t2 in np.linspace(0.3, 0.99, 70):
            n, r, h, _ = run("val", t1, t2)
            if r and h / r >= target and (best is None or r > best[0]):
                best = (r, t1, t2)
    _, t1, t2 = best
    n, r, h, calls = run("test", t1, t2)
    lo, hi = wilson(h, r)
    print(f"target {target:.0%}: M2>={t1:.2f}, else Haiku>={t2:.2f} -> test auto-routes {r/n:.1%} ({r}/{n}), "
          f"precision {h/r:.3f} [{lo:.2f},{hi:.2f}], LLM called on {calls/n:.0%} of issues")
