"""How stable are Haiku's answers? Compare the first run with repeat runs (b2_zeroshot.py --run r2 ...).

    python3 stability.py r2 [r3 ...]
"""
import json, sys
import numpy as np
from router.eval import wilson
from shadow import decide

rows = {json.loads(l)["number"]: json.loads(l) for l in open("data/processed/dataset.jsonl")}
load = lambda tag: {p["number"]: p for sp in ("val", "test") for p in json.load(open(f"data/preds/B2-claude-haiku-5-5{tag}_{sp}.json"))}
runs = [load("")] + [load("-" + t) for t in sys.argv[1:]]
m2 = {p["number"]: p for sp in ("val", "test") for p in json.load(open("data/preds/M2s0_test.json")) + json.load(open("data/preds/M2s0_val.json"))}
test = [n for n in runs[0] if rows[n]["split"] == "test" and rows[n]["gold"] and all(n in r for r in runs)]
print(f"{len(runs)} runs, {len(test)} labeled test issues answered in every run")
for i, r in enumerate(runs):
    print(f"  run {i + 1} top-1 {np.mean([r[n]['sig'] in rows[n]['gold'] for n in test]):.3f}")
same = [n for n in test if len({r[n]['sig'] for r in runs}) == 1]
print(f"same answer in every run: {len(same) / len(test):.1%} of issues")
print(f"  accuracy when all runs agree: {np.mean([runs[0][n]['sig'] in rows[n]['gold'] for n in same]):.3f}")
flip = [n for n in test if n not in same]
if flip:
    print(f"  accuracy of run 1 when runs disagree: {np.mean([runs[0][n]['sig'] in rows[n]['gold'] for n in flip]):.3f} (n={len(flip)})")
# how much does the cascade's decision move between runs?
dec = [[decide(m2[n]['sig'], m2[n]['confidence'], r.get(n), rows[n]['text']) for n in test] for r in runs]
changed = sum(len({d[k] for d in dec}) > 1 for k in range(len(test)))
print(f"cascade layers 1-2 decision changes between runs on {changed} of {len(test)} issues ({changed / len(test):.1%})")
for i, d in enumerate(dec):
    auto = [(k, s) for k, (dd, s) in enumerate(d) if dd == 'auto']
    right = sum(s in rows[test[k]]['gold'] for k, s in auto)
    lo, hi = wilson(right, len(auto))
    print(f"  run {i + 1}: auto {len(auto) / len(test):.1%}, precision {right / len(auto):.3f} [{lo:.2f},{hi:.2f}]")
if len(runs) >= 3:   # self-consistency: a vote across runs as the second opinion
    from collections import Counter
    auto = right = 0
    for n in test:
        v, c = Counter(r[n]['sig'] for r in runs).most_common(1)[0]
        if m2[n]['confidence'] >= 0.99 or (v == m2[n]['sig'] and c == len(runs) and m2[n]['confidence'] >= 0.69):
            auto += 1; right += m2[n]['sig'] in rows[n]['gold']
    print(f"self-consistency (all {len(runs)} runs agree with ModernBERT): auto {auto / len(test):.1%}, precision {right / auto:.3f}")
