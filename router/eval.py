"""One function, one table: how good is a router?

A prediction is {"number": int, "sig": str, "confidence": float}.
A prediction is "correct" when its SIG is in the issue's 30-day gold set
(the owning SIG can pull others in later, so naming any owner counts).
Only issues with a non-empty gold set are scored here.
"""
import math
from collections import defaultdict

import numpy as np


def wilson(k, n, z=1.96):
    """95% interval for a proportion; honest at small n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - r) / d, (c + r) / d)


def coverage_at_precision(conf, correct, target):
    """Largest share of issues we can auto-route while keeping precision >= target.
    Returns (coverage, threshold, precision, n_routed)."""
    order = np.argsort(-conf)
    hits = np.cumsum(correct[order])
    n = np.arange(1, len(order) + 1)
    prec = hits / n
    ok = np.where(prec >= target)[0]
    if len(ok) == 0:
        return 0.0, float("inf"), float("nan"), 0
    k = ok[-1]
    # ties: a threshold must route every issue at that confidence
    thr = conf[order][k]
    routed = conf >= thr
    p = correct[routed].mean()
    if p < target:  # tie at the cut pushed precision below target; step up
        higher = np.unique(conf[conf > thr])
        for t in higher:
            routed = conf >= t
            if correct[routed].mean() >= target:
                thr, p = t, correct[routed].mean()
                break
        else:
            return 0.0, float("inf"), float("nan"), 0
    return routed.mean(), float(thr), float(p), int(routed.sum())


def ece(conf, correct, bins=10):
    """Expected calibration error: average gap between stated confidence and accuracy."""
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return total


def fill_missing(preds, gold):
    """Every gold issue gets a prediction. A router that failed or answered "none" on an issue
    that does have an owner is counted as "sent to a human": wrong for top-1, never auto-routed."""
    have = {x["number"]: x for x in preds if x["number"] in gold}
    # score only the split(s) this prediction file covers
    splits = {gold[n]["split"] for n in have}
    return [have.get(n) or {"number": n, "sig": "__missing__", "confidence": -1.0}
            for n, r in gold.items() if r["split"] in splits]


def evaluate(preds, rows, subset=None):
    """preds: list of dicts; rows: dataset rows. Returns a flat metrics dict."""
    gold = {r["number"]: r for r in rows if r["gold"] and (subset is None or subset(r))}
    p = fill_missing(preds, gold)
    if not p:
        return {"n": 0}
    conf = np.array([x["confidence"] for x in p], dtype=float)
    correct = np.array([x["sig"] in gold[x["number"]]["gold"] for x in p], dtype=float)

    per_sig = defaultdict(lambda: [0, 0])  # gold primary SIG -> [hits, total]
    for x, c in zip(p, correct):
        for s in gold[x["number"]]["gold"]:
            per_sig[s][0] += c
            per_sig[s][1] += 1
    macro = np.mean([h / t for h, t in per_sig.values()])

    out = {"n": len(p), "top1": correct.mean(), "top1_ci": wilson(int(correct.sum()), len(p)),
           "macro_recall": macro, "ece": ece(conf, correct)}
    for t in (0.90, 0.95):
        cov, thr, prec, k = coverage_at_precision(conf, correct, t)
        out[f"cov@{int(t*100)}"] = cov
        out[f"thr@{int(t*100)}"] = thr
        out[f"prec@{int(t*100)}_ci"] = wilson(int(round(prec * k)), k) if k else (float("nan"),) * 2
    return out


SUBSETS = {
    "all": None,
    "needs_routing": lambda r: r["routed_by"] != "body",   # author did not name a SIG in the issue
    "routed_by_other": lambda r: r["routed_by"] == "other",
}


def report(name, preds, rows):
    lines = []
    for sname, f in SUBSETS.items():
        m = evaluate(preds, rows, f)
        if not m["n"]:
            continue
        lo, hi = m["top1_ci"]
        lines.append(
            f"{name:22s} {sname:16s} n={m['n']:4d}  top1 {m['top1']:.3f} [{lo:.2f},{hi:.2f}]  "
            f"macroR {m['macro_recall']:.3f}  cov@90 {m['cov@90']:.1%}  cov@95 {m['cov@95']:.1%}  "
            f"ECE {m['ece']:.3f}")
    return "\n".join(lines)


def honest_report(name, val_preds, test_preds, rows, targets=(0.90, 0.95), subset=None):
    """Pick the auto-route threshold on val, apply it unchanged to test.
    This is the number that would hold in production; the same-set numbers above are optimistic."""
    def arrays(preds):
        gold = {r["number"]: r for r in rows if r["gold"] and (subset is None or subset(r))}
        p = fill_missing(preds, gold)
        return (np.array([x["confidence"] for x in p], dtype=float),
                np.array([x["sig"] in gold[x["number"]]["gold"] for x in p], dtype=float))
    cv, okv = arrays(val_preds)
    ct, okt = arrays(test_preds)
    out = []
    for t in targets:
        _, thr, _, _ = coverage_at_precision(cv, okv, t)
        routed = ct >= thr
        k, n = int(okt[routed].sum()), int(routed.sum())
        prec = k / n if n else float("nan")
        lo, hi = wilson(k, n) if n else (float("nan"),) * 2
        out.append(f"target {t:.0%}: thr {thr:.3f} -> test auto-routes {routed.mean():.1%} "
                   f"({n}/{len(ct)}), precision {prec:.3f} [{lo:.2f},{hi:.2f}]")
    return f"{name}\n  " + "\n  ".join(out)
