"""M2: ModernBERT logits -> calibrated top-1 predictions.
Temperature scaling: one number T fitted on val so that softmax(logits/T) is honest
(a "0.9" really means right ~90% of the time). Then the usual honest val->test report.

python m2_eval.py [run_name]"""
import json, sys
import numpy as np
from router.eval import report, honest_report, ece

run = sys.argv[1] if len(sys.argv) > 1 else "modernbert-base"
meta = json.load(open(f"data/runs/{run}/meta.json"))
C = meta["classes"]
rows = [json.loads(l) for l in open("data/processed/dataset.jsonl")]
by = {r["number"]: r for r in rows}


def softmax(L, T):
    z = L / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def nll(L, nums, T):
    P = softmax(L, T)
    out = []
    for p, n in zip(P, nums):
        g = by[n]["gold"]
        if g:
            out.append(-np.log(max(sum(p[C.index(s)] for s in g if s in C), 1e-9)))
    return np.mean(out)


Lv, Lt = np.load(f"data/runs/{run}/val_logits.npy"), np.load(f"data/runs/{run}/test_logits.npy")
nv, nt = meta["val_numbers"], meta["test_numbers"]
grid = np.exp(np.linspace(np.log(0.05), np.log(20), 400))
T = float(grid[np.argmin([nll(Lv, nv, t) for t in grid])])


def preds(L, nums, T):
    P = softmax(L, T)
    return [{"number": n, "sig": C[p.argmax()], "confidence": float(p.max())} for p, n in zip(P, nums)]


raw_v, raw_t = preds(Lv, nv, 1.0), preds(Lt, nt, 1.0)
cal_v, cal_t = preds(Lv, nv, T), preds(Lt, nt, T)
json.dump(cal_v, open(f"data/preds/M2_val.json", "w")); json.dump(cal_t, open(f"data/preds/M2_test.json", "w"))
print(f"temperature T = {T:.3f}")
print(report("M2 raw (T=1)", raw_t, rows))
print(report("M2 calibrated", cal_t, rows))
print(honest_report("M2 calibrated", cal_v, cal_t, rows))
