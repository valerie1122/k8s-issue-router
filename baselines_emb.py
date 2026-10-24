"""B1 with real embeddings: kNN vote and logistic regression on cached embeddings.
python baselines_emb.py data/emb/BAAI__bge-base-en-v1.5.npy"""
import json, sys
from collections import Counter
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from router.eval import report, honest_report

rows = [json.loads(l) for l in open("data/processed/dataset.jsonl")]
E = np.load(sys.argv[1])
E = E / np.linalg.norm(E, axis=1, keepdims=True)
tr = [i for i, r in enumerate(rows) if r["split"] == "train" and r["gold"]]
mlb = MultiLabelBinarizer(); Y = mlb.fit_transform([rows[i]["gold"] for i in tr])
clf = OneVsRestClassifier(LogisticRegression(C=4, max_iter=3000, class_weight="balanced"), n_jobs=-1).fit(E[tr], Y)
preds = {}
for sp in ("val", "test"):
    ix = [i for i, r in enumerate(rows) if r["split"] == sp]
    P = clf.predict_proba(E[ix]); P = P / P.sum(1, keepdims=True)
    lr = [{"number": rows[i]["number"], "sig": mlb.classes_[P[k].argmax()], "confidence": float(P[k].max())} for k, i in enumerate(ix)]
    S = E[ix] @ E[tr].T
    knn = []
    for k, i in enumerate(ix):
        top = np.argsort(-S[k])[:15]; v = Counter()
        for j in top:
            for s in rows[tr[j]]["gold"]: v[s] += float(S[k, j])
        s, w = v.most_common(1)[0]
        knn.append({"number": rows[i]["number"], "sig": s, "confidence": w / sum(v.values())})
    preds[sp] = {"B1 emb-kNN": knn, "B1b emb+logreg": lr}
    for n, p in preds[sp].items():
        json.dump(p, open(f"data/preds/{n.split()[0]}_{sp}.json", "w"))
    print(f"[{sp}]"); [print(report(n, p, rows)) for n, p in preds[sp].items()]
print()
for n in preds["val"]:
    print(honest_report(n, preds["val"][n], preds["test"][n], rows))
