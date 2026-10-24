"""B0: TF-IDF + logistic regression, and TF-IDF nearest-neighbour vote.
Cheap baselines that need no model downloads.

python baselines_tfidf.py  -> results table on val and test, preds saved to data/preds/
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from router.eval import report

rows = [json.loads(l) for l in open("data/processed/dataset.jsonl")]
train = [r for r in rows if r["split"] == "train" and r["gold"]]
evalsets = {sp: [r for r in rows if r["split"] == sp] for sp in ("val", "test")}

vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2, max_features=200_000,
                      token_pattern=r"(?u)\b[\w./-]{2,}\b")
Xtr = vec.fit_transform([r["text"] for r in train])
mlb = MultiLabelBinarizer()
Ytr = mlb.fit_transform([r["gold"] for r in train])

clf = OneVsRestClassifier(LogisticRegression(C=8, max_iter=2000, class_weight="balanced"), n_jobs=-1)
clf.fit(Xtr, Ytr)

Path("data/preds").mkdir(parents=True, exist_ok=True)
for sp, ev in evalsets.items():
    X = vec.transform([r["text"] for r in ev])

    # logistic regression: softmax-normalise one-vs-rest scores into a top-1 confidence
    P = clf.predict_proba(X)
    P = P / P.sum(axis=1, keepdims=True)
    lr = [{"number": r["number"], "sig": mlb.classes_[P[i].argmax()], "confidence": float(P[i].max())}
          for i, r in enumerate(ev)]

    # kNN vote over the 15 most similar training issues, weighted by similarity
    S = (X @ Xtr.T).toarray()
    knn = []
    for i, r in enumerate(ev):
        top = np.argsort(-S[i])[:15]
        votes = Counter()
        for j in top:
            for s in train[j]["gold"]:
                votes[s] += S[i, j]
        sig, w = votes.most_common(1)[0] if votes else ("node", 0.0)
        knn.append({"number": r["number"], "sig": sig,
                    "confidence": w / (sum(votes.values()) or 1)})

    # majority class: the floor any model must beat
    maj = Counter(s for r in train for s in r["gold"]).most_common(1)[0][0]
    base = [{"number": r["number"], "sig": maj, "confidence": 0.5} for r in ev]

    for name, preds in (("majority(" + maj + ")", base), ("B0 tfidf+logreg", lr), ("B1 tfidf-kNN", knn)):
        json.dump(preds, open(f"data/preds/{name.split()[0].split('(')[0]}_{sp}.json", "w"))
        print(f"[{sp}]\n" + report(name, preds, rows))
    print()
