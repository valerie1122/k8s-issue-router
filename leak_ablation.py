"""How much would the leaks have inflated the numbers? Same model (TF-IDF + logistic regression),
same split and labels; only the input text changes.

  clean          : creation-time title/body, routing commands stripped   (what everything else uses)
  edited text    : current (post-edit) title/body, commands stripped
  no cleaning    : current title/body, commands left in

python leak_ablation.py
"""
import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from router.data import clean_text
from router.eval import evaluate, honest_report

rows = [json.loads(l) for l in open("data/processed/dataset.jsonl")]
raw = {json.loads(l)["number"]: json.loads(l) for l in open("data/interim/issues_3y_orig.jsonl")}
variants = {
    "clean": lambda r: r["text"],
    "edited text": lambda r: clean_text(raw[r["number"]]["current_title"], raw[r["number"]]["current_body"]),
    "no cleaning": lambda r: raw[r["number"]]["current_title"] + "\n\n" + (raw[r["number"]]["current_body"] or ""),
}
train = [r for r in rows if r["split"] == "train" and r["gold"]]
for name, text in variants.items():
    vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2, max_features=200_000,
                          token_pattern=r"(?u)\b[\w./-]{2,}\b")
    X = vec.fit_transform([text(r) for r in train])
    mlb = MultiLabelBinarizer(); Y = mlb.fit_transform([r["gold"] for r in train])
    clf = OneVsRestClassifier(LogisticRegression(C=8, max_iter=2000, class_weight="balanced"), n_jobs=-1).fit(X, Y)
    preds = {}
    for sp in ("val", "test"):
        ev = [r for r in rows if r["split"] == sp]
        P = clf.predict_proba(vec.transform([text(r) for r in ev])); P = P / P.sum(1, keepdims=True)
        preds[sp] = [{"number": r["number"], "sig": mlb.classes_[P[i].argmax()], "confidence": float(P[i].max())}
                     for i, r in enumerate(ev)]
    m = evaluate(preds["test"], rows)
    need = evaluate(preds["test"], rows, lambda r: not r["preassigned"])
    print(f"{name:12s} test top-1 {m['top1']:.3f} | issues the author did not route: {need['top1']:.3f}")
    print("   " + honest_report("", preds["val"], preds["test"], rows).split("\n")[2].strip())
