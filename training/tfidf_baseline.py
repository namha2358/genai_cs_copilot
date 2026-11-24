"""Optional baseline: logistic regression on TF-IDF features (no LLM), same splits and metrics.

Usage (from the repo root): python training/tfidf_baseline.py
Writes training/results/{preds,metrics,confusion}_tfidf.*
"""

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_intents import RESULTS, save_confusion_png, score  # noqa: E402

train = pd.read_csv("data/bitext/train.csv")
val = pd.read_csv("data/bitext/val.csv")
test = pd.read_csv("data/bitext/test.csv")

# Word 1-2-grams capture phrases ("cancel order"); char 3-5-grams cope with Bitext's typos.
features = FeatureUnion([
    ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)),
    ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2)),
])

# Pick the regularisation strength C on the validation split, never on test.
best = None
for C in (1, 3, 10, 30):
    clf = make_pipeline(features, LogisticRegression(C=C, max_iter=2000))
    clf.fit(train["instruction"], train["intent"])
    acc = (clf.predict(val["instruction"]) == val["intent"]).mean()
    print(f"C={C}: val accuracy {acc:.4f}")
    if best is None or acc > best[0]:
        best = (acc, C, clf)

val_acc, C, clf = best
test_pred = clf.predict(test["instruction"]).tolist()
pd.DataFrame({"text": test["instruction"], "gold": test["intent"], "pred": test_pred}).to_csv(
    RESULTS / "preds_tfidf.csv", index=False)
metrics, cm = score(test["intent"].tolist(), test_pred)
metrics.update({"name": "tfidf", "model": f"TF-IDF word1-2+char3-5 + LogisticRegression(C={C})",
                "val_accuracy": round(val_acc, 4), "train_rows": len(train)})
(RESULTS / "metrics_tfidf.json").write_text(json.dumps(metrics, indent=2))
cm.to_csv(RESULTS / "confusion_tfidf.csv")
save_confusion_png(cm, RESULTS / "confusion_tfidf.png", "tfidf (app labels)")
print(json.dumps(metrics, indent=2))
