"""Group near-duplicate Bitext messages and split by group into train / validation / test.

Why: Bitext was generated from templates, so many messages differ only by a placeholder
or punctuation ("cancel order {{Order Number}}" vs "cancel order {{Order Number}}!").
A random row split would put near-copies in both train and test and inflate accuracy.

Method:
 1. Normalize: lowercase, drop {{placeholders}}, drop punctuation, collapse whitespace.
 2. Within each intent, link two messages if the cosine similarity of their TF-IDF
    character 3-5-gram vectors is >= THRESHOLD; each connected component is one group.
    (Exact duplicates after normalization have similarity 1.0, so they are always grouped.)
 3. Within each intent, shuffle the groups and fill test, then validation, up to 10% of
    that intent's rows each; the rest go to train. Groups never cross splits, and every
    intent gets ~80/10/10, i.e. the split is stratified by intent.

Usage (from the repo root, after download_bitext.py):
    python training/make_splits.py [--threshold 0.9] [--seed 42]

Writes data/bitext/{train,val,test}.csv and training/results/split_summary.csv
"""

import argparse
import re
import string
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer

DATA = Path("data/bitext")
RESULTS = Path("training/results")
PLACEHOLDER = re.compile(r"\{\{.*?\}\}")
PUNCT = str.maketrans("", "", string.punctuation)


def normalize(text: str) -> str:
    text = PLACEHOLDER.sub(" ", text.lower())  # "{{Order Number}}" -> " "
    text = text.translate(PUNCT)  # strip punctuation
    return " ".join(text.split())  # collapse whitespace


def group_intent(norm_texts: list[str], threshold: float) -> np.ndarray:
    """Connected components of the 'similarity >= threshold' graph for one intent."""
    # Character n-grams are robust to typos, which Bitext injects on purpose (flag "Z").
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    X = vec.fit_transform(norm_texts)  # rows are L2-normalized, so X @ X.T is cosine similarity
    adjacency = (X @ X.T) >= threshold  # ~1000 x 1000 per intent: small enough to do directly
    _, labels = connected_components(adjacency, directed=False)
    return labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    df = pd.read_csv(DATA / "bitext_raw.csv")
    df["norm"] = df["instruction"].astype(str).map(normalize)
    df["group"] = ""
    df["split"] = ""

    for intent, idx in df.groupby("intent").groups.items():
        sub = df.loc[idx]
        comp = group_intent(sub["norm"].tolist(), args.threshold)
        df.loc[idx, "group"] = [f"{intent}:{c}" for c in comp]

        # Shuffle groups, then fill test and val up to their target row counts.
        sizes = pd.Series(comp).value_counts()
        order = rng.permutation(sizes.index.to_numpy())
        n = len(sub)
        targets = {"test": args.test_frac * n, "val": args.val_frac * n}
        filled = {"test": 0, "val": 0}
        assign = {}
        for g in order:
            for split in ("test", "val"):
                if filled[split] + sizes[g] <= targets[split]:
                    assign[g] = split
                    filled[split] += sizes[g]
                    break
            else:
                assign[g] = "train"
        df.loc[idx, "split"] = [assign[c] for c in comp]

    # Sanity check: no group may appear in two splits.
    leaks = df.groupby("group")["split"].nunique()
    assert (leaks == 1).all(), f"{(leaks > 1).sum()} groups cross splits"

    cols = ["instruction", "intent", "category", "flags", "group"]
    for split in ("train", "val", "test"):
        df.loc[df["split"] == split, cols].to_csv(DATA / f"{split}.csv", index=False)

    # Summary for the report.
    RESULTS.mkdir(parents=True, exist_ok=True)
    summary = df.pivot_table(index="intent", columns="split", values="instruction", aggfunc="count")
    summary = summary[["train", "val", "test"]]
    summary["groups"] = df.groupby("intent")["group"].nunique()
    summary.to_csv(RESULTS / "split_summary.csv")

    n_groups = df["group"].nunique()
    exact_dupes = len(df) - df["norm"].nunique()
    print(f"threshold={args.threshold} seed={args.seed}")
    print(f"rows={len(df):,} groups={n_groups:,} (mean size {len(df) / n_groups:.2f}, "
          f"largest {df['group'].value_counts().max()}); exact duplicates after normalizing: {exact_dupes:,}")
    print(df["split"].value_counts().reindex(["train", "val", "test"]).to_string())
    print("\nPer intent (first rows):\n" + summary.head(8).to_string())


if __name__ == "__main__":
    main()
