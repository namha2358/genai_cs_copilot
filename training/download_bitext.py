"""Download the Bitext customer-support dataset into data/ and summarise its intents.

Usage (from the repo root):
    python training/download_bitext.py

Writes (all git-ignored):
    data/bitext/bitext_raw.csv        the full dataset, one row per customer message
    data/bitext/intent_counts.csv     every intent with its category and row count
"""

from pathlib import Path

from datasets import load_dataset

DATASET_ID = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
OUT_DIR = Path("data/bitext")

# Intents the copilot handles (proposal: cancellation, refund, refund status).
# These names are checked against the real data below rather than trusted.
IN_SCOPE_CANDIDATES = ["cancel_order", "get_refund", "track_refund", "check_refund_policy"]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Bitext ships a single "train" split; we make our own train/val/test later (make_splits.py).
    ds = load_dataset(DATASET_ID, split="train")
    df = ds.to_pandas()
    df.to_csv(OUT_DIR / "bitext_raw.csv", index=False)
    print(f"{len(df):,} rows, columns: {list(df.columns)}")

    # Row count per intent, keeping the category so the report table can group them.
    counts = (
        df.groupby(["category", "intent"]).size().reset_index(name="rows").sort_values(["category", "intent"])
    )
    counts.to_csv(OUT_DIR / "intent_counts.csv", index=False)
    print(f"\n{df['intent'].nunique()} intents in {df['category'].nunique()} categories:\n")
    print(counts.to_string(index=False))

    # Confirm the in-scope labels exist exactly as spelled (they go in the decisions log).
    present = set(df["intent"])
    print("\nIn-scope label check:")
    for label in IN_SCOPE_CANDIDATES:
        print(f"  {label:22s} {'FOUND' if label in present else 'MISSING'}")


if __name__ == "__main__":
    main()
