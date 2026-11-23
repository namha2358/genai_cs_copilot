"""Score an intent model: accuracy, macro-F1, in-scope metrics, confusion matrix.

The same script scores every model, so the numbers are comparable:
    # few-shot baseline (base model, examples in the prompt) on the Bitext test split
    python training/eval_intents.py --name fewshot_base --model-id Qwen/Qwen2.5-1.5B-Instruct --prompt few_shot
    # fine-tuned model (after publishing), pinned revision
    python training/eval_intents.py --name finetuned --model-id <user>/<repo> --revision <sha> --prompt zero_shot
    # C's hand-made set (column names are arguments because the file isn't written yet)
    python training/eval_intents.py --name finetuned_handmade --model-id ... --csv eval/<file>.csv \
        --text-col message --label-col expected_intent
    # score predictions that already exist (e.g. downloaded from the Colab run), no model needed
    python training/eval_intents.py --name finetuned --preds training/results/preds_finetuned.csv

Metrics (all on the same rows):
  overall_accuracy, overall_macro_f1  : over the 27 Bitext labels; an unparsable output counts as wrong
  inscope_accuracy                    : accuracy on rows whose true intent is in scope
  inscope_macro_f1                    : F1 for each of the 4 in-scope labels, averaged, computed on ALL rows
                                        after collapsing to app labels, so out-of-scope messages wrongly
                                        called "get_refund" lower the score (false positives count)
  unknown_rate                        : share of outputs that were not a valid label
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so "app" imports from the repo root
from app.intent_prompt import APP_LABELS, BITEXT_LABELS, IN_SCOPE, UNKNOWN, to_app_label  # noqa: E402

RESULTS = Path("training/results")


def predict(args, texts):
    from app.intent_model import IntentModel  # imported here so --preds works without torch

    model = IntentModel(args.model_id, revision=args.revision, dtype=args.dtype,
                        few_shot=(args.prompt == "few_shot"))
    # Sort by length so each batch pads to similar lengths (much faster), then restore the order.
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    preds, raws = [None] * len(texts), [None] * len(texts)
    start = time.perf_counter()
    for b in range(0, len(order), args.batch_size):
        idx = order[b:b + args.batch_size]
        for i, (label, raw) in zip(idx, model.classify_bitext_batch([texts[i] for i in idx])):
            preds[i], raws[i] = label, raw
        done = min(b + args.batch_size, len(order))
        rate = (time.perf_counter() - start) / done
        print(f"{done}/{len(order)}  {rate:.2f} s/message, ~{rate * (len(order) - done) / 60:.0f} min left",
              flush=True)
    return preds, raws, (time.perf_counter() - start) / len(texts)


def score(gold, pred_bitext):
    """gold: true labels (Bitext or app labels). pred_bitext: parsed Bitext label or None."""
    gold_is_bitext = set(gold) <= set(BITEXT_LABELS)
    pred_app = [to_app_label(p) for p in pred_bitext]
    gold_app = [to_app_label(g) if gold_is_bitext else g for g in gold]
    m = {"rows": len(gold), "unknown_rate": round(sum(p is None for p in pred_bitext) / len(gold), 4)}
    if gold_is_bitext:  # overall 27-way metrics only make sense when gold labels are Bitext intents
        pred_filled = [p if p is not None else UNKNOWN for p in pred_bitext]
        m["overall_accuracy"] = round(accuracy_score(gold, pred_filled), 4)
        m["overall_macro_f1"] = round(f1_score(gold, pred_filled, labels=BITEXT_LABELS, average="macro",
                                               zero_division=0), 4)
    else:
        m["app_accuracy"] = round(accuracy_score(gold_app, pred_app), 4)
    in_rows = [i for i, g in enumerate(gold_app) if g in IN_SCOPE]
    m["inscope_rows"] = len(in_rows)
    m["inscope_accuracy"] = round(accuracy_score([gold_app[i] for i in in_rows], [pred_app[i] for i in in_rows]), 4)
    m["inscope_macro_f1"] = round(f1_score(gold_app, pred_app, labels=IN_SCOPE, average="macro",
                                           zero_division=0), 4)
    for lab in IN_SCOPE:  # per-label F1, useful for the report
        m[f"f1_{lab}"] = round(f1_score(gold_app, pred_app, labels=[lab], average="macro", zero_division=0), 4)
    cm = pd.DataFrame(confusion_matrix(gold_app, pred_app, labels=APP_LABELS),
                      index=[f"true_{l}" for l in APP_LABELS], columns=[f"pred_{l}" for l in APP_LABELS])
    return m, cm


def save_confusion_png(cm, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [c.removeprefix("pred_") for c in cm.columns]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm.values, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = cm.values[i, j]
            ax.text(j, i, str(v), ha="center", va="center",
                    color="white" if v > cm.values.max() / 2 else "black", fontsize=9)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="run name used in output file names")
    p.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--revision")
    p.add_argument("--prompt", choices=["few_shot", "zero_shot"], default="few_shot")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--csv", default="data/bitext/test.csv")
    p.add_argument("--text-col", default="instruction")
    p.add_argument("--label-col", default="intent")
    p.add_argument("--limit", type=int, help="score only the first N rows (quick checks)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--preds", help="existing predictions CSV to re-score instead of running a model")
    args = p.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)

    if args.preds:
        df = pd.read_csv(args.preds)
        df["pred"] = df["pred"].where(df["pred"].notna(), None)
        sec = None
    else:
        df = pd.read_csv(args.csv)
        if args.limit:
            df = df.head(args.limit)
        df = df.rename(columns={args.text_col: "text", args.label_col: "gold"})[["text", "gold"]]
        df["pred"], df["raw_output"], sec = predict(args, df["text"].astype(str).tolist())
        df.to_csv(RESULTS / f"preds_{args.name}.csv", index=False)

    metrics, cm = score(df["gold"].tolist(), df["pred"].tolist())
    metrics.update({"name": args.name, "model_id": args.model_id, "revision": args.revision,
                    "prompt": args.prompt, "data": args.preds or args.csv,
                    "sec_per_message": round(sec, 3) if sec else None})
    (RESULTS / f"metrics_{args.name}.json").write_text(json.dumps(metrics, indent=2))
    cm.to_csv(RESULTS / f"confusion_{args.name}.csv")
    save_confusion_png(cm, RESULTS / f"confusion_{args.name}.png", f"{args.name} (app labels)")
    print(json.dumps(metrics, indent=2))
    print(cm)


if __name__ == "__main__":
    main()
