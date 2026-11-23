"""CPU speed test: how long does one intent classification and one 150-token draft take?

Runs each (model, dtype) combination in a fresh subprocess so peak memory is measured
per configuration and is not polluted by the previous model.

Usage (from the repo root, outside Docker):
    python training/cpu_speed_test.py                 # all four configs
    python training/cpu_speed_test.py --only Qwen/Qwen2.5-0.5B-Instruct bfloat16

Writes training/results/cpu_speed_test.csv
"""

import argparse
import csv
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

MODELS = ["Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"]
DTYPES = ["float32", "bfloat16"]  # float16 is skipped on purpose: slow or unsupported on CPU
OUT = Path("training/results/cpu_speed_test.csv")

# The 27 Bitext labels (from download_bitext.py) plus "unknown" for anything else.
LABELS = [
    "cancel_order", "change_order", "change_shipping_address", "check_cancellation_fee", "check_invoice",
    "check_payment_methods", "check_refund_policy", "complaint", "contact_customer_service",
    "contact_human_agent", "create_account", "delete_account", "delivery_options", "delivery_period",
    "edit_account", "get_invoice", "get_refund", "newsletter_subscription", "payment_issue", "place_order",
    "recover_password", "registration_problems", "review", "set_up_shipping_address", "switch_account",
    "track_order", "track_refund", "unknown",
]
MESSAGE = "Hi, I ordered a jacket last week but it arrived damaged. I want my money back please."
ORDER_CONTEXT = "Order 10042: jacket, $89.00, delivered 5 days ago. Policy R1: damaged items get a full refund within 30 days."


def run_one(model_id, dtype_name):
    """Load one model and time the two tasks. Runs inside the child process."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, dtype_name)
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).eval()
    load_s = time.perf_counter() - t0

    def chat(system, user):
        # Instruct models expect their chat template; add_generation_prompt opens the assistant turn.
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return tok(text, return_tensors="pt")

    @torch.no_grad()
    def timed_generate(enc, min_new, max_new):
        start = time.perf_counter()
        out = model.generate(**enc, do_sample=False, min_new_tokens=min_new, max_new_tokens=max_new,
                             pad_token_id=tok.eos_token_id)
        elapsed = time.perf_counter() - start
        new_ids = out[0, enc["input_ids"].shape[1]:]
        return elapsed, len(new_ids), tok.decode(new_ids, skip_special_tokens=True)

    clf_enc = chat(
        "Classify the customer message into exactly one of these intents: " + ", ".join(LABELS)
        + ". Answer with the label only.",
        MESSAGE,
    )
    draft_enc = chat(
        "You are a customer support assistant. Draft a short, polite reply that cites the policy.",
        f"{ORDER_CONTEXT}\nCustomer: {MESSAGE}",
    )

    timed_generate(clf_enc, 1, 2)  # warm-up: the first call pays one-off setup costs
    clf_s, clf_tokens, clf_text = timed_generate(clf_enc, 1, 10)
    # min = max = 150 forces exactly 150 new tokens, so every config does the same amount of work.
    draft_s, draft_tokens, _ = timed_generate(draft_enc, 150, 150)

    # ru_maxrss is the peak resident memory of this process, in KB on Linux (bytes on macOS).
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_gb = peak / (1024**3 if sys.platform == "darwin" else 1024**2)
    return {
        "model": model_id, "dtype": dtype_name, "load_s": round(load_s, 1),
        "classify_s": round(clf_s, 2), "classify_output": clf_text.strip(),
        "draft_150_s": round(draft_s, 1), "draft_tokens": draft_tokens,
        "draft_tok_per_s": round(draft_tokens / draft_s, 2), "peak_ram_gb": round(peak_gb, 2),
        "threads": torch.get_num_threads(), "cpu": platform.processor() or platform.machine(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs=2, metavar=("MODEL", "DTYPE"))
    p.add_argument("--child", nargs=2, help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.child:  # child mode: run a single config and print JSON for the parent
        print("RESULT " + json.dumps(run_one(*args.child)))
        return

    configs = [tuple(args.only)] if args.only else [(m, d) for m in MODELS for d in DTYPES]
    rows = []
    for model_id, dtype_name in configs:
        print(f"\n=== {model_id} {dtype_name} ===", flush=True)
        proc = subprocess.run([sys.executable, __file__, "--child", model_id, dtype_name],
                              capture_output=True, text=True, env=os.environ)
        line = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
        if not line:
            print(proc.stderr[-2000:])
            continue
        row = json.loads(line[0][7:])
        print(row, flush=True)
        rows.append(row)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
