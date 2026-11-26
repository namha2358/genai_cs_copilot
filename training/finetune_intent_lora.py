# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     name: python3
#   accelerator: GPU
# ---

# %% [markdown]
# # LoRA fine-tuning of Qwen2.5-1.5B-Instruct for intent identification (Member A)
#
# Adapted from the course notebook **Module 9 Practical 2: Fine-Tuning an LLM**
# (`mmerler/applied_genai_notebooks`, `notebooks/Module_9_Practical_2_finetuning.py`).
#
# **Kept from the course notebook:** the training loop (`train_causal_lm`, unchanged), loss on the
# answer only (prompt tokens get label -100), LoRA rank 16, alpha 32, dropout 0.05, all linear layers,
# learning rate 2e-4, batch 4 x gradient accumulation 2, 1 epoch, linear schedule with 5% warm-up,
# gradient checkpointing, automatic float16 on a T4 / bfloat16 on newer GPUs.
#
# **Changed for this project:**
# - Task: SQuAD question answering -> intent identification. Each example is the instruction
#   ("classify this message", with the 27 labels) plus the customer message; the answer is the label.
# - Model: `Qwen/Qwen2.5-1.5B-Instruct` instead of the base `Qwen2.5-1.5B`, so examples are written with
#   the model's chat template. The prompt comes from `app/intent_prompt.py`, the same file the app uses.
# - Data: our grouped Bitext splits (`training/make_splits.py`), capped per intent so training stays short.
# - Evaluation: label accuracy and macro-F1 instead of EM/F1; a merge-and-publish step at the end.
#
# **How to run in Colab:** Runtime -> Change runtime type -> T4 GPU, then Run all. To publish, first add a
# Hugging Face *write* token as a Colab secret named `HF_TOKEN` (key icon in the left bar) and set `HF_REPO`.
# On a laptop CPU, set `QUICK_RUN = True` for a smoke test (also gives time per step for the L4 table).

# %%
import os
import subprocess
import sys

IN_COLAB = "google.colab" in sys.modules
REPO_URL = "https://github.com/namha2358/genai_cs_copilot"
REPO_BRANCH = "main"

if IN_COLAB:
    # Colab already has torch; add the Hugging Face libraries and get the team repo (public).
    subprocess.run("pip install -q -U transformers peft datasets accelerate scikit-learn", shell=True, check=True)
    if not os.path.exists("genai_cs_copilot"):
        subprocess.run(f"git clone -q --branch {REPO_BRANCH} {REPO_URL}", shell=True, check=True)
    os.chdir("genai_cs_copilot")
else:
    # Locally: run from the repo root or from training/
    if os.path.basename(os.getcwd()) == "training":
        os.chdir("..")

# Rebuild exactly the same splits as everyone else (the scripts are deterministic: fixed seed).
if not os.path.exists("data/bitext/test.csv"):
    subprocess.run([sys.executable, "training/download_bitext.py"], check=True)
    subprocess.run([sys.executable, "training/make_splits.py"], check=True)
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "training"))
print("working directory:", os.getcwd())

# %%
import json
import math
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, PeftModel, get_peft_model

from app.intent_prompt import BITEXT_LABELS, build_messages, parse_bitext_label
from eval_intents import save_confusion_png, score  # the same metrics as the baselines

torch.manual_seed(42)
np.random.seed(42)

device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
)
# Half precision on GPU: bfloat16 on recent GPUs (Ampere or newer), float16 on older ones such as the T4.
if device.type == "cuda":
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
else:
    dtype = torch.float32
print(f"Using device: {device}, compute dtype: {dtype}")
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# %% [markdown]
# ## Configuration

# %%
QUICK_RUN = False  # True: tiny subsets for a fast end-to-end check (laptop / CPU)

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
TRAIN_PER_INTENT = 10 if QUICK_RUN else 200  # cap per intent (27 intents -> 5,400 training examples)
N_VAL = 54 if QUICK_RUN else None  # None = the whole validation split
N_TEST = 54 if QUICK_RUN else None  # None = the whole test split
MAX_LEN = 256  # prompt (~150 tokens with the label list) + message + label
EVAL_BATCH = 32 if device.type == "cuda" else 8

# LoRA settings: unchanged from the course notebook
LORA_BATCH_SIZE = 4
LORA_GRAD_ACCUM = 2  # effective batch size 8
LORA_LR = 2e-4
LORA_RANK = 16
LORA_ALPHA = 32
EPOCHS = 1

HF_REPO = "YOUR_HF_USERNAME/qwen2.5-1.5b-instruct-bitext-intent"  # set before publishing
OUT_DIR = "checkpoints"  # git-ignored

# %% [markdown]
# ## 1. Data
#
# Stratified cap: the same number of training examples from every intent (fixed seed). Validation and test
# stay complete, so the results are comparable with the few-shot and TF-IDF baselines.

# %%
train_df = pd.read_csv("data/bitext/train.csv")
val_df = pd.read_csv("data/bitext/val.csv")
test_df = pd.read_csv("data/bitext/test.csv")

# GroupBy.sample / .head keep every column (unlike groupby.apply, whose behaviour changed in pandas 3).
train_df = (
    train_df.groupby("intent").sample(n=TRAIN_PER_INTENT, random_state=42)
    .sample(frac=1, random_state=42)  # shuffle intents together
    .reset_index(drop=True)
)
if N_VAL:
    val_df = val_df.groupby("intent").head(max(1, N_VAL // 27)).reset_index(drop=True)
if N_TEST:
    test_df = test_df.groupby("intent").head(max(1, N_TEST // 27)).reset_index(drop=True)
print(f"train {len(train_df):,} (cap {TRAIN_PER_INTENT}/intent), val {len(val_df):,}, test {len(test_df):,}")

# %%
tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"  # for generation; training batches are padded on the right by our collate


def prompt_text(message):
    """The chat-formatted prompt, ending with the opened assistant turn (identical to the app)."""
    return tok.apply_chat_template(build_messages(message, few_shot=False), tokenize=False,
                                   add_generation_prompt=True)


def encode_example(message, label, max_len=MAX_LEN):
    """Token ids for prompt + answer, with labels = -100 on the prompt so only the answer is learned."""
    prompt_ids = tok(prompt_text(message), add_special_tokens=False)["input_ids"]
    # For Qwen Instruct the end-of-turn token <|im_end|> is the EOS: learning it teaches the model to stop.
    answer_ids = tok(label, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    if len(prompt_ids) + len(answer_ids) > max_len:
        return None  # too long: skip
    return {"input_ids": prompt_ids + answer_ids, "labels": [-100] * len(prompt_ids) + answer_ids}


def make_collate(pad_id):
    """Pad a batch on the right; padding gets label -100 and attention mask 0. (Course notebook.)"""

    def collate(batch):
        length = max(len(ex["input_ids"]) for ex in batch)
        input_ids = torch.full((len(batch), length), pad_id, dtype=torch.long)
        labels = torch.full((len(batch), length), -100, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), length), dtype=torch.long)
        for i, ex in enumerate(batch):
            n = len(ex["input_ids"])
            input_ids[i, :n] = torch.tensor(ex["input_ids"])
            labels[i, :n] = torch.tensor(ex["labels"])
            attention_mask[i, :n] = 1
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

    return collate


train_data = [e for e in (encode_example(m, l) for m, l in zip(train_df["instruction"], train_df["intent"])) if e]
print(f"encoded {len(train_data):,} training examples ({len(train_df) - len(train_data)} skipped as too long)")
_ex = train_data[0]
_n_prompt = sum(1 for l in _ex["labels"] if l == -100)
print(f"example: {len(_ex['input_ids'])} tokens, {_n_prompt} prompt tokens (masked); learned part: "
      f"{tok.decode(_ex['input_ids'][_n_prompt:])!r}")

# %% [markdown]
# ## 2. Training loop (unchanged from the course notebook)

# %%
def autocast():
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=(device.type == "cuda"))


def train_causal_lm(model, train_data, collate, lr, epochs, batch_size, grad_accum=1, desc="training"):
    """Fine-tune any Hugging Face causal LM; returns the list of per-batch training losses."""
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate)
    total_steps = epochs * math.ceil(len(loader) / grad_accum)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.05 * total_steps), total_steps)
    # float16 needs loss scaling to avoid underflowing gradients; bfloat16 and float32 do not
    scaler = torch.amp.GradScaler("cuda") if (device.type == "cuda" and dtype == torch.float16) else None

    model.train()
    losses = []
    for epoch in range(epochs):
        progress = tqdm(loader, desc=f"{desc} (epoch {epoch + 1}/{epochs})")
        for i, batch in enumerate(progress):
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast():
                loss = model(**batch).loss / grad_accum
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (i + 1) % grad_accum == 0 or (i + 1) == len(loader):
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            losses.append(loss.item() * grad_accum)
            if i % 20 == 0:
                progress.set_postfix(loss=f"{sum(losses[-20:]) / len(losses[-20:]):.3f}")
    model.eval()
    return losses


@torch.no_grad()
def classify(model, messages, batch_size=EVAL_BATCH):
    """Greedy label generation; returns (parsed Bitext label or None) per message."""
    model.eval()
    preds = []
    for start in tqdm(range(0, len(messages), batch_size), desc="classifying"):
        prompts = [prompt_text(m) for m in messages[start:start + batch_size]]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with autocast():
            out = model.generate(**enc, max_new_tokens=8, do_sample=False,
                                 pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        for row in out[:, enc["input_ids"].shape[1]:]:
            preds.append(parse_bitext_label(tok.decode(row, skip_special_tokens=True)))
    return preds

# %% [markdown]
# ## 3. LoRA fine-tuning

# %%
_base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype).to(device)
# Recompute activations in the backward pass instead of storing them: less memory, a bit slower
_base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
_base.config.use_cache = False

lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0.05,
    target_modules="all-linear",  # attention and MLP projections (the LM head is excluded)
    task_type="CAUSAL_LM",
)
lora_model = get_peft_model(_base, lora_config)
lora_trainable, lora_total = lora_model.get_nb_trainable_parameters()
print(f"Trainable parameters: {lora_trainable:,} of {lora_total:,} ({100 * lora_trainable / lora_total:.2f}%)")

# %%
if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats()
_t0 = time.perf_counter()
lora_losses = train_causal_lm(
    lora_model, train_data, make_collate(tok.pad_token_id), lr=LORA_LR, epochs=EPOCHS,
    batch_size=LORA_BATCH_SIZE, grad_accum=LORA_GRAD_ACCUM, desc="LoRA",
)
train_seconds = time.perf_counter() - _t0
n_batches = math.ceil(len(train_data) / LORA_BATCH_SIZE)
peak_gpu_gb = torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else None
lora_model.config.use_cache = True  # back on for fast generation
os.makedirs(OUT_DIR, exist_ok=True)
lora_model.save_pretrained(f"{OUT_DIR}/qwen-intent-lora")  # saves only the adapter weights (tens of MB)
print(f"time per epoch {train_seconds / EPOCHS / 60:.1f} min, {train_seconds / n_batches:.2f} s per batch, "
      f"peak GPU memory {peak_gpu_gb if peak_gpu_gb is None else round(peak_gpu_gb, 2)} GB")

# %% [markdown]
# ## 4. Validation and test

# %%
os.makedirs("training/results", exist_ok=True)
tag = "finetuned_quick" if QUICK_RUN else "finetuned"
all_metrics = {}
for split_name, df in [("val", val_df), ("test", test_df)]:
    preds = classify(lora_model, df["instruction"].tolist())
    m, cm = score(df["intent"].tolist(), preds)
    all_metrics[split_name] = m
    pd.DataFrame({"text": df["instruction"], "gold": df["intent"], "pred": preds}).to_csv(
        f"training/results/preds_{tag}_{split_name}.csv", index=False)
    if split_name == "test":
        cm.to_csv(f"training/results/confusion_{tag}.csv")
        save_confusion_png(cm, f"training/results/confusion_{tag}.png", f"{tag} (app labels)")
    print(split_name, json.dumps(m, indent=1))

# %%
run_record = {
    "model": MODEL_ID, "device": str(device),
    "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    "compute_dtype": str(dtype), "quick_run": QUICK_RUN,
    "train_examples": len(train_data), "train_per_intent": TRAIN_PER_INTENT,
    "lora": {"rank": LORA_RANK, "alpha": LORA_ALPHA, "dropout": 0.05, "target_modules": "all-linear"},
    "lr": LORA_LR, "batch_size": LORA_BATCH_SIZE, "grad_accum": LORA_GRAD_ACCUM, "epochs": EPOCHS,
    "max_len": MAX_LEN, "trainable_params": lora_trainable, "total_params": lora_total,
    "time_per_epoch_min": round(train_seconds / EPOCHS / 60, 2),
    "sec_per_batch": round(train_seconds / n_batches, 3),
    "peak_gpu_memory_gb": None if peak_gpu_gb is None else round(peak_gpu_gb, 2),
    "final_train_loss_avg_last50": round(float(np.mean(lora_losses[-50:])), 4),
    "metrics": all_metrics,
    "versions": {"torch": torch.__version__, "transformers": __import__("transformers").__version__,
                 "peft": __import__("peft").__version__},
}
with open(f"training/results/train_run_{tag}.json", "w") as f:
    json.dump(run_record, f, indent=2)
pd.Series(lora_losses).rolling(20, min_periods=1).mean().plot(
    title="training loss on label tokens (moving average)", xlabel="batch").get_figure().savefig(
    f"training/results/train_loss_{tag}.png", dpi=150)
print(json.dumps(run_record, indent=2))

# %% [markdown]
# ## 5. Merge the adapter and save in half precision
#
# We reload the base model in float32 and merge there (merging in float16 would round twice), then store
# the merged weights in **bfloat16**: the app loads bfloat16 on CPU (float16 is slow or unsupported there).

# %%
del lora_model, _base
if device.type == "cuda":
    torch.cuda.empty_cache()

merged = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
merged = PeftModel.from_pretrained(merged, f"{OUT_DIR}/qwen-intent-lora").merge_and_unload()  # W + (alpha/r)BA
merged = merged.to(torch.bfloat16)
MERGED_DIR = f"{OUT_DIR}/qwen-intent-merged"
merged.save_pretrained(MERGED_DIR)
tok.padding_side = "left"
tok.save_pretrained(MERGED_DIR)

# Sanity check: the merged copy must classify like the adapter version did.
_check = AutoModelForCausalLM.from_pretrained(MERGED_DIR, dtype=dtype).to(device)  # GPU compute dtype
for _m in ["I want to cancel order 1234", "where is my refund??", "what is your refund policy?"]:
    print(_m, "->", classify(_check, [_m])[0])
del _check

# %% [markdown]
# ## 6. Publish to Hugging Face (public model repo)
#
# Needs the `HF_TOKEN` Colab secret (a *write* token) and `HF_REPO` set above. Afterwards, add the printed
# repo id and revision to the decisions log; B pins exactly that revision in the Dockerfile.

# %%
if not QUICK_RUN and not HF_REPO.startswith("YOUR_"):
    from huggingface_hub import HfApi

    token = None
    if IN_COLAB:
        from google.colab import userdata
        token = userdata.get("HF_TOKEN")
    api = HfApi(token=token)  # falls back to `huggingface-cli login` outside Colab
    api.create_repo(HF_REPO, repo_type="model", private=False, exist_ok=True)
    with open(f"{MERGED_DIR}/README.md", "w") as f:
        f.write(f"""---
license: apache-2.0
base_model: {MODEL_ID}
datasets: [bitext/Bitext-customer-support-llm-chatbot-training-dataset]
pipeline_tag: text-generation
---
# Qwen2.5-1.5B-Instruct, LoRA-tuned for customer-support intent identification

Course project model. LoRA (r={LORA_RANK}, alpha={LORA_ALPHA}, all linear layers, lr {LORA_LR}, {EPOCHS} epoch,
{len(train_data):,} Bitext examples) merged into the base weights, stored in bfloat16.
Prompt format: see `app/intent_prompt.py` in the project repo (system prompt with the 27 labels, zero-shot).
Bitext test split (grouped near-duplicates): accuracy {all_metrics['test'].get('overall_accuracy')},
macro-F1 {all_metrics['test'].get('overall_macro_f1')}. Outputs are suggestions for staff, never actions.
""")
    commit = api.upload_folder(folder_path=MERGED_DIR, repo_id=HF_REPO, repo_type="model",
                               commit_message="Merged LoRA intent model")
    revision = commit.oid
    print(f"Published: {HF_REPO}\nRevision (put this in the decisions log): {revision}")
    with open("training/results/published_model.json", "w") as f:
        json.dump({"repo": HF_REPO, "revision": revision}, f, indent=2)
else:
    print("Skipped publishing (QUICK_RUN, or HF_REPO not set).")

# %% [markdown]
# ## 7. Download the results
#
# The small result files (metrics, confusion matrix, loss curve, run record) go back into the team repo under
# `training/results/`. Weights never go in git. In Colab, this zips them for download.

# %%
if IN_COLAB:
    subprocess.run("zip -q -r /content/member_a_results.zip training/results", shell=True, check=True)
    from google.colab import files
    files.download("/content/member_a_results.zip")
