# Handoff 5: model-loading module (Member A → B and C)

**Files**
- `app/intent_model.py`: loads the model, `classify`, `generate`.
- `app/intent_prompt.py`: the labels, the prompt and the output parser. It is pure Python and the single source of truth.

**Interface (kickoff Step 0, same as B's fake model)**
```python
from app.intent_model import load_model
model = load_model()                       # once, at app startup
model.classify("I want to cancel order 1234")   # -> "cancel_order"
model.generate(prompt_text, max_length=200)     # -> draft string
```
- `classify` returns one of:
  - `cancel_order`, `get_refund`, `track_refund`, `check_refund_policy`
  - `out_of_scope`: any other Bitext intent
  - `unknown`: the model's output wasn't a valid label
- `generate`:
  - `max_length` means **maximum new tokens**. ⚠ Please confirm this in the decisions log.
  - The prompt is sent as one user message through the model's chat template.
  - Decoding is greedy, so the same input always gives the same output.

**Settings (environment variables)** ⚠ The names are new. Please add them to the decisions log.

| variable | default | notes |
|---|---|---|
| `INTENT_MODEL_ID` | `Qwen/Qwen2.5-1.5B-Instruct` | a Hugging Face repo id or a local folder |
| `INTENT_MODEL_REVISION` | latest | pin the exact commit from the decisions log |
| `INTENT_MODEL_DTYPE` | `bfloat16` | or `float32`; float16 is refused because it is slow or unsupported on CPU |
| `INTENT_PROMPT` | auto | `few_shot` for the base model, `zero_shot` for the fine-tuned model |

How the fake model is selected is B's choice. Check your own setting before calling `load_model()`.

**For B (Docker / pyproject)**
- New dependency: `transformers` (tested with 5.24.0) on top of the CPU torch already in `pyproject.toml`.
- Memory: about 3.4 GB peak for 1.5B in bfloat16, measured outside Docker.
- Timings, measured on a 4-vCPU Xeon:
  - load: 5–30 s
  - `classify` with the base model and few-shot prompt: about 1–1.5 s
  - a 150-token draft: about 16 s

**For C**
- Your scenario tests can use the base model now: the defaults are the base model with the few-shot prompt.
- After the fine-tune, set `INTENT_MODEL_ID` and `INTENT_MODEL_REVISION` to the published repo. No code changes are needed.

**Check it yourself:** `python -m app.intent_model` classifies 3 messages and writes one sentence.
