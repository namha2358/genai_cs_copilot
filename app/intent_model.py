"""Model loading for the app (Member A, handoff 5).

Kickoff interface (Step 0), the same as B's fake model:
    classify(message: str) -> str                 one of app.intent_prompt.APP_LABELS
    generate(prompt: str, max_length: int) -> str max_length = maximum NEW tokens

Load ONCE at startup, e.g. in main.py:
    from app.intent_model import load_model
    model = load_model()          # reads the settings below; takes ~5-30 s on CPU

Settings (environment variables; B decides how "fake" is selected before calling this):
    INTENT_MODEL_ID        Hugging Face repo id or local folder.  Default: Qwen/Qwen2.5-1.5B-Instruct (base)
    INTENT_MODEL_REVISION  exact commit hash to pin (from the decisions log).  Default: latest
    INTENT_MODEL_DTYPE     bfloat16 (default) or float32.  Never float16: slow or unsupported on CPU.
    INTENT_PROMPT          few_shot or zero_shot.  Default: few_shot for the base model, zero_shot otherwise
                           (the fine-tuned model was trained on the zero_shot format).
"""

import os
import threading

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.intent_prompt import build_messages, parse_bitext_label, to_app_label

BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


class IntentModel:
    def __init__(self, model_id=BASE_MODEL_ID, revision=None, dtype="bfloat16", few_shot=None):
        if dtype not in ("bfloat16", "float32"):
            raise ValueError("dtype must be bfloat16 or float32 on CPU (float16 is slow or unsupported)")
        self.model_id, self.revision = model_id, revision
        # Base model needs examples in the prompt; the fine-tuned one was trained without them.
        self.few_shot = (model_id == BASE_MODEL_ID) if few_shot is None else few_shot

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.tokenizer.padding_side = "left"  # decoder-only: pad on the left so generation follows the prompt
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, dtype=getattr(torch, dtype)
        ).eval()  # eval(): turns off dropout
        # FastAPI runs sync endpoints in a thread pool; one generation at a time keeps memory flat.
        self._lock = threading.Lock()

    @torch.no_grad()  # inference only: no gradients, less memory
    def _generate_chat(self, chats: list[list[dict]], max_new_tokens: int) -> list[str]:
        texts = [
            # add_generation_prompt=True opens the assistant turn, so the model answers instead of continuing the user
            self.tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in chats
        ]
        enc = self.tokenizer(texts, return_tensors="pt", padding=True)
        with self._lock:
            out = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # greedy: the same input always gives the same output
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        new_tokens = out[:, enc["input_ids"].shape[1]:]  # drop the prompt, keep only what the model wrote
        return [self.tokenizer.decode(row, skip_special_tokens=True).strip() for row in new_tokens]

    def classify_bitext_batch(self, messages: list[str]) -> list[tuple[str | None, str]]:
        """For evaluation: (parsed Bitext label or None, raw model text) per message."""
        raw = self._generate_chat([build_messages(m, self.few_shot) for m in messages], max_new_tokens=8)
        return [(parse_bitext_label(r), r) for r in raw]

    def classify(self, message: str) -> str:
        """Kickoff interface: message -> in-scope label, out_of_scope, or unknown."""
        bitext_label, _ = self.classify_bitext_batch([message])[0]
        return to_app_label(bitext_label)

    def generate(self, prompt: str, max_length: int = 200) -> str:
        """Kickoff interface: C's drafting prompt -> draft text. max_length = max new tokens."""
        return self._generate_chat([[{"role": "user", "content": prompt}]], max_new_tokens=max_length)[0]


def load_model() -> IntentModel:
    """Build the model from environment settings. Call once at app startup."""
    model_id = os.getenv("INTENT_MODEL_ID", BASE_MODEL_ID)
    prompt = os.getenv("INTENT_PROMPT")  # None -> decided by model id
    return IntentModel(
        model_id=model_id,
        revision=os.getenv("INTENT_MODEL_REVISION") or None,
        dtype=os.getenv("INTENT_MODEL_DTYPE", "bfloat16"),
        few_shot=None if prompt is None else (prompt == "few_shot"),
    )


if __name__ == "__main__":  # quick manual check: python -m app.intent_model
    m = load_model()
    for msg in ["I want to cancel order 1234", "where is my refund??", "what's the weather like?"]:
        print(f"{msg!r} -> {m.classify(msg)}")
    print(m.generate("Write one polite sentence thanking a customer.", max_length=30))
