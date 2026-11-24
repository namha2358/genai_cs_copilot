"""Intent labels, the classification prompt and the output parser (Member A).

Pure Python, no torch: imported by the app's loading module (app/intent_model.py),
the few-shot baseline and the training notebook, so all three use the SAME prompt and
the SAME label rules. Change the prompt here and nowhere else (and retrain if you do).
"""

import re

# The 27 Bitext intents, exactly as spelled in the dataset (checked by training/download_bitext.py).
BITEXT_LABELS = [
    "cancel_order", "change_order", "change_shipping_address", "check_cancellation_fee", "check_invoice",
    "check_payment_methods", "check_refund_policy", "complaint", "contact_customer_service",
    "contact_human_agent", "create_account", "delete_account", "delivery_options", "delivery_period",
    "edit_account", "get_invoice", "get_refund", "newsletter_subscription", "payment_issue", "place_order",
    "recover_password", "registration_problems", "review", "set_up_shipping_address", "switch_account",
    "track_order", "track_refund",
]

# Kickoff Step 0: the intents the app handles. Everything else -> out_of_scope; unparsable -> unknown.
IN_SCOPE = ["cancel_order", "get_refund", "track_refund", "check_refund_policy"]
OUT_OF_SCOPE = "out_of_scope"
UNKNOWN = "unknown"
APP_LABELS = IN_SCOPE + [OUT_OF_SCOPE, UNKNOWN]

SYSTEM_PROMPT = (
    "You classify customer support messages for an online store. "
    "Reply with exactly one intent label from this list and nothing else:\n"
    + ", ".join(BITEXT_LABELS)
)

# Few-shot examples: 3 per in-scope intent, copied verbatim from the TRAIN split
# (never from val/test), interleaved so no intent sits last in the prompt every time.
FEW_SHOT_EXAMPLES = [
    ("how do i cancel order {{Order Number}}", "cancel_order"),
    ("where do I get a reimbursement?", "get_refund"),
    ("help me seeing the current status of my reimbursement", "track_refund"),
    ("can you help me checking your reimbursement policy?", "check_refund_policy"),
    ("problem with canceling order {{Order Number}}", "cancel_order"),
    ("help me obtain a compensation of money", "get_refund"),
    ("I do not know how to see the status of my rebate", "track_refund"),
    ("can i check how long refunds usually take", "check_refund_policy"),
    ("I want help cancelling purchase {{Order Number}}", "cancel_order"),
    ("I have got to ask for a rebate, can you help me?", "get_refund"),
    ("how to see if there is anything new on my compensation", "track_refund"),
    ("I want to see in which cases can I request to be refunded", "check_refund_policy"),
]


def build_messages(message: str, few_shot: bool) -> list[dict]:
    """Chat messages for one classification.

    few_shot=True  -> base model: examples are given as earlier user/assistant turns
                      (the multi-turn few-shot pattern from the Hugging Face chat-templating docs).
    few_shot=False -> fine-tuned model: the exact format it was trained on, no examples.
    """
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    if few_shot:
        for text, label in FEW_SHOT_EXAMPLES:
            msgs.append({"role": "user", "content": text})
            msgs.append({"role": "assistant", "content": label})
    msgs.append({"role": "user", "content": message})
    return msgs


# Longest labels first, so a label that is part of a longer one can never win by accident.
_LABEL_RE = re.compile(r"\b(" + "|".join(sorted(BITEXT_LABELS, key=len, reverse=True)) + r")\b")


def parse_bitext_label(output: str) -> str | None:
    """Model text -> one of the 27 Bitext labels, or None if no valid label is present."""
    text = output.strip().lower().replace("`", "").replace('"', "").replace("'", "")
    first = text.split()[0].strip(".,:;") if text.split() else ""
    if first in BITEXT_LABELS:  # the normal case: the reply is just the label
        return first
    match = _LABEL_RE.search(text)  # fallback: a valid label somewhere in a chattier reply
    return match.group(1) if match else None


def to_app_label(bitext_label: str | None) -> str:
    """Bitext label -> app label: in-scope stays, other Bitext intents -> out_of_scope, None -> unknown."""
    if bitext_label is None:
        return UNKNOWN
    return bitext_label if bitext_label in IN_SCOPE else OUT_OF_SCOPE
