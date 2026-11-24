"""Rule engine for the LUMON support copilot (Member C).

Pure Python (standard library only): no FastAPI, no model. The app (Member B) calls
check_request() for every customer message. Python rules decide eligibility here;
the model only explains the decision in the drafted reply.

All numbers (store date, refund window, review limit, refund timelines) come from
store/policies.json, never from this file.
"""

from __future__ import annotations          # lets "dict | None" type hints work on Python 3.9

import json                                 # reads policies.json
from dataclasses import dataclass, asdict   # small result object that converts to a dict
from datetime import date, timedelta        # date math for the refund window and timelines
from pathlib import Path                    # builds the path to policies.json next to this file

POLICIES_PATH = Path(__file__).parent / "policies.json"   # store/policies.json

# Intents the rules handle (same names as app/intent_prompt.py IN_SCOPE)
IN_SCOPE = ["cancel_order", "get_refund", "track_refund", "check_refund_policy"]


@dataclass
class RuleResult:
    """What the rules decided for one request (the agreed rule-engine output)."""
    decision: str        # eligible / ineligible / needs_clarification / needs_review / info
    policy_id: str       # the one policy the decision rests on, e.g. "REF-01"
    policy_text: str     # that policy's full text, quoted in the draft and the API response
    reason: str          # one plain sentence explaining the decision, used in the prompt

    def to_dict(self) -> dict:
        """Plain dict, ready to return as JSON from an endpoint."""
        return asdict(self)


def load_policies(path: Path = POLICIES_PATH) -> dict:
    """Read policies.json and add a lookup of policy text by ID."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    data["policy_by_id"] = {p["id"]: p["text"] for p in data["policies"]}   # "REF-01" -> text
    return data


def _result(policies: dict, decision: str, policy_id: str, reason: str) -> RuleResult:
    """Build a RuleResult and attach the policy text for the given ID."""
    return RuleResult(decision, policy_id, policies["policy_by_id"][policy_id], reason)


def _add_business_days(start: date, days: int) -> date:
    """Count forward a number of business days (Monday-Friday) from a start date."""
    current = start
    while days > 0:
        current += timedelta(days=1)          # move one calendar day
        if current.weekday() < 5:             # 0-4 are Monday-Friday
            days -= 1                         # only weekdays count
    return current


def _fmt(d: date) -> str:
    """Readable date for replies, e.g. 'December 4, 2026'."""
    return f"{d.strftime('%B')} {d.day}, {d.year}"


def check_request(intent: str, order: dict | None, policies: dict | None = None) -> RuleResult:
    """Decide what the store's policies allow for one request.

    intent: label from the model (cancel_order, get_refund, track_refund,
            check_refund_policy, out_of_scope or unknown).
    order:  one order dict with the agreed field names, or None if no order matched.
    """
    policies = policies or load_policies()                         # load from file unless given
    today = date.fromisoformat(policies["store_today"])            # fixed store date, not the real date
    window = policies["refund_window_days"]                        # 30
    review_over = policies["review_total_over"]                    # 1500.00

    # 1. Anything outside the four in-scope intents goes to a person.
    if intent not in IN_SCOPE:
        return _result(policies, "needs_review", "REV-01",
                       "The request is outside what the assistant handles, so a staff member should reply.")

    # 2. A general policy question needs no order: answer it from the refund policy.
    if intent == "check_refund_policy":
        return _result(policies, "info", "REF-01",
                       f"The customer is asking about the refund policy: refundable items can be refunded "
                       f"within {window} days of delivery, unworn and with tags attached.")

    # 3. Every other request needs a matching order first.
    if order is None:
        return _result(policies, "needs_clarification", "INF-01",
                       "No matching order was found, so the customer needs to provide their order ID.")

    status = order.get("ship_status")                  # not_shipped / shipped / delivered
    refund_status = order.get("refund_status")         # none / requested / pending / completed
    payment = order.get("payment_method")              # credit_card / paypal / gift_card / None

    # 4. High-value cancellations and refunds need a staff member (status questions do not).
    if intent in ("cancel_order", "get_refund") and order.get("total", 0) > review_over:
        return _result(policies, "needs_review", "REV-01",
                       f"The order total is over ${review_over:,.0f}, so a staff member must review the request.")

    # 5. Cancellation: allowed only before the order ships.
    if intent == "cancel_order":
        if status == "not_shipped":
            return _result(policies, "eligible", "CAN-01",
                           "The order has not shipped yet, so it can be cancelled at no charge.")
        return _result(policies, "ineligible", "CAN-01",
                       f"The order has already {'been delivered' if status == 'delivered' else 'shipped'}, "
                       "so it can no longer be cancelled; the customer can request a refund after delivery.")

    # 6. Refund status questions.
    if intent == "track_refund":
        if refund_status == "none":
            return _result(policies, "needs_clarification", "REF-02",
                           "There is no refund on file for this order, so the customer should confirm "
                           "the order ID or whether they have requested a return.")
        if payment is None or order.get("refund_date") is None:
            return _result(policies, "needs_review", "REV-01",
                           "The order record is missing the payment method or refund date needed to give a timeline.")
        changed = date.fromisoformat(order["refund_date"])          # date the refund last changed status
        if refund_status == "requested":
            return _result(policies, "info", "REF-02",
                           f"The return was requested on {_fmt(changed)} and is waiting for the item "
                           "to arrive and be approved by staff.")
        if refund_status == "pending":
            low, high = policies["refund_business_days"][payment]   # e.g. [5, 7] for credit cards
            first = _fmt(_add_business_days(changed, low))           # earliest arrival date
            last = _fmt(_add_business_days(changed, high))           # latest arrival date
            arrival = f"by {last}" if low == high else f"between {first} and {last}"
            return _result(policies, "info", "REF-02",
                           f"The refund was approved on {_fmt(changed)} and should reach the customer {arrival}.")
        return _result(policies, "info", "REF-02",                  # completed
                       f"The refund was completed on {_fmt(changed)}.")

    # 7. Refund requests (intent == "get_refund").
    if refund_status in ("requested", "pending"):
        return _result(policies, "info", "REF-02",
                       "A refund for this order is already in progress, so no new request is needed.")
    if refund_status == "completed":
        return _result(policies, "ineligible", "REF-02",
                       "This order has already been refunded.")
    if status == "not_shipped":
        return _result(policies, "ineligible", "CAN-01",
                       "The order has not shipped yet, so it cannot be refunded; it can be cancelled at no charge instead.")
    if status == "shipped":
        return _result(policies, "ineligible", "REF-01",
                       "The order has not been delivered yet, and the refund window starts on delivery.")
    if order.get("delivery_date") is None or payment is None:
        return _result(policies, "needs_review", "REV-01",
                       "The order record is missing the delivery date or payment method needed to decide.")

    refundable = [item["refundable"] for item in order["items"]]   # one True/False per item
    if not any(refundable):
        return _result(policies, "ineligible", "REF-03",
                       "The items in this order are not refundable.")
    if not all(refundable):
        return _result(policies, "needs_review", "REV-01",
                       "The order has both refundable and non-refundable items, so a staff member must review it.")

    days = (today - date.fromisoformat(order["delivery_date"])).days   # days since delivery
    if days > window:
        return _result(policies, "ineligible", "REF-01",
                       f"The order was delivered {days} days ago, past the {window}-day refund window.")
    return _result(policies, "eligible", "REF-01",
                   f"The order was delivered {days} days ago, within the {window}-day refund window, "
                   "so the items can be returned for a refund if unworn with tags attached.")