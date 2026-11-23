"""Commitment check for drafted replies (Member C).

check_draft() flags drafts that claim more than the rules allow. The app (Member B) sends
flagged drafts to needs_review; the scenario tests (Member C) count the same flags.
The check is deliberately strict: a false flag only means a person reads the draft.
"""

from __future__ import annotations   # lets "list[str]" type hints work on Python 3.9

import re   # pattern matching on the draft text

# Claims that an action already happened. The assistant never processes anything, so these are flagged
# whenever the decision is not about a past fact (see check_draft).
DONE_CLAIMS = re.compile(
    r"\b(has|have) been (refunded|cancell?ed|processed|issued|approved)\b"
    r"|\bwe('ve| have) (refunded|cancell?ed|processed|issued|approved)\b"
    r"|\brefund (was|has been) (issued|sent|processed)\b",
    re.IGNORECASE)

# Promises that only an "eligible" decision supports.
PROMISES = re.compile(
    r"\bwe('ll| will) (refund|cancel|issue|process)\b"
    r"|\byou('ll| will) (receive|get) (a|your|the) (full )?refund\b"
    r"|\b(?<!not )eligible for (a )?refund\b"
    r"|\b(?<!not )approved\b",
    re.IGNORECASE)

# Words no decision supports.
ALWAYS_FLAG = re.compile(r"\bguarantee(d)?\b", re.IGNORECASE)

# Wording that contradicts an "eligible" decision.
REFUSALS = re.compile(
    r"\bnot eligible\b|\bunable to (cancel|refund)\b|\bcannot be (cancell?ed|refunded)\b",
    re.IGNORECASE)


def check_draft(draft: str, decision: str, policy_id: str) -> tuple[bool, list[str]]:
    """Return (passed, reasons). passed is False if any problem was found."""
    reasons = []
    if policy_id.lower() not in draft.lower():                        # e.g. "REF-01" must appear
        reasons.append(f"Draft does not cite policy {policy_id}.")
    # Past-tense facts are fine for info and ineligible replies (e.g. "your refund was completed on ...")
    if decision in ("eligible", "needs_clarification", "needs_review") and DONE_CLAIMS.search(draft):
        reasons.append("Draft claims an action was already completed.")
    if ALWAYS_FLAG.search(draft):
        reasons.append("Draft uses a guarantee.")
    if decision == "eligible" and REFUSALS.search(draft):
        reasons.append("Draft refuses a request the rules allow.")
    if decision in ("ineligible", "needs_clarification", "needs_review") and PROMISES.search(draft):
        reasons.append(f"Draft promises something a '{decision}' decision does not support.")
    return (len(reasons) == 0, reasons)