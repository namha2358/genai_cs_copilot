from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from app.intent_prompt import APP_LABELS

from app.database import get_order, list_orders, save_draft, get_draft
from store.rules import check_request, load_policies
from store.prompt import build_prompt
from store.checks import check_draft

router = APIRouter(tags=["Customer Support"])


class HealthResponse(BaseModel):
    status: str
    model_mode: str
    model_loaded: bool
    store_data_loaded: bool


def public_order(order: dict) -> dict:
    """Exclude evaluation-only metadata from API responses."""
    return {
        key: value
        for key, value in order.items()
        if key != "test_case"
    }

@router.get("/health", response_model=HealthResponse)
def get_health(request: Request):
    """Report database availability and actual model-loading state."""
    model = getattr(request.app.state, "intent_model", None)

    return HealthResponse(
        status="ok",
        model_mode="real" if model is not None else "not_configured",
        model_loaded=model is not None,
        store_data_loaded=bool(list_orders()),
    )


@router.get("/orders")
def get_orders() -> list[dict]:
    """Return simulated orders for staff selection."""
    return [public_order(order) for order in list_orders()]


@router.get("/orders/{order_id}")
def get_order_details(order_id: str) -> dict:
    """Return one order or HTTP 404 if it does not exist."""
    order = get_order(order_id)

    if order is None:
        raise HTTPException(
            status_code=404,
            detail="Order not found.",
        )

    return public_order(order)

class IntentRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Message must not be blank.")
        return value


class IntentResponse(BaseModel):
    intent: str


@router.post("/classify_intent", response_model=IntentResponse)
def classify_intent(payload: IntentRequest, request: Request):
    """Classify a customer message without making an eligibility decision."""
    model = getattr(request.app.state, "intent_model", None)

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Intent model is not loaded.",
        )

    label = model.classify(payload.message)
    if label not in APP_LABELS:
        label = "unknown"

    return IntentResponse(intent=label)

DECISION_TO_STATUS = {
    "eligible": "pending_approval",
    "ineligible": "pending_approval",
    "info": "pending_approval",
    "needs_clarification": "needs_clarification",
    "needs_review": "needs_review",
}


class DraftRequest(IntentRequest):
    order_id: str | None = Field(default=None, max_length=100)

    @field_validator("order_id")
    @classmethod
    def normalize_order_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip() or None


class DraftResponse(BaseModel):
    intent: str
    order_id: str | None
    order_found: bool
    decision: str
    status: str
    policy_id: str
    policy_text: str
    reason: str
    draft: str
    checks_passed: bool
    check_reasons: list[str]

class SavedDraftResponse(DraftResponse):
    draft_id: str
    created_at: str
    updated_at: str
    version: int
    approved_at: str | None
    approved_by: str | None

@router.post("/draft_reply", response_model=SavedDraftResponse)
def draft_reply(payload: DraftRequest, request: Request):
    """Generate and check a draft without sending it or changing an order."""
    model = getattr(request.app.state, "intent_model", None)

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Intent model is not loaded.",
        )

    # Identify the request using Member A's model.
    intent = model.classify(payload.message)
    if intent not in APP_LABELS:
        intent = "unknown"

    # Missing orders are passed to the rules as None.
    order = get_order(payload.order_id) if payload.order_id else None
    order_for_prompt = public_order(order) if order is not None else None

    # Use the same policy snapshot for the decision and drafting prompt.
    policies = load_policies()
    result = check_request(intent, order, policies=policies)
    prompt = build_prompt(
        order_for_prompt,
        result,
        payload.message,
        policies=policies,
    )

    # The model explains the rule decision; it does not authorize actions.
    draft = model.generate(prompt, max_length=250).strip()

    passed, reasons = check_draft(
        draft,
        result.decision,
        result.policy_id,
    )

    if not draft:
        passed = False
        reasons.append("The model returned an empty draft.")
    
    # Temporary conservative checks for unsupported timing and delivery claims.
    draft_lower = draft.lower()

    if "immediately" in draft_lower:
        passed = False
        reasons.append(
            "Draft mentions immediate action; staff must verify this timing."
        )

    if "confirmation email" in draft_lower:
        passed = False
        reasons.append(
            "Draft mentions a confirmation email; the supplied policy "
            "does not specify this delivery channel."
        )

    # Preserve the rule decision even when the draft needs review.
    status = DECISION_TO_STATUS[result.decision]
    if not passed:
        status = "needs_review"

        response = DraftResponse(
        intent=intent,
        order_id=payload.order_id,
        order_found=order is not None,
        decision=result.decision,
        status=status,
        policy_id=result.policy_id,
        policy_text=result.policy_text,
        reason=result.reason,
        draft=draft,
        checks_passed=passed,
        check_reasons=reasons,
    )
        saved = save_draft(
        customer_message=payload.message,
        response=response.model_dump(),
        order_snapshot=order_for_prompt,
        model_id=model.model_id,
        model_revision=model.revision,
    )

    return SavedDraftResponse(**saved)

@router.get("/drafts/{draft_id}", response_model=SavedDraftResponse)
def get_draft_details(draft_id: str):
    """Retrieve a saved draft without generating a new reply."""
    record = get_draft(draft_id)

    if record is None:
        raise HTTPException(
            status_code=404,
            detail="Draft not found.",
        )

    return SavedDraftResponse(**record)