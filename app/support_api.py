from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.database import get_order, list_orders


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
def get_health():
    """Report service status; the language model is not connected yet."""
    return HealthResponse(
        status="ok",
        model_mode="not_configured",
        model_loaded=False,
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