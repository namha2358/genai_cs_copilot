from fastapi import APIRouter
from pydantic import BaseModel


router = APIRouter(tags=["Customer Support"])


class HealthResponse(BaseModel):
    status: str
    model_mode: str
    model_loaded: bool
    store_data_loaded: bool


@router.get("/health", response_model=HealthResponse)
def get_health():
    """Report the current development state of the support service."""
    return HealthResponse(
        status="ok",
        model_mode="not_configured",
        model_loaded=False,
        store_data_loaded=False,
    )