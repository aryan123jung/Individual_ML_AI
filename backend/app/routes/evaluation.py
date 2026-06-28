from fastapi import APIRouter

from app.services import evaluation_service


router = APIRouter(prefix="/evaluation", tags=["evaluation"])


@router.get("/metrics")
def get_metrics():
    return evaluation_service.get_model_metrics()
