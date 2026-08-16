from fastapi import APIRouter

from app.schemas.inference import CustomInferenceRequest
from app.services import inference_service


router = APIRouter(prefix="/inference", tags=["inference"])


@router.get("/domains")
def get_domains():
    return {"domains": inference_service.available_domains()}


@router.get("/{domain}/players/{player_name}")
def infer_existing_player(domain: str, player_name: str):
    return inference_service.infer_existing_player(domain, player_name)


@router.post("/{domain}/custom")
def infer_custom_player(domain: str, payload: CustomInferenceRequest):
    return inference_service.infer_custom_player(domain, payload.feature_values)
