from fastapi import APIRouter

from app.services import common_service


router = APIRouter(prefix="/system", tags=["system"])


@router.get("/summary")
def get_system_summary():
    return common_service.get_system_summary()


@router.post("/refresh")
def refresh_caches():
    return common_service.refresh_data_caches()
