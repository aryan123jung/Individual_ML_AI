from fastapi import APIRouter, Query

from app.services import recommendation_service


router = APIRouter(prefix="/recommendations", tags=["recommendations"])


@router.get("/")
def get_recommendations(limit: int | None = Query(default=50, ge=1, le=500)):
    return recommendation_service.get_all_recommendations(limit=limit)


@router.get("/teams/summary")
def get_team_summary():
    return recommendation_service.get_team_summary()


@router.get("/teams/{team_name}")
def get_team_recommendations(team_name: str, limit: int | None = Query(default=25, ge=1, le=200)):
    return recommendation_service.get_team_recommendations(team_name, limit=limit)


@router.get("/roles/{role_name}")
def get_role_recommendations(role_name: str, limit: int | None = Query(default=25, ge=1, le=200)):
    return recommendation_service.get_role_recommendations(role_name, limit=limit)


@router.get("/gaps")
def get_squad_gaps(team: str | None = None):
    return recommendation_service.get_squad_gaps(team=team)
