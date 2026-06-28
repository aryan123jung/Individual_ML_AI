from fastapi import APIRouter

from app.services import team_service


router = APIRouter(prefix="/teams", tags=["teams"])


@router.get("/")
def get_teams():
    return team_service.list_teams()


@router.get("/{team_name}")
def get_team(team_name: str):
    return team_service.get_team_detail(team_name)
