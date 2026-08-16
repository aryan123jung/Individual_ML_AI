from fastapi import APIRouter, Query

from app.services import player_service


router = APIRouter(prefix="/players", tags=["players"])


@router.get("/compare/")
def compare_players(
    player_a: str = Query(..., min_length=1),
    player_b: str = Query(..., min_length=1),
):
    return player_service.compare_players(player_a, player_b)


@router.get("/{player_name}")
def get_player(player_name: str):
    return player_service.get_player(player_name)
