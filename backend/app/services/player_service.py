from __future__ import annotations

from app.core import loaders


def get_player(player_name: str):
    recommendations = loaders.load_recommendations()
    current_squad = loaders.load_current_squad()
    similarity = loaders.load_similarity()

    rec_rows = recommendations[
        recommendations["domestic_player"].str.lower() == player_name.lower()
    ].to_dict(orient="records")
    squad_rows = current_squad[
        current_squad["player"].str.lower() == player_name.lower()
    ].to_dict(orient="records")
    similarity_rows = similarity[
        similarity["domestic_player"].str.lower() == player_name.lower()
    ].to_dict(orient="records")

    return {
        "player": player_name,
        "recommendations": rec_rows,
        "current_squad_profile": squad_rows,
        "similarity_matches": similarity_rows,
    }


def compare_players(player_a: str, player_b: str):
    recommendations = loaders.load_recommendations()
    result = {}
    for player in [player_a, player_b]:
        result[player] = recommendations[
            recommendations["domestic_player"].str.lower() == player.lower()
        ].to_dict(orient="records")
    return result
