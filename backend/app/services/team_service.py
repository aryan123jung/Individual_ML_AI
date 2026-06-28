from __future__ import annotations

from app.core import loaders


def list_teams():
    df = loaders.load_team_summary()
    if df.empty:
        return []
    return sorted(df["team"].dropna().unique().tolist())


def get_team_detail(team: str):
    return {
        "team": team,
        "summary": [
            row
            for row in loaders.load_team_summary().to_dict(orient="records")
            if str(row.get("team", "")).lower() == team.lower()
        ],
        "gaps": [
            row
            for row in loaders.load_squad_gaps().to_dict(orient="records")
            if str(row.get("team", "")).lower() == team.lower()
        ],
        "underperformers": [
            row
            for row in loaders.load_underperformers().to_dict(orient="records")
            if str(row.get("team", "")).lower() == team.lower()
        ],
        "replacements": [
            row
            for row in loaders.load_replacements().to_dict(orient="records")
            if str(row.get("team", "")).lower() == team.lower()
        ],
    }
