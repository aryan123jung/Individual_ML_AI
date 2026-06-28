from __future__ import annotations

from app.core import loaders


def get_all_recommendations(limit: int | None = None):
    df = loaders.load_recommendations()
    if limit:
        df = df.head(limit)
    return df.to_dict(orient="records")


def get_team_recommendations(team: str, limit: int | None = None):
    df = loaders.load_recommendations()
    df = df[df["team"].str.lower() == team.lower()].copy()
    if limit:
        df = df.head(limit)
    return df.to_dict(orient="records")


def get_role_recommendations(role: str, limit: int | None = None):
    df = loaders.load_recommendations()
    df = df[df["target_role"].str.lower() == role.lower()].copy()
    if limit:
        df = df.head(limit)
    return df.to_dict(orient="records")


def get_team_summary():
    return loaders.load_team_summary().to_dict(orient="records")


def get_squad_gaps(team: str | None = None):
    df = loaders.load_squad_gaps()
    if team:
        df = df[df["team"].str.lower() == team.lower()].copy()
    return df.to_dict(orient="records")
