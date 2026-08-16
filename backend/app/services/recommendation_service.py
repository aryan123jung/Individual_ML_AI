from __future__ import annotations

import pandas as pd
from fastapi import HTTPException

from app.core import loaders


def _json_safe_record(record):
    safe = {}
    for key, value in record.items():
        if pd.isna(value):
            safe[key] = None
        else:
            safe[key] = value.item() if hasattr(value, "item") else value
    return safe


def _json_safe_records(df: pd.DataFrame):
    return [_json_safe_record(row) for row in df.to_dict(orient="records")]


def get_all_recommendations(limit: int | None = None):
    df = loaders.load_recommendations()
    if "overall_team_rank" in df.columns:
        df = df.sort_values(["team", "overall_team_rank"], ascending=[True, True])
    if limit:
        df = df.head(limit)
    return _json_safe_records(df)


def get_team_recommendations(team: str, limit: int | None = None, recommendation_type: str | None = None):
    loaders.clear_caches()
    df = loaders.load_recommendations()
    if df.empty:
        raise HTTPException(status_code=404, detail="No recommendation data available.")

    team_names = df["team"].dropna().astype(str)
    if team_names.str.lower().eq(team.lower()).sum() == 0:
        raise HTTPException(status_code=404, detail=f"Team not found in recommendations: {team}")

    df = df[df["team"].str.lower() == team.lower()].copy()
    if recommendation_type:
        df = df[df["recommendation_type"].str.lower() == recommendation_type.lower()].copy()
        if df.empty:
            return []
    if "overall_team_rank" in df.columns:
        df = df.sort_values("overall_team_rank", ascending=True)
    if limit:
        df = df.head(limit)
    return _json_safe_records(df)


def get_role_recommendations(role: str, limit: int | None = None):
    loaders.clear_caches()
    df = loaders.load_recommendations()
    if df.empty:
        raise HTTPException(status_code=404, detail="No recommendation data available.")
    df = df[df["target_role"].str.lower() == role.lower()].copy()
    if df.empty:
        raise HTTPException(status_code=404, detail=f"Role not found in recommendations: {role}")
    if limit:
        df = df.head(limit)
    return _json_safe_records(df)


def get_team_summary():
    loaders.clear_caches()
    return _json_safe_records(loaders.load_team_summary())


def get_squad_gaps(team: str | None = None):
    loaders.clear_caches()
    df = loaders.load_squad_gaps()
    if team:
        df = df[df["team"].str.lower() == team.lower()].copy()
        if df.empty:
            raise HTTPException(status_code=404, detail=f"Team not found in squad gaps: {team}")
    return _json_safe_records(df)


def get_recommendation_health():
    loaders.clear_caches()
    qa_df = loaders.load_recommendation_qa_summary()
    candidate_pool_df = loaders.load_candidate_pool_summary()
    recommendations_df = loaders.load_recommendations()

    checks = _json_safe_records(qa_df)
    candidate_pool = _json_safe_records(candidate_pool_df)
    pass_count = sum(1 for row in checks if str(row.get("status", "")).lower() == "pass")
    total_checks = len(checks)

    return {
        "checks": checks,
        "candidate_pool": candidate_pool,
        "summary": {
            "total_checks": total_checks,
            "passed_checks": pass_count,
            "failed_checks": max(total_checks - pass_count, 0),
            "recommendation_rows": int(len(recommendations_df)),
        },
    }
