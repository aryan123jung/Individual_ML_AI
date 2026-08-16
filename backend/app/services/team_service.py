from __future__ import annotations

import pandas as pd
from fastapi import HTTPException

from app.core import loaders


def list_teams():
    df = loaders.load_team_summary()
    if df.empty:
        return []
    return sorted(df["team"].dropna().unique().tolist())


def _json_safe_record(record):
    safe = {}
    for key, value in record.items():
        if pd.isna(value):
            safe[key] = None
        else:
            safe[key] = value.item() if hasattr(value, "item") else value
    return safe


def _severity_rank(value):
    order = {"high": 3, "medium": 2, "low": 1}
    return order.get(str(value or "").lower(), 0)


def _action_rank(value):
    order = {"replace_now": 3, "watchlist": 2, "competition_needed": 1}
    return order.get(str(value or "").lower(), 0)


def get_team_detail(team: str):
    loaders.clear_caches()
    current_squad = loaders.load_current_squad()
    squad_team_column = "team" if "team" in current_squad.columns else None
    available_teams = set(str(name).lower() for name in list_teams())
    if team.lower() not in available_teams:
        raise HTTPException(status_code=404, detail=f"Team not found: {team}")
    all_watch_rows = [
        row
        for row in loaders.load_underperformers().to_dict(orient="records")
        if str(row.get("team", "")).lower() == team.lower()
    ]
    gap_rows = [
        _json_safe_record(row)
        for row in loaders.load_squad_gaps().to_dict(orient="records")
        if str(row.get("team", "")).lower() == team.lower()
    ]
    all_watch_rows = sorted(
        all_watch_rows,
        key=lambda row: (
            -_action_rank(row.get("action_level")),
            -_severity_rank(row.get("severity")),
            -float(row.get("underperformance_score", 0) or 0),
            float(row.get("replacement_priority_rank", 999) or 999),
        ),
    )
    underperformers = [
        row
        for row in all_watch_rows
        if str(row.get("action_level", "")).lower() in {"watchlist", "replace_now"}
    ]
    monitoring_players = [
        row
        for row in all_watch_rows
        if str(row.get("action_level", "")).lower() == "competition_needed"
    ]

    actionable_watch_rows = [
        row
        for row in all_watch_rows
        if str(row.get("action_level", "")).lower() in {"competition_needed", "watchlist", "replace_now"}
    ]
    underperforming_players = {str(row.get("player", "")) for row in actionable_watch_rows}
    replacements = [
        row
        for row in loaders.load_replacements().to_dict(orient="records")
        if str(row.get("team", "")).lower() == team.lower()
        and str(row.get("underperforming_player", "")) in underperforming_players
    ]
    replacements = sorted(
        replacements,
        key=lambda row: (
            str(row.get("underperforming_player", "")),
            float(row.get("replacement_rank", 999) or 999),
            -float(row.get("replacement_score", 0) or 0),
        ),
    )
    limited_replacements = []
    seen_counts = {}
    for row in replacements:
        player = str(row.get("underperforming_player", ""))
        seen_counts[player] = seen_counts.get(player, 0) + 1
        if seen_counts[player] <= 3:
            limited_replacements.append(row)

    positive_gap_rows = [row for row in gap_rows if float(row.get("deficit", 0) or 0) > 0]
    gap_summary = {
        "total_gap_roles": len(positive_gap_rows),
        "total_missing_slots": int(sum(float(row.get("deficit", 0) or 0) for row in positive_gap_rows)),
        "critical_gap_roles": sum(1 for row in positive_gap_rows if float(row.get("deficit", 0) or 0) >= 2),
    }

    return {
        "team": team,
        "current_squad": (
            [
                _json_safe_record(row)
                for row in current_squad.to_dict(orient="records")
                if str(row.get(squad_team_column, "")).lower() == team.lower()
            ]
            if squad_team_column
            else []
        ),
        "summary": [
            _json_safe_record(row)
            for row in loaders.load_team_summary().to_dict(orient="records")
            if str(row.get("team", "")).lower() == team.lower()
        ],
        "gaps": gap_rows,
        "gap_summary": gap_summary,
        "underperformers": [_json_safe_record(row) for row in underperformers],
        "monitoring_players": [_json_safe_record(row) for row in monitoring_players],
        "replacements": [_json_safe_record(row) for row in limited_replacements],
    }
