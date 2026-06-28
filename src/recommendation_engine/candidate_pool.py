import pandas as pd
import numpy as np

from recommendation_engine.config import (
    SMAT_MIN_RECENT_BATTING_MATCHES,
    SMAT_MIN_RECENT_BOWLING_MATCHES,
    SMAT_RECENT_SEASON_WINDOW,
)


def _safe_rate(numerator, denominator, multiplier=1.0, digits=2):
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce").replace(0, np.nan)
    return ((num / den) * multiplier).round(digits)


def _recent_match_counts(raw_df, match_threshold):
    seasons = pd.to_numeric(raw_df["season_start_year"], errors="coerce").dropna()
    latest_season = int(seasons.max()) if not seasons.empty else 0
    min_season = latest_season - SMAT_RECENT_SEASON_WINDOW + 1

    recent_df = raw_df[pd.to_numeric(raw_df["season_start_year"], errors="coerce") >= min_season].copy()
    counts = (
        recent_df.groupby("player")
        .agg(
            recent_matches=("match_file", "nunique"),
            latest_active_season=("season_start_year", "max"),
        )
        .reset_index()
    )
    counts["is_recent_active"] = counts["recent_matches"] >= match_threshold
    return counts, recent_df, latest_season, min_season


def build_active_candidate_pools(smat_batting, smat_bowling, smat_allrounder, smat_batting_raw, smat_bowling_raw):
    batting_recent, recent_batting_raw, latest_batting_season, min_batting_season = _recent_match_counts(
        smat_batting_raw, SMAT_MIN_RECENT_BATTING_MATCHES
    )
    bowling_recent, recent_bowling_raw, latest_bowling_season, min_bowling_season = _recent_match_counts(
        smat_bowling_raw, SMAT_MIN_RECENT_BOWLING_MATCHES
    )

    recent_batting_form = (
        recent_batting_raw.groupby("player")
        .agg(
            recent_runs=("runs", "sum"),
            recent_balls_faced=("balls_faced", "sum"),
            recent_avg_runs=("runs", "mean"),
            recent_pp_strike_rate=("pp_runs", lambda s: 0),
            recent_mid_strike_rate=("mid_runs", lambda s: 0),
            recent_death_strike_rate=("death_runs", lambda s: 0),
        )
        .reset_index()
    )
    batting_phase = recent_batting_raw.groupby("player").agg(
        recent_pp_runs=("pp_runs", "sum"),
        recent_pp_balls=("pp_balls", "sum"),
        recent_mid_runs=("mid_runs", "sum"),
        recent_mid_balls=("mid_balls", "sum"),
        recent_death_runs=("death_runs", "sum"),
        recent_death_balls=("death_balls", "sum"),
    )
    recent_batting_form = recent_batting_form.drop(
        columns=["recent_pp_strike_rate", "recent_mid_strike_rate", "recent_death_strike_rate"]
    ).merge(batting_phase.reset_index(), on="player", how="left")
    recent_batting_form["recent_strike_rate"] = _safe_rate(
        recent_batting_form["recent_runs"], recent_batting_form["recent_balls_faced"], 100
    )
    recent_batting_form["recent_pp_strike_rate"] = _safe_rate(
        recent_batting_form["recent_pp_runs"], recent_batting_form["recent_pp_balls"], 100
    )
    recent_batting_form["recent_mid_strike_rate"] = _safe_rate(
        recent_batting_form["recent_mid_runs"], recent_batting_form["recent_mid_balls"], 100
    )
    recent_batting_form["recent_death_strike_rate"] = _safe_rate(
        recent_batting_form["recent_death_runs"], recent_batting_form["recent_death_balls"], 100
    )

    recent_bowling_form = (
        recent_bowling_raw.groupby("player")
        .agg(
            recent_wickets=("wickets", "sum"),
            recent_balls_bowled=("balls_bowled", "sum"),
            recent_runs_conceded=("runs_conceded", "sum"),
            recent_avg_wickets_per_match=("wickets", "mean"),
            recent_pp_economy=("pp_runs", lambda s: 0),
            recent_mid_economy=("mid_runs", lambda s: 0),
            recent_death_economy=("death_runs", lambda s: 0),
        )
        .reset_index()
    )
    bowling_phase = recent_bowling_raw.groupby("player").agg(
        recent_pp_runs=("pp_runs", "sum"),
        recent_pp_balls=("pp_balls", "sum"),
        recent_mid_runs=("mid_runs", "sum"),
        recent_mid_balls=("mid_balls", "sum"),
        recent_death_runs=("death_runs", "sum"),
        recent_death_balls=("death_balls", "sum"),
    )
    recent_bowling_form = recent_bowling_form.drop(
        columns=["recent_pp_economy", "recent_mid_economy", "recent_death_economy"]
    ).merge(bowling_phase.reset_index(), on="player", how="left")
    recent_bowling_form["recent_economy_rate"] = _safe_rate(
        recent_bowling_form["recent_runs_conceded"], recent_bowling_form["recent_balls_bowled"], 6
    )
    recent_bowling_form["recent_pp_economy"] = _safe_rate(
        recent_bowling_form["recent_pp_runs"], recent_bowling_form["recent_pp_balls"], 6
    )
    recent_bowling_form["recent_mid_economy"] = _safe_rate(
        recent_bowling_form["recent_mid_runs"], recent_bowling_form["recent_mid_balls"], 6
    )
    recent_bowling_form["recent_death_economy"] = _safe_rate(
        recent_bowling_form["recent_death_runs"], recent_bowling_form["recent_death_balls"], 6
    )

    smat_batting = smat_batting.merge(batting_recent, on="player", how="left")
    smat_batting = smat_batting.merge(recent_batting_form, on="player", how="left")
    smat_bowling = smat_bowling.merge(bowling_recent, on="player", how="left")
    smat_bowling = smat_bowling.merge(recent_bowling_form, on="player", how="left")

    allrounder_recent = batting_recent.merge(
        bowling_recent,
        on="player",
        how="outer",
        suffixes=("_bat", "_bowl"),
    )
    allrounder_recent["recent_batting_matches"] = allrounder_recent["recent_matches_bat"].fillna(0)
    allrounder_recent["recent_bowling_matches"] = allrounder_recent["recent_matches_bowl"].fillna(0)
    allrounder_recent["latest_active_season"] = allrounder_recent[
        ["latest_active_season_bat", "latest_active_season_bowl"]
    ].max(axis=1)
    allrounder_recent["is_recent_active"] = (
        (allrounder_recent["recent_batting_matches"] >= SMAT_MIN_RECENT_BATTING_MATCHES)
        | (allrounder_recent["recent_bowling_matches"] >= SMAT_MIN_RECENT_BOWLING_MATCHES)
    )

    smat_allrounder = smat_allrounder.merge(
        allrounder_recent[
            [
                "player",
                "recent_batting_matches",
                "recent_bowling_matches",
                "latest_active_season",
                "is_recent_active",
            ]
        ],
        on="player",
        how="left",
    )
    smat_allrounder = smat_allrounder.merge(
        recent_batting_form[
            [
                "player",
                "recent_runs",
                "recent_balls_faced",
                "recent_avg_runs",
                "recent_strike_rate",
                "recent_pp_strike_rate",
                "recent_mid_strike_rate",
                "recent_death_strike_rate",
            ]
        ],
        on="player",
        how="left",
    )
    smat_allrounder = smat_allrounder.merge(
        recent_bowling_form[
            [
                "player",
                "recent_wickets",
                "recent_balls_bowled",
                "recent_runs_conceded",
                "recent_avg_wickets_per_match",
                "recent_economy_rate",
                "recent_pp_economy",
                "recent_mid_economy",
                "recent_death_economy",
            ]
        ],
        on="player",
        how="left",
    )

    smat_batting["recent_matches"] = smat_batting["recent_matches"].fillna(0).astype(int)
    smat_bowling["recent_matches"] = smat_bowling["recent_matches"].fillna(0).astype(int)
    smat_allrounder["recent_batting_matches"] = smat_allrounder["recent_batting_matches"].fillna(0).astype(int)
    smat_allrounder["recent_bowling_matches"] = smat_allrounder["recent_bowling_matches"].fillna(0).astype(int)

    smat_batting["latest_active_season"] = smat_batting["latest_active_season"].fillna(0).astype(int)
    smat_bowling["latest_active_season"] = smat_bowling["latest_active_season"].fillna(0).astype(int)
    smat_allrounder["latest_active_season"] = smat_allrounder["latest_active_season"].fillna(0).astype(int)

    smat_batting["is_recent_active"] = smat_batting["is_recent_active"].fillna(False)
    smat_bowling["is_recent_active"] = smat_bowling["is_recent_active"].fillna(False)
    smat_allrounder["is_recent_active"] = smat_allrounder["is_recent_active"].fillna(False)

    smat_batting = smat_batting[smat_batting["is_recent_active"]].copy()
    smat_bowling = smat_bowling[smat_bowling["is_recent_active"]].copy()
    smat_allrounder = smat_allrounder[smat_allrounder["is_recent_active"]].copy()

    meta = {
        "latest_batting_season": latest_batting_season,
        "latest_bowling_season": latest_bowling_season,
        "min_batting_season": min_batting_season,
        "min_bowling_season": min_bowling_season,
        "recent_batting_players": smat_batting["player"].nunique(),
        "recent_bowling_players": smat_bowling["player"].nunique(),
        "recent_allrounder_players": smat_allrounder["player"].nunique(),
    }
    return smat_batting, smat_bowling, smat_allrounder, meta
