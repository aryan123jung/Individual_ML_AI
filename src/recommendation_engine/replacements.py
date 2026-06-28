import numpy as np
import pandas as pd


def _safe_ratio_gap(current, benchmark, reverse=False):
    if pd.isna(current) or pd.isna(benchmark) or benchmark == 0:
        return np.nan
    if reverse:
        return max((current - benchmark) / benchmark, 0)
    return max((benchmark - current) / benchmark, 0)


def _severity(score):
    if score >= 35:
        return "high"
    if score >= 20:
        return "medium"
    return "low"


def _action_level(score):
    if score >= 35:
        return "replace_now"
    if score >= 20:
        return "watchlist"
    if score >= 10:
        return "competition_needed"
    return "stable"


def _mean_score(values):
    cleaned = [value for value in values if not pd.isna(value)]
    if not cleaned:
        return np.nan
    return float(np.mean(cleaned))


def _batting_phase_column(role):
    if role in ["aggressive_opener", "anchor_opener"]:
        return "pp_strike_rate"
    if role in ["top_order_anchor", "top_order_aggressor", "middle_order"]:
        return "mid_strike_rate"
    if role in ["finisher", "lower_order_hitter"]:
        return "death_strike_rate"
    return "strike_rate"


def _bowling_phase_column(role):
    if role == "powerplay_bowler":
        return "pp_economy"
    if role == "middle_overs_bowler":
        return "mid_economy"
    if role == "death_bowler":
        return "death_economy"
    return "economy"


def _current_allrounder_frame(ipl_2026_batting, ipl_2026_bowling, ipl_allrounder):
    hist_roles = ipl_allrounder[
        [
            "player",
            "team",
            "allrounder_role",
            "batting_strength_score",
            "bowling_strength_score",
            "allrounder_index",
        ]
    ].drop_duplicates(subset=["player"])
    current = hist_roles.merge(
        ipl_2026_batting[
            [
                "player",
                "matches",
                "runs",
                "average",
                "strike_rate",
                "pp_strike_rate",
                "mid_strike_rate",
                "death_strike_rate",
            ]
        ],
        on="player",
        how="left",
    ).merge(
        ipl_2026_bowling[
            [
                "player",
                "wickets",
                "economy",
                "pp_economy",
                "mid_economy",
                "death_economy",
            ]
        ],
        on="player",
        how="left",
    )
    current["matches"] = current["matches"].fillna(0)
    current["current_batting_strength"] = (
        current["average"].fillna(0) * 0.35
        + current["strike_rate"].fillna(0) * 0.35
        + current["runs"].fillna(0).clip(upper=250) * 0.30 / 2.5
    ).round(2)
    current["current_bowling_strength"] = (
        (100 - current["economy"].fillna(100)).clip(lower=0) * 0.40
        + current["wickets"].fillna(0).clip(upper=20) * 3.5
        + (100 - current["pp_economy"].fillna(current["economy"].fillna(100))).clip(lower=0) * 0.10
    ).round(2)
    current["current_allrounder_index"] = (
        current["current_batting_strength"] * 0.5 + current["current_bowling_strength"] * 0.5
    ).round(2)
    return current


def build_replacement_watchlist(
    ipl_current_squad,
    ipl_2026_batting,
    ipl_2026_bowling,
    ipl_batting,
    ipl_bowling,
    ipl_allrounder,
):
    watch_rows = []
    primary_roles = (
        ipl_current_squad[["team", "player_name", "primary_role"]]
        .drop_duplicates(subset=["team", "player_name"])
        .copy()
    )

    current_batting = ipl_current_squad[["team", "player_name"]].merge(
        ipl_2026_batting,
        left_on="player_name",
        right_on="player",
        how="left",
        suffixes=("", "_2026"),
    ).merge(
        ipl_batting[
            ["player", "batting_role", "avg_runs", "strike_rate", "pp_strike_rate", "mid_strike_rate", "death_strike_rate"]
        ].drop_duplicates(subset=["player"]),
        left_on="player_name",
        right_on="player",
        how="left",
        suffixes=("", "_hist"),
    )
    current_batting = current_batting.merge(
        primary_roles,
        on=["team", "player_name"],
        how="left",
    )
    current_batting = current_batting.rename(
        columns={
            "strike_rate": "strike_rate_2026",
            "pp_strike_rate": "pp_strike_rate_2026",
            "mid_strike_rate": "mid_strike_rate_2026",
            "death_strike_rate": "death_strike_rate_2026",
            "team_2026": "current_team_2026",
        }
    )

    batting_benchmarks = (
        ipl_batting[ipl_batting["eligible_for_model"]]
        .groupby("batting_role", as_index=False)
        .agg(
            benchmark_avg_runs=("avg_runs", "median"),
            benchmark_strike_rate=("strike_rate", "median"),
            benchmark_pp_strike_rate=("pp_strike_rate", "median"),
            benchmark_mid_strike_rate=("mid_strike_rate", "median"),
            benchmark_death_strike_rate=("death_strike_rate", "median"),
        )
    )
    current_batting = current_batting.merge(batting_benchmarks, on="batting_role", how="left")

    for _, row in current_batting.iterrows():
        role = row.get("batting_role")
        matches = row.get("matches", 0)
        primary_role = str(row.get("primary_role", "")).strip().lower()
        batting_allowed_roles = {"batter", "wicket keeper", "wicketkeeper", "all rounder"}
        if (
            pd.isna(role)
            or matches < 5
            or role == "tailender"
            or primary_role not in batting_allowed_roles
        ):
            continue
        phase_col = _batting_phase_column(role)
        phase_benchmark_col = f"benchmark_{phase_col}"
        score = _mean_score(
            [
                _safe_ratio_gap(row.get("average"), row.get("benchmark_avg_runs")),
                _safe_ratio_gap(row.get("strike_rate_2026"), row.get("benchmark_strike_rate")),
                _safe_ratio_gap(row.get(f"{phase_col}_2026"), row.get(phase_benchmark_col)),
            ]
        )
        score = 0 if pd.isna(score) else round(score * 100, 2)
        if score < 10:
            continue
        watch_rows.append(
            {
                "team": row["team"],
                "player": row["player_name"],
                "replacement_type": "batting",
                "target_role": role,
                "matches_played": int(matches),
                "underperformance_score": score,
                "severity": _severity(score),
                "action_level": _action_level(score),
                "reason": f"Current batting output below IPL benchmark for {role}",
            }
        )

    current_bowling = ipl_current_squad[["team", "player_name"]].merge(
        ipl_2026_bowling,
        left_on="player_name",
        right_on="player",
        how="left",
        suffixes=("", "_2026"),
    ).merge(
        ipl_bowling[
            ["player", "bowling_role", "avg_wickets_per_match", "economy_rate", "pp_economy", "mid_economy", "death_economy"]
        ].drop_duplicates(subset=["player"]),
        left_on="player_name",
        right_on="player",
        how="left",
        suffixes=("", "_hist"),
    )
    current_bowling = current_bowling.rename(
        columns={
            "economy": "economy_2026",
            "pp_economy": "pp_economy_2026",
            "mid_economy": "mid_economy_2026",
            "death_economy": "death_economy_2026",
            "team_2026": "current_team_2026",
        }
    )

    bowling_benchmarks = (
        ipl_bowling[ipl_bowling["eligible_for_model"]]
        .groupby("bowling_role", as_index=False)
        .agg(
            benchmark_avg_wickets_per_match=("avg_wickets_per_match", "median"),
            benchmark_economy=("economy_rate", "median"),
            benchmark_pp_economy=("pp_economy", "median"),
            benchmark_mid_economy=("mid_economy", "median"),
            benchmark_death_economy=("death_economy", "median"),
        )
    )
    current_bowling = current_bowling.merge(bowling_benchmarks, on="bowling_role", how="left")
    current_bowling["current_avg_wickets_per_match"] = (
        current_bowling["wickets"] / current_bowling["matches"].replace(0, np.nan)
    )

    for _, row in current_bowling.iterrows():
        role = row.get("bowling_role")
        matches = row.get("matches", 0)
        if pd.isna(role) or matches < 5 or role == "part_time_bowler":
            continue
        phase_col = _bowling_phase_column(role)
        phase_benchmark_col = f"benchmark_{phase_col}"
        score = _mean_score(
            [
                _safe_ratio_gap(
                    row.get("current_avg_wickets_per_match"),
                    row.get("benchmark_avg_wickets_per_match"),
                ),
                _safe_ratio_gap(row.get("economy_2026"), row.get("benchmark_economy"), reverse=True),
                _safe_ratio_gap(row.get(f"{phase_col}_2026"), row.get(phase_benchmark_col), reverse=True),
            ]
        )
        score = 0 if pd.isna(score) else round(score * 100, 2)
        if score < 10:
            continue
        watch_rows.append(
            {
                "team": row["team"],
                "player": row["player_name"],
                "replacement_type": "bowling",
                "target_role": role,
                "matches_played": int(matches),
                "underperformance_score": score,
                "severity": _severity(score),
                "action_level": _action_level(score),
                "reason": f"Current bowling output below IPL benchmark for {role}",
            }
        )

    current_allrounder = _current_allrounder_frame(ipl_2026_batting, ipl_2026_bowling, ipl_allrounder)
    allrounder_benchmarks = (
        ipl_allrounder.groupby("allrounder_role", as_index=False)
        .agg(
            benchmark_allrounder_index=("allrounder_index", "median"),
            benchmark_batting_strength=("batting_strength_score", "median"),
            benchmark_bowling_strength=("bowling_strength_score", "median"),
        )
    )
    current_allrounder = current_allrounder.merge(allrounder_benchmarks, on="allrounder_role", how="left")

    for _, row in current_allrounder.iterrows():
        role = row.get("allrounder_role")
        matches = row.get("matches", 0)
        batting_volume = row.get("runs", 0) >= 40 or row.get("average", 0) >= 15
        bowling_volume = row.get("wickets", 0) >= 1 or pd.notna(row.get("economy"))
        if pd.isna(role) or matches < 5 or not batting_volume or not bowling_volume:
            continue
        score = _mean_score(
            [
                _safe_ratio_gap(
                    row.get("current_allrounder_index"),
                    row.get("benchmark_allrounder_index"),
                ),
                _safe_ratio_gap(
                    row.get("current_batting_strength"),
                    row.get("benchmark_batting_strength"),
                ),
                _safe_ratio_gap(
                    row.get("current_bowling_strength"),
                    row.get("benchmark_bowling_strength"),
                ),
            ]
        )
        score = 0 if pd.isna(score) else round(score * 100, 2)
        if score < 10:
            continue
        watch_rows.append(
            {
                "team": row["team"],
                "player": row["player"],
                "replacement_type": "allrounder",
                "target_role": role,
                "matches_played": int(matches),
                "underperformance_score": score,
                "severity": _severity(score),
                "action_level": _action_level(score),
                "reason": f"Current all-round impact below IPL benchmark for {role}",
            }
        )

    watchlist = pd.DataFrame(watch_rows)
    if watchlist.empty:
        return watchlist, watchlist

    watchlist = watchlist.sort_values(
        ["team", "underperformance_score"], ascending=[True, False]
    ).reset_index(drop=True)
    watchlist["replacement_priority_rank"] = watchlist.groupby("team").cumcount() + 1
    return watchlist


def build_replacement_recommendations(watchlist, recommendations, top_n=5):
    if watchlist.empty or recommendations.empty:
        return pd.DataFrame()

    rows = []
    for _, under in watchlist.iterrows():
        matches = recommendations[
            (recommendations["team"] == under["team"])
            & (recommendations["recommendation_type"] == under["replacement_type"])
            & (recommendations["target_role"] == under["target_role"])
        ].sort_values("final_recommendation_score", ascending=False).head(top_n)

        for rank, (_, rec) in enumerate(matches.iterrows(), start=1):
            rows.append(
                {
                    "team": under["team"],
                    "underperforming_player": under["player"],
                    "replacement_type": under["replacement_type"],
                    "target_role": under["target_role"],
                    "underperformance_score": under["underperformance_score"],
                    "severity": under["severity"],
                    "action_level": under["action_level"],
                    "replacement_rank": rank,
                    "recommended_player": rec["domestic_player"],
                    "recommended_team": rec["domestic_team"],
                    "replacement_score": rec["final_recommendation_score"],
                    "recent_form_score": rec.get("recent_form_score"),
                    "role_fit_score": rec.get("role_fit_score"),
                    "closest_ipl_benchmark": rec.get("closest_ipl_benchmark"),
                    "reason": under["reason"],
                }
            )
    return pd.DataFrame(rows)
