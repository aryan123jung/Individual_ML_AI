import pandas as pd


def percentile_score(series, higher_is_better=True):
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() <= 1:
        return pd.Series(50, index=series.index, dtype=float)
    ranks = values.rank(pct=True) * 100
    if not higher_is_better:
        ranks = 100 - ranks
    return ranks.fillna(0).clip(0, 100)


def batting_quality_score(df):
    role = df["batting_role"].fillna("")
    base = (
        percentile_score(df["avg_runs"]) * 0.20
        + percentile_score(df["strike_rate"]) * 0.20
        + percentile_score(df["boundary_rate"]) * 0.15
        + percentile_score(df["innings_above_30_pct"]) * 0.15
        + percentile_score(df["sample_reliability_score"]) * 0.10
    )
    phase = pd.Series(0.0, index=df.index)
    phase += role.isin(["aggressive_opener", "anchor_opener"]).astype(float) * (
        percentile_score(df["pp_strike_rate"]) * 0.20
    )
    phase += role.isin(["middle_order", "top_order_anchor", "top_order_aggressor"]).astype(float) * (
        percentile_score(df["mid_strike_rate"]) * 0.20
    )
    phase += role.isin(["finisher", "lower_order_hitter"]).astype(float) * (
        percentile_score(df["death_strike_rate"]) * 0.20
    )
    phase += role.eq("tailender").astype(float) * (percentile_score(df["strike_rate"]) * 0.20)
    return (base + phase).round(2).clip(0, 100)


def bowling_quality_score(df):
    role = df["bowling_role"].fillna("")
    base = (
        percentile_score(df["economy_rate"], higher_is_better=False) * 0.20
        + percentile_score(df["avg_wickets_per_match"]) * 0.20
        + percentile_score(df["bowling_dot_rate"]) * 0.20
        + percentile_score(df["two_plus_haul_rate"]) * 0.15
        + percentile_score(df["sample_reliability_score"]) * 0.10
    )
    phase = pd.Series(0.0, index=df.index)
    phase += role.eq("powerplay_bowler").astype(float) * (
        percentile_score(df["pp_economy"], higher_is_better=False) * 0.15
    )
    phase += role.eq("middle_overs_bowler").astype(float) * (
        percentile_score(df["mid_economy"], higher_is_better=False) * 0.15
    )
    phase += role.eq("death_bowler").astype(float) * (
        percentile_score(df["death_economy"], higher_is_better=False) * 0.15
    )
    phase += role.isin(["all_phases_bowler", "part_time_bowler"]).astype(float) * (
        percentile_score(df["economy_rate"], higher_is_better=False) * 0.15
    )
    return (base + phase).round(2).clip(0, 100)


def allrounder_quality_score(df):
    return (
        percentile_score(df["allrounder_index"]) * 0.45
        + percentile_score(df["batting_strength_score"]) * 0.25
        + percentile_score(df["bowling_strength_score"]) * 0.25
        + percentile_score(df["total_balls_bowled"]) * 0.05
    ).round(2).clip(0, 100)


def batting_recent_form_score(df):
    return (
        percentile_score(df.get("recent_avg_runs", 0)) * 0.35
        + percentile_score(df.get("recent_strike_rate", 0)) * 0.35
        + percentile_score(df.get("recent_matches", 0)) * 0.15
        + percentile_score(df.get("recent_runs", 0)) * 0.15
    ).round(2).clip(0, 100)


def bowling_recent_form_score(df):
    return (
        percentile_score(df.get("recent_avg_wickets_per_match", 0)) * 0.35
        + percentile_score(df.get("recent_economy_rate", 0), higher_is_better=False) * 0.35
        + percentile_score(df.get("recent_matches", 0)) * 0.15
        + percentile_score(df.get("recent_wickets", 0)) * 0.15
    ).round(2).clip(0, 100)


def allrounder_recent_form_score(df):
    return (
        percentile_score(df.get("recent_avg_runs", 0)) * 0.20
        + percentile_score(df.get("recent_strike_rate", 0)) * 0.20
        + percentile_score(df.get("recent_avg_wickets_per_match", 0)) * 0.25
        + percentile_score(df.get("recent_economy_rate", 0), higher_is_better=False) * 0.25
        + percentile_score(
            df.get("recent_batting_matches", df.get("recent_matches", 0))
            + df.get("recent_bowling_matches", 0)
        ) * 0.10
    ).round(2).clip(0, 100)


def batting_role_fit_score(df, target_role):
    target = str(target_role or "")
    if target in ["aggressive_opener", "anchor_opener"]:
        score = percentile_score(df.get("recent_pp_strike_rate", df["pp_strike_rate"]))
    elif target in ["top_order_anchor", "top_order_aggressor", "middle_order"]:
        score = percentile_score(df.get("recent_mid_strike_rate", df["mid_strike_rate"]))
    elif target in ["finisher", "lower_order_hitter"]:
        score = percentile_score(df.get("recent_death_strike_rate", df["death_strike_rate"]))
    else:
        score = percentile_score(df["strike_rate"])
    return score.round(2).clip(0, 100)


def bowling_role_fit_score(df, target_role):
    target = str(target_role or "")
    if target == "powerplay_bowler":
        score = percentile_score(df.get("recent_pp_economy", df["pp_economy"]), higher_is_better=False)
    elif target == "middle_overs_bowler":
        score = percentile_score(df.get("recent_mid_economy", df["mid_economy"]), higher_is_better=False)
    elif target == "death_bowler":
        score = percentile_score(df.get("recent_death_economy", df["death_economy"]), higher_is_better=False)
    else:
        score = percentile_score(df.get("recent_economy_rate", df["economy_rate"]), higher_is_better=False)
    return score.round(2).clip(0, 100)


def allrounder_role_fit_score(df, target_role):
    target = str(target_role or "")
    if target == "batting_allrounder":
        score = (
            percentile_score(df.get("recent_avg_runs", df["avg_runs"])) * 0.5
            + percentile_score(df.get("recent_strike_rate", df["strike_rate"])) * 0.5
        )
    elif target == "bowling_allrounder":
        score = (
            percentile_score(df.get("recent_avg_wickets_per_match", df["avg_wickets_per_match"])) * 0.5
            + percentile_score(df.get("recent_economy_rate", df["economy_rate"]), higher_is_better=False) * 0.5
        )
    else:
        score = (
            percentile_score(df.get("recent_avg_runs", df["avg_runs"])) * 0.25
            + percentile_score(df.get("recent_strike_rate", df["strike_rate"])) * 0.25
            + percentile_score(df.get("recent_avg_wickets_per_match", df["avg_wickets_per_match"])) * 0.25
            + percentile_score(df.get("recent_economy_rate", df["economy_rate"]), higher_is_better=False) * 0.25
        )
    return score.round(2).clip(0, 100)
