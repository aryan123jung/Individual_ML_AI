import json
from pathlib import Path

import numpy as np
import pandas as pd


BATTING_RAW = "data/processed/ipl_batting_raw.csv"
BOWLING_RAW = "data/processed/ipl_bowling_raw.csv"
JSON_DIR = "data/raw/ipl_raw"
MIN_BAT_MATCHES = 5
MIN_BOWL_MATCHES = 5

OUTPUTS = {
    "batting": "data/processed/ipl_batting_features_v2.csv",
    "bowling": "data/processed/ipl_bowling_features_v2.csv",
    "allrounder": "data/processed/ipl_allrounder_features_v2.csv",
    "positions": "data/processed/ipl_batting_positions_v2.csv",
}


def safe_divide(numerator, denominator, multiply_by=1.0, round_digits=3):
    denominator = denominator.replace(0, np.nan)
    return ((numerator / denominator) * multiply_by).round(round_digits)


def assign_position_role(position):
    if pd.isna(position):
        return "unknown"
    if position <= 2:
        return "opener"
    if position <= 4:
        return "top_order"
    if position <= 6:
        return "middle_order"
    if position <= 8:
        return "lower_order"
    return "tailender"


def extract_batting_positions():
    records = []
    json_path = Path(JSON_DIR)

    for match_file in sorted(json_path.glob("*.json")):
        with open(match_file, "r") as f:
            match = json.load(f)

        for inning in match.get("innings", []):
            batting_team = " ".join(str(inning.get("team", "Unknown")).split())
            seen = []
            seen_set = set()

            for over_data in inning.get("overs", []):
                for delivery in over_data.get("deliveries", []):
                    for player_key in ("batter", "non_striker"):
                        player = delivery.get(player_key)
                        if player:
                            player = " ".join(str(player).split())
                        if player and player not in seen_set:
                            seen.append(player)
                            seen_set.add(player)

            for idx, player in enumerate(seen, start=1):
                records.append(
                    {
                        "player": player,
                        "match_file": match_file.name,
                        "team": batting_team,
                        "position": idx,
                    }
                )

    return pd.DataFrame(records)


def aggregate_batting_positions(position_df):
    if position_df.empty:
        return pd.DataFrame(
            columns=[
                "player",
                "avg_position",
                "median_position",
                "highest_position",
                "lowest_position",
                "typical_position",
                "position_role",
            ]
        )

    grouped = (
        position_df.groupby("player")["position"]
        .agg(
            avg_position="mean",
            median_position="median",
            highest_position="min",
            lowest_position="max",
        )
        .reset_index()
    )

    typical = (
        position_df.groupby("player")["position"]
        .agg(lambda x: x.mode().iat[0] if not x.mode().empty else x.iloc[0])
        .reset_index(name="typical_position")
    )

    grouped = grouped.merge(typical, on="player", how="left")
    grouped["typical_position"] = grouped["typical_position"].round().astype("Int64")
    grouped["position_role"] = grouped["typical_position"].apply(assign_position_role)

    return grouped


def compute_batting_features(df):
    df = df.copy()
    df["is_duck_dismissal"] = ((df["runs"] == 0) & (df["dismissed"] > 0)).astype(int)
    df["is_30_plus"] = (df["runs"] >= 30).astype(int)
    df["is_50_plus"] = (df["runs"] >= 50).astype(int)

    g = df.groupby("player")
    features = pd.DataFrame(index=g.size().index)

    features["total_runs"] = g["runs"].sum()
    features["total_matches"] = g["match_file"].nunique()
    features["total_balls_faced"] = g["balls_faced"].sum()
    features["total_fours"] = g["fours"].sum()
    features["total_sixes"] = g["sixes"].sum()
    features["total_dismissed"] = g["dismissed"].sum()

    features["avg_runs"] = g["runs"].mean().round(2)
    features["highest_score"] = g["runs"].max()
    features["lowest_score"] = g["runs"].min()

    features["not_out_rate"] = (
        1 - safe_divide(features["total_dismissed"], features["total_matches"], round_digits=3)
    ).round(3)

    features["duck_count"] = g["is_duck_dismissal"].sum()
    features["duck_rate"] = safe_divide(
        features["duck_count"], features["total_matches"], round_digits=3
    )

    features["consistency_score"] = g["runs"].std().fillna(0).round(2)
    features["innings_above_30"] = g["is_30_plus"].sum()
    features["innings_above_50"] = g["is_50_plus"].sum()
    features["innings_above_30_pct"] = safe_divide(
        features["innings_above_30"], features["total_matches"], round_digits=3
    )
    features["innings_above_50_pct"] = safe_divide(
        features["innings_above_50"], features["total_matches"], round_digits=3
    )

    features["strike_rate"] = safe_divide(
        features["total_runs"], features["total_balls_faced"], multiply_by=100, round_digits=2
    )
    features["boundary_rate"] = safe_divide(
        features["total_fours"] + features["total_sixes"],
        features["total_balls_faced"],
        round_digits=3,
    )
    features["six_rate"] = safe_divide(
        features["total_sixes"], features["total_balls_faced"], round_digits=3
    )
    features["four_rate"] = safe_divide(
        features["total_fours"], features["total_balls_faced"], round_digits=3
    )

    boundary_runs = (features["total_fours"] * 4) + (features["total_sixes"] * 6)
    features["boundary_runs_pct"] = safe_divide(
        boundary_runs, features["total_runs"], round_digits=3
    )
    features["dot_ball_rate"] = safe_divide(
        g["dot_balls"].sum(), features["total_balls_faced"], round_digits=3
    )

    pp_runs = g["pp_runs"].sum()
    mid_runs = g["mid_runs"].sum()
    death_runs = g["death_runs"].sum()
    pp_balls = g["pp_balls"].sum()
    mid_balls = g["mid_balls"].sum()
    death_balls = g["death_balls"].sum()

    features["pp_runs"] = pp_runs
    features["mid_runs"] = mid_runs
    features["death_runs"] = death_runs
    features["pp_balls_faced"] = pp_balls
    features["mid_balls_faced"] = mid_balls
    features["death_balls_faced"] = death_balls

    features["pp_runs_pct"] = safe_divide(pp_runs, features["total_runs"], round_digits=3)
    features["mid_runs_pct"] = safe_divide(mid_runs, features["total_runs"], round_digits=3)
    features["death_runs_pct"] = safe_divide(death_runs, features["total_runs"], round_digits=3)

    features["pp_strike_rate"] = safe_divide(
        pp_runs, pp_balls, multiply_by=100, round_digits=2
    )
    features["mid_strike_rate"] = safe_divide(
        mid_runs, mid_balls, multiply_by=100, round_digits=2
    )
    features["death_strike_rate"] = safe_divide(
        death_runs, death_balls, multiply_by=100, round_digits=2
    )

    phase_sr = pd.DataFrame(
        {
            "powerplay": features["pp_strike_rate"].fillna(0),
            "middle": features["mid_strike_rate"].fillna(0),
            "death": features["death_strike_rate"].fillna(0),
        }
    )
    features["dominant_phase"] = phase_sr.idxmax(axis=1)
    features.loc[features["total_runs"] == 0, "dominant_phase"] = "unknown"

    win_mask = df["team"] == df["winner"]
    loss_mask = df["team"] != df["winner"]
    features["avg_runs_in_wins"] = df.loc[win_mask].groupby("player")["runs"].mean().round(2)
    features["avg_runs_in_losses"] = df.loc[loss_mask].groupby("player")["runs"].mean().round(2)

    features["balls_faced_per_match"] = safe_divide(
        features["total_balls_faced"], features["total_matches"], round_digits=2
    )
    features["sample_reliability_score"] = (
        np.minimum(features["total_matches"] / 8, 1.0) * 0.5
        + np.minimum(features["total_balls_faced"] / 120, 1.0) * 0.5
    ).round(3)

    features["team"] = g["team"].agg(lambda x: x.value_counts().index[0])
    features["season_start_year"] = g["season_start_year"].max()

    return features.reset_index()


def compute_bowling_features(df):
    df = df.copy()
    df["is_two_plus_haul"] = (df["wickets"] >= 2).astype(int)

    g = df.groupby("player")
    features = pd.DataFrame(index=g.size().index)

    features["total_wickets"] = g["wickets"].sum()
    features["total_matches_bowled"] = g["match_file"].nunique()
    features["total_balls_bowled"] = g["balls_bowled"].sum()
    features["total_runs_conceded"] = g["runs_conceded"].sum()
    features["total_wides"] = g["wides"].sum()
    features["total_noballs"] = g["noballs"].sum()

    features["economy_rate"] = safe_divide(
        features["total_runs_conceded"],
        features["total_balls_bowled"] / 6,
        round_digits=2,
    )
    features["bowling_average"] = safe_divide(
        features["total_runs_conceded"], features["total_wickets"], round_digits=2
    )
    features["bowling_strike_rate"] = safe_divide(
        features["total_balls_bowled"], features["total_wickets"], round_digits=2
    )
    features["avg_wickets_per_match"] = safe_divide(
        features["total_wickets"], features["total_matches_bowled"], round_digits=3
    )

    features["bowling_consistency"] = g["wickets"].std().fillna(0).round(2)
    features["two_plus_hauls"] = g["is_two_plus_haul"].sum()
    features["two_plus_haul_rate"] = safe_divide(
        features["two_plus_hauls"], features["total_matches_bowled"], round_digits=3
    )
    features["bowling_dot_rate"] = safe_divide(
        g["dot_balls"].sum(), features["total_balls_bowled"], round_digits=3
    )
    features["wide_rate"] = safe_divide(
        features["total_wides"], features["total_matches_bowled"], round_digits=3
    )
    features["noball_rate"] = safe_divide(
        features["total_noballs"], features["total_matches_bowled"], round_digits=3
    )

    pp_runs = g["pp_runs"].sum()
    mid_runs = g["mid_runs"].sum()
    death_runs = g["death_runs"].sum()
    pp_balls = g["pp_balls"].sum()
    mid_balls = g["mid_balls"].sum()
    death_balls = g["death_balls"].sum()

    features["pp_runs_conceded"] = pp_runs
    features["mid_runs_conceded"] = mid_runs
    features["death_runs_conceded"] = death_runs
    features["pp_balls_bowled"] = pp_balls
    features["mid_balls_bowled"] = mid_balls
    features["death_balls_bowled"] = death_balls

    features["pp_economy"] = safe_divide(
        pp_runs, pp_balls, multiply_by=6, round_digits=2
    )
    features["mid_economy"] = safe_divide(
        mid_runs, mid_balls, multiply_by=6, round_digits=2
    )
    features["death_economy"] = safe_divide(
        death_runs, death_balls, multiply_by=6, round_digits=2
    )

    features["pp_workload_pct"] = safe_divide(
        pp_balls, features["total_balls_bowled"], round_digits=3
    )
    features["mid_workload_pct"] = safe_divide(
        mid_balls, features["total_balls_bowled"], round_digits=3
    )
    features["death_workload_pct"] = safe_divide(
        death_balls, features["total_balls_bowled"], round_digits=3
    )

    phase_eco = pd.DataFrame(
        {
            "powerplay": features["pp_economy"].fillna(999),
            "middle": features["mid_economy"].fillna(999),
            "death": features["death_economy"].fillna(999),
        }
    )
    features["dominant_phase"] = phase_eco.idxmin(axis=1)
    features.loc[features["total_balls_bowled"] == 0, "dominant_phase"] = "unknown"

    def assign_bowling_role(row):
        if row["total_balls_bowled"] < 60:
            return "part_time_bowler"
        if row["death_balls_bowled"] >= 24 and row["death_workload_pct"] >= 0.30:
            return "death_bowler"
        if row["pp_balls_bowled"] >= 24 and row["pp_workload_pct"] >= 0.30:
            return "powerplay_bowler"
        if row["mid_balls_bowled"] >= 36 and row["mid_workload_pct"] >= 0.40:
            return "middle_overs_bowler"
        return "all_phases_bowler"

    features["bowling_role"] = features.apply(assign_bowling_role, axis=1)

    win_mask = df["team"] == df["winner"]
    loss_mask = df["team"] != df["winner"]
    features["wickets_in_wins"] = df.loc[win_mask].groupby("player")["wickets"].mean().round(2)
    features["wickets_in_losses"] = df.loc[loss_mask].groupby("player")["wickets"].mean().round(2)

    features["balls_bowled_per_match"] = safe_divide(
        features["total_balls_bowled"], features["total_matches_bowled"], round_digits=2
    )
    features["sample_reliability_score"] = (
        np.minimum(features["total_matches_bowled"] / 8, 1.0) * 0.5
        + np.minimum(features["total_balls_bowled"] / 120, 1.0) * 0.5
    ).round(3)

    features["team"] = g["team"].agg(lambda x: x.value_counts().index[0])
    features["season_start_year"] = g["season_start_year"].max()

    return features.reset_index()


def assign_batting_role(row):
    pos = row.get("typical_position", pd.NA)
    pp_sr = row.get("pp_strike_rate", 0)
    death_sr = row.get("death_strike_rate", 0)
    strike_rate = row.get("strike_rate", 0)
    avg_runs = row.get("avg_runs", 0)

    if pd.isna(pos):
        return "unknown"
    if pos <= 2 and row.get("pp_balls_faced", 0) >= 20 and pp_sr >= 120:
        return "aggressive_opener"
    if pos <= 2:
        return "anchor_opener"
    if pos <= 4 and avg_runs >= 25:
        return "top_order_anchor"
    if pos <= 4 and strike_rate >= 140:
        return "top_order_aggressor"
    if pos <= 6 and row.get("death_balls_faced", 0) >= 20 and death_sr >= 150:
        return "finisher"
    if pos <= 6:
        return "middle_order"
    if row.get("total_balls_faced", 0) >= 40 and strike_rate >= 150:
        return "lower_order_hitter"
    return "tailender"


def merge_batting_positions(batting_features, position_agg):
    merged = batting_features.merge(
        position_agg[
            [
                "player",
                "avg_position",
                "median_position",
                "highest_position",
                "lowest_position",
                "typical_position",
                "position_role",
            ]
        ],
        on="player",
        how="left",
    )
    merged["batting_role"] = merged.apply(assign_batting_role, axis=1)
    return merged


def build_allrounder_features(batting_features, bowling_features):
    merged = batting_features.merge(
        bowling_features,
        on="player",
        how="outer",
        suffixes=("_bat", "_bowl"),
    )

    merged["team"] = merged["team_bat"].combine_first(merged["team_bowl"])
    merged["season_start_year"] = merged["season_start_year_bat"].combine_first(
        merged["season_start_year_bowl"]
    )

    merged["batting_strength_score"] = (
        merged["avg_runs"].fillna(0) * 0.35
        + merged["strike_rate"].fillna(0) * 0.35
        + (merged["innings_above_30_pct"].fillna(0) * 100) * 0.30
    ).round(2)

    merged["bowling_strength_score"] = (
        (100 - merged["economy_rate"].fillna(100)) * 0.40
        + (merged["avg_wickets_per_match"].fillna(0) * 100) * 0.35
        + (merged["bowling_dot_rate"].fillna(0) * 100) * 0.25
    ).round(2)

    merged["allrounder_index"] = (
        merged["batting_strength_score"].fillna(0) * 0.5
        + merged["bowling_strength_score"].fillna(0) * 0.5
    ).round(2)

    merged["is_batting_allrounder"] = (
        (merged["avg_runs"].fillna(0) >= 15)
        & (merged["strike_rate"].fillna(0) >= 120)
        & (merged["total_matches"].fillna(0) >= 5)
        & (merged["total_balls_bowled"].fillna(0) >= 72)
        & (merged["total_wickets"].fillna(0) >= 5)
    )
    merged["is_bowling_allrounder"] = (
        (merged["avg_wickets_per_match"].fillna(0) >= 0.50)
        & (merged["total_matches_bowled"].fillna(0) >= 5)
        & (merged["total_balls_bowled"].fillna(0) >= 72)
        & (merged["total_runs"].fillna(0) >= 150)
        & (merged["total_balls_faced"].fillna(0) >= 100)
    )

    def classify_role(row):
        if row["is_batting_allrounder"]:
            return "batting_allrounder"
        if row["is_bowling_allrounder"]:
            return "bowling_allrounder"
        batting_volume = row.get("total_runs", 0) >= 100 or row.get("total_balls_faced", 0) >= 80
        bowling_volume = row.get("total_balls_bowled", 0) >= 72 or row.get("total_wickets", 0) >= 5
        if (
            batting_volume
            and row.get("total_balls_bowled", 0) >= 36
            and row.get("total_wickets", 0) >= 2
        ):
            return "utility_player"
        if bowling_volume and (
            row.get("total_runs", 0) < 100
            or row.get("bowling_strength_score", 0) >= row.get("batting_strength_score", 0)
        ):
            return "pure_bowler"
        if batting_volume:
            return "pure_batter"
        if row.get("total_wickets", 0) > 0:
            return "pure_bowler"
        if row.get("total_runs", 0) > 0:
            return "pure_batter"
        return "unknown"

    merged["allrounder_role"] = merged.apply(classify_role, axis=1)

    keep_cols = [
        "player",
        "team",
        "season_start_year",
        "batting_strength_score",
        "bowling_strength_score",
        "allrounder_index",
        "is_batting_allrounder",
        "is_bowling_allrounder",
        "allrounder_role",
        "total_runs",
        "total_matches",
        "total_balls_faced",
        "avg_runs",
        "strike_rate",
        "batting_role",
        "total_wickets",
        "total_matches_bowled",
        "total_balls_bowled",
        "avg_wickets_per_match",
        "economy_rate",
        "bowling_role",
    ]
    existing_cols = [col for col in keep_cols if col in merged.columns]
    return merged[existing_cols].copy()


def add_model_eligibility(features_df, min_matches_col, min_matches):
    features_df = features_df.copy()
    features_df["eligible_for_model"] = features_df[min_matches_col] >= min_matches
    return features_df


def main():
    batting_df = pd.read_csv(BATTING_RAW)
    bowling_df = pd.read_csv(BOWLING_RAW)

    position_df = extract_batting_positions()
    position_agg = aggregate_batting_positions(position_df)
    batting_players = set(batting_df["player"])
    position_agg = position_agg[position_agg["player"].isin(batting_players)].copy()

    batting_features = compute_batting_features(batting_df)
    batting_features = merge_batting_positions(batting_features, position_agg)
    batting_features = add_model_eligibility(
        batting_features, "total_matches", MIN_BAT_MATCHES
    )

    bowling_features = compute_bowling_features(bowling_df)
    bowling_features = add_model_eligibility(
        bowling_features, "total_matches_bowled", MIN_BOWL_MATCHES
    )

    allrounder_features = build_allrounder_features(batting_features, bowling_features)

    Path(OUTPUTS["batting"]).parent.mkdir(parents=True, exist_ok=True)
    batting_features.to_csv(OUTPUTS["batting"], index=False)
    bowling_features.to_csv(OUTPUTS["bowling"], index=False)
    allrounder_features.to_csv(OUTPUTS["allrounder"], index=False)
    position_agg.to_csv(OUTPUTS["positions"], index=False)

    print("\nIPL complete")
    print(f"Batting features  : {batting_features.shape} -> {OUTPUTS['batting']}")
    print(f"Bowling features  : {bowling_features.shape} -> {OUTPUTS['bowling']}")
    print(f"All-rounder table : {allrounder_features.shape} -> {OUTPUTS['allrounder']}")
    print(f"Position table    : {position_agg.shape} -> {OUTPUTS['positions']}")


if __name__ == "__main__":
    main()
