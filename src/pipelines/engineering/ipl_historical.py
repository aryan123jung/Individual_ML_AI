from __future__ import annotations

from pathlib import Path

import pandas as pd

from pipelines.engineering.feature_utils import (
    MIN_BAT_MATCHES,
    MIN_BOWL_MATCHES,
    OUTPUTS,
    add_model_eligibility,
    aggregate_batting_positions,
    build_allrounder_features,
    compute_batting_features,
    compute_bowling_features,
    merge_batting_positions,
)


BASE_DIR = Path(__file__).resolve().parents[3]
RAW_DIR = BASE_DIR / "data" / "ipl-2008-2026"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

BALL_BY_BALL_PATH = RAW_DIR / "ball_by_ball_data.csv"
MATCHES_PATH = RAW_DIR / "ipl_matches_data.csv"
TEAMS_PATH = RAW_DIR / "teams_data.csv"
ALIASES_PATH = RAW_DIR / "team_aliases.csv"

RAW_OUTPUTS = {
    "batting": PROCESSED_DIR / "ipl_batting_raw.csv",
    "bowling": PROCESSED_DIR / "ipl_bowling_raw.csv",
}

DISMISSAL_TYPES_NOT_CHARGED_TO_BOWLER = {
    "run out",
    "retired hurt",
    "obstructing the field",
}

CURRENT_TEAM_ALIASES = {
    "royal challengers bangalore": "Royal Challengers Bengaluru",
    "bangalore": "Royal Challengers Bengaluru",
    "bengaluru": "Royal Challengers Bengaluru",
    "kings xi punjab": "Punjab Kings",
    "delhi daredevils": "Delhi Capitals",
}


def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split()).strip()


def normalize_key(value) -> str:
    return clean_text(value).lower()


def phase_for_over(over_number: int) -> str:
    if over_number <= 5:
        return "powerplay"
    if over_number <= 14:
        return "middle"
    return "death"


def season_start_year(value) -> int:
    text = clean_text(value)
    if "/" in text:
        left = text.split("/", 1)[0]
        if left.isdigit():
            return int(left)
    if text.isdigit():
        return int(text)
    return -1


def build_team_lookup() -> dict[int, str]:
    teams = pd.read_csv(TEAMS_PATH)
    aliases = pd.read_csv(ALIASES_PATH)

    team_lookup = {int(row["team_id"]): clean_text(row["team_name"]) for _, row in teams.iterrows()}
    alias_lookup = {}
    for _, row in aliases.iterrows():
        alias = clean_text(row["alias_name"])
        canonical = clean_text(team_lookup.get(int(row["team_id"]), alias))
        if alias:
            alias_lookup[normalize_key(alias)] = canonical

    standardized = {}
    for team_id, team_name in team_lookup.items():
        canonical = alias_lookup.get(normalize_key(team_name), team_name)
        canonical = CURRENT_TEAM_ALIASES.get(normalize_key(canonical), canonical)
        standardized[team_id] = canonical
    return standardized


def load_historical_data():
    team_lookup = build_team_lookup()
    matches = pd.read_csv(MATCHES_PATH)
    balls = pd.read_csv(BALL_BY_BALL_PATH)

    balls = balls[~balls["is_super_over"].fillna(False)].copy()
    balls["batter"] = balls["batter"].map(clean_text)
    balls["non_striker"] = balls["non_striker"].map(clean_text)
    balls["bowler"] = balls["bowler"].map(clean_text)
    balls["player_out"] = balls["player_out"].map(clean_text)
    balls["team"] = balls["team_batting"].map(lambda x: team_lookup.get(int(x), str(x)))
    balls["opponent"] = balls["team_bowling"].map(lambda x: team_lookup.get(int(x), str(x)))
    balls["phase"] = balls["over_number"].map(phase_for_over)
    balls["is_legal_batting_ball"] = (~balls["is_wide_ball"].fillna(False)).astype(int)
    balls["is_legal_bowling_ball"] = (
        ~balls["is_wide_ball"].fillna(False) & ~balls["is_no_ball"].fillna(False)
    ).astype(int)
    balls["runs_conceded_by_bowler"] = (
        balls["batter_runs"].fillna(0).astype(int)
        + balls["wide_ball_runs"].fillna(0).astype(int)
        + balls["no_ball_runs"].fillna(0).astype(int)
    )
    balls["batting_dot_ball"] = (
        (balls["batter_runs"].fillna(0).astype(int) == 0) & (balls["is_legal_batting_ball"] == 1)
    ).astype(int)
    balls["bowling_dot_ball"] = (
        (balls["runs_conceded_by_bowler"].fillna(0).astype(int) == 0)
        & (balls["is_legal_bowling_ball"] == 1)
    ).astype(int)
    balls["wicket_kind"] = balls["wicket_kind"].map(clean_text)
    balls["is_bowler_wicket"] = (
        balls["player_out"].ne("")
        & ~balls["wicket_kind"].str.lower().isin(DISMISSAL_TYPES_NOT_CHARGED_TO_BOWLER)
    ).astype(int)
    balls["dismissed_batter"] = (
        balls["player_out"].map(normalize_key) == balls["batter"].map(normalize_key)
    ).astype(int)

    matches["winner"] = matches["match_winner"].map(
        lambda x: team_lookup.get(int(x), "No Result") if pd.notna(x) and x != 0 else "No Result"
    )
    matches["season"] = matches["season"].map(clean_text)
    matches["match_file"] = matches["match_id"].astype(str)
    matches = matches[
        ["match_id", "match_file", "match_date", "season", "venue", "winner"]
    ].rename(columns={"match_date": "date"})

    balls = balls.merge(matches, on="match_id", how="left")
    balls["season_start_year"] = balls["season"].map(season_start_year)
    return balls


def build_batting_raw(balls: pd.DataFrame) -> pd.DataFrame:
    phase_source = balls.copy()
    phase_source["pp_runs"] = phase_source["batter_runs"].where(
        phase_source["phase"] == "powerplay", 0
    )
    phase_source["pp_balls"] = phase_source["is_legal_batting_ball"].where(
        phase_source["phase"] == "powerplay", 0
    )
    phase_source["mid_runs"] = phase_source["batter_runs"].where(
        phase_source["phase"] == "middle", 0
    )
    phase_source["mid_balls"] = phase_source["is_legal_batting_ball"].where(
        phase_source["phase"] == "middle", 0
    )
    phase_source["death_runs"] = phase_source["batter_runs"].where(
        phase_source["phase"] == "death", 0
    )
    phase_source["death_balls"] = phase_source["is_legal_batting_ball"].where(
        phase_source["phase"] == "death", 0
    )

    batting = (
        phase_source.groupby(["match_id", "innings", "batter"], as_index=False)
        .agg(
            match_file=("match_file", "first"),
            date=("date", "first"),
            season=("season", "first"),
            venue=("venue", "first"),
            team=("team", "first"),
            opponent=("opponent", "first"),
            winner=("winner", "first"),
            runs=("batter_runs", "sum"),
            balls_faced=("is_legal_batting_ball", "sum"),
            fours=("batter_runs", lambda s: int((s == 4).sum())),
            sixes=("batter_runs", lambda s: int((s == 6).sum())),
            dismissed=("dismissed_batter", "max"),
            dot_balls=("batting_dot_ball", "sum"),
            pp_runs=("pp_runs", "sum"),
            pp_balls=("pp_balls", "sum"),
            mid_runs=("mid_runs", "sum"),
            mid_balls=("mid_balls", "sum"),
            death_runs=("death_runs", "sum"),
            death_balls=("death_balls", "sum"),
        )
        .rename(columns={"batter": "player"})
    )
    batting["season_start_year"] = batting["season"].map(season_start_year)
    return batting[
        [
            "player",
            "match_file",
            "date",
            "season",
            "season_start_year",
            "venue",
            "team",
            "opponent",
            "winner",
            "runs",
            "balls_faced",
            "fours",
            "sixes",
            "dismissed",
            "pp_runs",
            "pp_balls",
            "mid_runs",
            "mid_balls",
            "death_runs",
            "death_balls",
            "dot_balls",
        ]
    ]


def build_bowling_raw(balls: pd.DataFrame) -> pd.DataFrame:
    phase_source = balls.copy()
    phase_source["pp_runs"] = phase_source["runs_conceded_by_bowler"].where(
        phase_source["phase"] == "powerplay", 0
    )
    phase_source["pp_balls"] = phase_source["is_legal_bowling_ball"].where(
        phase_source["phase"] == "powerplay", 0
    )
    phase_source["mid_runs"] = phase_source["runs_conceded_by_bowler"].where(
        phase_source["phase"] == "middle", 0
    )
    phase_source["mid_balls"] = phase_source["is_legal_bowling_ball"].where(
        phase_source["phase"] == "middle", 0
    )
    phase_source["death_runs"] = phase_source["runs_conceded_by_bowler"].where(
        phase_source["phase"] == "death", 0
    )
    phase_source["death_balls"] = phase_source["is_legal_bowling_ball"].where(
        phase_source["phase"] == "death", 0
    )

    bowling = (
        phase_source.groupby(["match_id", "innings", "bowler"], as_index=False)
        .agg(
            match_file=("match_file", "first"),
            date=("date", "first"),
            season=("season", "first"),
            venue=("venue", "first"),
            team=("opponent", "first"),
            opponent=("team", "first"),
            winner=("winner", "first"),
            runs_conceded=("runs_conceded_by_bowler", "sum"),
            balls_bowled=("is_legal_bowling_ball", "sum"),
            wickets=("is_bowler_wicket", "sum"),
            wides=("wide_ball_runs", "sum"),
            noballs=("no_ball_runs", "sum"),
            dot_balls=("bowling_dot_ball", "sum"),
            pp_runs=("pp_runs", "sum"),
            pp_balls=("pp_balls", "sum"),
            mid_runs=("mid_runs", "sum"),
            mid_balls=("mid_balls", "sum"),
            death_runs=("death_runs", "sum"),
            death_balls=("death_balls", "sum"),
        )
        .rename(columns={"bowler": "player"})
    )
    bowling["season_start_year"] = bowling["season"].map(season_start_year)
    return bowling[
        [
            "player",
            "match_file",
            "date",
            "season",
            "season_start_year",
            "venue",
            "team",
            "opponent",
            "winner",
            "runs_conceded",
            "balls_bowled",
            "wickets",
            "wides",
            "noballs",
            "dot_balls",
            "pp_runs",
            "pp_balls",
            "mid_runs",
            "mid_balls",
            "death_runs",
            "death_balls",
        ]
    ]


def extract_batting_positions_from_csv(balls: pd.DataFrame) -> pd.DataFrame:
    ordered = balls.sort_values(["match_id", "innings", "over_number", "ball_number"])
    batter_entries = ordered[["match_id", "innings", "team", "batter", "over_number", "ball_number"]].rename(
        columns={"batter": "player"}
    )
    non_striker_entries = ordered[
        ["match_id", "innings", "team", "non_striker", "over_number", "ball_number"]
    ].rename(columns={"non_striker": "player"})
    player_entries = pd.concat([batter_entries, non_striker_entries], ignore_index=True)
    player_entries["player"] = player_entries["player"].map(clean_text)
    player_entries = player_entries[player_entries["player"] != ""]
    player_entries = player_entries.sort_values(
        ["match_id", "innings", "team", "over_number", "ball_number", "player"]
    )
    first_appearances = player_entries.drop_duplicates(
        subset=["match_id", "innings", "team", "player"], keep="first"
    ).copy()
    first_appearances["position"] = (
        first_appearances.groupby(["match_id", "innings", "team"]).cumcount() + 1
    )
    first_appearances["match_file"] = first_appearances["match_id"].astype(str)
    return first_appearances[["player", "match_file", "team", "position"]]


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    balls = load_historical_data()
    batting_raw = build_batting_raw(balls)
    bowling_raw = build_bowling_raw(balls)

    positions = extract_batting_positions_from_csv(balls)
    position_agg = aggregate_batting_positions(positions)
    position_agg = position_agg[position_agg["player"].isin(set(batting_raw["player"]))].copy()

    batting_features = compute_batting_features(batting_raw)
    batting_features = merge_batting_positions(batting_features, position_agg)
    batting_features = add_model_eligibility(
        batting_features, "total_matches", MIN_BAT_MATCHES
    )

    bowling_features = compute_bowling_features(bowling_raw)
    bowling_features = add_model_eligibility(
        bowling_features, "total_matches_bowled", MIN_BOWL_MATCHES
    )

    allrounder_features = build_allrounder_features(batting_features, bowling_features)

    batting_raw.to_csv(RAW_OUTPUTS["batting"], index=False)
    bowling_raw.to_csv(RAW_OUTPUTS["bowling"], index=False)
    batting_features.to_csv(OUTPUTS["batting"], index=False)
    bowling_features.to_csv(OUTPUTS["bowling"], index=False)
    allrounder_features.to_csv(OUTPUTS["allrounder"], index=False)
    position_agg.to_csv(OUTPUTS["positions"], index=False)

    print("Historical IPL CSV engineering complete")
    print(f"Batting raw       : {batting_raw.shape} -> {RAW_OUTPUTS['batting']}")
    print(f"Bowling raw       : {bowling_raw.shape} -> {RAW_OUTPUTS['bowling']}")
    print(f"Batting features  : {batting_features.shape} -> {OUTPUTS['batting']}")
    print(f"Bowling features  : {bowling_features.shape} -> {OUTPUTS['bowling']}")
    print(f"All-rounder table : {allrounder_features.shape} -> {OUTPUTS['allrounder']}")
    print(f"Position table    : {position_agg.shape} -> {OUTPUTS['positions']}")


if __name__ == "__main__":
    main()
