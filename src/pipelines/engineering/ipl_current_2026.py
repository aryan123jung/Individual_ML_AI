from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[3]
CURRENT_DATA_DIR = BASE_DIR / "data" / "ipl2026_kaggle"
HISTORICAL_DATA_DIR = BASE_DIR / "data" / "ipl-2008-2026"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

OUTPUTS = {
    "batting_raw": PROCESSED_DIR / "ipl_2026_batting_raw.csv",
    "bowling_raw": PROCESSED_DIR / "ipl_2026_bowling_raw.csv",
    "batting_features": PROCESSED_DIR / "ipl_2026_batting_features.csv",
    "bowling_features": PROCESSED_DIR / "ipl_2026_bowling_features.csv",
    "current_squad": PROCESSED_DIR / "ipl_2026_current_squad.csv",
    "player_master": PROCESSED_DIR / "ipl_2026_player_master.csv",
    "match_summary": PROCESSED_DIR / "ipl_2026_match_summary.csv",
}

CURRENT_TEAM_ALIASES = {
    "mi": "Mumbai Indians",
    "mumbai indians": "Mumbai Indians",
    "rcb": "Royal Challengers Bengaluru",
    "royal challengers bangalore": "Royal Challengers Bengaluru",
    "royal challengers bengaluru": "Royal Challengers Bengaluru",
    "srh": "Sunrisers Hyderabad",
    "sunrisers hyderabad": "Sunrisers Hyderabad",
    "csk": "Chennai Super Kings",
    "chennai super kings": "Chennai Super Kings",
    "kkr": "Kolkata Knight Riders",
    "kolkata knight riders": "Kolkata Knight Riders",
    "kolkata night riders": "Kolkata Knight Riders",
    "rr": "Rajasthan Royals",
    "rajasthan royals": "Rajasthan Royals",
    "dc": "Delhi Capitals",
    "delhi capitals": "Delhi Capitals",
    "delhi daredevils": "Delhi Capitals",
    "pbks": "Punjab Kings",
    "punjab kings": "Punjab Kings",
    "kings xi punjab": "Punjab Kings",
    "gt": "Gujarat Titans",
    "gujarat titans": "Gujarat Titans",
    "gujrat titans": "Gujarat Titans",
    "lsg": "Lucknow Super Giants",
    "lucknow super giants": "Lucknow Super Giants",
}

DISMISSAL_TYPES_NOT_CHARGED_TO_BOWLER = {
    "run out",
    "retired hurt",
    "obstructing the field",
}


def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split()).strip()


def normalize_key(value) -> str:
    return clean_text(value).lower()


def phase_for_over(over_value: float | int) -> str:
    over_number = int(float(over_value))
    if over_number <= 5:
        return "powerplay"
    if over_number <= 14:
        return "middle"
    return "death"


def cricket_overs_from_balls(balls: int) -> float:
    if not balls:
        return 0.0
    overs, remainder = divmod(int(balls), 6)
    return float(f"{overs}.{remainder}")


def load_team_map() -> dict[str, str]:
    aliases_path = HISTORICAL_DATA_DIR / "team_aliases.csv"
    teams_path = HISTORICAL_DATA_DIR / "teams_data.csv"

    team_map: dict[str, str] = {}
    if aliases_path.exists() and teams_path.exists():
        aliases = pd.read_csv(aliases_path)
        teams = pd.read_csv(teams_path)
        merged = aliases.merge(teams, on="team_id", how="left")
        for _, row in merged.iterrows():
            alias = clean_text(row.get("alias_name"))
            team_name = clean_text(row.get("team_name"))
            if alias and team_name:
                team_map[normalize_key(alias)] = team_name

    team_map.update(CURRENT_TEAM_ALIASES)
    return team_map


def standardize_team(value, team_map: dict[str, str]) -> str:
    cleaned = clean_text(value)
    if not cleaned:
        return cleaned
    return team_map.get(normalize_key(cleaned), cleaned)


def build_player_master() -> pd.DataFrame:
    path = HISTORICAL_DATA_DIR / "players-data-updated.csv"
    if not path.exists():
        return pd.DataFrame(columns=["lookup_name", "player_name", "player_full_name", "bat_style", "bowl_style", "field_pos"])

    master = pd.read_csv(path)
    for column in ["player_name", "player_full_name", "bat_style", "bowl_style", "field_pos"]:
        if column not in master.columns:
            master[column] = ""
    master["lookup_name"] = master["player_name"].map(normalize_key)
    master["lookup_full_name"] = master["player_full_name"].map(normalize_key)
    return master


def build_squads(team_map: dict[str, str], player_master: pd.DataFrame) -> pd.DataFrame:
    squads_path = CURRENT_DATA_DIR / "squads.csv"
    squads = pd.read_csv(squads_path)
    squads["team"] = squads["team_name"].map(lambda x: standardize_team(x, team_map))
    squads["player_name"] = squads["player"].map(clean_text)
    squads["designation"] = squads["designation"].fillna("")
    squads["role"] = squads["role"].fillna("")

    if not player_master.empty:
        enriched = squads.copy()
        enriched["lookup_name"] = enriched["player_name"].map(normalize_key)

        direct = (
            player_master[["lookup_full_name", "bat_style", "bowl_style", "field_pos"]]
            .drop_duplicates(subset=["lookup_full_name"])
            .rename(columns={"lookup_full_name": "lookup_name"})
        )
        short = (
            player_master[["lookup_name", "bat_style", "bowl_style", "field_pos"]]
            .drop_duplicates(subset=["lookup_name"])
            .rename(columns={"lookup_name": "lookup_name_short"})
        )

        enriched = enriched.merge(direct, on="lookup_name", how="left")
        fallback = enriched[["lookup_name"]].merge(
            short.rename(columns={"lookup_name_short": "lookup_name"}),
            on="lookup_name",
            how="left",
        )
        for column in ["bat_style", "bowl_style", "field_pos"]:
            enriched[column] = enriched[column].combine_first(fallback[column])

        squads["bat_style"] = enriched["bat_style"].fillna("")
        squads["bowl_style"] = enriched["bowl_style"].fillna("")
        squads["field_pos"] = enriched["field_pos"].fillna("")
    else:
        squads["bat_style"] = ""
        squads["bowl_style"] = ""
        squads["field_pos"] = ""

    return squads[
        [
            "team_no",
            "team",
            "team_name",
            "player_name",
            "nationality",
            "role",
            "designation",
            "bat_style",
            "bowl_style",
            "field_pos",
        ]
    ].drop_duplicates()


def prepare_deliveries(team_map: dict[str, str]) -> pd.DataFrame:
    deliveries = pd.read_csv(CURRENT_DATA_DIR / "deliveries.csv")
    deliveries["player"] = deliveries["striker"].map(clean_text)
    deliveries["bowler_name"] = deliveries["bowler"].map(clean_text)
    deliveries["player_dismissed"] = deliveries["player_dismissed"].map(clean_text)
    deliveries["fielder"] = deliveries["fielder"].map(clean_text)
    deliveries["batting_team"] = deliveries["batting_team"].map(lambda x: standardize_team(x, team_map))
    deliveries["bowling_team"] = deliveries["bowling_team"].map(lambda x: standardize_team(x, team_map))
    deliveries["over_number"] = deliveries["over"].apply(lambda x: int(float(x)))
    deliveries["phase"] = deliveries["over_number"].map(phase_for_over)
    deliveries["is_legal_batting_ball"] = (deliveries["wide"].fillna(0).astype(int) == 0).astype(int)
    deliveries["is_legal_bowling_ball"] = (
        (deliveries["wide"].fillna(0).astype(int) == 0)
        & (deliveries["noballs"].fillna(0).astype(int) == 0)
    ).astype(int)
    deliveries["runs_conceded_by_bowler"] = (
        deliveries["runs_of_bat"].fillna(0).astype(int)
        + deliveries["wide"].fillna(0).astype(int)
        + deliveries["noballs"].fillna(0).astype(int)
    )
    deliveries["is_bowler_wicket"] = deliveries["wicket_type"].fillna("").map(
        lambda x: clean_text(x).lower() not in DISMISSAL_TYPES_NOT_CHARGED_TO_BOWLER and clean_text(x) != ""
    ).astype(int)
    deliveries["is_dot_ball"] = (
        (deliveries["runs_conceded_by_bowler"] == 0)
        & (deliveries["is_legal_bowling_ball"] == 1)
    ).astype(int)
    deliveries["dismissal_for_striker"] = (
        deliveries["player_dismissed"].map(normalize_key) == deliveries["player"].map(normalize_key)
    ).astype(int)
    return deliveries


def build_match_innings_tables(deliveries: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    batting = (
        deliveries.groupby(["match_no", "innings", "player"], as_index=False)
        .agg(
            date=("date", "first"),
            stage=("stage", "first"),
            venue=("venue", "first"),
            team=("batting_team", "first"),
            opponent=("bowling_team", "first"),
            runs=("runs_of_bat", "sum"),
            balls_faced=("is_legal_batting_ball", "sum"),
            fours=("runs_of_bat", lambda s: int((s == 4).sum())),
            sixes=("runs_of_bat", lambda s: int((s == 6).sum())),
            dismissed=("dismissal_for_striker", "max"),
            dot_balls=("is_dot_ball", "sum"),
            pp_runs=("runs_of_bat", lambda s: 0),
        )
        .copy()
    )

    phase_rows = []
    for (match_no, innings, player), frame in deliveries.groupby(["match_no", "innings", "player"]):
        phase_rows.append(
            {
                "match_no": match_no,
                "innings": innings,
                "player": player,
                "pp_runs": int(frame.loc[frame["phase"] == "powerplay", "runs_of_bat"].sum()),
                "pp_balls": int(frame.loc[frame["phase"] == "powerplay", "is_legal_batting_ball"].sum()),
                "mid_runs": int(frame.loc[frame["phase"] == "middle", "runs_of_bat"].sum()),
                "mid_balls": int(frame.loc[frame["phase"] == "middle", "is_legal_batting_ball"].sum()),
                "death_runs": int(frame.loc[frame["phase"] == "death", "runs_of_bat"].sum()),
                "death_balls": int(frame.loc[frame["phase"] == "death", "is_legal_batting_ball"].sum()),
            }
        )
    phase_df = pd.DataFrame(phase_rows)
    batting = batting.drop(columns=["pp_runs"]).merge(phase_df, on=["match_no", "innings", "player"], how="left")

    bowling = (
        deliveries.groupby(["match_no", "innings", "bowler_name"], as_index=False)
        .agg(
            date=("date", "first"),
            stage=("stage", "first"),
            venue=("venue", "first"),
            team=("bowling_team", "first"),
            opponent=("batting_team", "first"),
            runs_conceded=("runs_conceded_by_bowler", "sum"),
            balls_bowled=("is_legal_bowling_ball", "sum"),
            wickets=("is_bowler_wicket", "sum"),
            wides=("wide", "sum"),
            noballs=("noballs", "sum"),
            dot_balls=("is_dot_ball", "sum"),
        )
        .rename(columns={"bowler_name": "player"})
    )

    bowling_phase_rows = []
    for (match_no, innings, bowler), frame in deliveries.groupby(["match_no", "innings", "bowler_name"]):
        bowling_phase_rows.append(
            {
                "match_no": match_no,
                "innings": innings,
                "player": bowler,
                "pp_runs": int(frame.loc[frame["phase"] == "powerplay", "runs_conceded_by_bowler"].sum()),
                "pp_balls": int(frame.loc[frame["phase"] == "powerplay", "is_legal_bowling_ball"].sum()),
                "mid_runs": int(frame.loc[frame["phase"] == "middle", "runs_conceded_by_bowler"].sum()),
                "mid_balls": int(frame.loc[frame["phase"] == "middle", "is_legal_bowling_ball"].sum()),
                "death_runs": int(frame.loc[frame["phase"] == "death", "runs_conceded_by_bowler"].sum()),
                "death_balls": int(frame.loc[frame["phase"] == "death", "is_legal_bowling_ball"].sum()),
            }
        )
    bowling_phase_df = pd.DataFrame(bowling_phase_rows)
    bowling = bowling.merge(bowling_phase_df, on=["match_no", "innings", "player"], how="left")

    maiden_rows = []
    for (match_no, innings, bowler, over_number), frame in deliveries.groupby(
        ["match_no", "innings", "bowler_name", "over_number"]
    ):
        maiden_rows.append(
            {
                "match_no": match_no,
                "innings": innings,
                "player": bowler,
                "maiden_over": int(
                    frame.loc[frame["is_legal_bowling_ball"] == 1, "runs_conceded_by_bowler"].sum() == 0
                    and frame["is_legal_bowling_ball"].sum() > 0
                ),
            }
        )
    maiden_df = pd.DataFrame(maiden_rows)
    maiden_by_match = (
        maiden_df.groupby(["match_no", "innings", "player"], as_index=False)["maiden_over"].sum()
        if not maiden_df.empty
        else pd.DataFrame(columns=["match_no", "innings", "player", "maiden_over"])
    )
    bowling = bowling.merge(maiden_by_match, on=["match_no", "innings", "player"], how="left")
    bowling["maiden_over"] = bowling["maiden_over"].fillna(0)

    batting["dismissed"] = batting["dismissed"].fillna(0).astype(int)
    batting["not_out"] = (1 - batting["dismissed"]).astype(int)
    batting["high_score"] = batting["runs"].astype(int)
    batting["ducks"] = ((batting["runs"] == 0) & (batting["dismissed"] == 1)).astype(int)

    bowling["wickets"] = bowling["wickets"].fillna(0).astype(int)

    return batting, bowling


def aggregate_player_batting(batting_match: pd.DataFrame) -> pd.DataFrame:
    grouped = batting_match.groupby("player", as_index=False)
    batting = grouped.agg(
        team=("team", lambda s: s.value_counts().index[0]),
        matches=("match_no", "nunique"),
        innings=("innings", "count"),
        runs=("runs", "sum"),
        balls_faced=("balls_faced", "sum"),
        fours=("fours", "sum"),
        sixes=("sixes", "sum"),
        dismissals=("dismissed", "sum"),
        not_outs=("not_out", "sum"),
        high_score=("high_score", "max"),
        ducks=("ducks", "sum"),
        pp_runs=("pp_runs", "sum"),
        pp_balls=("pp_balls", "sum"),
        mid_runs=("mid_runs", "sum"),
        mid_balls=("mid_balls", "sum"),
        death_runs=("death_runs", "sum"),
        death_balls=("death_balls", "sum"),
        dot_balls=("dot_balls", "sum"),
    )
    batting["strike_rate"] = np.where(
        batting["balls_faced"] > 0, (batting["runs"] / batting["balls_faced"]) * 100, np.nan
    ).round(2)
    batting["average"] = np.where(
        batting["dismissals"] > 0, batting["runs"] / batting["dismissals"], np.nan
    ).round(2)
    batting["pp_strike_rate"] = np.where(
        batting["pp_balls"] > 0, (batting["pp_runs"] / batting["pp_balls"]) * 100, np.nan
    ).round(2)
    batting["mid_strike_rate"] = np.where(
        batting["mid_balls"] > 0, (batting["mid_runs"] / batting["mid_balls"]) * 100, np.nan
    ).round(2)
    batting["death_strike_rate"] = np.where(
        batting["death_balls"] > 0, (batting["death_runs"] / batting["death_balls"]) * 100, np.nan
    ).round(2)
    batting["batting_phase"] = batting[["pp_strike_rate", "mid_strike_rate", "death_strike_rate"]].fillna(-1).idxmax(axis=1)
    batting["sample_reliability_score"] = (
        np.minimum(batting["matches"] / 8, 1.0) * 0.5
        + np.minimum(batting["balls_faced"] / 120, 1.0) * 0.5
    ).round(3)
    batting["average"] = batting["average"].replace({np.inf: np.nan, -np.inf: np.nan})
    batting["balls_faced_per_match"] = np.where(
        batting["matches"] > 0, batting["balls_faced"] / batting["matches"], np.nan
    ).round(2)
    batting["boundary_rate"] = np.where(
        batting["balls_faced"] > 0, (batting["fours"] + batting["sixes"]) / batting["balls_faced"], np.nan
    ).round(3)
    batting["six_rate"] = np.where(
        batting["balls_faced"] > 0, batting["sixes"] / batting["balls_faced"], np.nan
    ).round(3)
    batting["four_rate"] = np.where(
        batting["balls_faced"] > 0, batting["fours"] / batting["balls_faced"], np.nan
    ).round(3)
    batting["dot_ball_rate"] = np.where(
        batting["balls_faced"] > 0, batting["dot_balls"] / batting["balls_faced"], np.nan
    ).round(3)
    batting["season_start_year"] = 2026
    return batting


def aggregate_player_bowling(bowling_match: pd.DataFrame) -> pd.DataFrame:
    grouped = bowling_match.groupby("player", as_index=False)
    bowling = grouped.agg(
        team=("team", lambda s: s.value_counts().index[0]),
        matches=("match_no", "nunique"),
        innings=("innings", "count"),
        wickets=("wickets", "sum"),
        balls_bowled=("balls_bowled", "sum"),
        runs_conceded=("runs_conceded", "sum"),
        wides=("wides", "sum"),
        noballs=("noballs", "sum"),
        dot_balls=("dot_balls", "sum"),
        maiden_overs=("maiden_over", "sum"),
        pp_runs=("pp_runs", "sum"),
        pp_balls=("pp_balls", "sum"),
        mid_runs=("mid_runs", "sum"),
        mid_balls=("mid_balls", "sum"),
        death_runs=("death_runs", "sum"),
        death_balls=("death_balls", "sum"),
    )
    bowling["overs"] = bowling["balls_bowled"].map(cricket_overs_from_balls)
    bowling["economy"] = np.where(
        bowling["balls_bowled"] > 0, (bowling["runs_conceded"] / bowling["balls_bowled"]) * 6, np.nan
    ).round(2)
    bowling["avg"] = np.where(
        bowling["wickets"] > 0, bowling["runs_conceded"] / bowling["wickets"], np.nan
    ).round(2)
    bowling["strike_rate"] = np.where(
        bowling["wickets"] > 0, bowling["balls_bowled"] / bowling["wickets"], np.nan
    ).round(2)
    bowling["pp_economy"] = np.where(
        bowling["pp_balls"] > 0, (bowling["pp_runs"] / bowling["pp_balls"]) * 6, np.nan
    ).round(2)
    bowling["mid_economy"] = np.where(
        bowling["mid_balls"] > 0, (bowling["mid_runs"] / bowling["mid_balls"]) * 6, np.nan
    ).round(2)
    bowling["death_economy"] = np.where(
        bowling["death_balls"] > 0, (bowling["death_runs"] / bowling["death_balls"]) * 6, np.nan
    ).round(2)
    bowling["pp_workload_pct"] = np.where(
        bowling["balls_bowled"] > 0, bowling["pp_balls"] / bowling["balls_bowled"], np.nan
    ).round(3)
    bowling["mid_workload_pct"] = np.where(
        bowling["balls_bowled"] > 0, bowling["mid_balls"] / bowling["balls_bowled"], np.nan
    ).round(3)
    bowling["death_workload_pct"] = np.where(
        bowling["balls_bowled"] > 0, bowling["death_balls"] / bowling["balls_bowled"], np.nan
    ).round(3)
    bowling["sample_reliability_score"] = (
        np.minimum(bowling["matches"] / 8, 1.0) * 0.5
        + np.minimum(bowling["balls_bowled"] / 120, 1.0) * 0.5
    ).round(3)
    bowling["bowling_phase"] = bowling[["pp_economy", "mid_economy", "death_economy"]].fillna(999).idxmin(axis=1)
    bowling["two_plus_hauls"] = bowling["wickets"].ge(2).astype(int)
    bowling["four_plus_hauls"] = bowling["wickets"].ge(4).astype(int)
    bowling["five_wicket_hauls"] = bowling["wickets"].ge(5).astype(int)
    bowling["season_start_year"] = 2026
    return bowling


def build_match_summary(deliveries: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    summary = (
        deliveries.groupby("match_no", as_index=False)
        .agg(
            batting_teams=("batting_team", lambda s: ", ".join(sorted(set(s)))),
            bowling_teams=("bowling_team", lambda s: ", ".join(sorted(set(s)))),
            total_runs=("runs_conceded_by_bowler", "sum"),
            wickets=("is_bowler_wicket", "sum"),
            balls=("is_legal_bowling_ball", "sum"),
        )
        .rename(columns={"match_no": "match_id"})
    )
    if "match_id" in matches.columns:
        summary = matches.merge(summary, on="match_id", how="left")
    return summary


def build_current_squad_table(squads: pd.DataFrame, batting: pd.DataFrame, bowling: pd.DataFrame) -> pd.DataFrame:
    batting = batting.rename(columns={"player": "player_name"})
    bowling = bowling.rename(columns={"player": "player_name"})
    merged = squads.merge(batting, on="player_name", how="left", suffixes=("", "_bat"))
    merged = merged.merge(bowling, on="player_name", how="left", suffixes=("", "_bowl"))

    merged["competition"] = "IPL"
    merged["season"] = "2026"
    merged["status"] = "Verified from Kaggle 2026 delivery data"
    merged["primary_role"] = merged["role"].replace({"All Rounder": "All Rounder", "Wicket Keeper": "Wicket Keeper"})
    merged["batting_role"] = ""
    merged["bowling_role"] = ""
    merged["matches_played"] = (
        merged[["matches", "matches_bowl"]]
        .fillna(0)
        .max(axis=1)
        .round(0)
        .astype(int)
    )
    merged["batting_innings"] = merged.get("innings", 0).fillna(0).round(0).astype(int)
    merged["bowling_innings"] = merged.get("innings_bowl", 0).fillna(0).round(0).astype(int)
    merged["runs"] = merged.get("runs", 0).fillna(0).round(0).astype(int)
    merged["balls_faced"] = merged.get("balls_faced", 0).fillna(0).round(0).astype(int)
    merged["fours"] = merged.get("fours", 0).fillna(0).round(0).astype(int)
    merged["sixes"] = merged.get("sixes", 0).fillna(0).round(0).astype(int)
    merged["dismissals"] = merged.get("dismissals", 0).fillna(0).round(0).astype(int)
    merged["runs_conceded"] = merged.get("runs_conceded", 0).fillna(0).round(0).astype(int)
    merged["wickets"] = merged.get("wickets", 0).fillna(0).round(0).astype(int)
    merged["dot_balls"] = merged.get("dot_balls", 0).fillna(0).round(0).astype(int)
    merged["balls_bowled"] = merged.get("balls_bowled", 0).fillna(0).round(0).astype(int)
    merged["batting_strike_rate"] = merged.get("strike_rate", np.nan)
    merged["batting_average"] = merged.get("average", np.nan)
    merged["bowling_economy"] = merged.get("economy", np.nan)
    merged["bowling_strike_rate"] = merged.get("strike_rate_bowl", np.nan)
    merged["notes"] = ""
    merged["source_note"] = "Derived from Kaggle IPL 2026 deliveries and squads data"

    keep_cols = [
        "competition",
        "season",
        "team",
        "player_name",
        "status",
        "primary_role",
        "batting_role",
        "bowling_role",
        "matches_played",
        "batting_innings",
        "runs",
        "balls_faced",
        "fours",
        "sixes",
        "dismissals",
        "batting_strike_rate",
        "batting_average",
        "bowling_innings",
        "balls_bowled",
        "runs_conceded",
        "wickets",
        "bowling_economy",
        "bowling_strike_rate",
        "dot_balls",
        "notes",
        "source_note",
    ]
    existing_cols = [column for column in keep_cols if column in merged.columns]
    return merged[existing_cols].sort_values(["team", "player_name"], kind="stable").reset_index(drop=True)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    team_map = load_team_map()
    player_master = build_player_master()
    squads = build_squads(team_map, player_master)
    deliveries = prepare_deliveries(team_map)
    matches = pd.read_csv(CURRENT_DATA_DIR / "matches.csv")

    batting_match, bowling_match = build_match_innings_tables(deliveries)
    batting_features = aggregate_player_batting(batting_match)
    bowling_features = aggregate_player_bowling(bowling_match)
    current_squad = build_current_squad_table(squads, batting_features, bowling_features)
    match_summary = build_match_summary(deliveries, matches)

    batting_match.to_csv(OUTPUTS["batting_raw"], index=False)
    bowling_match.to_csv(OUTPUTS["bowling_raw"], index=False)
    batting_features.to_csv(OUTPUTS["batting_features"], index=False)
    bowling_features.to_csv(OUTPUTS["bowling_features"], index=False)
    current_squad.to_csv(OUTPUTS["current_squad"], index=False)
    player_master.to_csv(OUTPUTS["player_master"], index=False)
    match_summary.to_csv(OUTPUTS["match_summary"], index=False)

    print("Kaggle IPL 2026 engineering complete")
    print(f"Batting raw      : {batting_match.shape} -> {OUTPUTS['batting_raw']}")
    print(f"Bowling raw      : {bowling_match.shape} -> {OUTPUTS['bowling_raw']}")
    print(f"Batting features  : {batting_features.shape} -> {OUTPUTS['batting_features']}")
    print(f"Bowling features  : {bowling_features.shape} -> {OUTPUTS['bowling_features']}")
    print(f"Current squad     : {current_squad.shape} -> {OUTPUTS['current_squad']}")
    print(f"Player master     : {player_master.shape} -> {OUTPUTS['player_master']}")
    print(f"Match summary     : {match_summary.shape} -> {OUTPUTS['match_summary']}")


if __name__ == "__main__":
    main()
