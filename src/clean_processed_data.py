from pathlib import Path

import pandas as pd


PROCESSED_DIR = Path("data/processed")

RAW_FILES = {
    "smat_batting_raw.csv": {
        "numeric": [
            "runs", "balls_faced", "fours", "sixes", "dismissed",
            "pp_runs", "pp_balls", "mid_runs", "mid_balls",
            "death_runs", "death_balls", "dot_balls",
        ],
        "text": ["player", "match_file", "date", "season", "venue", "team", "opponent", "winner"],
    },
    "smat_bowling_raw.csv": {
        "numeric": [
            "runs_conceded", "balls_bowled", "wickets", "wides", "noballs",
            "dot_balls", "pp_runs", "pp_balls", "mid_runs", "mid_balls",
            "death_runs", "death_balls",
        ],
        "text": ["player", "match_file", "date", "season", "venue", "team", "opponent", "winner"],
    },
    "ipl_batting_raw.csv": {
        "numeric": [
            "match_number", "runs", "balls_faced", "fours", "sixes", "dismissed",
            "pp_runs", "pp_balls", "mid_runs", "mid_balls",
            "death_runs", "death_balls", "dot_balls",
        ],
        "text": [
            "player", "match_file", "tournament", "date", "season",
            "venue", "city", "team", "opponent", "winner",
        ],
    },
    "ipl_bowling_raw.csv": {
        "numeric": [
            "match_number", "runs_conceded", "balls_bowled", "wickets", "wides",
            "noballs", "dot_balls", "pp_runs", "pp_balls",
            "mid_runs", "mid_balls", "death_runs", "death_balls",
        ],
        "text": [
            "player", "match_file", "tournament", "date", "season",
            "venue", "city", "team", "opponent", "winner",
        ],
    },
}


def normalize_text(value):
    if pd.isna(value):
        return pd.NA

    value = " ".join(str(value).strip().split())
    if not value or value.lower() in {"nan", "none", "null"}:
        return pd.NA
    return value


def normalize_season(value):
    if pd.isna(value):
        return pd.NA

    value = str(value).strip()
    if "/" in value:
        left, right = value.split("/", 1)
        if len(left) == 4 and len(right) == 2 and left.isdigit() and right.isdigit():
            return f"{left}/{right}"

    if value.isdigit() and len(value) == 4:
        return value

    return value


def season_start_year(value):
    if pd.isna(value):
        return pd.NA

    value = str(value)
    if "/" in value:
        left = value.split("/", 1)[0]
        return int(left) if left.isdigit() else pd.NA

    return int(value) if value.isdigit() else pd.NA


def infer_opponent_from_match(df):
    if "opponent" not in df.columns:
        df["opponent"] = pd.NA

    team_pairs = (
        df.groupby("match_file")["team"]
        .apply(lambda x: sorted(set(v for v in x.dropna() if v != "Unknown")))
        .to_dict()
    )

    inferred = []
    for _, row in df.iterrows():
        current = row.get("opponent")
        if pd.notna(current) and current != "Unknown":
            inferred.append(current)
            continue

        teams = team_pairs.get(row["match_file"], [])
        if len(teams) == 2 and row["team"] in teams:
            other = teams[0] if teams[1] == row["team"] else teams[1]
            inferred.append(other)
        else:
            inferred.append(pd.NA)

    df["opponent"] = inferred
    return df


def clean_raw_file(path, config):
    df = pd.read_csv(path)
    before_rows = len(df)

    for col in config["text"]:
        if col in df.columns:
            df[col] = df[col].apply(normalize_text)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    if "season" in df.columns:
        df["season"] = df["season"].apply(normalize_season)
        df["season_start_year"] = df["season"].apply(season_start_year)

    for col in config["numeric"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            df[col] = df[col].clip(lower=0)

    if "team" in df.columns:
        df["team"] = df["team"].fillna("Unknown")

    if "winner" in df.columns:
        df["winner"] = df["winner"].fillna("No Result")

    if "opponent" in config["text"]:
        df = infer_opponent_from_match(df)

    df = df.drop_duplicates().reset_index(drop=True)

    report = {
        "file": path.name,
        "rows_before": before_rows,
        "rows_after": len(df),
        "missing_player": int(df["player"].isna().sum()) if "player" in df.columns else 0,
        "unknown_team": int((df["team"] == "Unknown").sum()) if "team" in df.columns else 0,
        "missing_opponent": int(df["opponent"].isna().sum()) if "opponent" in df.columns else 0,
        "bad_dates": int(pd.to_datetime(df["date"], errors="coerce").isna().sum()) if "date" in df.columns else 0,
    }

    return df, report


def main():
    reports = []

    for filename, config in RAW_FILES.items():
        path = PROCESSED_DIR / filename
        if not path.exists():
            print(f"Skipping missing file: {filename}")
            continue

        cleaned_df, report = clean_raw_file(path, config)
        cleaned_df.to_csv(path, index=False)
        reports.append(report)

    report_df = pd.DataFrame(reports)
    report_path = PROCESSED_DIR / "cleaning_report.csv"
    report_df.to_csv(report_path, index=False)

    print("\nCleaning complete.")
    print(report_df.to_string(index=False))
    print(f"\nSaved report to {report_path}")


if __name__ == "__main__":
    main()
