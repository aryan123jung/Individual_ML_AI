from pathlib import Path

import pandas as pd


PROCESSED_DIR = Path("data/processed")
REPORT_DIR = Path("reports/data_engineering")

DATASETS = {
    "smat_batting": PROCESSED_DIR / "smat_batting_features_v2.csv",
    "smat_bowling": PROCESSED_DIR / "smat_bowling_features_v2.csv",
    "smat_allrounder": PROCESSED_DIR / "smat_allrounder_features_v2.csv",
    "smat_positions": PROCESSED_DIR / "smat_batting_positions_v2.csv",
    "ipl_batting": PROCESSED_DIR / "ipl_batting_features_v2.csv",
    "ipl_bowling": PROCESSED_DIR / "ipl_bowling_features_v2.csv",
    "ipl_allrounder": PROCESSED_DIR / "ipl_allrounder_features_v2.csv",
    "ipl_positions": PROCESSED_DIR / "ipl_batting_positions_v2.csv",
    "ipl_2026_batting": PROCESSED_DIR / "ipl_2026_batting_features.csv",
    "ipl_2026_bowling": PROCESSED_DIR / "ipl_2026_bowling_features.csv",
    "ipl_2026_current_squad": PROCESSED_DIR / "ipl_2026_current_squad.csv",
    "ipl_2026_match_summary": PROCESSED_DIR / "ipl_2026_match_summary.csv",
}

TOP_PLAYER_CONFIG = {
    "smat_batting": [
        ("total_runs", False),
        ("strike_rate", False),
        ("avg_runs", False),
        ("sample_reliability_score", False),
    ],
    "ipl_batting": [
        ("total_runs", False),
        ("strike_rate", False),
        ("avg_runs", False),
        ("sample_reliability_score", False),
    ],
    "smat_bowling": [
        ("total_wickets", False),
        ("economy_rate", True),
        ("bowling_dot_rate", False),
        ("sample_reliability_score", False),
    ],
    "ipl_bowling": [
        ("total_wickets", False),
        ("economy_rate", True),
        ("bowling_dot_rate", False),
        ("sample_reliability_score", False),
    ],
    "smat_allrounder": [
        ("allrounder_index", False),
        ("batting_strength_score", False),
        ("bowling_strength_score", False),
    ],
    "ipl_allrounder": [
        ("allrounder_index", False),
        ("batting_strength_score", False),
        ("bowling_strength_score", False),
    ],
}


def load_dataset(name, path):
    if not path.exists():
        return None
    return pd.read_csv(path)


def build_dataset_summary(frames):
    rows = []
    for name, df in frames.items():
        rows.append(
            {
                "dataset": name,
                "rows": len(df),
                "columns": len(df.columns),
                "unique_players": df["player"].nunique() if "player" in df.columns else None,
                "duplicate_rows": int(df.duplicated().sum()),
                "total_missing_values": int(df.isna().sum().sum()),
                "eligible_players": int(df["eligible_for_model"].sum())
                if "eligible_for_model" in df.columns
                else None,
            }
        )
    return pd.DataFrame(rows)


def build_feature_schema(frames):
    rows = []
    for name, df in frames.items():
        for column in df.columns:
            rows.append(
                {
                    "dataset": name,
                    "feature": column,
                    "dtype": str(df[column].dtype),
                    "missing_count": int(df[column].isna().sum()),
                    "missing_pct": round(df[column].isna().mean() * 100, 2),
                    "unique_values": int(df[column].nunique(dropna=True)),
                }
            )
    return pd.DataFrame(rows)


def build_numeric_stats(frames):
    rows = []
    for name, df in frames.items():
        numeric_cols = df.select_dtypes(include="number").columns
        for column in numeric_cols:
            series = df[column]
            rows.append(
                {
                    "dataset": name,
                    "feature": column,
                    "count": int(series.count()),
                    "missing_count": int(series.isna().sum()),
                    "min": round(series.min(), 3) if series.count() else None,
                    "max": round(series.max(), 3) if series.count() else None,
                    "mean": round(series.mean(), 3) if series.count() else None,
                    "median": round(series.median(), 3) if series.count() else None,
                    "std": round(series.std(), 3) if series.count() else None,
                }
            )
    return pd.DataFrame(rows)


def build_role_distributions(frames):
    role_columns = [
        "batting_role",
        "bowling_role",
        "allrounder_role",
        "position_role",
        "dominant_phase",
        "eligible_for_model",
    ]
    rows = []
    for name, df in frames.items():
        for column in role_columns:
            if column not in df.columns:
                continue
            counts = df[column].value_counts(dropna=False)
            for value, count in counts.items():
                rows.append(
                    {
                        "dataset": name,
                        "category": column,
                        "value": value,
                        "count": int(count),
                        "pct": round((count / len(df)) * 100, 2) if len(df) else 0,
                    }
                )
    return pd.DataFrame(rows)


def build_top_players(frames):
    rows = []
    for name, ranking_rules in TOP_PLAYER_CONFIG.items():
        df = frames[name].copy()
        if "eligible_for_model" in df.columns:
            df = df[df["eligible_for_model"]]

        for metric, ascending in ranking_rules:
            if metric not in df.columns:
                continue
            ranked = df.sort_values(metric, ascending=ascending).head(10)
            for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
                rows.append(
                    {
                        "dataset": name,
                        "metric": metric,
                        "rank": rank,
                        "player": row.get("player"),
                        "team": row.get("team"),
                        "value": row.get(metric),
                        "batting_role": row.get("batting_role"),
                        "bowling_role": row.get("bowling_role"),
                        "allrounder_role": row.get("allrounder_role"),
                    }
                )
    return pd.DataFrame(rows)


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    frames = {name: df for name, path in DATASETS.items() if (df := load_dataset(name, path)) is not None}

    outputs = {
        "dataset_summary.csv": build_dataset_summary(frames),
        "feature_schema.csv": build_feature_schema(frames),
        "numeric_feature_stats.csv": build_numeric_stats(frames),
        "role_distributions.csv": build_role_distributions(frames),
        "top_players_by_metric.csv": build_top_players(frames),
    }

    for filename, df in outputs.items():
        df.to_csv(REPORT_DIR / filename, index=False)

    print("Data engineering summary generated:")
    for filename in outputs:
        print(f"- {REPORT_DIR / filename}")


if __name__ == "__main__":
    main()
