import pandas as pd

from recommendation_engine.config import PROCESSED_DIR


def load_optional_csv(filename):
    path = PROCESSED_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_features():
    return {
        "smat_batting": pd.read_csv(PROCESSED_DIR / "smat_batting_features_v2.csv"),
        "smat_bowling": pd.read_csv(PROCESSED_DIR / "smat_bowling_features_v2.csv"),
        "smat_allrounder": pd.read_csv(PROCESSED_DIR / "smat_allrounder_features_v2.csv"),
        "smat_batting_raw": pd.read_csv(PROCESSED_DIR / "smat_batting_raw.csv"),
        "smat_bowling_raw": pd.read_csv(PROCESSED_DIR / "smat_bowling_raw.csv"),
        "ipl_batting": pd.read_csv(PROCESSED_DIR / "ipl_batting_features_v2.csv"),
        "ipl_bowling": pd.read_csv(PROCESSED_DIR / "ipl_bowling_features_v2.csv"),
        "ipl_allrounder": pd.read_csv(PROCESSED_DIR / "ipl_allrounder_features_v2.csv"),
        "ipl_batting_raw": pd.read_csv(PROCESSED_DIR / "ipl_batting_raw.csv"),
        "ipl_bowling_raw": pd.read_csv(PROCESSED_DIR / "ipl_bowling_raw.csv"),
        "ipl_current_squad": load_optional_csv("ipl_2026_current_squad.csv"),
        "ipl_2026_batting": load_optional_csv("ipl_2026_batting_features.csv"),
        "ipl_2026_bowling": load_optional_csv("ipl_2026_bowling_features.csv"),
        "smat_batting_ml_scores": load_optional_csv("smat_batting_ml_scores.csv"),
        "smat_bowling_ml_scores": load_optional_csv("smat_bowling_ml_scores.csv"),
        "smat_allrounder_ml_scores": load_optional_csv("smat_allrounder_ml_scores.csv"),
        "smat_batting_clusters": load_optional_csv("smat_batting_clusters.csv"),
        "smat_bowling_clusters": load_optional_csv("smat_bowling_clusters.csv"),
        "smat_allrounder_clusters": load_optional_csv("smat_allrounder_clusters.csv"),
    }
