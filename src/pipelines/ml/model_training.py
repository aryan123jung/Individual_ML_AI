from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from recommendation_engine.config import (
    ALLROUNDER_FEATURES,
    BATTING_FEATURES,
    BOWLING_FEATURES,
    PROCESSED_DIR,
)


BASE_DIR = Path(__file__).resolve().parents[3]
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports" / "ml"


def _load_frame(filename: str) -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / filename)


def _domain_config():
    return {
        "batting": {
            "historical_file": "ipl_batting_features_v2.csv",
            "candidate_file": "smat_batting_features_v2.csv",
            "features": BATTING_FEATURES,
            "label_builder": _build_batting_label,
        },
        "bowling": {
            "historical_file": "ipl_bowling_features_v2.csv",
            "candidate_file": "smat_bowling_features_v2.csv",
            "features": BOWLING_FEATURES,
            "label_builder": _build_bowling_label,
        },
        "allrounder": {
            "historical_file": "ipl_allrounder_features_v2.csv",
            "candidate_file": "smat_allrounder_features_v2.csv",
            "features": ALLROUNDER_FEATURES,
            "label_builder": _build_allrounder_label,
        },
    }


def _label_from_role_medians(df: pd.DataFrame, role_column: str, metric_rules: list[tuple[str, bool]], min_role_rows: int = 6):
    labeled = df.copy()
    labeled["success_label"] = 0

    for role, group in labeled.groupby(role_column):
        if pd.isna(role) or len(group) < min_role_rows:
            continue

        thresholds = {}
        for column, higher_is_better in metric_rules:
            thresholds[column] = group[column].median()

        def _row_success(row):
            score = 0
            for column, higher_is_better in metric_rules:
                value = row.get(column)
                threshold = thresholds[column]
                if pd.isna(value) or pd.isna(threshold):
                    continue
                if higher_is_better and value >= threshold:
                    score += 1
                if not higher_is_better and value <= threshold:
                    score += 1
            return int(score >= 2)

        labeled.loc[group.index, "success_label"] = group.apply(_row_success, axis=1)

    return labeled


def _build_batting_label(df: pd.DataFrame) -> pd.DataFrame:
    eligible = df[df["eligible_for_model"]].copy()
    eligible = _label_from_role_medians(
        eligible,
        "batting_role",
        [
            ("avg_runs", True),
            ("strike_rate", True),
            ("innings_above_30_pct", True),
        ],
    )
    return eligible


def _build_bowling_label(df: pd.DataFrame) -> pd.DataFrame:
    eligible = df[df["eligible_for_model"]].copy()
    eligible = _label_from_role_medians(
        eligible,
        "bowling_role",
        [
            ("avg_wickets_per_match", True),
            ("economy_rate", False),
            ("bowling_dot_rate", True),
        ],
    )
    return eligible


def _build_allrounder_label(df: pd.DataFrame) -> pd.DataFrame:
    eligible = df[df["allrounder_role"].notna()].copy()
    eligible = _label_from_role_medians(
        eligible,
        "allrounder_role",
        [
            ("allrounder_index", True),
            ("batting_strength_score", True),
            ("bowling_strength_score", True),
        ],
    )
    return eligible


def _candidate_models():
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=4,
            random_state=42,
            class_weight="balanced",
        ),
    }


def _build_pipeline(feature_columns, estimator):
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[("numeric", numeric, feature_columns)],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )


def _evaluate_predictions(y_true, y_pred, y_prob):
    metrics = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 4),
    }
    if y_prob is not None and len(set(y_true)) > 1:
        metrics["roc_auc"] = round(roc_auc_score(y_true, y_prob), 4)
    else:
        metrics["roc_auc"] = None
    return metrics


def build_labeled_domain_frame(config: dict) -> pd.DataFrame:
    historical = _load_frame(config["historical_file"])
    return config["label_builder"](historical)


def build_random_split(labeled: pd.DataFrame, feature_columns: list[str]):
    X = labeled[feature_columns].copy()
    y = labeled["success_label"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if y.nunique() > 1 else None,
    )
    return X_train, X_test, y_train, y_test


def build_time_based_split(labeled: pd.DataFrame, feature_columns: list[str], test_season_count: int = 3):
    if "season_start_year" not in labeled.columns:
        raise ValueError("season_start_year column is required for time-based split")

    ordered_seasons = sorted(int(s) for s in labeled["season_start_year"].dropna().unique())
    if len(ordered_seasons) < test_season_count + 2:
        raise ValueError("Not enough seasons available for time-based validation")

    test_seasons = ordered_seasons[-test_season_count:]
    train_df = labeled[~labeled["season_start_year"].isin(test_seasons)].copy()
    test_df = labeled[labeled["season_start_year"].isin(test_seasons)].copy()

    X_train = train_df[feature_columns].copy()
    X_test = test_df[feature_columns].copy()
    y_train = train_df["success_label"].astype(int)
    y_test = test_df["success_label"].astype(int)

    return X_train, X_test, y_train, y_test, test_seasons


def train_domain_model(domain_name: str, config: dict):
    historical = _load_frame(config["historical_file"])
    candidates = _load_frame(config["candidate_file"])
    labeled = config["label_builder"](historical)
    feature_columns = config["features"]

    X = labeled[feature_columns].copy()
    y = labeled["success_label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if y.nunique() > 1 else None,
    )

    model_rows = []
    best = None
    best_f1 = -1.0

    for model_name, estimator in _candidate_models().items():
        pipeline = _build_pipeline(feature_columns, estimator)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None
        metrics = _evaluate_predictions(y_test, y_pred, y_prob)
        model_rows.append(
            {
                "domain": domain_name,
                "model_name": model_name,
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "positive_rate_train": round(float(y_train.mean()), 4),
                "positive_rate_test": round(float(y_test.mean()), 4),
                **metrics,
            }
        )
        if metrics["f1_score"] > best_f1:
            best = pipeline
            best_f1 = metrics["f1_score"]

    best_model_name = max(model_rows, key=lambda row: row["f1_score"])["model_name"]
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, MODEL_DIR / f"{domain_name}_suitability_model.joblib")

    labeled_output = labeled.copy()
    labeled_output["predicted_success_probability"] = best.predict_proba(labeled[feature_columns])[:, 1]
    candidates_output = candidates.copy()
    candidates_output["ml_suitability_score"] = best.predict_proba(candidates[feature_columns])[:, 1] * 100

    labeled_output.to_csv(PROCESSED_DIR / f"training_{domain_name}_labeled.csv", index=False)
    candidates_output.to_csv(PROCESSED_DIR / f"smat_{domain_name}_ml_scores.csv", index=False)

    return (
        pd.DataFrame(model_rows),
        pd.DataFrame(
            [
                {
                    "domain": domain_name,
                    "best_model": best_model_name,
                    "best_f1_score": round(best_f1, 4),
                    "training_rows": len(labeled),
                    "positive_rate": round(float(y.mean()), 4),
                }
            ]
        ),
        candidates_output,
    )


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    comparison_rows = []
    best_rows = []
    candidate_summaries = []

    for domain_name, config in _domain_config().items():
        model_comparison, best_model, scored_candidates = train_domain_model(domain_name, config)
        comparison_rows.append(model_comparison)
        best_rows.append(best_model)
        candidate_summaries.append(
            pd.DataFrame(
                [
                    {
                        "domain": domain_name,
                        "candidates_scored": scored_candidates["player"].nunique(),
                        "avg_ml_suitability_score": round(
                            float(scored_candidates["ml_suitability_score"].mean()), 2
                        ),
                        "top_candidate": scored_candidates.sort_values(
                            "ml_suitability_score", ascending=False
                        ).iloc[0]["player"],
                    }
                ]
            )
        )

    comparison_df = pd.concat(comparison_rows, ignore_index=True)
    best_df = pd.concat(best_rows, ignore_index=True)
    candidate_df = pd.concat(candidate_summaries, ignore_index=True)

    comparison_df.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    best_df.to_csv(REPORT_DIR / "best_models.csv", index=False)
    candidate_df.to_csv(REPORT_DIR / "candidate_scoring_summary.csv", index=False)

    print("ML model training complete:")
    print(f"- {REPORT_DIR / 'model_comparison.csv'} {comparison_df.shape}")
    print(f"- {REPORT_DIR / 'best_models.csv'} {best_df.shape}")
    print(f"- {REPORT_DIR / 'candidate_scoring_summary.csv'} {candidate_df.shape}")
    print(f"- Saved models to {MODEL_DIR}")


if __name__ == "__main__":
    main()
