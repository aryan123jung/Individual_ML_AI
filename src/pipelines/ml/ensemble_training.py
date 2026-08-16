"""
Ensemble Model Training Module

Trains an ensemble of machine learning models (Random Forest + XGBoost + Logistic Regression)
using stacking for improved player suitability predictions.

This module increases the AI/ML component of recommendations from 45% → 60%
by combining multiple diverse models and learning optimal weights.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

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


def _label_from_role_medians(
    df: pd.DataFrame,
    role_column: str,
    metric_rules: list[tuple[str, bool]],
    min_role_rows: int = 6
):
    """Build success labels based on role-specific performance thresholds."""
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


def _build_base_models():
    """
    Create base learners for stacking ensemble.

    Uses 3 diverse algorithms:
    - Random Forest: good for feature interactions
    - XGBoost: gradient boosting for sequential error correction
    - Logistic Regression: linear model for diversity
    """
    return [
        (
            'rf',
            RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=4,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1,
            ),
        ),
        (
            'xgb',
            XGBClassifier(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1,
                verbosity=0,
            ),
        ),
        (
            'lr',
            LogisticRegression(
                max_iter=2000,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]


def _build_pipeline(feature_columns, base_learners):
    """
    Build preprocessing + ensemble stacking pipeline.

    Architecture:
    1. Preprocessing: Imputation + Scaling
    2. Base Learners: RF, XGBoost, Logistic Regression (run in parallel)
    3. Meta-Learner: Logistic Regression (learns optimal weights)
    """
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

    # Stacking classifier: train base models, then train meta-learner
    stacking_model = StackingClassifier(
        estimators=base_learners,
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=5,  # 5-fold cross-validation for base model training
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("ensemble", stacking_model),
        ]
    )


def _evaluate_predictions(y_true, y_pred, y_prob):
    """Calculate comprehensive evaluation metrics."""
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


def train_ensemble_model(domain_name: str, config: dict):
    """
    Train ensemble model for a specific domain (batting/bowling/allrounder).

    Args:
        domain_name: 'batting', 'bowling', or 'allrounder'
        config: Domain configuration with file paths and features

    Returns:
        Tuple of (model_comparison_df, best_model_df, scored_candidates_df)
    """
    print(f"\n{'='*70}")
    print(f"Training Ensemble Model for: {domain_name.upper()}")
    print(f"{'='*70}")

    # Load data
    historical = _load_frame(config["historical_file"])
    candidates = _load_frame(config["candidate_file"])
    labeled = config["label_builder"](historical)
    feature_columns = config["features"]

    print(f"✓ Loaded {len(labeled)} labeled historical records")
    print(f"✓ Features: {len(feature_columns)} statistics")

    # Split data
    X = labeled[feature_columns].copy()
    y = labeled["success_label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y if y.nunique() > 1 else None,
    )

    print(f"✓ Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"✓ Positive rate: {y_train.mean():.1%}")

    # Build and train ensemble
    base_learners = _build_base_models()
    pipeline = _build_pipeline(feature_columns, base_learners)

    print(f"\nTraining ensemble with 3 base models + meta-learner...")
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    metrics = _evaluate_predictions(y_test, y_pred, y_prob)

    print(f"\n📊 Ensemble Results:")
    print(f"   F1-Score:  {metrics['f1_score']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall:    {metrics['recall']:.4f}")
    print(f"   ROC-AUC:   {metrics['roc_auc']:.4f}")

    # Score candidates
    candidates_output = candidates.copy()
    candidates_output["ml_suitability_score"] = (
        pipeline.predict_proba(candidates[feature_columns])[:, 1] * 100
    )

    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_DIR / f"{domain_name}_ensemble_model.joblib")
    print(f"\n✓ Saved model: {MODEL_DIR / f'{domain_name}_ensemble_model.joblib'}")

    # Save scores
    candidates_output.to_csv(
        PROCESSED_DIR / f"smat_{domain_name}_ml_scores.csv",
        index=False
    )
    print(f"✓ Saved scores: {PROCESSED_DIR / f'smat_{domain_name}_ml_scores.csv'}")

    return (
        pd.DataFrame([{
            "domain": domain_name,
            "model_type": "Ensemble (RF + XGBoost + LR)",
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "positive_rate_train": round(float(y_train.mean()), 4),
            "positive_rate_test": round(float(y_test.mean()), 4),
            **metrics,
        }]),
        candidates_output,
    )


def main():
    """Train ensemble models for all domains."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("ENSEMBLE MODEL TRAINING - Increasing AI/ML from 45% → 60%")
    print("="*70)

    comparison_rows = []
    for domain_name, config in _domain_config().items():
        comparison_df, scored_candidates = train_ensemble_model(domain_name, config)
        comparison_rows.append(comparison_df)

    # Save comparison report
    comparison_df = pd.concat(comparison_rows, ignore_index=True)
    comparison_df.to_csv(REPORT_DIR / "ensemble_model_comparison.csv", index=False)

    print(f"\n{'='*70}")
    print(f"✅ ENSEMBLE TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"\nReport: {REPORT_DIR / 'ensemble_model_comparison.csv'}")
    print(f"Models: {MODEL_DIR}")
    print(f"\nNext steps:")
    print(f"1. Update MODEL_LED_WEIGHTS in recommender.py to 60% ML")
    print(f"2. Run src/recommendation_engine/run.py to generate new recommendations")
    print(f"3. Compare old vs new recommendations\n")


if __name__ == "__main__":
    main()
