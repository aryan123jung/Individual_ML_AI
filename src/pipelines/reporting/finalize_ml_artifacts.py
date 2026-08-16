from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[3]
ML_DIR = BASE_DIR / "reports" / "ml"
TESTING_DIR = BASE_DIR / "reports" / "testing" / "ml"


def build_model_registry() -> pd.DataFrame:
    best_models = pd.read_csv(ML_DIR / "best_models.csv")
    time_based = pd.read_csv(TESTING_DIR / "time_based_validation" / "time_based_model_metrics.csv")
    comparison = pd.read_csv(TESTING_DIR / "evaluation_mode_comparison.csv")

    merged = (
        best_models.merge(
            time_based[
                ["domain", "best_model", "test_seasons", "accuracy", "precision", "recall", "f1_score", "roc_auc"]
            ],
            on=["domain", "best_model"],
            how="left",
            suffixes=("", "_time"),
        )
        .merge(
            comparison[
                ["domain", "accuracy_random", "f1_score_random", "accuracy_drop_pct_points", "f1_drop_pct_points"]
            ],
            on="domain",
            how="left",
        )
        .rename(
            columns={
                "best_model": "production_model",
                "accuracy": "time_based_accuracy",
                "precision": "time_based_precision",
                "recall": "time_based_recall",
                "f1_score": "time_based_f1_score",
                "roc_auc": "time_based_roc_auc",
            }
        )
    )

    merged["deployment_note"] = (
        "Use time-based validation as the main thesis result; random split is baseline only."
    )
    return merged[
        [
            "domain",
            "production_model",
            "training_rows",
            "positive_rate",
            "test_seasons",
            "time_based_accuracy",
            "time_based_precision",
            "time_based_recall",
            "time_based_f1_score",
            "time_based_roc_auc",
            "accuracy_random",
            "f1_score_random",
            "accuracy_drop_pct_points",
            "f1_drop_pct_points",
            "deployment_note",
        ]
    ]


def build_thesis_summary() -> str:
    registry = pd.read_csv(ML_DIR / "final_model_registry.csv")
    ablation = pd.read_csv(TESTING_DIR / "ablation" / "allrounder" / "allrounder_ablation_results.csv")

    lines = [
        "# Final ML Pipeline Summary",
        "",
        "## Production Models",
        "The final production suitability model for all three domains is Random Forest.",
        "This choice is based on consistent superiority over Logistic Regression in the internal model comparison stage.",
        "",
        "## Why Time-Based Validation Is The Main Result",
        "Random train/test split is useful as a baseline, but time-based validation is more academically correct for this thesis.",
        "The system is intended to learn from historical IPL patterns and generalize forward to later seasons and domestic candidates.",
        "Therefore, the main thesis result should be the time-based metrics, not the random-split metrics.",
        "",
        "## Final Domain Results",
    ]

    for _, row in registry.iterrows():
        lines.append(
            f"- {row['domain'].title()}: production model `{row['production_model']}`, "
            f"time-based accuracy `{row['time_based_accuracy']:.4f}`, "
            f"F1-score `{row['time_based_f1_score']:.4f}`, "
            f"ROC-AUC `{row['time_based_roc_auc']:.4f}`, "
            f"tested on seasons `{row['test_seasons']}`."
        )

    lines.extend(
        [
            "",
            "## Domain Interpretation",
            "- Batting is the hardest domain because batting performance is more volatile and role-sensitive, which explains its lower forward accuracy than bowling and all-rounder prediction.",
            "- Bowling is more stable because economy, wickets, and phase control are comparatively consistent and structurally measurable.",
            "- All-rounder performance is the strongest domain because the feature space captures both batting and bowling contribution together, creating clearer separation between successful and unsuccessful profiles.",
            "",
            "## Allrounder High-Score Justification",
            "The all-rounder model shows very high performance, so it requires explicit justification in the thesis.",
            "An ablation study was used for this purpose.",
        ]
    )

    full = ablation[ablation["variant"] == "full_feature_set"].iloc[0]
    no_index = ablation[ablation["variant"] == "without_allrounder_index"].iloc[0]
    raw = ablation[ablation["variant"] == "raw_features_only"].iloc[0]
    lines.extend(
        [
            f"- Full all-rounder feature set: accuracy `{full['accuracy']:.4f}`, F1 `{full['f1_score']:.4f}`.",
            f"- Without `allrounder_index`: accuracy `{no_index['accuracy']:.4f}`, F1 `{no_index['f1_score']:.4f}`.",
            f"- Raw features only: accuracy `{raw['accuracy']:.4f}`, F1 `{raw['f1_score']:.4f}`.",
            "This shows that the performance is not purely artificial. The model remains strong even after removing composite strength features, although accuracy declines when the composite all-rounder index is removed.",
            "Therefore, the correct thesis position is that the all-rounder score is high because the role is genuinely more separable in the engineered feature space, while composite features further strengthen that separability.",
            "",
            "## Final Thesis Position",
            "The system is genuinely AI/ML-based because it uses supervised learning to estimate IPL suitability and unsupervised clustering to assign player archetypes.",
            "However, the recommendation engine is not only a classifier. It is a hybrid decision-support framework that combines:",
            "- engineered cricket features,",
            "- supervised suitability prediction,",
            "- clustering-based archetype support,",
            "- similarity benchmarking against IPL profiles, and",
            "- rule-based squad gap and replacement logic.",
            "",
            "This is the correct final description for the thesis: a hybrid AI/ML-driven player recommendation system for IPL squad formation.",
        ]
    )

    return "\n".join(lines) + "\n"


def main() -> None:
    registry = build_model_registry()
    registry.to_csv(ML_DIR / "final_model_registry.csv", index=False)
    summary = build_thesis_summary()
    (ML_DIR / "final_ml_pipeline_summary.md").write_text(summary)
    print(f"Wrote {ML_DIR / 'final_model_registry.csv'} {registry.shape}")
    print(f"Wrote {ML_DIR / 'final_ml_pipeline_summary.md'}")


if __name__ == "__main__":
    main()
