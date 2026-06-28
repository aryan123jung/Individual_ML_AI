from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

from pipelines.ml.model_training import (
    _build_pipeline,
    _candidate_models,
    _evaluate_predictions,
    build_labeled_domain_frame,
    build_time_based_split,
)


BASE_DIR = Path(__file__).resolve().parents[3]
REPORT_DIR = BASE_DIR / "reports" / "testing" / "ml" / "ablation" / "allrounder"
CHART_DIR = REPORT_DIR / "charts"


def _annotate_bar_values(ax):
    for patch in ax.patches:
        height = patch.get_height()
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            height + 0.01,
            f"{height * 100:.2f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def _allrounder_config():
    return {
        "historical_file": "ipl_allrounder_features_v2.csv",
        "candidate_file": "smat_allrounder_features_v2.csv",
        "features": [
            "batting_strength_score",
            "bowling_strength_score",
            "allrounder_index",
            "total_runs",
            "strike_rate",
            "total_wickets",
            "avg_wickets_per_match",
            "economy_rate",
        ],
        "label_builder": lambda df: df[df["allrounder_role"].notna()].copy(),
    }


def _labeled_allrounder_data():
    from pipelines.ml.model_training import _build_allrounder_label

    raw = pd.read_csv(BASE_DIR / "data" / "processed" / "ipl_allrounder_features_v2.csv")
    return _build_allrounder_label(raw)


def _variant_feature_sets():
    base = [
        "batting_strength_score",
        "bowling_strength_score",
        "allrounder_index",
        "total_runs",
        "strike_rate",
        "total_wickets",
        "avg_wickets_per_match",
        "economy_rate",
    ]
    return {
        "full_feature_set": base,
        "without_allrounder_index": [f for f in base if f != "allrounder_index"],
        "without_composite_strength_scores": [
            f for f in base if f not in {"batting_strength_score", "bowling_strength_score"}
        ],
        "raw_features_only": [
            "total_runs",
            "strike_rate",
            "total_wickets",
            "avg_wickets_per_match",
            "economy_rate",
        ],
    }


def _save_ablation_chart(results_df: pd.DataFrame):
    plot_df = results_df[["variant", "accuracy", "f1_score", "roc_auc"]].copy()
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_df.set_index("variant").plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_title("Allrounder Time-Based Ablation Results")
    ax.set_ylabel("Score")
    ax.set_xlabel("Feature Variant")
    ax.tick_params(axis="x", rotation=15)
    ax.legend(title="Metric")
    _annotate_bar_values(ax)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "allrounder_ablation_bar_chart.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    labeled = _labeled_allrounder_data()
    rows = []
    confusion_rows = []

    for variant, features in _variant_feature_sets().items():
        X_train, X_test, y_train, y_test, test_seasons = build_time_based_split(labeled, features)
        estimator = _candidate_models()["random_forest"]
        pipeline = _build_pipeline(features, estimator)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        metrics = _evaluate_predictions(y_test, y_pred, y_prob)
        rows.append(
            {
                "variant": variant,
                "feature_count": len(features),
                "features_used": ",".join(features),
                "test_seasons": ",".join(str(s) for s in test_seasons),
                **metrics,
            }
        )
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        confusion_rows.append(
            {
                "variant": variant,
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )

    results_df = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    confusion_df = pd.DataFrame(confusion_rows)
    results_df.to_csv(REPORT_DIR / "allrounder_ablation_results.csv", index=False)
    confusion_df.to_csv(REPORT_DIR / "allrounder_ablation_confusion_matrices.csv", index=False)
    _save_ablation_chart(results_df)

    summary_lines = [
        "# Allrounder Ablation Summary",
        "",
        "This test measures whether the allrounder model remains strong after removing composite features that may be closely aligned with the label.",
        "",
    ]
    for _, row in results_df.iterrows():
        summary_lines.append(
            f"- {row['variant']}: accuracy `{row['accuracy']:.4f}`, F1 `{row['f1_score']:.4f}`, ROC-AUC `{row['roc_auc']:.4f}`."
        )
    (REPORT_DIR / "allrounder_ablation_summary.md").write_text(
        "\n".join(summary_lines), encoding="utf-8"
    )

    print("Allrounder ablation complete:")
    print(f"- {REPORT_DIR / 'allrounder_ablation_results.csv'} {results_df.shape}")
    print(f"- {REPORT_DIR / 'allrounder_ablation_confusion_matrices.csv'} {confusion_df.shape}")
    print(f"- {CHART_DIR / 'allrounder_ablation_bar_chart.png'}")


if __name__ == "__main__":
    main()
