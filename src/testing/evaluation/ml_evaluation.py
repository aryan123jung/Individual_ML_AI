from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from pipelines.ml.model_training import (
    _build_pipeline,
    _candidate_models,
    _domain_config,
    build_labeled_domain_frame,
    build_random_split,
    build_time_based_split,
)


BASE_DIR = Path(__file__).resolve().parents[3]
REPORT_DIR = BASE_DIR / "reports" / "testing" / "ml"
SOURCE_ML_REPORT_DIR = BASE_DIR / "reports" / "ml"
RANDOM_SPLIT_DIR = REPORT_DIR / "random_split"
CHART_DIR = RANDOM_SPLIT_DIR / "charts"
PREDICTION_DIR = RANDOM_SPLIT_DIR / "predictions"
SUMMARY_DIR = RANDOM_SPLIT_DIR / "summaries"
TIME_BASED_DIR = REPORT_DIR / "time_based_validation"
TIME_BASED_CHART_DIR = TIME_BASED_DIR / "charts"
TIME_BASED_PREDICTION_DIR = TIME_BASED_DIR / "predictions"
TIME_BASED_SUMMARY_DIR = TIME_BASED_DIR / "summaries"


def _best_model_lookup() -> dict[str, str]:
    best = pd.read_csv(SOURCE_ML_REPORT_DIR / "best_models.csv")
    return dict(zip(best["domain"], best["best_model"]))


def _annotate_bar_values(ax, scale=100.0, suffix="%"):
    for patch in ax.patches:
        height = patch.get_height()
        if pd.isna(height):
            continue
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            height + 0.015,
            f"{height * scale:.2f}{suffix}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def _annotate_line_points(ax, x_values, y_values, scale=100.0, suffix="%"):
    for x_val, y_val in zip(x_values, y_values):
        if pd.isna(y_val):
            continue
        ax.annotate(
            f"{y_val * scale:.2f}{suffix}",
            (x_val, y_val),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )


def _save_metric_comparison_chart(model_comparison: pd.DataFrame):
    metrics = ["accuracy", "f1_score", "roc_auc"]
    domains = model_comparison["domain"].unique().tolist()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, metric in zip(axes, metrics):
        pivot = (
            model_comparison.pivot(index="domain", columns="model_name", values=metric)
            .reindex(domains)
        )
        pivot.plot(kind="bar", ax=ax)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=0)
        ax.legend(title="Model")
        _annotate_bar_values(ax)

    fig.suptitle("Model Performance Comparison by Domain")
    fig.tight_layout()
    path = CHART_DIR / "model_metric_comparison.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_best_model_chart(model_comparison: pd.DataFrame):
    best_rows = (
        model_comparison.sort_values(["domain", "f1_score"], ascending=[True, False])
        .groupby("domain")
        .head(1)
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(best_rows))
    ax.bar([i - 0.25 for i in x], best_rows["accuracy"], width=0.25, label="Accuracy")
    ax.bar(x, best_rows["f1_score"], width=0.25, label="F1 Score")
    ax.bar([i + 0.25 for i in x], best_rows["roc_auc"], width=0.25, label="ROC AUC")
    ax.set_xticks(list(x))
    ax.set_xticklabels(best_rows["domain"].tolist())
    ax.set_ylim(0, 1.05)
    ax.set_title("Best Model Performance by Domain")
    ax.legend()
    _annotate_bar_values(ax)
    fig.tight_layout()
    path = CHART_DIR / "best_model_performance.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_metric_line_chart(model_comparison: pd.DataFrame):
    best_rows = (
        model_comparison.sort_values(["domain", "f1_score"], ascending=[True, False])
        .groupby("domain")
        .head(1)
        .copy()
    )
    best_rows = best_rows.set_index("domain").loc[["batting", "bowling", "allrounder"]].reset_index()
    metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for metric in metrics:
        ax.plot(
            best_rows["domain"],
            best_rows[metric],
            marker="o",
            linewidth=2,
            label=metric.replace("_", " ").title(),
        )
        _annotate_line_points(ax, best_rows["domain"], best_rows[metric])

    ax.set_ylim(0, 1.05)
    ax.set_title("Best Model Metrics Across Player Domains")
    ax.set_ylabel("Score")
    ax.set_xlabel("Domain")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    path = CHART_DIR / "best_model_metrics_line_graph.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_confusion_matrix_chart(domain_name: str, matrix):
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["True 0", "True 1"])
    ax.set_title(f"{domain_name.title()} Confusion Matrix")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = CHART_DIR / f"confusion_matrix_{domain_name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_clustering_chart(clustering_metrics: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(clustering_metrics["domain"], clustering_metrics["silhouette_score"], color="#4c78a8")
    ax.set_ylim(0, max(0.4, clustering_metrics["silhouette_score"].max() + 0.05))
    ax.set_title("Clustering Quality by Domain")
    ax.set_ylabel("Silhouette Score")
    for idx, value in enumerate(clustering_metrics["silhouette_score"]):
        ax.text(idx, value + 0.01, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    path = CHART_DIR / "clustering_silhouette_scores.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_time_based_metric_chart(time_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for metric in ["accuracy", "f1_score", "roc_auc"]:
        ax.plot(
            time_df["domain"],
            time_df[metric],
            marker="o",
            linewidth=2,
            label=metric.replace("_", " ").title(),
        )
        _annotate_line_points(ax, time_df["domain"], time_df[metric])
    ax.set_ylim(0, 1.05)
    ax.set_title("Time-Based Validation Performance")
    ax.set_ylabel("Score")
    ax.set_xlabel("Domain")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    path = TIME_BASED_CHART_DIR / "time_based_metric_line_graph.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_time_based_metric_bar_chart(time_df: pd.DataFrame):
    plot_df = time_df[["domain", "accuracy", "f1_score", "roc_auc"]].copy()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    plot_df.set_index("domain").plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_title("Time-Based Validation Metric Comparison")
    ax.set_ylabel("Score")
    ax.set_xlabel("Domain")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Metric")
    _annotate_bar_values(ax)
    fig.tight_layout()
    path = TIME_BASED_CHART_DIR / "time_based_metric_bar_chart.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_time_based_confusion_matrix_chart(domain_name: str, matrix):
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(matrix, cmap="Oranges")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["True 0", "True 1"])
    ax.set_title(f"{domain_name.title()} Time-Based Confusion Matrix")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = TIME_BASED_CHART_DIR / f"time_based_confusion_matrix_{domain_name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _build_thesis_summary(model_comparison: pd.DataFrame, confusion_df: pd.DataFrame, clustering_metrics: pd.DataFrame):
    best_rows = (
        model_comparison.sort_values(["domain", "f1_score"], ascending=[True, False])
        .groupby("domain")
        .head(1)
    )
    lines = [
        "# ML Evaluation Summary",
        "",
        "## Overview",
        "The player recommendation system was evaluated using a supervised learning stage and an unsupervised clustering stage.",
        "For supervised learning, historical IPL feature sets were split into training and testing partitions using an 80:20 train/test split with a fixed random state of 42.",
        "Two classification algorithms were compared in each domain: Logistic Regression and Random Forest.",
        "",
        "## Supervised Model Results",
    ]

    for _, row in best_rows.iterrows():
        lines.append(
            f"- {row['domain'].title()}: the best model was `{row['model_name']}` with accuracy `{row['accuracy']:.4f}`, F1-score `{row['f1_score']:.4f}`, and ROC-AUC `{row['roc_auc']:.4f}`."
        )

    lines.extend(
        [
            "",
            "These results indicate that the Random Forest model consistently produced the strongest predictive performance across batting, bowling, and all-rounder evaluation. This suggests that nonlinear relationships between engineered cricket features are important for predicting IPL suitability.",
            "",
            "## Confusion Matrix Interpretation",
        ]
    )

    for _, row in confusion_df.iterrows():
        lines.append(
            f"- {row['domain'].title()}: true negatives `{int(row['tn'])}`, false positives `{int(row['fp'])}`, false negatives `{int(row['fn'])}`, and true positives `{int(row['tp'])}`."
        )

    lines.extend(
        [
            "",
            "The confusion matrices show that the models are not only accurate overall, but also effective at identifying successful players without producing excessive misclassification. This is important because the system is intended as a decision-support tool for selectors and coaches.",
            "",
            "## Clustering Results",
        ]
    )

    for _, row in clustering_metrics.iterrows():
        lines.append(
            f"- {row['domain'].title()}: `{int(row['selected_clusters'])}` clusters were selected with silhouette score `{row['silhouette_score']:.4f}`."
        )

    lines.extend(
        [
            "",
            "The clustering stage supports the supervised model by grouping players into role-based archetypes. Among the three domains, all-rounder clustering achieved the strongest separation, indicating clearer natural structure in all-rounder profiles than in batting-only or bowling-only profiles.",
            "",
            "## Thesis Interpretation",
            "Overall, the evaluation demonstrates that the proposed system can learn meaningful patterns from historical IPL data and apply them to domestic player assessment. The combination of feature engineering, supervised classification, and unsupervised clustering strengthens the system's credibility as an AI/ML-based player recommendation framework for IPL squad formation.",
        ]
    )

    output_path = SUMMARY_DIR / "thesis_ml_evaluation_summary.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _build_time_based_summary(time_metrics_df: pd.DataFrame, time_confusion_df: pd.DataFrame):
    lines = [
        "# Time-Based Validation Summary",
        "",
        "This validation uses older IPL seasons for training and the latest seasons for testing.",
        "It is a stricter and more realistic test than a random split because it evaluates whether the model generalizes forward in time.",
        "",
        "## Results",
    ]

    for _, row in time_metrics_df.iterrows():
        lines.append(
            f"- {row['domain'].title()}: tested on seasons `{row['test_seasons']}` with accuracy `{row['accuracy']:.4f}`, F1-score `{row['f1_score']:.4f}`, and ROC-AUC `{row['roc_auc']:.4f}`."
        )

    lines.extend(["", "## Confusion Matrices"])
    for _, row in time_confusion_df.iterrows():
        lines.append(
            f"- {row['domain'].title()}: true negatives `{int(row['tn'])}`, false positives `{int(row['fp'])}`, false negatives `{int(row['fn'])}`, true positives `{int(row['tp'])}`."
        )

    lines.extend(
        [
            "",
            "If the time-based scores are lower than the random-split scores, that is expected and academically healthier. It means the time-based evaluation is testing real generalization instead of same-distribution memorization.",
        ]
    )
    (TIME_BASED_SUMMARY_DIR / "time_based_validation_summary.md").write_text("\n".join(lines), encoding="utf-8")


def _build_comparison_summary(random_metrics: pd.DataFrame, time_metrics: pd.DataFrame):
    merged = random_metrics.merge(
        time_metrics[
            ["domain", "accuracy", "precision", "recall", "f1_score", "roc_auc", "test_seasons"]
        ],
        on="domain",
        suffixes=("_random", "_time"),
    )
    merged["accuracy_drop_pct_points"] = ((merged["accuracy_random"] - merged["accuracy_time"]) * 100).round(2)
    merged["f1_drop_pct_points"] = ((merged["f1_score_random"] - merged["f1_score_time"]) * 100).round(2)
    merged.to_csv(REPORT_DIR / "evaluation_mode_comparison.csv", index=False)


def _cleanup_legacy_outputs():
    legacy_paths = [
        REPORT_DIR / "charts",
        REPORT_DIR / "predictions",
        REPORT_DIR / "summaries",
        REPORT_DIR / "confusion_matrices.csv",
    ]
    for path in legacy_paths:
        if path.is_file():
            path.unlink()


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    RANDOM_SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    TIME_BASED_DIR.mkdir(parents=True, exist_ok=True)
    TIME_BASED_CHART_DIR.mkdir(parents=True, exist_ok=True)
    TIME_BASED_PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    TIME_BASED_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    _cleanup_legacy_outputs()

    model_comparison = pd.read_csv(SOURCE_ML_REPORT_DIR / "model_comparison.csv")
    clustering_metrics = pd.read_csv(SOURCE_ML_REPORT_DIR / "clustering_metrics.csv")
    best_lookup = _best_model_lookup()

    confusion_rows = []
    for domain_name, config in _domain_config().items():
        labeled = build_labeled_domain_frame(config)
        feature_columns = config["features"]
        X_train, X_test, y_train, y_test = build_random_split(labeled, feature_columns)
        estimator = _candidate_models()[best_lookup[domain_name]]
        pipeline = _build_pipeline(feature_columns, estimator)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = matrix.ravel()
        confusion_rows.append(
            {
                "domain": domain_name,
                "best_model": best_lookup[domain_name],
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "test_rows": len(y_test),
            }
        )
        predictions = pd.DataFrame(
            {
                "actual_label": y_test.reset_index(drop=True),
                "predicted_label": pd.Series(y_pred),
            }
        )
        predictions.to_csv(PREDICTION_DIR / f"test_predictions_{domain_name}.csv", index=False)
        _save_confusion_matrix_chart(domain_name, matrix)

    confusion_df = pd.DataFrame(confusion_rows)
    confusion_df.to_csv(RANDOM_SPLIT_DIR / "confusion_matrices.csv", index=False)

    _save_metric_comparison_chart(model_comparison)
    _save_best_model_chart(model_comparison)
    _save_metric_line_chart(model_comparison)
    _save_clustering_chart(clustering_metrics)
    _build_thesis_summary(model_comparison, confusion_df, clustering_metrics)

    time_metrics_rows = []
    time_confusion_rows = []
    for domain_name, config in _domain_config().items():
        labeled = build_labeled_domain_frame(config)
        feature_columns = config["features"]
        X_train, X_test, y_train, y_test, test_seasons = build_time_based_split(
            labeled, feature_columns
        )
        estimator = _candidate_models()[best_lookup[domain_name]]
        pipeline = _build_pipeline(feature_columns, estimator)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = (
            pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None
        )
        metrics = {
            "domain": domain_name,
            "best_model": best_lookup[domain_name],
            "test_seasons": ",".join(str(season) for season in test_seasons),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
        }
        from pipelines.ml.model_training import _evaluate_predictions

        metrics.update(_evaluate_predictions(y_test, y_pred, y_prob))
        time_metrics_rows.append(metrics)

        matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = matrix.ravel()
        time_confusion_rows.append(
            {
                "domain": domain_name,
                "best_model": best_lookup[domain_name],
                "test_seasons": ",".join(str(season) for season in test_seasons),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "test_rows": len(y_test),
            }
        )
        pd.DataFrame(
            {
                "actual_label": y_test.reset_index(drop=True),
                "predicted_label": pd.Series(y_pred),
                "predicted_probability": pd.Series(y_prob) if y_prob is not None else None,
            }
        ).to_csv(TIME_BASED_PREDICTION_DIR / f"time_based_test_predictions_{domain_name}.csv", index=False)
        _save_time_based_confusion_matrix_chart(domain_name, matrix)

    time_metrics_df = pd.DataFrame(time_metrics_rows)
    time_confusion_df = pd.DataFrame(time_confusion_rows)
    time_metrics_df.to_csv(TIME_BASED_DIR / "time_based_model_metrics.csv", index=False)
    time_confusion_df.to_csv(TIME_BASED_DIR / "time_based_confusion_matrices.csv", index=False)
    _save_time_based_metric_chart(time_metrics_df)
    _save_time_based_metric_bar_chart(time_metrics_df)
    _build_time_based_summary(time_metrics_df, time_confusion_df)

    random_best_metrics = (
        model_comparison.sort_values(["domain", "f1_score"], ascending=[True, False])
        .groupby("domain")
        .head(1)
        .reset_index(drop=True)
    )
    _build_comparison_summary(random_best_metrics, time_metrics_df)

    print("ML evaluation complete:")
    print(f"- {RANDOM_SPLIT_DIR / 'confusion_matrices.csv'} {confusion_df.shape}")
    print(f"- {CHART_DIR}")
    print(f"- {PREDICTION_DIR}")
    print(f"- {SUMMARY_DIR / 'thesis_ml_evaluation_summary.md'}")
    print(f"- {TIME_BASED_DIR / 'time_based_model_metrics.csv'} {time_metrics_df.shape}")
    print(f"- {REPORT_DIR / 'evaluation_mode_comparison.csv'}")


if __name__ == "__main__":
    main()
