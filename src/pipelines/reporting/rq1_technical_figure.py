from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mplconfig-codex")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[3]
DATA_ENGINEERING_DIR = BASE_DIR / "reports" / "data_engineering"
ML_DIR = BASE_DIR / "reports" / "ml"
TESTING_DIR = BASE_DIR / "reports" / "testing" / "ml"
CHART_DIR = ML_DIR / "charts"

THESIS_BG = "#fcfaf4"
PANEL_BG = "#ffffff"
TEXT = "#173b2f"
SUBTEXT = "#5b756b"
GRID = "#d9ddd4"
ACCENT = "#1f6d5a"
ACCENT_2 = "#c96a48"
ACCENT_3 = "#2362d3"
ACCENT_4 = "#8e5bd6"
EDGE = "#d8d7cc"


def _style_axes(ax):
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_color(EDGE)
        spine.set_linewidth(1.0)
    ax.tick_params(colors=TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)


def _load_feature_counts() -> dict[str, int]:
    df = pd.read_csv(DATA_ENGINEERING_DIR / "dataset_summary.csv")
    mapping = {
        "Batting features": "smat_batting",
        "Bowling features": "smat_bowling",
        "All-rounder features": "smat_allrounder",
    }
    out = {}
    for label, dataset in mapping.items():
        row = df.loc[df["dataset"] == dataset]
        out[label] = int(row["columns"].iloc[0])
    return out


def _load_time_based_metrics() -> pd.DataFrame:
    df = pd.read_csv(TESTING_DIR / "evaluation_mode_comparison.csv")
    ordered = (
        df[["domain", "accuracy_time", "roc_auc_time"]]
        .copy()
        .reset_index(drop=True)
    )
    domain_order = {"batting": 0, "bowling": 1, "allrounder": 2}
    ordered["sort_key"] = ordered["domain"].map(domain_order)
    return ordered.sort_values("sort_key").drop(columns=["sort_key"]).reset_index(drop=True)


def _load_clustering_metrics() -> pd.DataFrame:
    df = pd.read_csv(ML_DIR / "clustering_metrics.csv")
    domain_order = {"batting": 0, "bowling": 1, "allrounder": 2}
    df["sort_key"] = df["domain"].map(domain_order)
    return df.sort_values("sort_key").drop(columns=["sort_key"]).reset_index(drop=True)


def _draw_box(ax, x: float, y: float, w: float, h: float, title: str, body: str, face: str):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.03",
        linewidth=1.2,
        edgecolor=EDGE,
        facecolor=face,
    )
    ax.add_patch(box)
    ax.text(x + 0.02, y + h * 0.67, title, fontsize=12.5, fontweight="bold", color=TEXT, va="center")
    ax.text(x + 0.02, y + h * 0.33, body, fontsize=10.2, color=SUBTEXT, va="center", linespacing=1.35)


def _draw_arrow(ax, x1: float, y1: float, x2: float, y2: float):
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=1.8,
        color=ACCENT,
        shrinkA=6,
        shrinkB=6,
    )
    ax.add_patch(arrow)


def save_rq1_technical_figure() -> Path:
    feature_counts = _load_feature_counts()
    time_df = _load_time_based_metrics()
    cluster_df = _load_clustering_metrics()

    fig = plt.figure(figsize=(16, 11.2), facecolor=THESIS_BG)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], hspace=0.28, wspace=0.22)
    ax_flow = fig.add_subplot(gs[0, :])
    ax_acc = fig.add_subplot(gs[1, 0])
    ax_cluster = fig.add_subplot(gs[1, 1])

    fig.suptitle(
        "RQ1 Technical Finding: From SMAT Data to IPL Role-Based Recommendations",
        fontsize=21,
        fontweight="bold",
        color=TEXT,
        y=0.985,
    )

    ax_flow.set_facecolor(THESIS_BG)
    ax_flow.set_xlim(0, 1)
    ax_flow.set_ylim(0, 1)
    ax_flow.axis("off")

    ax_flow.text(
        0.0,
        0.97,
        "Integrated evidence pipeline used to transform domestic T20 performance data into franchise recommendation outputs",
        fontsize=11.2,
        color=SUBTEXT,
        va="top",
    )

    bodies = [
        "SMAT batting,\nbowling, and\nall-rounder tables",
        (
            f"Batting: {feature_counts['Batting features']}\n"
            f"Bowling: {feature_counts['Bowling features']}\n"
            f"All-rounder: {feature_counts['All-rounder features']}"
        ),
        "Random Forest\nsuitability models\ntrained on IPL labels",
        "KMeans role\narchetypes and\ncluster structure",
        "Domestic vs IPL\nbenchmark similarity\nand reliability signals",
        "Squad-gap aware,\neligible franchise\nshortlists",
    ]
    titles = [
        "1. SMAT Input",
        "2. Feature Engineering",
        "3. Supervised Learning",
        "4. Clustering",
        "5. Similarity Context",
        "6. Final Recommendation",
    ]
    colors = ["#eef6f1", "#f4f8ef", "#eef4fb", "#f5effa", "#fbf3ee", "#edf7f5"]

    box_w = 0.275
    box_h = 0.22
    top_y = 0.56
    bottom_y = 0.24
    xs = [0.02, 0.36, 0.70]

    top_positions = [(xs[0], top_y), (xs[1], top_y), (xs[2], top_y)]
    bottom_positions = [(xs[0], bottom_y), (xs[1], bottom_y), (xs[2], bottom_y)]
    positions = top_positions + bottom_positions

    for (x, y), title, body, color in zip(positions, titles, bodies, colors):
        _draw_box(ax_flow, x, y, box_w, box_h, title, body, color)

    _draw_arrow(ax_flow, xs[0] + box_w, top_y + box_h / 2, xs[1], top_y + box_h / 2)
    _draw_arrow(ax_flow, xs[1] + box_w, top_y + box_h / 2, xs[2], top_y + box_h / 2)
    _draw_arrow(ax_flow, xs[2] + box_w / 2, top_y, xs[2] + box_w / 2, bottom_y + box_h)
    _draw_arrow(ax_flow, xs[2], bottom_y + box_h / 2, xs[1] + box_w, bottom_y + box_h / 2)
    _draw_arrow(ax_flow, xs[1], bottom_y + box_h / 2, xs[0] + box_w, bottom_y + box_h / 2)

    ax_flow.text(
        0.5,
        0.08,
        "Meaningful recommendation does not come from one score alone; it emerges from connected stages of representation, prediction, archetype discovery, benchmarking, and franchise need analysis.",
        ha="center",
        va="center",
        fontsize=10.8,
        color=TEXT,
    )

    _style_axes(ax_acc)
    acc_domains = [d.title().replace("Allrounder", "All-rounder") for d in time_df["domain"]]
    acc_values = time_df["accuracy_time"] * 100
    auc_values = time_df["roc_auc_time"] * 100
    x = list(range(len(acc_domains)))
    width = 0.34
    b1 = ax_acc.bar([i - width / 2 for i in x], acc_values, width=width, color=ACCENT, label="Time-based accuracy")
    b2 = ax_acc.bar([i + width / 2 for i in x], auc_values, width=width, color=ACCENT_3, label="Time-based ROC-AUC")
    ax_acc.set_title("Temporal model transfer performance", fontsize=14, fontweight="bold", loc="left", pad=12)
    ax_acc.set_ylabel("Score (%)")
    ax_acc.set_ylim(0, 110)
    ax_acc.set_xticks(x)
    ax_acc.set_xticklabels(acc_domains)
    ax_acc.legend(frameon=False, loc="upper left")
    for bars in (b1, b2):
        for bar in bars:
            height = bar.get_height()
            ax_acc.text(
                bar.get_x() + bar.get_width() / 2,
                height + 1.6,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=9.6,
                color=TEXT,
                fontweight="semibold",
            )

    _style_axes(ax_cluster)
    cluster_domains = [d.title().replace("Allrounder", "All-rounder") for d in cluster_df["domain"]]
    sil_values = cluster_df["silhouette_score"]
    clusters = cluster_df["selected_clusters"]
    colors = [ACCENT_2, ACCENT_4, ACCENT]
    bars = ax_cluster.bar(cluster_domains, sil_values, color=colors)
    ax_cluster.set_title("Archetype quality from unsupervised clustering", fontsize=14, fontweight="bold", loc="left", pad=12)
    ax_cluster.set_ylabel("Silhouette score")
    ax_cluster.set_ylim(0, max(0.4, sil_values.max() + 0.06))
    for bar, value, k in zip(bars, sil_values, clusters):
        ax_cluster.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.01,
            f"{value:.4f}\n{k} clusters",
            ha="center",
            va="bottom",
            fontsize=9.5,
            color=TEXT,
            fontweight="semibold",
        )

    fig.text(
        0.5,
        0.02,
        "Source files: dataset_summary.csv, evaluation_mode_comparison.csv, and clustering_metrics.csv generated by the project pipeline.",
        ha="center",
        fontsize=10.2,
        color=SUBTEXT,
    )

    CHART_DIR.mkdir(parents=True, exist_ok=True)
    path = CHART_DIR / "rq1_feature_to_recommendation_pipeline.png"
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor=THESIS_BG)
    plt.close(fig)
    return path


def main():
    path = save_rq1_technical_figure()
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
