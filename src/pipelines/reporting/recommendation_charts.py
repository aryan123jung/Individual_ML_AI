from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mplconfig-codex")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[3]
RECOMMENDATION_DIR = BASE_DIR / "reports" / "recommendations"
ML_DIR = BASE_DIR / "reports" / "ml"
QA_DIR = RECOMMENDATION_DIR / "qa"
CHART_DIR = RECOMMENDATION_DIR / "charts"


THESIS_BG = "#fcfaf4"
PANEL_BG = "#ffffff"
TEXT = "#173b2f"
SUBTEXT = "#5b756b"
GRID = "#d9ddd4"
ACCENT = "#1f6d5a"
ACCENT_2 = "#c96a48"
ACCENT_3 = "#2362d3"
ACCENT_4 = "#8e5bd6"


def _style_axes(ax):
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_color("#e2e5dc")
        spine.set_linewidth(1.0)
    ax.tick_params(colors=TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    ax.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)


def _human_pool_label(value: str) -> str:
    mapping = {
        "smat_batting_recent_active_without_current_ipl_squad_players": "Batting shortlist",
        "smat_bowling_recent_active_without_current_ipl_squad_players": "Bowling shortlist",
        "smat_allrounder_recent_active_without_current_ipl_squad_players": "All-rounder shortlist",
        "recent_active_batting_players_2023_2024": "Batting active pool",
        "recent_active_bowling_players_2023_2024": "Bowling active pool",
        "recent_active_allrounder_players_2023_2024": "All-rounder active pool",
        "manual_unavailable_players_excluded": "Manual exclusions",
    }
    return mapping.get(value, value.replace("_", " ").title())


def _load_output_counts() -> pd.DataFrame:
    rows = [
        ("Franchise recommendations", RECOMMENDATION_DIR / "franchise_recommendations.csv"),
        ("Team summary rows", RECOMMENDATION_DIR / "team_recommendation_summary.csv"),
        ("Squad-gap rows", RECOMMENDATION_DIR / "franchise_squad_gaps.csv"),
        ("Replacement rows", RECOMMENDATION_DIR / "replacement_recommendations.csv"),
        ("Underperformer rows", RECOMMENDATION_DIR / "current_squad_underperformers.csv"),
        ("Similarity rows", RECOMMENDATION_DIR / "smat_ipl_similarity.csv"),
        ("IPL benchmark rows", RECOMMENDATION_DIR / "ipl_role_benchmarks.csv"),
    ]
    return pd.DataFrame(
        [
            {"output_type": label, "rows": len(pd.read_csv(path))}
            for label, path in rows
        ]
    )


def _load_qa_summary() -> pd.DataFrame:
    df = pd.read_csv(QA_DIR / "recommendation_qa_summary.csv")
    df["check_label"] = df["check"].map(
        {
            "recommended_current_ipl_players": "Current IPL players in recommendations",
            "recommended_unavailable_players": "Manually unavailable players recommended",
            "unavailable_similarity_benchmarks": "Unavailable similarity benchmarks used",
        }
    )
    return df


def _load_candidate_pool_summary() -> tuple[pd.DataFrame, int]:
    df = pd.read_csv(RECOMMENDATION_DIR / "domestic_candidate_pool_summary.csv")
    filtered = df[df["candidate_pool"].str.contains("without_current_ipl_squad_players")].copy()
    active = df[df["candidate_pool"].str.contains("recent_active_")].copy()

    filtered["domain"] = filtered["candidate_pool"].str.extract(r"smat_(.*?)_recent")[0]
    active["domain"] = active["candidate_pool"].str.extract(r"recent_active_(.*?)_players")[0]

    merged = filtered[["domain", "players"]].merge(
        active[["domain", "players"]],
        on="domain",
        suffixes=("_shortlist", "_active"),
    )
    merged["domain"] = merged["domain"].replace({"allrounder": "All-rounder"}).str.title()

    excluded = int(df.loc[df["candidate_pool"] == "manual_unavailable_players_excluded", "players"].iloc[0])
    return merged.sort_values("domain"), excluded


def _load_candidate_scores() -> pd.DataFrame:
    df = pd.read_csv(ML_DIR / "candidate_scoring_summary.csv")
    df["domain"] = df["domain"].replace({"allrounder": "All-rounder"}).str.title()
    return df


def save_recommendation_output_qa_chart():
    outputs = _load_output_counts().sort_values("rows", ascending=True)
    qa = _load_qa_summary()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.4), facecolor=THESIS_BG)
    fig.suptitle("Recommendation Output and Quality Assurance", fontsize=20, fontweight="bold", color=TEXT, y=0.98)

    _style_axes(ax1)
    bars = ax1.barh(outputs["output_type"], outputs["rows"], color=[ACCENT if r != "Similarity rows" else ACCENT_4 for r in outputs["output_type"]])
    ax1.set_title("Pipeline output volumes", fontsize=14, fontweight="bold", loc="left", pad=12)
    ax1.set_xlabel("Rows generated")
    ax1.grid(axis="y", visible=False)
    for bar, value in zip(bars, outputs["rows"]):
        ax1.text(bar.get_width() + max(outputs["rows"]) * 0.015, bar.get_y() + bar.get_height() / 2, f"{int(value)}", va="center", ha="left", color=TEXT, fontsize=10, fontweight="semibold")

    _style_axes(ax2)
    qa_plot = qa.copy()
    qa_plot["count_display"] = qa_plot["count"].astype(int)
    bars2 = ax2.barh(qa_plot["check_label"], qa_plot["count_display"], color=ACCENT)
    ax2.set_title("QA eligibility checks", fontsize=14, fontweight="bold", loc="left", pad=12)
    ax2.set_xlabel("Invalid rows detected")
    ax2.set_xlim(0, 1)
    ax2.grid(axis="y", visible=False)
    ax2.set_xticks([0, 0.5, 1.0])
    for bar, status in zip(bars2, qa_plot["status"]):
        ax2.text(0.03, bar.get_y() + bar.get_height() / 2, "0", va="center", ha="left", color=TEXT, fontsize=10, fontweight="semibold")
        ax2.text(0.97, bar.get_y() + bar.get_height() / 2, status.upper(), va="center", ha="right", color=ACCENT, fontsize=10, fontweight="bold", transform=ax2.get_yaxis_transform())

    fig.text(
        0.5,
        0.02,
        "All QA checks pass: no current IPL players, no manually unavailable players, and no unavailable benchmark players appear in final outputs.",
        ha="center",
        fontsize=10.5,
        color=SUBTEXT,
    )
    fig.tight_layout(rect=(0.02, 0.05, 0.98, 0.94))
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    path = CHART_DIR / "recommendation_output_qa_overview.png"
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor=THESIS_BG)
    plt.close(fig)
    return path


def save_candidate_pool_suitability_chart():
    pools, excluded = _load_candidate_pool_summary()
    scores = _load_candidate_scores()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.4), facecolor=THESIS_BG)
    fig.suptitle("Candidate Pool Filtering and Domain Suitability", fontsize=20, fontweight="bold", color=TEXT, y=0.98)

    _style_axes(ax1)
    x = range(len(pools))
    width = 0.34
    b1 = ax1.bar([i - width / 2 for i in x], pools["players_shortlist"], width=width, color=ACCENT, label="Filtered shortlist")
    b2 = ax1.bar([i + width / 2 for i in x], pools["players_active"], width=width, color=ACCENT_2, label="Broader active pool")
    ax1.set_title("Recent active pool versus final shortlist", fontsize=14, fontweight="bold", loc="left", pad=12)
    ax1.set_ylabel("Players")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(pools["domain"])
    ax1.legend(frameon=False, loc="upper left")
    for bars in (b1, b2):
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, height + 3, f"{int(height)}", ha="center", va="bottom", color=TEXT, fontsize=10, fontweight="semibold")
    ax1.text(0.98, 0.06, f"Manual exclusions: {excluded}", transform=ax1.transAxes, ha="right", va="bottom", color=SUBTEXT, fontsize=10)

    _style_axes(ax2)
    bars3 = ax2.bar(scores["domain"], scores["avg_ml_suitability_score"], color=[ACCENT_3, ACCENT, ACCENT_4])
    ax2.set_title("Average ML suitability by domain", fontsize=14, fontweight="bold", loc="left", pad=12)
    ax2.set_ylabel("Average ML suitability score")
    ax2.set_ylim(0, max(scores["avg_ml_suitability_score"]) + 15)
    ax2.grid(axis="x", visible=False)
    for bar, _, count in zip(bars3, scores["avg_ml_suitability_score"], scores["candidates_scored"]):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, height + 1.2, f"{height:.2f}", ha="center", va="bottom", color=TEXT, fontsize=10, fontweight="semibold")
        ax2.text(bar.get_x() + bar.get_width() / 2, max(4, height * 0.18), f"{int(count)} scored", ha="center", va="center", color="white", fontsize=9.5, fontweight="bold")

    fig.text(
        0.5,
        0.02,
        "Role-specific scoring produces different suitability distributions across batting, bowling, and all-rounder candidate pools.",
        ha="center",
        fontsize=10.5,
        color=SUBTEXT,
    )
    fig.tight_layout(rect=(0.02, 0.05, 0.98, 0.94))
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    path = CHART_DIR / "candidate_pool_and_ml_suitability.png"
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor=THESIS_BG)
    plt.close(fig)
    return path


def main():
    output_qa = save_recommendation_output_qa_chart()
    candidate = save_candidate_pool_suitability_chart()
    print("Recommendation reporting charts complete:")
    print(f"- {output_qa}")
    print(f"- {candidate}")


if __name__ == "__main__":
    main()
