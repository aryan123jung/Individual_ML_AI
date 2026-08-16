import pandas as pd

from recommendation_engine.config import REPORT_DIR
from recommendation_engine.data_loader import load_features
from recommendation_engine.recommender import _name_aliases
from recommendation_engine.squad_gaps import build_latest_ipl_rosters


def _alias_set(values):
    aliases = set()
    for value in values:
        aliases.update(_name_aliases(value))
    return aliases


def main():
    data = load_features()
    recommendations_path = REPORT_DIR / "franchise_recommendations.csv"
    similarity_path = REPORT_DIR / "smat_ipl_similarity.csv"

    recommendations = pd.read_csv(recommendations_path) if recommendations_path.exists() else pd.DataFrame()
    similarity = pd.read_csv(similarity_path) if similarity_path.exists() else pd.DataFrame()

    roster = build_latest_ipl_rosters(
        data.get("ipl_current_squad"),
        data["ipl_batting_raw"],
        data["ipl_bowling_raw"],
    )
    roster_aliases = _alias_set(roster["player"].dropna().astype(str).tolist())
    unavailable = data.get("manual_unavailable_players", pd.DataFrame())
    unavailable_aliases = _alias_set(
        unavailable.get("player", pd.Series(dtype=str)).dropna().astype(str).tolist()
    )

    if recommendations.empty:
        current_squad_hits = pd.DataFrame()
        unavailable_hits = pd.DataFrame()
    else:
        current_squad_hits = recommendations[
            recommendations["domestic_player"].astype(str).map(
                lambda name: bool(_name_aliases(name) & roster_aliases)
            )
        ].copy()
        unavailable_hits = recommendations[
            recommendations["domestic_player"].astype(str).map(
                lambda name: bool(_name_aliases(name) & unavailable_aliases)
            )
        ].copy()

    if similarity.empty:
        unavailable_benchmark_hits = pd.DataFrame()
    else:
        unavailable_benchmark_hits = similarity[
            similarity["ipl_player"].astype(str).map(
                lambda name: bool(_name_aliases(name) & unavailable_aliases)
            )
        ].copy()

    summary = pd.DataFrame(
        [
            {
                "check": "recommended_current_ipl_players",
                "count": len(current_squad_hits),
                "status": "pass" if current_squad_hits.empty else "fail",
            },
            {
                "check": "recommended_unavailable_players",
                "count": len(unavailable_hits),
                "status": "pass" if unavailable_hits.empty else "fail",
            },
            {
                "check": "unavailable_similarity_benchmarks",
                "count": len(unavailable_benchmark_hits),
                "status": "pass" if unavailable_benchmark_hits.empty else "fail",
            },
        ]
    )

    qa_dir = REPORT_DIR / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(qa_dir / "recommendation_qa_summary.csv", index=False)
    current_squad_hits.to_csv(qa_dir / "recommended_current_ipl_players.csv", index=False)
    unavailable_hits.to_csv(qa_dir / "recommended_unavailable_players.csv", index=False)
    unavailable_benchmark_hits.to_csv(qa_dir / "unavailable_similarity_benchmarks.csv", index=False)

    print("Recommendation QA complete:")
    print(f"- {qa_dir / 'recommendation_qa_summary.csv'} {summary.shape}")
    print(f"- {qa_dir / 'recommended_current_ipl_players.csv'} {current_squad_hits.shape}")
    print(f"- {qa_dir / 'recommended_unavailable_players.csv'} {unavailable_hits.shape}")
    print(f"- {qa_dir / 'unavailable_similarity_benchmarks.csv'} {unavailable_benchmark_hits.shape}")


if __name__ == "__main__":
    main()
