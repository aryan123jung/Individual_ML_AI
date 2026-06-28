import pandas as pd

from recommendation_engine.benchmarks import build_role_benchmarks
from recommendation_engine.config import (
    ALLROUNDER_FEATURES,
    BATTING_FEATURES,
    BOWLING_FEATURES,
    REPORT_DIR,
)
from recommendation_engine.candidate_pool import build_active_candidate_pools
from recommendation_engine.data_loader import load_features
from recommendation_engine.recommender import (
    build_recommendations,
    build_team_recommendation_summary,
    remove_current_squad_players,
)
from recommendation_engine.replacements import (
    build_replacement_recommendations,
    build_replacement_watchlist,
)
from recommendation_engine.similarity import best_role_matches
from recommendation_engine.squad_gaps import build_latest_ipl_rosters, build_squad_gaps


def merge_ml_scores(base_df, ml_df):
    if ml_df is None or ml_df.empty or "ml_suitability_score" not in ml_df.columns:
        result = base_df.copy()
        result["ml_suitability_score"] = 0.0
        return result
    keep_cols = ["player", "ml_suitability_score"]
    merged = base_df.merge(
        ml_df[keep_cols].drop_duplicates(subset=["player"]),
        on="player",
        how="left",
    )
    merged["ml_suitability_score"] = merged["ml_suitability_score"].fillna(0.0)
    return merged


def merge_cluster_scores(base_df, cluster_df):
    result = base_df.copy()
    if cluster_df is None or cluster_df.empty:
        result["cluster_label"] = None
        result["archetype_role"] = None
        result["archetype_role_match_score"] = 0.0
        return result
    keep_cols = [
        "player",
        "cluster_label",
        "archetype_role",
        "archetype_role_match_score",
    ]
    available_cols = [col for col in keep_cols if col in cluster_df.columns]
    merged = result.merge(
        cluster_df[available_cols].drop_duplicates(subset=["player"]),
        on="player",
        how="left",
    )
    if "archetype_role_match_score" not in merged.columns:
        merged["archetype_role_match_score"] = 0.0
    merged["archetype_role_match_score"] = merged["archetype_role_match_score"].fillna(0.0)
    return merged


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_features()
    (
        data["smat_batting"],
        data["smat_bowling"],
        data["smat_allrounder"],
        recent_pool_meta,
    ) = build_active_candidate_pools(
        data["smat_batting"],
        data["smat_bowling"],
        data["smat_allrounder"],
        data["smat_batting_raw"],
        data["smat_bowling_raw"],
    )
    data["smat_batting"] = merge_ml_scores(data["smat_batting"], data.get("smat_batting_ml_scores"))
    data["smat_bowling"] = merge_ml_scores(data["smat_bowling"], data.get("smat_bowling_ml_scores"))
    data["smat_allrounder"] = merge_ml_scores(data["smat_allrounder"], data.get("smat_allrounder_ml_scores"))
    data["smat_batting"] = merge_cluster_scores(
        data["smat_batting"], data.get("smat_batting_clusters")
    )
    data["smat_bowling"] = merge_cluster_scores(
        data["smat_bowling"], data.get("smat_bowling_clusters")
    )
    data["smat_allrounder"] = merge_cluster_scores(
        data["smat_allrounder"], data.get("smat_allrounder_clusters")
    )

    role_benchmarks = build_role_benchmarks(
        data["ipl_batting"], data["ipl_bowling"], data["ipl_allrounder"]
    )

    batting_similarity = best_role_matches(
        data["smat_batting"],
        data["ipl_batting"],
        "batting_role",
        BATTING_FEATURES,
        "batting",
    )
    bowling_similarity = best_role_matches(
        data["smat_bowling"],
        data["ipl_bowling"],
        "bowling_role",
        BOWLING_FEATURES,
        "bowling",
    )
    allrounder_similarity = best_role_matches(
        data["smat_allrounder"],
        data["ipl_allrounder"],
        "allrounder_role",
        ALLROUNDER_FEATURES,
        "allrounder",
    )
    similarity = pd.concat(
        [batting_similarity, bowling_similarity, allrounder_similarity],
        ignore_index=True,
    )

    roster = build_latest_ipl_rosters(
        data.get("ipl_current_squad"),
        data["ipl_batting_raw"],
        data["ipl_bowling_raw"],
    )
    squad_gaps = build_squad_gaps(
        roster, data["ipl_batting"], data["ipl_bowling"], data["ipl_allrounder"]
    )
    smat_batting_candidates, smat_bowling_candidates, smat_allrounder_candidates = remove_current_squad_players(
        data["smat_batting"],
        data["smat_bowling"],
        data["smat_allrounder"],
        roster,
    )
    recommendations = build_recommendations(
        squad_gaps,
        smat_batting_candidates,
        smat_bowling_candidates,
        smat_allrounder_candidates,
        similarity,
    )
    team_summary = build_team_recommendation_summary(recommendations)
    replacement_watchlist = build_replacement_watchlist(
        data["ipl_current_squad"],
        data["ipl_2026_batting"],
        data["ipl_2026_bowling"],
        data["ipl_batting"],
        data["ipl_bowling"],
        data["ipl_allrounder"],
    )
    replacement_recommendations = build_replacement_recommendations(
        replacement_watchlist, recommendations
    )

    outputs = {
        "ipl_role_benchmarks.csv": role_benchmarks,
        "smat_ipl_similarity.csv": similarity,
        "franchise_squad_gaps.csv": squad_gaps,
        "franchise_recommendations.csv": recommendations,
        "team_recommendation_summary.csv": team_summary,
        "current_squad_underperformers.csv": replacement_watchlist,
        "replacement_recommendations.csv": replacement_recommendations,
        "replacement_action_summary.csv": (
            replacement_watchlist.groupby(["team", "action_level"], as_index=False)
            .size()
            .rename(columns={"size": "players"})
            if not replacement_watchlist.empty
            else pd.DataFrame(columns=["team", "action_level", "players"])
        ),
        "domestic_candidate_pool_summary.csv": pd.DataFrame(
            [
                {
                    "candidate_pool": "smat_batting_recent_active_without_current_ipl_squad_players",
                    "players": smat_batting_candidates["player"].nunique(),
                },
                {
                    "candidate_pool": "smat_bowling_recent_active_without_current_ipl_squad_players",
                    "players": smat_bowling_candidates["player"].nunique(),
                },
                {
                    "candidate_pool": "smat_allrounder_recent_active_without_current_ipl_squad_players",
                    "players": smat_allrounder_candidates["player"].nunique(),
                },
                {
                    "candidate_pool": f"recent_active_batting_players_{recent_pool_meta['min_batting_season']}_{recent_pool_meta['latest_batting_season']}",
                    "players": recent_pool_meta["recent_batting_players"],
                },
                {
                    "candidate_pool": f"recent_active_bowling_players_{recent_pool_meta['min_bowling_season']}_{recent_pool_meta['latest_bowling_season']}",
                    "players": recent_pool_meta["recent_bowling_players"],
                },
                {
                    "candidate_pool": f"recent_active_allrounder_players_{min(recent_pool_meta['min_batting_season'], recent_pool_meta['min_bowling_season'])}_{max(recent_pool_meta['latest_batting_season'], recent_pool_meta['latest_bowling_season'])}",
                    "players": recent_pool_meta["recent_allrounder_players"],
                },
            ]
        ),
    }

    for filename, df in outputs.items():
        df.to_csv(REPORT_DIR / filename, index=False)

    print("Recommendation engine complete:")
    for filename, df in outputs.items():
        print(f"- {REPORT_DIR / filename} {df.shape}")


if __name__ == "__main__":
    main()
