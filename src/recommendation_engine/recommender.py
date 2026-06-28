import pandas as pd

from recommendation_engine.scoring import (
    allrounder_quality_score,
    allrounder_recent_form_score,
    allrounder_role_fit_score,
    batting_quality_score,
    batting_recent_form_score,
    batting_role_fit_score,
    bowling_quality_score,
    bowling_recent_form_score,
    bowling_role_fit_score,
)
from recommendation_engine.similarity import best_similarity_lookup


def recommendation_rows_for_gap(
    gap_row,
    smat_df,
    role_column,
    comparison_type,
    quality_column,
    recent_form_column,
    role_fit_fn,
    similarity_lookup,
):
    role = gap_row["role"]
    candidates = smat_df[smat_df[role_column] == role].copy()
    if "eligible_for_model" in candidates.columns:
        candidates = candidates[candidates["eligible_for_model"]].copy()
    if candidates.empty:
        return []

    candidates["role_fit_score"] = role_fit_fn(candidates, role)

    rows = []
    for _, candidate in candidates.iterrows():
        lookup_key = (comparison_type, role, candidate["player"])
        sim_row = similarity_lookup.get(lookup_key)
        similarity_score = float(sim_row["similarity_score"]) if sim_row is not None else 0
        benchmark_player = sim_row["ipl_player"] if sim_row is not None else None
        benchmark_team = sim_row["ipl_team"] if sim_row is not None else None
        reliability = float(candidate.get("sample_reliability_score", 0)) * 100
        quality = float(candidate.get(quality_column, 0))
        ml_score = float(candidate.get("ml_suitability_score", 0))
        recent_form = float(candidate.get(recent_form_column, 0))
        role_fit = float(candidate.get("role_fit_score", 0))
        archetype_match = float(candidate.get("archetype_role_match_score", 0))
        gap_score = float(gap_row["gap_severity"]) * 100
        final_score = (
            (quality * 0.20)
            + (similarity_score * 0.15)
            + (gap_score * 0.15)
            + (reliability * 0.10)
            + (recent_form * 0.10)
            + (role_fit * 0.10)
            + (ml_score * 0.10)
            + (archetype_match * 0.10)
        )

        rows.append(
            {
                "team": gap_row["team"],
                "recommendation_type": comparison_type,
                "target_role": role,
                "domestic_player": candidate["player"],
                "domestic_team": candidate.get("team"),
                "quality_score": round(quality, 2),
                "similarity_score": round(similarity_score, 2),
                "gap_score": round(gap_score, 2),
                "reliability_score": round(reliability, 2),
                "recent_form_score": round(recent_form, 2),
                "role_fit_score": round(role_fit, 2),
                "ml_suitability_score": round(ml_score, 2),
                "archetype_role_match_score": round(archetype_match, 2),
                "cluster_label": candidate.get("cluster_label"),
                "archetype_role": candidate.get("archetype_role"),
                "final_recommendation_score": round(final_score, 2),
                "closest_ipl_benchmark": benchmark_player,
                "closest_ipl_benchmark_team": benchmark_team,
                "current_role_count": gap_row["current_count"],
                "target_role_count": gap_row["target_count"],
                "role_deficit": gap_row["deficit"],
            }
        )
    return rows


def build_recommendations(squad_gaps, smat_batting, smat_bowling, smat_allrounder, similarity_df):
    smat_batting = smat_batting.copy()
    smat_bowling = smat_bowling.copy()
    smat_allrounder = smat_allrounder.copy()

    smat_batting["quality_score"] = batting_quality_score(smat_batting)
    smat_batting["recent_form_score"] = batting_recent_form_score(smat_batting)
    smat_bowling["quality_score"] = bowling_quality_score(smat_bowling)
    smat_bowling["recent_form_score"] = bowling_recent_form_score(smat_bowling)
    smat_allrounder["quality_score"] = allrounder_quality_score(smat_allrounder)
    smat_allrounder["recent_form_score"] = allrounder_recent_form_score(smat_allrounder)

    similarity_lookup = best_similarity_lookup(similarity_df)
    rows = []

    for _, gap in squad_gaps[squad_gaps["deficit"] > 0].iterrows():
        if gap["gap_type"] == "batting":
            rows.extend(
                recommendation_rows_for_gap(
                    gap,
                    smat_batting,
                    "batting_role",
                    "batting",
                    "quality_score",
                    "recent_form_score",
                    batting_role_fit_score,
                    similarity_lookup,
                )
            )
        elif gap["gap_type"] == "bowling":
            rows.extend(
                recommendation_rows_for_gap(
                    gap,
                    smat_bowling,
                    "bowling_role",
                    "bowling",
                    "quality_score",
                    "recent_form_score",
                    bowling_role_fit_score,
                    similarity_lookup,
                )
            )
        elif gap["gap_type"] == "allrounder":
            rows.extend(
                recommendation_rows_for_gap(
                    gap,
                    smat_allrounder,
                    "allrounder_role",
                    "allrounder",
                    "quality_score",
                    "recent_form_score",
                    allrounder_role_fit_score,
                    similarity_lookup,
                )
            )

    recommendations = pd.DataFrame(rows)
    if recommendations.empty:
        return recommendations

    recommendations = recommendations.sort_values(
        ["team", "target_role", "final_recommendation_score"],
        ascending=[True, True, False],
    )
    recommendations["role_rank"] = (
        recommendations.groupby(["team", "recommendation_type", "target_role"]).cumcount() + 1
    )
    recommendations["overall_team_rank"] = (
        recommendations.sort_values(["team", "final_recommendation_score"], ascending=[True, False])
        .groupby("team")
        .cumcount()
        + 1
    )
    return recommendations


def remove_current_squad_players(smat_batting, smat_bowling, smat_allrounder, current_roster):
    current_squad_players = set(current_roster["player"])
    return (
        smat_batting[~smat_batting["player"].isin(current_squad_players)].copy(),
        smat_bowling[~smat_bowling["player"].isin(current_squad_players)].copy(),
        smat_allrounder[~smat_allrounder["player"].isin(current_squad_players)].copy(),
    )


def build_team_recommendation_summary(recommendations):
    if recommendations.empty:
        return recommendations
    top = recommendations[recommendations["overall_team_rank"] <= 10].copy()
    return top.sort_values(["team", "overall_team_rank"])
