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


# Ensemble Model Led Weighting (Increased to 60% AI/ML - Aug 2026)
# Uses ensemble of Random Forest + XGBoost + Logistic Regression
# ML suitability is the dominant signal (60%), cricket analytics support (40%)
MODEL_LED_WEIGHTS = {
    "ml_score": 0.50,           # ↑ 42% → 50% (Ensemble of 3 models)
    "archetype_match": 0.10,    # ↑ 3% → 10% (ML clustering)
    "quality": 0.10,            # ↓ 14% → 10% (Cricket analytics)
    "similarity_score": 0.06,   # ↓ 11% → 6% (Reduced domain logic)
    "recent_form": 0.04,        # ↓ 8% → 4% (Less emphasis)
    "gap_score": 0.10,          # ↑ 7% → 10% (Squad context)
    "reliability": 0.10,        # ↑ 5% → 10% (Data quality)
    "role_fit": 0.00,           # ✗ 10% → 0% (REMOVED - too cricket-specific)
}
# Total AI/ML: ml_score (50%) + archetype_match (10%) = 60%
# Total Cricket/Rules: quality (10%) + similarity (6%) + gap (10%) + reliability (10%) + recent_form (4%) = 40%


def _normalize_name(value):
    return "".join(ch.lower() for ch in str(value or "") if ch.isalnum() or ch.isspace()).strip()


def _name_aliases(value):
    normalized = _normalize_name(value)
    parts = normalized.split()
    aliases = {normalized.replace(" ", "")}
    if parts:
        aliases.add("".join(parts))
    if len(parts) >= 2:
        aliases.add(f"{parts[0][0]}{parts[-1]}")
        aliases.add(f"{parts[0][0]}{''.join(parts[1:])}")
        aliases.add("".join(parts[1:]))
    if len(parts) >= 3 and len(parts[0]) == 1:
        aliases.add("".join(parts[1:]))
        aliases.add(f"{parts[1][0]}{parts[-1]}")
    return {alias for alias in aliases if alias}


def _exclude_current_squad(df, current_squad_players):
    if df.empty:
        return df.copy()
    current_aliases = set()
    for player in current_squad_players:
        current_aliases.update(_name_aliases(player))
    keep_mask = ~df["player"].astype(str).map(lambda name: bool(_name_aliases(name) & current_aliases))
    return df[keep_mask].copy()


def exclude_unavailable_players(df, unavailable_players):
    if df.empty or unavailable_players is None or unavailable_players.empty:
        return df.copy()
    player_col = "player" if "player" in unavailable_players.columns else unavailable_players.columns[0]
    blocked_aliases = set()
    for player in unavailable_players[player_col].dropna().astype(str):
        blocked_aliases.update(_name_aliases(player))
    keep_mask = ~df["player"].astype(str).map(lambda name: bool(_name_aliases(name) & blocked_aliases))
    return df[keep_mask].copy()


def _shortlist_limit(deficit):
    deficit = int(deficit or 0)
    if deficit <= 1:
        return 4
    if deficit == 2:
        return 6
    return 8


def _model_led_recommendation_score(
    quality,
    similarity_score,
    gap_score,
    reliability,
    recent_form,
    role_fit,
    ml_score,
    archetype_match,
):
    weighted_score = (
        (ml_score * MODEL_LED_WEIGHTS["ml_score"])
        + (quality * MODEL_LED_WEIGHTS["quality"])
        + (similarity_score * MODEL_LED_WEIGHTS["similarity_score"])
        + (role_fit * MODEL_LED_WEIGHTS["role_fit"])
        + (recent_form * MODEL_LED_WEIGHTS["recent_form"])
        + (gap_score * MODEL_LED_WEIGHTS["gap_score"])
        + (reliability * MODEL_LED_WEIGHTS["reliability"])
        + (archetype_match * MODEL_LED_WEIGHTS["archetype_match"])
    )

    # Penalties are lighter now; they guard against low-evidence outliers
    # instead of overwhelming the ML model.
    penalty = (
        max(0, 35 - reliability) * 0.22
        + max(0, 40 - recent_form) * 0.16
        + max(0, 40 - quality) * 0.12
    )
    return max(weighted_score - penalty, 0)


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
    if "sample_reliability_score" in candidates.columns:
        candidates = candidates[candidates["sample_reliability_score"].fillna(0) >= 0.45].copy()
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
        if ml_score < 50:
            continue
        if reliability < 35:
            continue

        final_score = _model_led_recommendation_score(
            quality=quality,
            similarity_score=similarity_score,
            gap_score=gap_score,
            reliability=reliability,
            recent_form=recent_form,
            role_fit=role_fit,
            ml_score=ml_score,
            archetype_match=archetype_match,
        )
        if final_score < 50:
            continue

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
    if "matches" in smat_batting.columns:
        smat_batting = smat_batting[smat_batting["matches"].fillna(0) >= 4].copy()
    if "matches" in smat_bowling.columns:
        smat_bowling = smat_bowling[smat_bowling["matches"].fillna(0) >= 4].copy()
    if "matches" in smat_allrounder.columns:
        smat_allrounder = smat_allrounder[smat_allrounder["matches"].fillna(0) >= 4].copy()

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
    recommendations["shortlist_limit"] = recommendations["role_deficit"].map(_shortlist_limit)
    recommendations = recommendations[
        recommendations["role_rank"] <= recommendations["shortlist_limit"]
    ].copy()
    recommendations = recommendations.sort_values(
        ["team", "domestic_player", "final_recommendation_score", "quality_score", "role_fit_score"],
        ascending=[True, True, False, False, False],
    )
    recommendations = recommendations.drop_duplicates(
        subset=["team", "domestic_player"],
        keep="first",
    ).copy()
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
    recommendations = recommendations.drop(columns=["shortlist_limit"])
    return recommendations


def remove_current_squad_players(smat_batting, smat_bowling, smat_allrounder, current_roster):
    current_squad_players = set(current_roster["player"].dropna().astype(str))
    return (
        _exclude_current_squad(smat_batting, current_squad_players),
        _exclude_current_squad(smat_bowling, current_squad_players),
        _exclude_current_squad(smat_allrounder, current_squad_players),
    )


def build_team_recommendation_summary(recommendations):
    if recommendations.empty:
        return recommendations
    top = recommendations[recommendations["overall_team_rank"] <= 10].copy()
    return top.sort_values(["team", "overall_team_rank"])
