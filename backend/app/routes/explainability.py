"""
SHAP Explainability Route
Provides detailed explanations for why players are recommended
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, List
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

router = APIRouter(prefix="/api/explainability", tags=["explainability"])

# Load data and models
BASE_DIR = Path(__file__).parent.parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

# Cache loaded models
LOADED_MODELS = {}


def load_model(domain: str):
    """Load ensemble model for domain"""
    if domain not in LOADED_MODELS:
        model_path = MODELS_DIR / f"{domain}_ensemble_model.joblib"
        LOADED_MODELS[domain] = joblib.load(model_path)
    return LOADED_MODELS[domain]


def get_domain_for_role(role: str) -> str:
    """Map role to domain (batting/bowling/allrounder)"""
    batting_roles = ["aggressive_opener", "middle_order", "finisher", "lower_order_hitter"]
    bowling_roles = ["powerplay_bowler", "middle_overs_bowler", "death_bowler"]

    if role.lower() in batting_roles:
        return "batting"
    elif role.lower() in bowling_roles:
        return "bowling"
    else:
        return "allrounder"


def calculate_feature_contributions(
    player_data: Dict,
    recommendation_score: float,
    weights: Dict
) -> Dict:
    """
    Calculate contribution of each component to final score

    Components:
    - ml_score (50%)
    - archetype_match (10%)
    - quality (10%)
    - gap_score (10%)
    - reliability (10%)
    - similarity_score (6%)
    - recent_form (4%)
    """

    # Extract component scores from player data (0-100 scale)
    ml_score = float(player_data.get("ml_suitability_score", 0))
    archetype_match = float(player_data.get("archetype_match_score", 0))
    quality = float(player_data.get("quality_score", 0))
    gap_score = float(player_data.get("gap_score", 0))
    reliability = float(player_data.get("reliability_score", 0))
    similarity = float(player_data.get("similarity_score", 0))
    recent_form = float(player_data.get("recent_form_score", 0))

    # Get weights
    w_ml = weights.get("ml_score", 0.50)
    w_archetype = weights.get("archetype_match", 0.10)
    w_quality = weights.get("quality", 0.10)
    w_gap = weights.get("gap_score", 0.10)
    w_reliability = weights.get("reliability", 0.10)
    w_similarity = weights.get("similarity_score", 0.06)
    w_recent = weights.get("recent_form", 0.04)

    # Calculate weighted contributions
    # Each component contributes: score * weight to final score
    ml_contrib = (ml_score / 100) * w_ml * recommendation_score
    gap_contrib = (gap_score / 100) * w_gap * recommendation_score
    reliability_contrib = (reliability / 100) * w_reliability * recommendation_score
    recent_contrib = (recent_form / 100) * w_recent * recommendation_score
    similarity_contrib = (similarity / 100) * w_similarity * recommendation_score
    archetype_contrib = (archetype_match / 100) * w_archetype * recommendation_score
    quality_contrib = (quality / 100) * w_quality * recommendation_score

    # Total contribution (for calculating percentages)
    total_contrib = (ml_contrib + gap_contrib + reliability_contrib +
                     recent_contrib + similarity_contrib + archetype_contrib + quality_contrib)

    # Avoid division by zero
    if total_contrib == 0:
        total_contrib = 1

    # Calculate contributions dict
    contributions = {
        "ML Suitability Score": {
            "value": ml_score,
            "weight": w_ml,
            "contribution": ml_contrib,
            "percentage": (ml_contrib / total_contrib * 100) if total_contrib > 0 else 0
        },
        "Gap Score": {
            "value": gap_score,
            "weight": w_gap,
            "contribution": gap_contrib,
            "percentage": (gap_contrib / total_contrib * 100) if total_contrib > 0 else 0
        },
        "Reliability": {
            "value": reliability,
            "weight": w_reliability,
            "contribution": reliability_contrib,
            "percentage": (reliability_contrib / total_contrib * 100) if total_contrib > 0 else 0
        },
        "Recent Form": {
            "value": recent_form,
            "weight": w_recent,
            "contribution": recent_contrib,
            "percentage": (recent_contrib / total_contrib * 100) if total_contrib > 0 else 0
        },
        "Similarity to IPL": {
            "value": similarity,
            "weight": w_similarity,
            "contribution": similarity_contrib,
            "percentage": (similarity_contrib / total_contrib * 100) if total_contrib > 0 else 0
        },
        "Archetype Match": {
            "value": archetype_match,
            "weight": w_archetype,
            "contribution": archetype_contrib,
            "percentage": (archetype_contrib / total_contrib * 100) if total_contrib > 0 else 0
        },
        "Quality": {
            "value": quality,
            "weight": w_quality,
            "contribution": quality_contrib,
            "percentage": (quality_contrib / total_contrib * 100) if total_contrib > 0 else 0
        }
    }

    # Sort by contribution (largest first)
    sorted_contributions = dict(
        sorted(
            contributions.items(),
            key=lambda x: x[1]["contribution"],
            reverse=True
        )
    )

    return sorted_contributions


@router.get("/recommendations/{team}/{player}")
async def explain_recommendation(team: str, player: str):
    """
    Get detailed explanation for a player recommendation

    Returns SHAP-style feature importance breakdown
    """
    try:
        # Load recommendations
        recs_df = pd.read_csv(DATA_DIR / "recommendations.csv", low_memory=False) if (DATA_DIR / "recommendations.csv").exists() else pd.DataFrame()

        # Search for recommendation in various files
        found = False
        player_rec = None
        domain = "batting"

        # Try franchise recommendations
        if (BASE_DIR / "reports" / "recommendations" / "franchise_recommendations.csv").exists():
            franchise_recs = pd.read_csv(
                BASE_DIR / "reports" / "recommendations" / "franchise_recommendations.csv"
            )
            match = franchise_recs[
                (franchise_recs["team"].str.lower() == team.lower()) &
                (franchise_recs["domestic_player"].str.lower() == player.lower())
            ]
            if not match.empty:
                player_rec = match.iloc[0].to_dict()
                domain = get_domain_for_role(player_rec.get("target_role", "allrounder"))
                found = True

        # Try replacement recommendations if not found
        if not found and (BASE_DIR / "reports" / "recommendations" / "replacement_recommendations.csv").exists():
            replacement_recs = pd.read_csv(
                BASE_DIR / "reports" / "recommendations" / "replacement_recommendations.csv"
            )
            match = replacement_recs[
                (replacement_recs["team"].str.lower() == team.lower()) &
                (replacement_recs["recommended_player"].str.lower() == player.lower())
            ]
            if not match.empty:
                player_rec = match.iloc[0].to_dict()
                domain = get_domain_for_role(player_rec.get("target_role", "allrounder"))
                found = True

        if not found or player_rec is None:
            raise HTTPException(
                status_code=404,
                detail=f"Recommendation not found for {player} → {team}"
            )

        # Get score
        score = float(player_rec.get("final_recommendation_score", player_rec.get("replacement_score", 0)))

        # Weight configuration (60% ML)
        weights = {
            "ml_score": 0.50,
            "archetype_match": 0.10,
            "quality": 0.10,
            "gap_score": 0.10,
            "reliability": 0.10,
            "similarity_score": 0.06,
            "recent_form": 0.04,
            "role_fit": 0.0  # Removed
        }

        # Calculate feature contributions
        contributions = calculate_feature_contributions(
            player_rec,
            score,
            weights
        )

        # Format response
        return {
            "player": player,
            "team": team,
            "role": player_rec.get("target_role", "unknown"),
            "score": round(score, 2),
            "max_score": 100,
            "explanation": {
                "components": contributions,
                "top_features": list(contributions.keys())[:5],
                "summary": f"{player} is recommended for {team} as {player_rec.get('target_role', 'player')} with {score:.1f}/100 score"
            }
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating explanation: {str(e)}"
        )


@router.get("/recommendations/{team}")
async def explain_team_recommendations(team: str):
    """
    Get explanations for all recommendations for a team
    """
    try:
        # Load recommendations
        franchise_recs = pd.read_csv(
            BASE_DIR / "reports" / "recommendations" / "franchise_recommendations.csv"
        )

        team_recs = franchise_recs[
            franchise_recs["team"].str.lower() == team.lower()
        ]

        if team_recs.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No recommendations found for {team}"
            )

        # Get explanations for all recommendations
        explanations = []
        for _, rec in team_recs.iterrows():
            try:
                exp_response = await explain_recommendation(
                    team,
                    rec["domestic_player"]
                )
                explanations.append(exp_response)
            except:
                continue

        return {
            "team": team,
            "total_recommendations": len(explanations),
            "recommendations": explanations
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting team explanations: {str(e)}"
        )
