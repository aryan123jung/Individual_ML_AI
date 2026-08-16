from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import HTTPException

from app.core import loaders


DOMAIN_CONFIG = {
    "batting": {
        "feature_loader": loaders.load_smat_batting_features,
        "role_column": "batting_role",
    },
    "bowling": {
        "feature_loader": loaders.load_smat_bowling_features,
        "role_column": "bowling_role",
    },
    "allrounder": {
        "feature_loader": loaders.load_smat_allrounder_features,
        "role_column": "allrounder_role",
    },
}


def _get_domain_config(domain: str) -> dict[str, Any]:
    if domain not in DOMAIN_CONFIG:
        raise HTTPException(status_code=404, detail=f"Unsupported domain: {domain}")
    return DOMAIN_CONFIG[domain]


def _cluster_metadata(domain: str, cluster_id: int, role_value: str | None):
    cluster_bundle = loaders.load_clustering_model(domain)
    label = f"{domain}_{role_value or 'unknown'}_cluster_{cluster_id}"
    return {
        "cluster_id": int(cluster_id),
        "cluster_label": label,
        "archetype_role": role_value,
        "clustering_feature_columns": cluster_bundle["feature_columns"],
    }


def _json_safe_dict(record: dict[str, Any]) -> dict[str, Any]:
    safe = {}
    for key, value in record.items():
        if pd.isna(value):
            safe[key] = None
        else:
            safe[key] = value.item() if hasattr(value, "item") else value
    return safe


def _predict(domain: str, row_df: pd.DataFrame, role_value: str | None, player_name: str | None = None):
    model = loaders.load_suitability_model(domain)
    clustering_bundle = loaders.load_clustering_model(domain)
    model_meta = loaders.get_production_model_metadata(domain)

    probability = float(model.predict_proba(row_df)[0][1])
    suitability_score = round(probability * 100, 2)

    cluster_features = clustering_bundle["feature_columns"]
    cluster_input = row_df[cluster_features].copy()
    cluster_id = int(clustering_bundle["model"].predict(clustering_bundle["preprocessor"].transform(cluster_input))[0])
    cluster_meta = _cluster_metadata(domain, cluster_id, role_value)

    return {
        "domain": domain,
        "player": player_name,
        "suitability_score": suitability_score,
        "predicted_probability": round(probability, 4),
        "cluster_id": cluster_meta["cluster_id"],
        "cluster_label": cluster_meta["cluster_label"],
        "archetype_role": cluster_meta["archetype_role"],
        "model_family": model_meta["model_family"],
        "model_name": model_meta["model_name"],
        "model_artifact": model_meta["artifact"],
        "feature_values": _json_safe_dict(row_df.iloc[0].to_dict()),
    }


def available_domains():
    return sorted(DOMAIN_CONFIG.keys())


def infer_existing_player(domain: str, player_name: str):
    config = _get_domain_config(domain)
    df = config["feature_loader"]()
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No feature data found for domain: {domain}")

    matches = df[df["player"].str.lower() == player_name.lower()].copy()
    if matches.empty:
        raise HTTPException(status_code=404, detail=f"Player not found in {domain} features: {player_name}")

    row = matches.iloc[[0]].copy()
    role_value = row.iloc[0].get(config["role_column"])
    feature_columns = loaders.load_clustering_model(domain)["feature_columns"]
    model_feature_columns = row.select_dtypes(include=["number", "bool"]).columns.tolist()
    # Keep only columns accepted by the pipeline by reading them from the clustering model plus known numeric frame;
    # suitability pipeline ignores extras after ColumnTransformer column selection.
    input_df = row[model_feature_columns].copy()
    return _predict(domain, input_df, role_value, player_name=player_name)


def infer_custom_player(domain: str, feature_values: dict[str, float]):
    config = _get_domain_config(domain)
    cluster_features = loaders.load_clustering_model(domain)["feature_columns"]
    required_features = sorted(set(cluster_features) | set(feature_values.keys()))
    row = {feature: float(feature_values.get(feature, 0.0)) for feature in required_features}
    input_df = pd.DataFrame([row])
    return _predict(domain, input_df, role_value=None, player_name=None)
