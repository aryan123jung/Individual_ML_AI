from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from recommendation_engine.config import (
    ALLROUNDER_FEATURES,
    BATTING_FEATURES,
    BOWLING_FEATURES,
    PROCESSED_DIR,
)


BASE_DIR = Path(__file__).resolve().parents[3]
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports" / "ml"


def _load_frame(filename: str) -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / filename)


def _domain_config():
    return {
        "batting": {
            "historical_file": "ipl_batting_features_v2.csv",
            "candidate_file": "smat_batting_features_v2.csv",
            "features": BATTING_FEATURES,
            "role_column": "batting_role",
        },
        "bowling": {
            "historical_file": "ipl_bowling_features_v2.csv",
            "candidate_file": "smat_bowling_features_v2.csv",
            "features": BOWLING_FEATURES,
            "role_column": "bowling_role",
        },
        "allrounder": {
            "historical_file": "ipl_allrounder_features_v2.csv",
            "candidate_file": "smat_allrounder_features_v2.csv",
            "features": ALLROUNDER_FEATURES,
            "role_column": "allrounder_role",
        },
    }


def _eligible_frame(domain_name: str, df: pd.DataFrame) -> pd.DataFrame:
    if domain_name in {"batting", "bowling"} and "eligible_for_model" in df.columns:
        return df[df["eligible_for_model"]].copy()
    return df.copy()


def _build_preprocessor(feature_columns: list[str]) -> ColumnTransformer:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[("numeric", numeric, feature_columns)],
        remainder="drop",
    )


def _best_kmeans(X_transformed):
    best_model = None
    best_score = -1.0
    best_k = None

    upper_k = min(6, max(3, len(X_transformed) // 25))
    for k in range(3, upper_k + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(X_transformed)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(X_transformed, labels)
        if score > best_score:
            best_model = model
            best_score = score
            best_k = k

    return best_model, best_k, round(best_score, 4) if best_score >= 0 else None


def _cluster_role_profiles(domain_name: str, historical: pd.DataFrame, role_column: str) -> tuple[pd.DataFrame, dict]:
    profile_rows = []
    label_map = {}

    for cluster_id, group in historical.groupby("cluster_id"):
        role_counts = group[role_column].fillna("unknown").value_counts()
        top_role = role_counts.index[0] if not role_counts.empty else "unknown"
        top_role_share = float(role_counts.iloc[0] / len(group)) if len(group) else 0.0
        cluster_label = f"{domain_name}_{top_role}_cluster_{int(cluster_id)}"
        label_map[cluster_id] = cluster_label
        profile_rows.append(
            {
                "domain": domain_name,
                "cluster_id": int(cluster_id),
                "cluster_label": cluster_label,
                "dominant_role": top_role,
                "dominant_role_share": round(top_role_share * 100, 2),
                "players_in_cluster": len(group),
            }
        )

    return pd.DataFrame(profile_rows), label_map


def train_domain_clustering(domain_name: str, config: dict):
    historical = _eligible_frame(domain_name, _load_frame(config["historical_file"]))
    candidates = _eligible_frame(domain_name, _load_frame(config["candidate_file"]))
    feature_columns = config["features"]
    role_column = config["role_column"]

    preprocessor = _build_preprocessor(feature_columns)
    X_hist = preprocessor.fit_transform(historical[feature_columns])
    model, best_k, silhouette = _best_kmeans(X_hist)

    historical_out = historical.copy()
    candidates_out = candidates.copy()
    historical_out["cluster_id"] = model.predict(X_hist)
    candidates_out["cluster_id"] = model.predict(preprocessor.transform(candidates[feature_columns]))

    cluster_profiles, label_map = _cluster_role_profiles(domain_name, historical_out, role_column)
    historical_out["cluster_label"] = historical_out["cluster_id"].map(label_map)
    historical_out["archetype_role"] = historical_out["cluster_id"].map(
        cluster_profiles.set_index("cluster_id")["dominant_role"]
    )
    candidates_out["cluster_label"] = candidates_out["cluster_id"].map(label_map)
    candidates_out["archetype_role"] = candidates_out["cluster_id"].map(
        cluster_profiles.set_index("cluster_id")["dominant_role"]
    )
    candidates_out["archetype_role_match_score"] = (
        (candidates_out[role_column] == candidates_out["archetype_role"]).astype(float) * 100
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"preprocessor": preprocessor, "model": model, "feature_columns": feature_columns},
        MODEL_DIR / f"{domain_name}_clustering_model.joblib",
    )

    historical_out.to_csv(PROCESSED_DIR / f"ipl_{domain_name}_clusters.csv", index=False)
    candidates_out.to_csv(PROCESSED_DIR / f"smat_{domain_name}_clusters.csv", index=False)

    metrics = pd.DataFrame(
        [
            {
                "domain": domain_name,
                "selected_clusters": best_k,
                "silhouette_score": silhouette,
                "historical_rows": len(historical_out),
                "candidate_rows": len(candidates_out),
            }
        ]
    )

    return metrics, cluster_profiles, candidates_out


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    metrics_rows = []
    profile_rows = []
    summary_rows = []

    for domain_name, config in _domain_config().items():
        metrics, profiles, candidates = train_domain_clustering(domain_name, config)
        metrics_rows.append(metrics)
        profile_rows.append(profiles)
        summary_rows.append(
            pd.DataFrame(
                [
                    {
                        "domain": domain_name,
                        "clusters_created": int(metrics.iloc[0]["selected_clusters"]),
                        "top_candidate": candidates.sort_values(
                            "archetype_role_match_score", ascending=False
                        )["player"].iloc[0],
                    }
                ]
            )
        )

    metrics_df = pd.concat(metrics_rows, ignore_index=True)
    profiles_df = pd.concat(profile_rows, ignore_index=True)
    summary_df = pd.concat(summary_rows, ignore_index=True)

    metrics_path = REPORT_DIR / "clustering_metrics.csv"
    profiles_path = REPORT_DIR / "cluster_profiles.csv"
    summary_path = REPORT_DIR / "cluster_candidate_summary.csv"

    metrics_df.to_csv(metrics_path, index=False)
    profiles_df.to_csv(profiles_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("Player clustering complete:")
    print(f"- {metrics_path} {metrics_df.shape}")
    print(f"- {profiles_path} {profiles_df.shape}")
    print(f"- {summary_path} {summary_df.shape}")


if __name__ == "__main__":
    main()
