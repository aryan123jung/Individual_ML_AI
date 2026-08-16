from __future__ import annotations

from functools import lru_cache

import joblib
import pandas as pd

from app.core.config import ML_REPORT_DIR, MODEL_DIR, PROCESSED_DIR, RECOMMENDATION_DIR, TESTING_DIR


def _read_csv(path):
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _preferred_model_path(domain: str):
    ensemble_path = MODEL_DIR / f"{domain}_ensemble_model.joblib"
    if ensemble_path.exists():
        return ensemble_path
    return MODEL_DIR / f"{domain}_suitability_model.joblib"


def get_production_model_metadata(domain: str) -> dict[str, str]:
    path = _preferred_model_path(domain)
    model_family = "ensemble" if path.name.endswith("_ensemble_model.joblib") else "legacy_single_model"
    model_name = "Ensemble (RF + XGBoost + LR)" if model_family == "ensemble" else "Legacy suitability model"
    return {
        "domain": domain,
        "model_family": model_family,
        "model_name": model_name,
        "artifact": path.name,
    }


@lru_cache(maxsize=1)
def load_recommendations():
    return _read_csv(RECOMMENDATION_DIR / "franchise_recommendations.csv")


@lru_cache(maxsize=1)
def load_team_summary():
    return _read_csv(RECOMMENDATION_DIR / "team_recommendation_summary.csv")


@lru_cache(maxsize=1)
def load_squad_gaps():
    return _read_csv(RECOMMENDATION_DIR / "franchise_squad_gaps.csv")


@lru_cache(maxsize=1)
def load_replacements():
    return _read_csv(RECOMMENDATION_DIR / "replacement_recommendations.csv")


@lru_cache(maxsize=1)
def load_underperformers():
    return _read_csv(RECOMMENDATION_DIR / "current_squad_underperformers.csv")


@lru_cache(maxsize=1)
def load_similarity():
    return _read_csv(RECOMMENDATION_DIR / "smat_ipl_similarity.csv")


@lru_cache(maxsize=1)
def load_role_benchmarks():
    return _read_csv(RECOMMENDATION_DIR / "ipl_role_benchmarks.csv")


@lru_cache(maxsize=1)
def load_candidate_pool_summary():
    return _read_csv(RECOMMENDATION_DIR / "domestic_candidate_pool_summary.csv")


@lru_cache(maxsize=1)
def load_recommendation_qa_summary():
    return _read_csv(RECOMMENDATION_DIR / "qa" / "recommendation_qa_summary.csv")


@lru_cache(maxsize=1)
def load_current_squad():
    return _read_csv(PROCESSED_DIR / "ipl_2026_current_squad.csv")


@lru_cache(maxsize=1)
def load_player_master():
    return _read_csv(PROCESSED_DIR / "ipl_2026_player_master.csv")


@lru_cache(maxsize=1)
def load_model_comparison():
    return _read_csv(ML_REPORT_DIR / "model_comparison.csv")


@lru_cache(maxsize=1)
def load_best_models():
    return _read_csv(ML_REPORT_DIR / "best_models.csv")


@lru_cache(maxsize=1)
def load_ensemble_model_comparison():
    return _read_csv(ML_REPORT_DIR / "ensemble_model_comparison.csv")


@lru_cache(maxsize=1)
def load_final_model_registry():
    return _read_csv(ML_REPORT_DIR / "final_model_registry.csv")


@lru_cache(maxsize=1)
def load_runtime_model_registry():
    registry = load_final_model_registry().copy()
    if registry.empty:
        rows = []
        for domain in ["batting", "bowling", "allrounder"]:
            rows.append(get_production_model_metadata(domain))
        return pd.DataFrame(rows)

    runtime_rows = []
    for _, row in registry.iterrows():
        domain = str(row.get("domain", "")).strip().lower()
        if not domain:
            continue
        model_meta = get_production_model_metadata(domain)
        merged = row.to_dict()
        merged["production_model"] = model_meta["model_name"]
        merged["model_family"] = model_meta["model_family"]
        merged["model_artifact"] = model_meta["artifact"]
        runtime_rows.append(merged)
    return pd.DataFrame(runtime_rows)


@lru_cache(maxsize=1)
def load_random_split_confusion():
    return _read_csv(TESTING_DIR / "random_split" / "confusion_matrices.csv")


@lru_cache(maxsize=1)
def load_time_based_metrics():
    return _read_csv(TESTING_DIR / "time_based_validation" / "time_based_model_metrics.csv")


@lru_cache(maxsize=1)
def load_time_based_confusion():
    return _read_csv(TESTING_DIR / "time_based_validation" / "time_based_confusion_matrices.csv")


@lru_cache(maxsize=1)
def load_evaluation_comparison():
    return _read_csv(TESTING_DIR / "evaluation_mode_comparison.csv")


@lru_cache(maxsize=1)
def load_smat_batting_features():
    return _read_csv(PROCESSED_DIR / "smat_batting_features_v2.csv")


@lru_cache(maxsize=1)
def load_smat_bowling_features():
    return _read_csv(PROCESSED_DIR / "smat_bowling_features_v2.csv")


@lru_cache(maxsize=1)
def load_smat_allrounder_features():
    return _read_csv(PROCESSED_DIR / "smat_allrounder_features_v2.csv")


@lru_cache(maxsize=1)
def load_ipl_batting_features():
    return _read_csv(PROCESSED_DIR / "ipl_batting_features_v2.csv")


@lru_cache(maxsize=1)
def load_ipl_bowling_features():
    return _read_csv(PROCESSED_DIR / "ipl_bowling_features_v2.csv")


@lru_cache(maxsize=1)
def load_ipl_allrounder_features():
    return _read_csv(PROCESSED_DIR / "ipl_allrounder_features_v2.csv")


@lru_cache(maxsize=1)
def load_ipl_batting_raw():
    return _read_csv(PROCESSED_DIR / "ipl_batting_raw.csv")


@lru_cache(maxsize=1)
def load_ipl_bowling_raw():
    return _read_csv(PROCESSED_DIR / "ipl_bowling_raw.csv")


@lru_cache(maxsize=None)
def load_suitability_model(domain: str):
    return joblib.load(_preferred_model_path(domain))


@lru_cache(maxsize=None)
def load_legacy_suitability_model(domain: str):
    return joblib.load(MODEL_DIR / f"{domain}_suitability_model.joblib")


@lru_cache(maxsize=None)
def load_ensemble_model(domain: str):
    return joblib.load(MODEL_DIR / f"{domain}_ensemble_model.joblib")


@lru_cache(maxsize=None)
def load_clustering_model(domain: str):
    return joblib.load(MODEL_DIR / f"{domain}_clustering_model.joblib")


def clear_caches():
    for fn in [
        load_recommendations,
        load_team_summary,
        load_squad_gaps,
        load_replacements,
        load_underperformers,
        load_similarity,
        load_role_benchmarks,
        load_candidate_pool_summary,
        load_recommendation_qa_summary,
        load_current_squad,
        load_player_master,
        load_model_comparison,
        load_best_models,
        load_ensemble_model_comparison,
        load_final_model_registry,
        load_runtime_model_registry,
        load_random_split_confusion,
        load_time_based_metrics,
        load_time_based_confusion,
        load_evaluation_comparison,
        load_smat_batting_features,
        load_smat_bowling_features,
        load_smat_allrounder_features,
        load_ipl_batting_features,
        load_ipl_bowling_features,
        load_ipl_allrounder_features,
        load_ipl_batting_raw,
        load_ipl_bowling_raw,
    ]:
        fn.cache_clear()
    load_suitability_model.cache_clear()
    load_legacy_suitability_model.cache_clear()
    load_ensemble_model.cache_clear()
    load_clustering_model.cache_clear()
