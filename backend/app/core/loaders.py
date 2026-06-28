from __future__ import annotations

from functools import lru_cache

import pandas as pd

from app.core.config import ML_REPORT_DIR, PROCESSED_DIR, RECOMMENDATION_DIR, TESTING_DIR


def _read_csv(path):
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


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


def clear_caches():
    for fn in [
        load_recommendations,
        load_team_summary,
        load_squad_gaps,
        load_replacements,
        load_underperformers,
        load_similarity,
        load_current_squad,
        load_player_master,
        load_model_comparison,
        load_best_models,
        load_random_split_confusion,
        load_time_based_metrics,
        load_time_based_confusion,
        load_evaluation_comparison,
    ]:
        fn.cache_clear()
