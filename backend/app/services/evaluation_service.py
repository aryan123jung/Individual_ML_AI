from __future__ import annotations

from app.core import loaders


def get_model_metrics():
    return {
        "model_comparison": loaders.load_model_comparison().to_dict(orient="records"),
        "best_models": loaders.load_best_models().to_dict(orient="records"),
        "random_split_confusion": loaders.load_random_split_confusion().to_dict(orient="records"),
        "time_based_metrics": loaders.load_time_based_metrics().to_dict(orient="records"),
        "time_based_confusion": loaders.load_time_based_confusion().to_dict(orient="records"),
        "evaluation_mode_comparison": loaders.load_evaluation_comparison().to_dict(orient="records"),
    }
