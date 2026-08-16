from __future__ import annotations

from app.core import loaders


def refresh_data_caches():
    loaders.clear_caches()
    return {"status": "ok", "message": "Data caches refreshed successfully."}


def get_system_summary():
    recommendations = loaders.load_recommendations()
    teams = loaders.load_team_summary()
    current_squad = loaders.load_current_squad()
    best_models = loaders.load_best_models()
    ensemble_models = loaders.load_ensemble_model_comparison()
    final_model_registry = loaders.load_runtime_model_registry()
    time_based = loaders.load_time_based_metrics()
    recommendation_qa = loaders.load_recommendation_qa_summary()
    candidate_pool = loaders.load_candidate_pool_summary()
    production_models = [
        loaders.get_production_model_metadata(domain)
        for domain in ["batting", "bowling", "allrounder"]
    ]

    qa_records = recommendation_qa.to_dict(orient="records")
    passed_checks = sum(1 for row in qa_records if str(row.get("status", "")).lower() == "pass")

    return {
        "status": "ok",
        "teams_count": int(teams["team"].nunique()) if not teams.empty and "team" in teams.columns else 0,
        "recommendation_rows": int(len(recommendations)),
        "current_squad_rows": int(len(current_squad)),
        "production_models": production_models,
        "best_models": best_models.to_dict(orient="records"),
        "ensemble_model_comparison": ensemble_models.to_dict(orient="records"),
        "final_model_registry": final_model_registry.to_dict(orient="records"),
        "time_based_metrics": time_based.to_dict(orient="records"),
        "recommendation_qa": qa_records,
        "candidate_pool": candidate_pool.to_dict(orient="records"),
        "qa_summary": {
            "total_checks": len(qa_records),
            "passed_checks": passed_checks,
            "failed_checks": max(len(qa_records) - passed_checks, 0),
        },
    }
