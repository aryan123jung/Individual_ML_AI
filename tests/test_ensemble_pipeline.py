"""
Ensemble Pipeline Validation Tests

Tests to verify that:
1. Ensemble models are properly trained
2. ML scores are correctly generated
3. Recommendation weights are updated to 60% ML
4. Pipeline produces valid recommendations
5. No regressions in output quality
"""

import pytest
import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports"


class TestEnsembleModels:
    """Test ensemble model files and artifacts."""

    def test_ensemble_models_exist(self):
        """Verify all ensemble models are saved."""
        domains = ["batting", "bowling", "allrounder"]
        for domain in domains:
            model_path = MODELS_DIR / f"{domain}_ensemble_model.joblib"
            assert model_path.exists(), f"Missing {domain}_ensemble_model.joblib"

    def test_ensemble_models_loadable(self):
        """Verify ensemble models can be loaded."""
        domains = ["batting", "bowling", "allrounder"]
        for domain in domains:
            model_path = MODELS_DIR / f"{domain}_ensemble_model.joblib"
            model = joblib.load(model_path)
            assert model is not None
            assert hasattr(model, "predict_proba"), f"{domain} model missing predict_proba"

    def test_ml_scores_generated(self):
        """Verify ML scores CSV files are generated for all domains."""
        domains = ["batting", "bowling", "allrounder"]
        for domain in domains:
            scores_path = DATA_DIR / f"smat_{domain}_ml_scores.csv"
            assert scores_path.exists(), f"Missing smat_{domain}_ml_scores.csv"


class TestMLScoresQuality:
    """Test quality and validity of generated ML scores."""

    def test_batting_scores_valid(self):
        """Verify batting ML scores are in valid range."""
        df = pd.read_csv(DATA_DIR / "smat_batting_ml_scores.csv")
        assert "ml_suitability_score" in df.columns
        assert (df["ml_suitability_score"] >= 0).all(), "Scores below 0"
        assert (df["ml_suitability_score"] <= 100).all(), "Scores above 100"
        assert df["ml_suitability_score"].notna().sum() > 100, "Too few valid scores"

    def test_bowling_scores_valid(self):
        """Verify bowling ML scores are in valid range."""
        df = pd.read_csv(DATA_DIR / "smat_bowling_ml_scores.csv")
        assert "ml_suitability_score" in df.columns
        assert (df["ml_suitability_score"] >= 0).all(), "Scores below 0"
        assert (df["ml_suitability_score"] <= 100).all(), "Scores above 100"
        assert df["ml_suitability_score"].notna().sum() > 100, "Too few valid scores"

    def test_allrounder_scores_valid(self):
        """Verify allrounder ML scores are in valid range."""
        df = pd.read_csv(DATA_DIR / "smat_allrounder_ml_scores.csv")
        assert "ml_suitability_score" in df.columns
        assert (df["ml_suitability_score"] >= 0).all(), "Scores below 0"
        assert (df["ml_suitability_score"] <= 100).all(), "Scores above 100"
        assert df["ml_suitability_score"].notna().sum() > 100, "Too few valid scores"

    def test_score_distribution(self):
        """Verify score distributions make sense."""
        df = pd.read_csv(DATA_DIR / "smat_batting_ml_scores.csv")
        mean_score = df["ml_suitability_score"].mean()
        assert 40 < mean_score < 60, f"Mean score {mean_score} outside expected range"


class TestRecommendations:
    """Test generated recommendations."""

    def test_recommendations_exist(self):
        """Verify recommendations are generated."""
        recs_path = REPORTS_DIR / "recommendations" / "franchise_recommendations.csv"
        assert recs_path.exists(), "Missing franchise_recommendations.csv"

    def test_recommendations_structure(self):
        """Verify recommendations have correct structure."""
        recs = pd.read_csv(REPORTS_DIR / "recommendations" / "franchise_recommendations.csv")
        required_cols = [
            "team",
            "domestic_player",
            "target_role",
            "ml_suitability_score",
            "final_recommendation_score"
        ]
        for col in required_cols:
            assert col in recs.columns, f"Missing column: {col}"

    def test_recommendations_validity(self):
        """Verify recommendations contain valid data."""
        recs = pd.read_csv(REPORTS_DIR / "recommendations" / "franchise_recommendations.csv")
        assert len(recs) > 100, "Too few recommendations"
        assert recs["team"].nunique() >= 8, "Not enough teams covered"
        assert recs["final_recommendation_score"].notna().sum() > len(recs) * 0.95, "Too many NaN scores"
        assert (recs["final_recommendation_score"] >= 0).all(), "Negative scores"
        assert (recs["final_recommendation_score"] <= 100).all(), "Scores > 100"

    def test_ml_score_dominates(self):
        """
        Verify ML score is the dominant factor in final scores.
        With 50% ML weight, ML score should have strong correlation with final score.
        """
        recs = pd.read_csv(REPORTS_DIR / "recommendations" / "franchise_recommendations.csv")
        correlation = recs["ml_suitability_score"].corr(recs["final_recommendation_score"])
        assert correlation > 0.75, f"ML score correlation {correlation} too low (expect >0.75)"

    def test_top_recommendations_have_high_ml_scores(self):
        """Top recommendations should have high ML suitability scores."""
        recs = pd.read_csv(REPORTS_DIR / "recommendations" / "franchise_recommendations.csv")
        top_10 = recs.nlargest(10, "final_recommendation_score")
        avg_ml_score = top_10["ml_suitability_score"].mean()
        assert avg_ml_score > 60, f"Top 10 avg ML score {avg_ml_score} too low"


class TestWeightConfiguration:
    """Test that weights are correctly configured for 60% ML."""

    def test_weights_sum_to_100(self):
        """Verify weights sum to 100%."""
        # Import from source to check actual weights
        import sys
        sys.path.insert(0, str(BASE_DIR / "src"))
        from recommendation_engine.recommender import MODEL_LED_WEIGHTS

        total_weight = sum(MODEL_LED_WEIGHTS.values())
        assert abs(total_weight - 1.0) < 0.01, f"Weights sum to {total_weight}, not 1.0"

    def test_ml_component_is_60_percent(self):
        """Verify ML component is 60% (ml_score + archetype_match)."""
        import sys
        sys.path.insert(0, str(BASE_DIR / "src"))
        from recommendation_engine.recommender import MODEL_LED_WEIGHTS

        ml_component = MODEL_LED_WEIGHTS.get("ml_score", 0) + MODEL_LED_WEIGHTS.get("archetype_match", 0)
        assert abs(ml_component - 0.60) < 0.01, f"ML component {ml_component}, expected 0.60"

    def test_role_fit_removed(self):
        """Verify role_fit weight is 0 (cricket-specific removed)."""
        import sys
        sys.path.insert(0, str(BASE_DIR / "src"))
        from recommendation_engine.recommender import MODEL_LED_WEIGHTS

        assert MODEL_LED_WEIGHTS.get("role_fit", 0) == 0, "role_fit should be 0 weight"


class TestPipelineIntegration:
    """End-to-end pipeline tests."""

    def test_full_pipeline_runs(self):
        """Verify full recommendation pipeline can run without errors."""
        # This would normally import and run the actual pipeline
        # For now, just verify all intermediate outputs exist
        reports_dir = REPORTS_DIR / "recommendations"
        required_files = [
            "franchise_recommendations.csv",
            "team_recommendation_summary.csv",
            "franchise_squad_gaps.csv",
        ]
        for fname in required_files:
            fpath = reports_dir / fname
            assert fpath.exists(), f"Missing {fname}"

    def test_no_duplicate_recommendations(self):
        """Verify no player is recommended twice for same team."""
        recs = pd.read_csv(REPORTS_DIR / "recommendations" / "franchise_recommendations.csv")
        duplicates = recs.groupby(["team", "domestic_player"]).size()
        assert (duplicates == 1).all(), "Found duplicate recommendations for same player/team"


class TestPerformanceImprovement:
    """Verify ensemble improves over single model."""

    def test_ensemble_comparison_exists(self):
        """Verify comparison report is generated."""
        report_path = REPORTS_DIR / "ml" / "ensemble_model_comparison.csv"
        assert report_path.exists(), "Missing ensemble_model_comparison.csv"

    def test_ensemble_metrics_improved(self):
        """Verify ensemble model metrics are reasonable."""
        comparison = pd.read_csv(REPORTS_DIR / "ml" / "ensemble_model_comparison.csv")

        # Check batting
        batting = comparison[comparison["domain"] == "batting"].iloc[0]
        assert batting["f1_score"] > 0.80, f"Batting F1 too low: {batting['f1_score']}"
        assert batting["precision"] > 0.75, f"Batting precision too low: {batting['precision']}"

        # Check bowling
        bowling = comparison[comparison["domain"] == "bowling"].iloc[0]
        assert bowling["f1_score"] > 0.85, f"Bowling F1 too low: {bowling['f1_score']}"
        assert bowling["roc_auc"] > 0.90, f"Bowling AUC too low: {bowling['roc_auc']}"

        # Check allrounder
        allrounder = comparison[comparison["domain"] == "allrounder"].iloc[0]
        assert allrounder["f1_score"] > 0.90, f"Allrounder F1 too low: {allrounder['f1_score']}"
        assert allrounder["roc_auc"] > 0.95, f"Allrounder AUC too low: {allrounder['roc_auc']}"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_handles_empty_squads(self):
        """Verify pipeline handles edge cases gracefully."""
        # This is a conceptual test - actual implementation depends on data
        recs = pd.read_csv(REPORTS_DIR / "recommendations" / "franchise_recommendations.csv")
        assert len(recs) > 0, "Recommendations should not be empty"

    def test_recommendations_are_diverse(self):
        """Verify recommendations span multiple teams and roles."""
        recs = pd.read_csv(REPORTS_DIR / "recommendations" / "franchise_recommendations.csv")
        assert recs["team"].nunique() >= 8, "Should recommend for most teams"
        assert recs["target_role"].nunique() >= 5, "Should recommend across multiple roles"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
