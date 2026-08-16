#!/usr/bin/env python3
"""
Ensemble Pipeline Validation Script

Validates that:
1. Ensemble models are trained and saved
2. ML scores are valid and updated
3. Recommendations are generated with 60% ML weighting
4. All outputs meet quality standards
"""

import sys
from pathlib import Path
import pandas as pd
import joblib

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports"

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'

test_passed = 0
test_failed = 0


def test(name, condition, error_msg=""):
    """Simple test function."""
    global test_passed, test_failed
    if condition:
        print(f"{GREEN}✓{RESET} {name}")
        test_passed += 1
    else:
        print(f"{RED}✗{RESET} {name}")
        if error_msg:
            print(f"  {RED}Error: {error_msg}{RESET}")
        test_failed += 1


def main():
    """Run all validation tests."""
    global test_passed, test_failed

    print(f"\n{BOLD}{BLUE}{'='*70}{RESET}")
    print(f"{BOLD}{BLUE}ENSEMBLE PIPELINE VALIDATION{RESET}")
    print(f"{BOLD}{BLUE}{'='*70}{RESET}\n")

    # 1. Test ensemble models exist
    print(f"{BOLD}1. Ensemble Models{RESET}")
    for domain in ["batting", "bowling", "allrounder"]:
        model_path = MODELS_DIR / f"{domain}_ensemble_model.joblib"
        test(f"  {domain.capitalize()} model exists", model_path.exists())

    # 2. Test models are loadable
    print(f"\n{BOLD}2. Model Loading{RESET}")
    for domain in ["batting", "bowling", "allrounder"]:
        try:
            model_path = MODELS_DIR / f"{domain}_ensemble_model.joblib"
            model = joblib.load(model_path)
            has_predict = hasattr(model, "predict_proba")
            test(f"  {domain.capitalize()} model loads and has predict_proba", has_predict)
        except Exception as e:
            test(f"  {domain.capitalize()} model loads", False, str(e))

    # 3. Test ML scores files exist
    print(f"\n{BOLD}3. ML Scores Generation{RESET}")
    for domain in ["batting", "bowling", "allrounder"]:
        scores_path = DATA_DIR / f"smat_{domain}_ml_scores.csv"
        test(f"  {domain.capitalize()} ML scores generated", scores_path.exists())

    # 4. Test ML scores quality
    print(f"\n{BOLD}4. ML Scores Quality{RESET}")
    for domain in ["batting", "bowling", "allrounder"]:
        try:
            df = pd.read_csv(DATA_DIR / f"smat_{domain}_ml_scores.csv")
            has_col = "ml_suitability_score" in df.columns
            test(f"  {domain.capitalize()} has ml_suitability_score column", has_col)

            if has_col:
                valid_range = (df["ml_suitability_score"] >= 0).all() and (df["ml_suitability_score"] <= 100).all()
                test(f"  {domain.capitalize()} scores in range [0-100]", valid_range)

                num_valid = df["ml_suitability_score"].notna().sum()
                test(f"  {domain.capitalize()} has {num_valid} valid scores", num_valid > 100, f"Only {num_valid} scores")
        except Exception as e:
            test(f"  {domain.capitalize()} scores valid", False, str(e))

    # 5. Test recommendations exist
    print(f"\n{BOLD}5. Recommendations Generation{RESET}")
    recs_path = REPORTS_DIR / "recommendations" / "franchise_recommendations.csv"
    test(f"  Recommendations file exists", recs_path.exists())

    # 6. Test recommendations structure
    print(f"\n{BOLD}6. Recommendations Structure{RESET}")
    try:
        recs = pd.read_csv(recs_path)
        required_cols = ["team", "domestic_player", "target_role", "ml_suitability_score", "final_recommendation_score"]
        for col in required_cols:
            test(f"  Has '{col}' column", col in recs.columns)

        # Test data validity
        print(f"\n{BOLD}7. Recommendations Data Quality{RESET}")
        test(f"  Has {len(recs)} recommendations", len(recs) > 100, f"Only {len(recs)} recs")
        test(f"  Covers {recs['team'].nunique()} teams", recs["team"].nunique() >= 8, f"Only {recs['team'].nunique()} teams")
        test(f"  Covers {recs['target_role'].nunique()} roles", recs["target_role"].nunique() >= 5, f"Only {recs['target_role'].nunique()} roles")

        # Test score ranges
        all_valid = recs["final_recommendation_score"].notna().all()
        test(f"  All final scores are valid (not NaN)", all_valid)

        score_range = (recs["final_recommendation_score"] >= 0).all() and (recs["final_recommendation_score"] <= 100).all()
        test(f"  Final scores in range [0-100]", score_range)

        # Test ML score dominance
        correlation = recs["ml_suitability_score"].corr(recs["final_recommendation_score"])
        test(f"  ML score correlates with final score (r={correlation:.3f})", correlation > 0.70, f"Correlation only {correlation:.3f}")

        # Test top recommendations have high ML scores
        top_10_ml = recs.nlargest(10, "final_recommendation_score")["ml_suitability_score"].mean()
        test(f"  Top 10 recommendations avg ML score: {top_10_ml:.1f}", top_10_ml > 55, f"ML score {top_10_ml:.1f}")

    except Exception as e:
        test(f"  Recommendations valid", False, str(e))

    # 7. Test weight configuration
    print(f"\n{BOLD}8. Weight Configuration (60% ML){RESET}")
    try:
        sys.path.insert(0, str(BASE_DIR / "src"))
        from recommendation_engine.recommender import MODEL_LED_WEIGHTS

        total = sum(MODEL_LED_WEIGHTS.values())
        test(f"  Weights sum to 1.0 (got {total:.3f})", abs(total - 1.0) < 0.01)

        ml_component = MODEL_LED_WEIGHTS.get("ml_score", 0) + MODEL_LED_WEIGHTS.get("archetype_match", 0)
        test(f"  ML component is 60% (got {ml_component*100:.0f}%)", abs(ml_component - 0.60) < 0.01, f"Got {ml_component*100:.1f}%")

        role_fit = MODEL_LED_WEIGHTS.get("role_fit", 0)
        test(f"  Role fit weight is 0 (cricket-specific removed)", role_fit == 0, f"Got {role_fit}")

        # Print weight breakdown
        print(f"\n  {BOLD}Weight Breakdown:{RESET}")
        for key, val in sorted(MODEL_LED_WEIGHTS.items(), key=lambda x: x[1], reverse=True):
            if val > 0:
                print(f"    - {key}: {val*100:5.1f}%")

    except Exception as e:
        test(f"  Weight configuration valid", False, str(e))

    # 8. Test performance metrics
    print(f"\n{BOLD}9. Ensemble Performance Metrics{RESET}")
    try:
        metrics_path = REPORTS_DIR / "ml" / "ensemble_model_comparison.csv"
        metrics = pd.read_csv(metrics_path)

        print(f"\n  {BOLD}Domain Metrics:{RESET}")
        for idx, row in metrics.iterrows():
            domain = row["domain"]
            f1 = row["f1_score"]
            precision = row["precision"]
            recall = row["recall"]
            auc = row["roc_auc"]
            print(f"    {domain.upper():12} F1={f1:.4f} Precision={precision:.4f} Recall={recall:.4f} AUC={auc:.4f}")

        # Test minimum thresholds
        batting = metrics[metrics["domain"] == "batting"].iloc[0]
        test(f"  Batting F1 >= 0.80", batting["f1_score"] >= 0.80, f"Got {batting['f1_score']:.4f}")

        bowling = metrics[metrics["domain"] == "bowling"].iloc[0]
        test(f"  Bowling F1 >= 0.85", bowling["f1_score"] >= 0.85, f"Got {bowling['f1_score']:.4f}")

        allrounder = metrics[metrics["domain"] == "allrounder"].iloc[0]
        test(f"  Allrounder F1 >= 0.90", allrounder["f1_score"] >= 0.90, f"Got {allrounder['f1_score']:.4f}")

    except Exception as e:
        test(f"  Performance metrics valid", False, str(e))

    # Summary
    print(f"\n{BOLD}{BLUE}{'='*70}{RESET}")
    print(f"{BOLD}VALIDATION SUMMARY{RESET}")
    print(f"{BOLD}{BLUE}{'='*70}{RESET}")
    total_tests = test_passed + test_failed
    print(f"{GREEN}{test_passed}/{total_tests}{RESET} tests passed")

    if test_failed > 0:
        print(f"{RED}{test_failed} tests failed{RESET}\n")
        return 1
    else:
        print(f"\n{GREEN}{BOLD}✓ ALL VALIDATIONS PASSED - ENSEMBLE PIPELINE READY FOR PRODUCTION{RESET}\n")
        return 0


if __name__ == "__main__":
    sys.exit(main())
