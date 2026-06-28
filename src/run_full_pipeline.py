from pipelines.engineering.ipl_current_2026 import main as run_ipl_2026_engineering
from pipelines.engineering.ipl_historical import main as run_ipl_historical_engineering
from pipelines.engineering.smat import main as run_smat_engineering
from pipelines.ml.clustering import main as run_player_clustering
from pipelines.ml.model_training import main as run_ml_model_training
from pipelines.reporting.data_summary import main as run_data_summary
from pipelines.reporting.primary_workbook import main as run_workbook_builder
from recommendation_engine.run import main as run_recommendation_engine
from testing.evaluation.ml_evaluation import main as run_ml_evaluation


def main():
    print("1/9 Historical IPL benchmark engineering")
    run_ipl_historical_engineering()

    print("\n2/9 Current IPL 2026 squad engineering")
    run_ipl_2026_engineering()

    print("\n3/9 SMAT feature engineering")
    run_smat_engineering()

    print("\n4/9 ML model training")
    run_ml_model_training()

    print("\n5/9 Player clustering")
    run_player_clustering()

    print("\n6/9 Recommendation engine")
    run_recommendation_engine()

    print("\n7/9 ML evaluation")
    run_ml_evaluation()

    print("\n8/9 Data engineering summary")
    run_data_summary()

    print("\n9/9 Primary workbook")
    run_workbook_builder()

    print("\nFull pipeline complete")


if __name__ == "__main__":
    main()
