from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_DIR.parents[0]
REPORTS_DIR = PROJECT_ROOT / "reports"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

RECOMMENDATION_DIR = REPORTS_DIR / "recommendations"
ML_REPORT_DIR = REPORTS_DIR / "ml"
TESTING_DIR = REPORTS_DIR / "testing" / "ml"
