"""Project-wide paths and constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "online_retail.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

KMEANS_MODEL_PATH = MODELS_DIR / "kmeans_rfm_model.joblib"
SCALER_MODEL_PATH = MODELS_DIR / "rfm_scaler.joblib"
CLUSTER_LABELS_PATH = MODELS_DIR / "cluster_labels.joblib"
SIMILARITY_MATRIX_PATH = MODELS_DIR / "product_similarity.joblib"
PRODUCT_NAMES_PATH = MODELS_DIR / "product_names.joblib"

RANDOM_STATE = 42
TOP_N_RECOMMENDATIONS = 5
