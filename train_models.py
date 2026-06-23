"""Train and save all models for Streamlit deployment."""

import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CLUSTER_LABELS_PATH,
    DATA_PATH,
    KMEANS_MODEL_PATH,
    PRODUCT_NAMES_PATH,
    RANDOM_STATE,
    SCALER_MODEL_PATH,
    SIMILARITY_MATRIX_PATH,
)
from src.data_loader import load_raw_data, preprocess_retail_data
from src.recommender import build_product_user_matrix, compute_item_similarity
from src.rfm import build_rfm_table, prepare_rfm_features, train_kmeans_segmentation


def select_best_k(scaled_features: np.ndarray, k_candidates: list[int]) -> int:
    """Pick K with highest silhouette score."""
    best_k = k_candidates[0]
    best_score = -1.0
    for k in k_candidates:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = model.fit_predict(scaled_features)
        score = silhouette_score(scaled_features, labels)
        if score > best_score:
            best_score = score
            best_k = k
    return best_k


def main() -> None:
    print("Loading and preprocessing data...")
    df = preprocess_retail_data(load_raw_data(DATA_PATH))

    print("Building RFM features...")
    rfm = build_rfm_table(df)
    upper = rfm["Monetary"].quantile(0.99)
    rfm["Monetary"] = rfm["Monetary"].clip(upper=upper)

    scaled, _ = prepare_rfm_features(rfm)
    best_k = select_best_k(scaled.values, [3, 4, 5, 6])
    print(f"Best K from tuning: {best_k}")

    print("Training final segmentation model...")
    model, scaler, labels, _ = train_kmeans_segmentation(
        rfm, n_clusters=best_k, random_state=RANDOM_STATE
    )

    print("Building recommendation similarity matrix...")
    product_matrix = build_product_user_matrix(df, min_customers=5)
    similarity = compute_item_similarity(product_matrix)
    product_names = similarity.index.tolist()

    joblib.dump(model, KMEANS_MODEL_PATH)
    joblib.dump(scaler, SCALER_MODEL_PATH)
    joblib.dump(labels, CLUSTER_LABELS_PATH)
    joblib.dump(similarity, SIMILARITY_MATRIX_PATH)
    joblib.dump(product_names, PRODUCT_NAMES_PATH)

    print("Saved model artifacts in models/ folder.")
    print("Done.")


if __name__ == "__main__":
    main()
