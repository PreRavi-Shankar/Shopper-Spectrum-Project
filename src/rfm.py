"""RFM feature engineering and customer segmentation."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def build_rfm_table(df: pd.DataFrame, reference_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """Compute Recency, Frequency, and Monetary values per customer."""
    if reference_date is None:
        reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    invoice_level = (
        df.groupby(["CustomerID", "InvoiceNo"], as_index=False)
        .agg(
            InvoiceDate=("InvoiceDate", "max"),
            InvoiceAmount=("TotalAmount", "sum"),
        )
    )

    rfm = invoice_level.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("InvoiceAmount", "sum"),
    )

    return rfm.reset_index()


def prepare_rfm_features(rfm: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    """Transform RFM values for clustering (log + standardization)."""
    features = rfm[["Recency", "Frequency", "Monetary"]].copy()
    features["Recency"] = np.log1p(features["Recency"])
    features["Frequency"] = np.log1p(features["Frequency"])
    features["Monetary"] = np.log1p(features["Monetary"])

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    scaled_df = pd.DataFrame(scaled, columns=["Recency", "Frequency", "Monetary"], index=rfm.index)
    return scaled_df, scaler


def label_clusters_from_centroids(centroids: np.ndarray) -> Dict[int, str]:
    """
    Assign business-friendly segment labels using centroid RFM patterns.
    Lower scaled Recency (log-days) implies more recent purchases.
    """
    recency = centroids[:, 0]
    frequency = centroids[:, 1]
    monetary = centroids[:, 2]

    labels: Dict[int, str] = {}
    for idx in range(len(centroids)):
        recent = recency[idx] <= np.median(recency)
        frequent = frequency[idx] >= np.median(frequency)
        high_spend = monetary[idx] >= np.median(monetary)

        if recent and frequent and high_spend:
            labels[idx] = "High-Value"
        elif recent and not frequent and not high_spend:
            labels[idx] = "At-Risk"
        elif not recent and not frequent and not high_spend:
            labels[idx] = "Occasional"
        else:
            labels[idx] = "Regular"

    return labels


def train_kmeans_segmentation(
    rfm: pd.DataFrame,
    n_clusters: int = 4,
    random_state: int = 42,
) -> Tuple[KMeans, StandardScaler, Dict[int, str], pd.DataFrame]:
    """Fit KMeans on scaled RFM features and return labeled customer segments."""
    scaled_features, scaler = prepare_rfm_features(rfm)

    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_ids = model.fit_predict(scaled_features)

    segment_labels = label_clusters_from_centroids(model.cluster_centers_)
    rfm_result = rfm.copy()
    rfm_result["Cluster"] = cluster_ids
    rfm_result["Segment"] = rfm_result["Cluster"].map(segment_labels)

    return model, scaler, segment_labels, rfm_result


def predict_segment(
    recency: float,
    frequency: float,
    monetary: float,
    model: KMeans,
    scaler: StandardScaler,
    segment_labels: Dict[int, str],
) -> Tuple[int, str]:
    """Predict customer segment from raw RFM inputs."""
    features = pd.DataFrame(
        [[np.log1p(recency), np.log1p(frequency), np.log1p(monetary)]],
        columns=["Recency", "Frequency", "Monetary"],
    )
    scaled = scaler.transform(features)
    cluster_id = int(model.predict(scaled)[0])
    return cluster_id, segment_labels[cluster_id]
