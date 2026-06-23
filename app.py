"""Shopper Spectrum - Streamlit deployment app."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CLUSTER_LABELS_PATH,
    KMEANS_MODEL_PATH,
    PRODUCT_NAMES_PATH,
    SCALER_MODEL_PATH,
    SIMILARITY_MATRIX_PATH,
    TOP_N_RECOMMENDATIONS,
)
from src.recommender import recommend_similar_products
from src.rfm import predict_segment


@st.cache_resource
def load_artifacts():
    """Load persisted ML artifacts produced by the notebook."""
    return {
        "kmeans": joblib.load(KMEANS_MODEL_PATH),
        "scaler": joblib.load(SCALER_MODEL_PATH),
        "cluster_labels": joblib.load(CLUSTER_LABELS_PATH),
        "similarity": joblib.load(SIMILARITY_MATRIX_PATH),
        "product_names": joblib.load(PRODUCT_NAMES_PATH),
    }


def main() -> None:
    st.set_page_config(
        page_title="Shopper Spectrum",
        page_icon="🛒",
        layout="wide",
    )

    st.title("🛒 Shopper Spectrum")
    st.markdown(
        "Customer segmentation and product recommendations for e-commerce retail analytics."
    )

    try:
        artifacts = load_artifacts()
    except FileNotFoundError:
        st.error(
            "Model files not found. Please run the Jupyter notebook first to train and save models in the `models/` folder."
        )
        st.stop()

    tab_reco, tab_segment = st.tabs(
        ["Product Recommendations", "Customer Segmentation"]
    )

    with tab_reco:
        st.subheader("Product Recommendation Module")
        st.write(
            "Enter a product name to get the top 5 similar products using item-based collaborative filtering."
        )

        sample_products = artifacts["product_names"][:10]
        product_query = st.text_input(
            "Product Name",
            placeholder="e.g., WHITE HANGING HEART T-LIGHT HOLDER",
        )
        st.caption(f"Sample products: {', '.join(sample_products[:3])} ...")

        if st.button("Get Recommendations", type="primary"):
            if not product_query.strip():
                st.warning("Please enter a product name.")
            else:
                recommendations = recommend_similar_products(
                    product_query,
                    artifacts["similarity"],
                    top_n=TOP_N_RECOMMENDATIONS,
                )
                if recommendations.empty:
                    st.error("Product not found. Try a partial product name from the dataset.")
                else:
                    st.success("Top 5 recommended products")
                    for idx, row in recommendations.iterrows():
                        st.markdown(
                            f"**{idx + 1}. {row['Product']}**  \nSimilarity Score: `{row['SimilarityScore']}`"
                        )

    with tab_segment:
        st.subheader("Customer Segmentation Module")
        st.write(
            "Enter Recency, Frequency, and Monetary values to predict the customer segment."
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            recency = st.number_input("Recency (days since last purchase)", min_value=0, value=30)
        with col2:
            frequency = st.number_input("Frequency (number of purchases)", min_value=1, value=5)
        with col3:
            monetary = st.number_input("Monetary (total spend)", min_value=0.0, value=500.0)

        if st.button("Predict Cluster", type="primary"):
            cluster_id, segment = predict_segment(
                recency,
                frequency,
                monetary,
                artifacts["kmeans"],
                artifacts["scaler"],
                artifacts["cluster_labels"],
            )
            st.success(f"Predicted Segment: **{segment}** (Cluster ID: {cluster_id})")

            segment_info = {
                "High-Value": "Recent, frequent, and high-spending customers.",
                "Regular": "Steady purchasers with moderate engagement.",
                "Occasional": "Infrequent buyers with lower spend.",
                "At-Risk": "Customers showing signs of churn or reduced activity.",
            }
            st.info(segment_info.get(segment, "Segment profile unavailable."))


if __name__ == "__main__":
    main()
