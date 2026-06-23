"""Item-based collaborative filtering recommendation engine."""

from __future__ import annotations

from typing import List

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def build_product_user_matrix(df: pd.DataFrame, min_customers: int = 5) -> pd.DataFrame:
    """Create customer-product purchase matrix for similarity computation."""
    matrix = (
        df.groupby(["Description", "CustomerID"])["Quantity"]
        .sum()
        .unstack(fill_value=0)
    )

    customer_counts = (matrix > 0).sum(axis=0)
    valid_customers = customer_counts[customer_counts >= 1].index
    matrix = matrix[valid_customers]

    product_counts = (matrix > 0).sum(axis=1)
    matrix = matrix.loc[product_counts >= min_customers]
    return matrix


def compute_item_similarity(product_matrix: pd.DataFrame) -> pd.DataFrame:
    """Compute cosine similarity between products."""
    similarity = cosine_similarity(product_matrix)
    return pd.DataFrame(similarity, index=product_matrix.index, columns=product_matrix.index)


def find_product_name(query: str, product_names: List[str]) -> str | None:
    """Case-insensitive partial match for product lookup."""
    query_lower = query.strip().lower()
    if not query_lower:
        return None

    for name in product_names:
        if query_lower == name.lower():
            return name

    partial_matches = [name for name in product_names if query_lower in name.lower()]
    return partial_matches[0] if partial_matches else None


def recommend_similar_products(
    product_name: str,
    similarity_df: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    """Return top-N similar products for a given product."""
    matched = find_product_name(product_name, similarity_df.index.tolist())
    if matched is None:
        return pd.DataFrame(columns=["Product", "SimilarityScore"])

    scores = similarity_df.loc[matched].sort_values(ascending=False)
    scores = scores[scores.index != matched].head(top_n)

    return pd.DataFrame(
        {"Product": scores.index, "SimilarityScore": scores.values.round(4)}
    ).reset_index(drop=True)
