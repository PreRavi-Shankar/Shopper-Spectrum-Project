"""Data loading and preprocessing utilities."""

from __future__ import annotations

import pandas as pd


def load_raw_data(path: str) -> pd.DataFrame:
    """Load the online retail CSV dataset."""
    try:
        return pd.read_csv(path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Dataset not found at: {path}") from exc


def preprocess_retail_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean transaction data following project guidelines:
    - Remove rows with missing CustomerID
    - Exclude cancelled invoices (InvoiceNo starting with 'C')
    - Remove negative or zero quantities and prices
    """
    cleaned = df.copy()

    cleaned["InvoiceNo"] = cleaned["InvoiceNo"].astype(str)
    cleaned["InvoiceDate"] = pd.to_datetime(cleaned["InvoiceDate"], errors="coerce")
    cleaned["CustomerID"] = cleaned["CustomerID"].astype("Int64")

    cleaned = cleaned.dropna(subset=["CustomerID", "InvoiceDate"])
    cleaned = cleaned[~cleaned["InvoiceNo"].str.startswith("C", na=False)]
    cleaned = cleaned[(cleaned["Quantity"] > 0) & (cleaned["UnitPrice"] > 0)]
    cleaned = cleaned.dropna(subset=["Description"])

    cleaned["CustomerID"] = cleaned["CustomerID"].astype(int)
    cleaned["TotalAmount"] = cleaned["Quantity"] * cleaned["UnitPrice"]
    cleaned["YearMonth"] = cleaned["InvoiceDate"].dt.to_period("M").astype(str)
    cleaned["DayOfWeek"] = cleaned["InvoiceDate"].dt.day_name()
    cleaned["Month"] = cleaned["InvoiceDate"].dt.month_name()
    cleaned["Hour"] = cleaned["InvoiceDate"].dt.hour

    return cleaned.reset_index(drop=True)
