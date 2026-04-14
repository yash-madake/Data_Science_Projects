from __future__ import annotations

import re

import pandas as pd


def _parse_total_sqft(value: object) -> float | None:
    if pd.isna(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    if "-" in text:
        parts = [p.strip() for p in text.split("-", maxsplit=1)]
        try:
            left = float(parts[0])
            right = float(parts[1])
            return (left + right) / 2.0
        except (TypeError, ValueError):
            return None

    match = re.search(r"\d+(\.\d+)?", text)
    if not match:
        return None

    return float(match.group(0))


def preprocess_dataframe(df: pd.DataFrame, min_location_count: int = 10) -> pd.DataFrame:
    work = df.copy()
    work.columns = [col.strip() for col in work.columns]

    required_cols = {"location", "size", "bath", "balcony", "total_sqft", "price"}
    missing = required_cols - set(work.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    work = work.dropna(subset=["location", "size", "bath", "total_sqft", "price"])
    work["location"] = work["location"].astype(str).str.strip()
    work["total_sqft"] = work["total_sqft"].apply(_parse_total_sqft)
    work["bhk"] = work["size"].astype(str).str.extract(r"(\d+)").astype(float)

    work["bath"] = pd.to_numeric(work["bath"], errors="coerce")
    work["balcony"] = pd.to_numeric(work["balcony"], errors="coerce").fillna(0)
    work["price"] = pd.to_numeric(work["price"], errors="coerce")

    work = work.dropna(subset=["total_sqft", "bhk", "bath", "price"])
    work = work[(work["total_sqft"] > 200) & (work["bath"] > 0) & (work["bhk"] > 0)]

    # Remove obvious outliers where bathroom count is unrealistically large.
    work = work[work["bath"] <= work["bhk"] + 2]

    location_counts = work["location"].value_counts()
    rare_locations = location_counts[location_counts < min_location_count].index
    work.loc[work["location"].isin(rare_locations), "location"] = "other"

    columns = ["location", "total_sqft", "bath", "balcony", "bhk", "price"]
    return work[columns].reset_index(drop=True)
