from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

from preprocessing import preprocess_dataframe


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained model.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/Bengaluru_House_Data.csv"),
        help="Path to source CSV data.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/model.pkl"),
        help="Path to trained model.",
    )
    args = parser.parse_args()

    payload = joblib.load(args.model_path)
    model = payload["model"]

    raw_df = pd.read_csv(args.data_path)
    df = preprocess_dataframe(raw_df)

    X = df[["location", "total_sqft", "bath", "balcony", "bhk"]]
    y = df["price"]

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    predictions = model.predict(X_test)

    rmse = root_mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"Model file: {args.model_path}")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE : {mae:.3f}")
    print(f"R2  : {r2:.3f}")


if __name__ == "__main__":
    main()
