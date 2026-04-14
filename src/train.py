from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from preprocessing import preprocess_dataframe


def build_pipeline() -> Pipeline:
    categorical_features = ["location"]
    numeric_features = ["total_sqft", "bath", "balcony", "bhk"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            ),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Bengaluru housing price model.")
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
        help="Output path for the trained model.",
    )
    args = parser.parse_args()

    raw_df = pd.read_csv(args.data_path)
    df = preprocess_dataframe(raw_df)

    X = df[["location", "total_sqft", "bath", "balcony", "bhk"]]
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    metrics = {
        "rmse": float(root_mean_squared_error(y_test, predictions)),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
    }

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": pipeline,
        "feature_order": list(X.columns),
        "known_locations": sorted(df["location"].unique().tolist()),
        "metrics": metrics,
    }
    joblib.dump(payload, args.model_path)

    print(f"Saved model to: {args.model_path}")
    print("Evaluation on holdout set:")
    print(f"RMSE: {metrics['rmse']:.3f}")
    print(f"MAE : {metrics['mae']:.3f}")
    print(f"R2  : {metrics['r2']:.3f}")


if __name__ == "__main__":
    main()
