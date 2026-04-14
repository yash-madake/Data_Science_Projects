from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict Bengaluru housing price.")
    parser.add_argument("--location", type=str, required=True)
    parser.add_argument("--total-sqft", type=float, required=True)
    parser.add_argument("--bath", type=float, required=True)
    parser.add_argument("--balcony", type=float, default=0.0)
    parser.add_argument("--bhk", type=float, required=True)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/model.pkl"),
        help="Path to the trained model file.",
    )
    args = parser.parse_args()

    payload = joblib.load(args.model_path)
    model = payload["model"]

    sample = pd.DataFrame(
        [
            {
                "location": args.location.strip(),
                "total_sqft": args.total_sqft,
                "bath": args.bath,
                "balcony": args.balcony,
                "bhk": args.bhk,
            }
        ]
    )

    prediction = float(model.predict(sample)[0])
    print(f"Predicted price: {prediction:.2f} Lakhs INR")


if __name__ == "__main__":
    main()
