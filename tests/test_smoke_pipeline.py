from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from preprocessing import preprocess_dataframe  # noqa: E402
from train import build_pipeline  # noqa: E402


class TestPipelineSmoke(unittest.TestCase):
    def setUp(self) -> None:
        self.raw_df = pd.DataFrame(
            [
                {
                    "location": "Whitefield",
                    "size": "2 BHK",
                    "bath": 2,
                    "balcony": 1,
                    "total_sqft": "1056",
                    "price": 39.07,
                },
                {
                    "location": "Whitefield",
                    "size": "3 BHK",
                    "bath": 2,
                    "balcony": 2,
                    "total_sqft": "1440",
                    "price": 62.0,
                },
                {
                    "location": "Electronic City",
                    "size": "2 BHK",
                    "bath": 2,
                    "balcony": 1,
                    "total_sqft": "1200-1300",
                    "price": 52.0,
                },
                {
                    "location": "Yelahanka",
                    "size": "3 BHK",
                    "bath": 3,
                    "balcony": 2,
                    "total_sqft": "1521 sqft",
                    "price": 95.0,
                },
            ]
        )

    def test_preprocess_dataframe_outputs_expected_columns(self) -> None:
        processed = preprocess_dataframe(self.raw_df, min_location_count=1)

        self.assertGreater(len(processed), 0)
        self.assertListEqual(
            list(processed.columns),
            ["location", "total_sqft", "bath", "balcony", "bhk", "price"],
        )
        self.assertTrue((processed["total_sqft"] > 0).all())
        self.assertTrue((processed["bhk"] > 0).all())

    def test_training_pipeline_can_fit_and_predict(self) -> None:
        processed = preprocess_dataframe(self.raw_df, min_location_count=1)
        X = processed[["location", "total_sqft", "bath", "balcony", "bhk"]]
        y = processed["price"]

        model = build_pipeline()
        model.fit(X, y)
        prediction = float(model.predict(X.head(1))[0])

        self.assertGreater(prediction, 0.0)


if __name__ == "__main__":
    unittest.main()
