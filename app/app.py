from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "model.pkl"


@st.cache_resource
def load_model_payload() -> dict:
    return joblib.load(MODEL_PATH)


def main() -> None:
    st.set_page_config(page_title="Bengaluru House Price Predictor")
    st.title("Bengaluru House Price Predictor")
    st.caption("Predicting property price in Lakhs INR using a trained ML model.")

    if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size == 0:
        st.error("Model is missing. Run `python src/train.py` first.")
        st.stop()

    payload = load_model_payload()
    model = payload["model"]
    locations = payload.get("known_locations", ["other"])

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            location = st.selectbox("Location", options=locations, index=0)
            total_sqft = st.number_input(
                "Total Sqft", min_value=200.0, max_value=10000.0, value=1200.0, step=50.0
            )
        with col2:
            bhk = st.number_input("BHK", min_value=1, max_value=12, value=2, step=1)
            bath = st.number_input("Bathrooms", min_value=1, max_value=12, value=2, step=1)
            balcony = st.number_input("Balconies", min_value=0, max_value=6, value=1, step=1)

        submitted = st.form_submit_button("Predict Price")

    if submitted:
        sample = pd.DataFrame(
            [
                {
                    "location": location,
                    "total_sqft": float(total_sqft),
                    "bath": float(bath),
                    "balcony": float(balcony),
                    "bhk": float(bhk),
                }
            ]
        )

        price_lakhs = float(model.predict(sample)[0])
        price_crore = price_lakhs / 100.0

        st.success(f"Estimated Price: {price_lakhs:.2f} Lakhs INR")
        st.info(f"Approx: {price_crore:.2f} Crore INR")

    metrics = payload.get("metrics")
    if metrics:
        st.markdown("### Last Training Metrics")
        st.write(
            {
                "RMSE": round(metrics.get("rmse", 0.0), 3),
                "MAE": round(metrics.get("mae", 0.0), 3),
                "R2": round(metrics.get("r2", 0.0), 3),
            }
        )


if __name__ == "__main__":
    main()
