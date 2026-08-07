"""Streamlit application for supply-chain risk classification."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.pkl"
METADATA_PATH = ROOT / "model_metadata.json"
SCALER_PATH = ROOT / "scaler.pkl"


st.set_page_config(
    page_title="Supply Chain Risk Predictor",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """<style>
    .block-container {max-width: 1250px; padding-top: 2.5rem;}
    .hero {padding: 1.4rem 1.6rem; border-radius: 18px; color: white;
           background: linear-gradient(115deg, #0b3b60, #087e8b); margin-bottom: 1.5rem;}
    .hero h1 {margin: 0; font-size: 2.2rem;}.hero p {margin: .5rem 0 0; opacity: .92;}
    [data-testid="stSidebar"] {background: #f4f8fb;}
    </style>""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError("model.pkl or model_metadata.json is missing.")
    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    scaler = joblib.load(SCALER_PATH) if SCALER_PATH.exists() else None
    return model, metadata, scaler


def friendly_name(column: str) -> str:
    return column.replace("_", " ").title()


def build_input_form(metadata: dict) -> pd.DataFrame | None:
    """Build controls in exact model feature order and return one prepared row."""
    with st.form("prediction_form"):
        st.subheader("Shipment and operating conditions")
        values: dict[str, float] = {}
        columns = st.columns(3)
        for index, feature in enumerate(metadata["feature_columns"]):
            container = columns[index % 3]
            with container:
                if feature == metadata["timestamp_feature"]:
                    date_value = st.date_input("Event date", value=datetime(2021, 1, 1).date())
                    time_value = st.time_input("Event time", value=datetime(2021, 1, 1).time())
                    values[feature] = pd.Timestamp.combine(date_value, time_value).timestamp()
                else:
                    stats = metadata["numeric_stats"][feature]
                    span = stats["max"] - stats["min"]
                    step = max(span / 100, 0.01)
                    values[feature] = st.number_input(
                        friendly_name(feature),
                        min_value=float(stats["min"]),
                        max_value=float(stats["max"]),
                        value=float(stats["median"]),
                        step=float(step),
                        format="%.4f",
                    )
        submitted = st.form_submit_button("Predict risk classification", type="primary", use_container_width=True)

    if not submitted:
        return None
    return pd.DataFrame([[values[column] for column in metadata["feature_columns"]]], columns=metadata["feature_columns"])


def render_prediction(model, metadata: dict, scaler) -> None:
    st.markdown('<div class="hero"><h1>Supply Chain Risk Predictor</h1><p>Estimate shipment risk from operational and logistics signals.</p></div>', unsafe_allow_html=True)
    st.info("Enter the shipment details below. Inputs are validated against the dataset range used for training.")
    input_frame = build_input_form(metadata)
    if input_frame is None:
        return
    try:
        model_input = scaler.transform(input_frame) if scaler is not None else input_frame
        predicted_code = int(model.predict(model_input)[0])
        predicted_label = metadata["target_encoding"]["classes"][predicted_code]
        if "High" in predicted_label:
            st.warning(f"### Predicted classification: {predicted_label}")
        elif "Low" in predicted_label:
            st.success(f"### Predicted classification: {predicted_label}")
        else:
            st.info(f"### Predicted classification: {predicted_label}")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(model_input)[0]
            probability_frame = pd.DataFrame({
                "Risk classification": metadata["target_encoding"]["classes"],
                "Probability": probabilities,
            })
            st.plotly_chart(
                px.bar(probability_frame, x="Risk classification", y="Probability", text_auto=".1%", color="Risk classification", title="Prediction confidence"),
                use_container_width=True,
            )
        else:
            st.caption("This model does not expose prediction probabilities.")
    except Exception as exc:
        st.error(f"Prediction could not be completed: {exc}")


def render_overview(metadata: dict) -> None:
    st.title("Dataset overview")
    dataset = metadata["dataset"]
    metric_one, metric_two, metric_three = st.columns(3)
    metric_one.metric("Records", f"{dataset['rows']:,}")
    metric_two.metric("Original columns", dataset["columns"])
    metric_three.metric("Model features", len(metadata["feature_columns"]))
    st.subheader("Risk-class distribution")
    distribution = pd.DataFrame(dataset["class_distribution"].items(), columns=["Risk classification", "Records"])
    st.plotly_chart(px.bar(distribution, x="Risk classification", y="Records", color="Risk classification"), use_container_width=True)
    st.caption("Excluded from modeling: " + ", ".join(metadata["excluded_features"]) + ".")


def render_model_info(model, metadata: dict) -> None:
    st.title("Model information")
    details = metadata["model"]
    st.write(f"**Selected model:** {details['name']}")
    st.write(f"**Reproducible hold-out accuracy:** {details['test_accuracy']:.2%}")
    st.write(f"**Preprocessing:** {metadata['timestamp_preprocessing']}")
    st.write("**Feature scaling:** Not used by the notebook workflow.")
    st.subheader("Model input order")
    st.code("\n".join(f"{number + 1}. {feature}" for number, feature in enumerate(metadata["feature_columns"])), language="text")
    if hasattr(model, "feature_importances_"):
        st.subheader("Feature importance")
        importance = pd.DataFrame({"Feature": metadata["feature_columns"], "Importance": model.feature_importances_}).sort_values("Importance", ascending=False)
        st.plotly_chart(px.bar(importance.head(15), x="Importance", y="Feature", orientation="h", title="Top 15 Random Forest features"), use_container_width=True)


def main() -> None:
    try:
        model, metadata, scaler = load_artifacts()
    except Exception as exc:
        st.error(f"Unable to load deployment artifacts: {exc}")
        st.stop()
    st.sidebar.title("🚚 Navigation")
    page = st.sidebar.radio("Choose a page", ["Risk prediction", "Dataset overview", "Model information"])
    st.sidebar.divider()
    st.sidebar.caption("Built for the Supply Chain & Logistics classification project.")
    if page == "Risk prediction":
        render_prediction(model, metadata, scaler)
    elif page == "Dataset overview":
        render_overview(metadata)
    else:
        render_model_info(model, metadata)


if __name__ == "__main__":
    main()
