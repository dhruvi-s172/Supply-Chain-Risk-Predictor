"""Rebuild the deployment model from dataset.csv and write its metadata.

This script mirrors the final executable preprocessing in notebook.ipynb:
it removes the two leakage-prone fields, converts timestamp to Unix seconds,
and label-encodes risk_classification before fitting a RandomForestClassifier.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "dataset.csv"
MODEL_PATH = ROOT / "model.pkl"
METADATA_PATH = ROOT / "model_metadata.json"
TARGET = "risk_classification"
EXCLUDED_FEATURES = ["disruption_likelihood_score", "delay_probability"]


def main() -> None:
    data = pd.read_csv(DATA_PATH).drop_duplicates()
    feature_columns = [
        column for column in data.columns if column not in [TARGET, *EXCLUDED_FEATURES]
    ]

    features = data[feature_columns].copy()
    features["timestamp"] = pd.to_datetime(features["timestamp"], errors="raise")
    features["timestamp"] = features["timestamp"].astype("int64") // 10**9

    target_encoder = LabelEncoder()
    labels = target_encoder.fit_transform(data[TARGET])

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)
    test_accuracy = accuracy_score(y_test, model.predict(x_test))
    joblib.dump(model, MODEL_PATH)

    numeric_stats = {
        column: {
            "min": float(data[column].min()),
            "max": float(data[column].max()),
            "median": float(data[column].median()),
        }
        for column in feature_columns
        if column != "timestamp"
    }
    metadata = {
        "target_column": TARGET,
        "feature_columns": feature_columns,
        "excluded_features": EXCLUDED_FEATURES,
        "timestamp_feature": "timestamp",
        "timestamp_preprocessing": "Convert ISO datetime input to Unix timestamp in seconds.",
        "target_encoding": {
            "encoder": "LabelEncoder",
            "classes": target_encoder.classes_.tolist(),
        },
        "scaling": {"used": False, "reason": "No StandardScaler appears in the notebook workflow."},
        "model": {
            "name": "Random Forest Classifier",
            "parameters": {"n_estimators": 100, "random_state": 42},
            "test_accuracy": round(float(test_accuracy), 4),
            "training_rows": int(len(x_train)),
            "test_rows": int(len(x_test)),
        },
        "dataset": {
            "rows": int(len(data)),
            "columns": int(len(data.columns)),
            "class_distribution": {
                key: int(value) for key, value in data[TARGET].value_counts().items()
            },
        },
        "numeric_stats": numeric_stats,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {MODEL_PATH.name}; hold-out accuracy: {test_accuracy:.2%}")


if __name__ == "__main__":
    main()
