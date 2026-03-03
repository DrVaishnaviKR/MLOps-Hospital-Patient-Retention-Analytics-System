"""
FastAPI application for Patient Churn Prediction.

Exposes endpoints for health checks, ML predictions,
and Prometheus metrics collection.
"""

import time
import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
import joblib
import pandas as pd
from prometheus_client import (
    Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
)

from backend.schemas import PatientData, PredictionResponse
from src.training.feature_engineering import engineer_features

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Patient Churn Prediction API",
    description="Predict patient churn using ML model",
    version="1.0.0"
)

# Load trained model and reference date (gracefully)
model = None
reference_date = None
try:
    model = joblib.load("models/best_model.joblib")
    reference_date = joblib.load("models/reference_date.joblib")
    logger.info("Model loaded successfully")
except (OSError, ValueError, KeyError) as exc:
    logger.warning("Could not load model: %s. /predict will return an error.", exc)

# Prometheus metrics
REQUEST_COUNT = Counter(
    "prediction_request_count_total",
    "Total prediction API requests"
)

REQUEST_LATENCY = Histogram(
    "prediction_request_latency_seconds",
    "Prediction request latency in seconds",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]
)

PREDICTION_CLASSES = Counter(
    "prediction_class_total",
    "Count of predictions by class",
    ["predicted_class"]
)


@app.get("/health")
def health():
    """Return API health status and model availability."""
    return {
        "status": "API is running",
        "model_loaded": model is not None
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(data: PatientData):
    """Accept patient data and return churn prediction with probability."""
    REQUEST_COUNT.inc()
    start_time = time.time()

    if model is None or reference_date is None:
        REQUEST_LATENCY.observe(time.time() - start_time)
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please retrain or check model files."
        )

    # Convert Pydantic model to DataFrame
    input_dict = data.model_dump()
    input_df = pd.DataFrame([input_dict])

    # Apply same feature engineering as training
    input_df = engineer_features(input_df, reference_date=reference_date)

    # Model inference
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    PREDICTION_CLASSES.labels(predicted_class=str(int(prediction))).inc()
    REQUEST_LATENCY.observe(time.time() - start_time)

    return {
        "prediction": int(prediction),
        "churn_probability": round(float(probability), 4)
    }


@app.get("/metrics")
def metrics():
    """Expose Prometheus metrics for scraping."""
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )
