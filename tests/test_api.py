"""Unit tests for the Patient Churn Prediction API."""

import os

import joblib
from fastapi.testclient import TestClient
from backend.app import app


client = TestClient(app)

# Sample valid patient data for testing
VALID_PATIENT = {
    "Age": 45,
    "Gender": "Male",
    "State": "California",
    "Tenure_Months": 24,
    "Specialty": "Cardiology",
    "Insurance_Type": "Private",
    "Visits_Last_Year": 6,
    "Missed_Appointments": 1,
    "Days_Since_Last_Visit": 30,
    "Last_Interaction_Date": "15-01-2025",
    "Overall_Satisfaction": 3.5,
    "Wait_Time_Satisfaction": 3.0,
    "Staff_Satisfaction": 4.0,
    "Provider_Rating": 3.8,
    "Avg_Out_Of_Pocket_Cost": 150.0,
    "Billing_Issues": 0,
    "Portal_Usage": 1,
    "Referrals_Made": 2,
    "Distance_To_Facility_Miles": 10.5,
}


# Test 1: Health endpoint
def test_health_endpoint():
    """Test that /health returns 200 with status and model info."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "API is running"
    assert "model_loaded" in data


# Test 2: Predict with valid input
def test_predict_valid_input():
    response = client.post("/predict", json=VALID_PATIENT)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "churn_probability" in data
    assert data["prediction"] in [0, 1]
    assert 0.0 <= data["churn_probability"] <= 1.0


# Test 3: Predict with invalid input (missing fields)
def test_predict_invalid_input():
    incomplete_data = {"Age": 45, "Gender": "Male"}
    response = client.post("/predict", json=incomplete_data)
    assert response.status_code == 422


# Test 4: Metrics endpoint returns Prometheus format
def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    content = response.text
    assert "prediction_request_count_total" in content or "request" in content.lower()


# Test 5: Prediction response schema validation
def test_prediction_response_schema():
    response = client.post("/predict", json=VALID_PATIENT)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["prediction"], int)
    assert isinstance(data["churn_probability"], float)


# Test 6: Model file exists and loads
def test_model_loads():
    model_path = "models/best_model.joblib"
    assert os.path.exists(model_path), "Model file not found"
    model = joblib.load(model_path)
    assert hasattr(model, "predict"), "Model missing predict method"
    assert hasattr(model, "predict_proba"), "Model missing predict_proba method"


# Test 7: Predict with edge-case values
def test_predict_edge_case_values():
    edge_case = VALID_PATIENT.copy()
    edge_case["Age"] = 18
    edge_case["Tenure_Months"] = 0
    edge_case["Visits_Last_Year"] = 0
    edge_case["Missed_Appointments"] = 0
    response = client.post("/predict", json=edge_case)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in [0, 1]
