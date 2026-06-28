from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 5, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 85.7, "TotalCharges": 428.5,
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_valid_input():
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["churn"] in ("Yes", "No")
    assert 0.0 <= body["churn_probability"] <= 1.0


def test_predict_invalid_category():
    bad = {**VALID_PAYLOAD, "Contract": "Lifetime"}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_predict_missing_field():
    incomplete = {k: v for k, v in VALID_PAYLOAD.items() if k != "tenure"}
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422