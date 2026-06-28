from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI

from app.schema import ChurnResponse, CustomerFeatures

PREPROCESSOR_PATH = Path("artifacts/data_transformation/preprocessor.joblib")
MODEL_PATH = Path("artifacts/model_trainer/model.joblib")

app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="Predicts whether a customer will churn, using the trained model.",
    version="1.0.0",
)

# load artifacts once at startup (the SAME preprocessor used in training, so
# inference applies identical encoding/scaling)
if not PREPROCESSOR_PATH.exists() or not MODEL_PATH.exists():
    raise RuntimeError(
        "Model artifacts not found. Run `python main.py` first to generate "
        f"'{PREPROCESSOR_PATH}' and '{MODEL_PATH}'."
    )
preprocessor = joblib.load(PREPROCESSOR_PATH)
model = joblib.load(MODEL_PATH)


@app.get("/")
def root():
    return {"message": "Telco Churn Prediction API. See /docs to try it out."}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}


@app.post("/predict", response_model=ChurnResponse)
def predict(features: CustomerFeatures):
    # one-row DataFrame with the same column names the preprocessor was fit on
    df = pd.DataFrame([features.model_dump()])
    X = preprocessor.transform(df)
    # restore feature names (the model was trained on named columns)
    X = pd.DataFrame(X, columns=preprocessor.get_feature_names_out())
    probability = float(model.predict_proba(X)[0, 1])
    label = "Yes" if probability >= 0.5 else "No"
    return ChurnResponse(churn=label, churn_probability=round(probability, 4))