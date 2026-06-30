# Telco Customer Churn Prediction — End-to-End MLOps

> Predict which telecom customers are about to leave — and act on it. This project takes a churn model from raw data all the way to a live, containerized prediction app with continuous integration.

[![CI](https://github.com/akhilsai007/Telco-Churn-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/akhilsai007/Telco-Churn-mlops/actions/workflows/ci.yml)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Hugging%20Face%20Spaces-FFD21E?logo=huggingface&logoColor=000)](https://akhilsai07-telco-churn-api.hf.space/)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)](Dockerfile)

## 🚀 Live demo

**Try it now → https://akhilsai07-telco-churn-api.hf.space/**

Enter a customer's plan details and get an instant churn-risk read on a visual gauge — no setup, runs in your browser.

![High churn risk — customer likely to leave](assets/demo.png)

![Low churn risk — customer likely to stay](assets/demo-2.png)

*Free tier: if the app has been idle it may take a few seconds to wake on the first visit.*

## Overview

This project implements the full ML lifecycle as a reproducible, modular pipeline:

- **Data ingestion & validation** — load the raw data and validate it against a schema (column presence, types, allowed categories, and data-quality checks such as the dataset's blank `TotalCharges` values).
- **Preprocessing & feature engineering** — clean the data, encode categorical features, scale numerics, and split train/test *before* fitting any transformer (no data leakage).
- **Model training & tracking** — train and compare multiple models, logging every run's parameters, metrics, and model artifact to MLflow.
- **Serving** — expose the best model through a FastAPI REST API with input validation, interactive docs, and a web UI.
- **Web interface** — a lightweight, self-serve form (served by the API) that shows churn risk on a gauge for non-technical users.
- **Containerization** — package the API and trained model into a lean Docker image.
- **Continuous integration** — automatically train the model and run the test suite on every push via GitHub Actions.

**Dataset:** [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (IBM sample data) — ~7,043 customers, 21 features, target `Churn` (Yes/No). The classes are imbalanced (~26.5% churn), which the modeling step handles explicitly.

## Tech stack

| Area | Tools |
|------|-------|
| Data & modeling | pandas, scikit-learn, XGBoost |
| Experiment tracking | MLflow |
| Serving | FastAPI, Uvicorn, Pydantic |
| Containerization | Docker |
| Testing & CI | pytest, GitHub Actions |
| Deployment | Hugging Face Spaces |

## Results

Three models were trained with class-imbalance handling (`class_weight="balanced"` for the scikit-learn models, `scale_pos_weight` for XGBoost) and evaluated on a held-out, stratified test set. **Logistic regression** performed best by ROC-AUC and was selected for serving.

| Model | ROC-AUC | Recall | F1 |
|-------|--------:|-------:|----:|
| **Logistic Regression** | **0.8413** | **0.7834** | **0.6136** |
| XGBoost | 0.8290 | 0.6925 | 0.6023 |
| Random Forest | 0.8212 | 0.6283 | 0.5890 |

Recall is the metric of interest here: the business goal is to catch customers likely to churn, and the selected model identifies roughly **78%** of them.

## Project structure

```
telco-churn/
├── churn/                      # pipeline package
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── config_entity.py        # typed config dataclasses
│   ├── configuration.py        # reads config.yaml -> config objects
│   ├── logger.py
│   └── utils.py
├── app/                        # FastAPI serving layer
│   ├── main.py                 # API endpoints + web UI
│   └── schema.py               # request/response models
├── config/
│   └── config.yaml             # paths, data schema, hyperparameters
├── tests/
│   └── test_api.py
├── assets/                     # README screenshot(s)
├── data/raw/                   # dataset goes here
├── .github/workflows/ci.yml    # CI pipeline
├── Dockerfile
├── requirements.txt            # full dev/training dependencies
├── requirements-docker.txt     # lean serving-only dependencies
└── main.py                     # runs the full training pipeline
```

## Setup

```bash
# clone and enter the project
git clone https://github.com/akhilsai007/Telco-Churn-mlops.git
cd Telco-Churn-mlops

# create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
```

**Data:** download the dataset from Kaggle and place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in `data/raw/`.

## Usage

### Train the model

Runs the full pipeline (ingestion → validation → transformation → training) and saves the best model:

```bash
python main.py
```

Outputs land in `artifacts/`: the fitted preprocessor, processed train/test sets, the best model (`model.joblib`), and a metrics report.

### View experiments in MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open `http://127.0.0.1:5000` and switch to the **Model training** tab to compare runs.

### Run the app locally

```bash
uvicorn app.main:app --reload
```

- Web form: `http://127.0.0.1:8000/`
- Interactive API docs (Swagger): `http://127.0.0.1:8000/docs`

### Call the API

```bash
curl -X POST https://akhilsai07-telco-churn-api.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 5, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 85.7, "TotalCharges": 428.5
  }'
```

Response:

```json
{ "churn": "Yes", "churn_probability": 0.7874 }
```

## How it's deployed

The trained model and preprocessor are baked into a Docker image (defined by the `Dockerfile`) and served by FastAPI. The image is deployed to Hugging Face Spaces, which rebuilds and redeploys automatically on every push to the Space. GitHub Actions runs the training pipeline and test suite on every push to this repo.