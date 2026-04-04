# Customer Churn Prediction Service

Simple End-to-end machine learning project that predicts customer churn using XGBoost and serves predictions via a FastAPI API.

## Overview

This project builds a production-style ML pipeline for predicting customer churn using the Telco Customer Churn dataset. It includes data preprocessing, model training, threshold tuning, API deployment, logging, and data drift detection.

The goal is to simulate a realistic ML engineering workflow rather than just training a model.

---

## Features

- End-to-end ML pipeline (EDA → training → deployment)
- XGBoost model with preprocessing pipeline
- Threshold tuning to optimise recall for churn detection
- FastAPI service with `/predict` endpoint
- Prediction logging to CSV
- Data drift detection using KS test
- Dockerised API for deployment

---

## Dataset

- Source: Kaggle Telco Customer Churn
- ~7,000 customers
- Binary classification: churn vs non-churn
- Mix of categorical and numeric features

---

## Model

- Algorithm: XGBoost
- Preprocessing:
  - One-hot encoding for categorical features
  - Numeric features passed through
- Train/test split with stratification

### Performance (Test Set)

- ROC-AUC: ~0.83  
- F1 Score: ~0.61  
- Recall: ~0.76  
- Precision: ~0.51  

### Threshold Tuning

Default threshold (0.5) resulted in low recall.  
Lowering threshold to 0.3 significantly improved churn detection.

---

## Project Structure

```
customer-churn-prediction-service/
├── app/
├── src/
├── artifacts/
├── data/
│   └── raw/
├── notebooks/
├── logs/
├── Dockerfile
├── requirements.txt
└── README.md
```


---

## API

### Run locally

uvicorn app.main:app --reload

### Run with Docker

docker build -t churn-api .
docker run -p 8000:8000 churn-api

### Endpoints

GET /health  
POST /predict  

### Example Request

{
  "gender": "Female",
  "SeniorCitizen": "0",
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "DSL",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 70.5,
  "TotalCharges": 845.5
}

### Example Response

{
  "churn_probability": 0.7421,
  "predicted_class": 1,
  "threshold": 0.3
}

---

## Training

python src/train.py

---

## Drift Detection

python src/drift.py

---

## Tech Stack

Python  
XGBoost  
scikit-learn  
FastAPI  
Docker  
pandas / numpy
