from pathlib import Path
import csv
import joblib
import pandas as pd
from fastapi import FastAPI
from app.schemas import ChurnInput

app = FastAPI(title="Churn Prediction API")

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

model = joblib.load(ARTIFACTS_DIR / "xgb_churn_model.joblib")
threshold = joblib.load(ARTIFACTS_DIR / "decision_threshold.joblib")

LOG_FILE = LOGS_DIR / "predictions.csv"

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: ChurnInput):
    input_dict = data.model_dump()
    input_df = pd.DataFrame([input_dict])

    proba = model.predict_proba(input_df)[:, 1][0]
    pred = int(proba > threshold)

    log_row = {
        **input_dict,
        "churn_probability": round(float(proba), 4),
        "predicted_class": pred,
        "threshold": threshold
    }

    file_exists = LOG_FILE.exists()
    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=log_row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_row)

    return {
        "churn_probability": round(float(proba), 4),
        "predicted_class": pred,
        "threshold": threshold
    }