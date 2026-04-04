from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "telco_churn.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

MODEL_PATH = ARTIFACTS_DIR / "xgb_churn_model.joblib"
THRESHOLD_PATH = ARTIFACTS_DIR / "decision_threshold.joblib"
REFERENCE_PATH = ARTIFACTS_DIR / "training_reference.csv"


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Fix numeric column stored as string
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Drop small number of rows with missing TotalCharges
    df = df.dropna().copy()

    # Clean and encode target
    df["Churn"] = (
        df["Churn"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"yes": 1, "no": 0})
    )

    # Treat SeniorCitizen as categorical to match training assumptions
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)

    # Drop ID column
    df = df.drop(columns=["customerID"])

    return df


def build_pipeline(categorical_cols, numeric_cols) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
            )),
        ]
    )

    return pipeline


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, threshold: float) -> None:
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba > threshold).astype(int)

    print(f"Threshold: {threshold}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))
    print("F1:", f1_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))


def main() -> None:
    df = load_data(DATA_PATH)
    df = clean_data(df)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    categorical_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=["object", "string"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    model = build_pipeline(categorical_cols, numeric_cols)
    model.fit(X_train, y_train)

    threshold = 0.3
    evaluate_model(model, X_test, y_test, threshold)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(threshold, THRESHOLD_PATH)
    X_train.to_csv(REFERENCE_PATH, index=False)

    print(f"\nSaved model to: {MODEL_PATH}")
    print(f"Saved threshold to: {THRESHOLD_PATH}")
    print(f"Saved training reference data to: {REFERENCE_PATH}")


if __name__ == "__main__":
    main()