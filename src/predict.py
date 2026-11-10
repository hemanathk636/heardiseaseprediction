"""
Inference Module for Heart Disease Risk Prediction.
Loads the serialized model and performs individual and batch predictions.
"""

import os
import joblib
import pandas as pd
from typing import Dict, Any, Tuple


FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal"
]


def load_model(model_path: str = None):
    """Loads the serialized model pipeline."""
    if model_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        model_path = os.path.join(project_root, "models", "heart_disease_model.joblib")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. Please run `python src/train.py` first to train and save the model."
        )

    return joblib.load(model_path)


def predict_risk(patient_data: Dict[str, Any], model=None) -> Tuple[int, str, float]:
    """
    Predicts heart disease risk for a single patient dictionary.
    
    Returns:
        (prediction, label, probability_percentage)
    """
    if model is None:
        model = load_model()

    df_input = pd.DataFrame([patient_data])[FEATURE_NAMES]
    pred = model.predict(df_input)[0]

    # Calculate probability if model supports it
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df_input)[0][1] * 100
    else:
        proba = 100.0 if pred == 1 else 0.0

    label = "High Risk of Heart Disease" if pred == 1 else "Low Risk / Healthy"
    return int(pred), label, round(proba, 2)


if __name__ == "__main__":
    print("\n[*] Running Sample Patient Inferences:")
    print("-" * 55)

    try:
        model = load_model()
    except FileNotFoundError as e:
        print(f"[!] {e}")
        exit(1)

    # Sample Patient 1 (Low Risk Profile: young, active, normal BP and cholesterol)
    patient_low_risk = {
        "age": 35, "sex": 1, "cp": 0, "trestbps": 115, "chol": 180,
        "fbs": 0, "restecg": 0, "thalach": 172, "exang": 0, "oldpeak": 0.0,
        "slope": 2, "ca": 0, "thal": 2
    }

    # Sample Patient 2 (High Risk Profile: older, high BP, high cholesterol, angina)
    patient_high_risk = {
        "age": 64, "sex": 1, "cp": 2, "trestbps": 160, "chol": 286,
        "fbs": 1, "restecg": 1, "thalach": 108, "exang": 1, "oldpeak": 2.6,
        "slope": 1, "ca": 2, "thal": 3
    }

    p1_pred, p1_label, p1_prob = predict_risk(patient_low_risk, model)
    print(f"Patient 1 (Age 35, Resting BP 115):")
    print(f"  Result:      {p1_label}")
    print(f"  Confidence:  {p1_prob}% probability of disease\n")

    p2_pred, p2_label, p2_prob = predict_risk(patient_high_risk, model)
    print(f"Patient 2 (Age 64, Resting BP 160, Angina):")
    print(f"  Result:      {p2_label}")
    print(f"  Confidence:  {p2_prob}% probability of disease")
    print("-" * 55)
