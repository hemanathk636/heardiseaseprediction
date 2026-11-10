"""
Interactive CLI Application for Heart Disease Prediction.
Allows users to evaluate patient clinical indicators in real-time.
"""

import sys
import os

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from predict import load_model, predict_risk


def run_interactive_assessment():
    print("\n" + "=" * 60)
    print(" HEART DISEASE RISK ASSESSMENT SYSTEM (CLI)")
    print("=" * 60)
    print("Please enter patient clinical values (press Enter for defaults):\n")

    try:
        model = load_model()
    except FileNotFoundError:
        print("[!] Model file not found. Running training pipeline first...")
        from train import train_and_evaluate
        train_and_evaluate()
        model = load_model()

    def get_input(prompt, default, cast_type=float):
        val = input(f"{prompt} [default: {default}]: ").strip()
        if not val:
            return default
        try:
            return cast_type(val)
        except ValueError:
            print(f"Invalid input, using default {default}")
            return default

    age = get_input("Age in years", 52, int)
    sex = get_input("Sex (1 = male, 0 = female)", 1, int)
    cp = get_input("Chest pain type (0: typical, 1: atypical, 2: non-anginal, 3: asymptomatic)", 0, int)
    trestbps = get_input("Resting blood pressure (mm Hg)", 130, int)
    chol = get_input("Serum cholesterol (mg/dl)", 240, int)
    fbs = get_input("Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)", 0, int)
    restecg = get_input("Resting ECG results (0: normal, 1: ST-T wave, 2: left vent)", 1, int)
    thalach = get_input("Maximum heart rate achieved", 150, int)
    exang = get_input("Exercise-induced angina (1 = yes, 0 = no)", 0, int)
    oldpeak = get_input("ST depression induced by exercise (oldpeak)", 1.2, float)
    slope = get_input("Slope of peak exercise ST segment (0-2)", 1, int)
    ca = get_input("Number of major vessels (0-3)", 0, int)
    thal = get_input("Thalassemia (1: normal, 2: fixed defect, 3: reversible defect)", 2, int)

    patient_payload = {
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
    }

    pred, label, proba = predict_risk(patient_payload, model)

    print("\n" + "=" * 60)
    print(" ASSESSMENT RESULTS")
    print("=" * 60)
    print(f"Prediction: {label}")
    print(f"Estimated Risk Probability: {proba}%")
    if pred == 1:
        print("\n[!] Recommendation: Clinical follow-up recommended with a cardiologist.")
    else:
        print("\n[+] Recommendation: Biomarkers within normal expected ranges.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_interactive_assessment()
