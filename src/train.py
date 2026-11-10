"""
Training Pipeline for Heart Disease Risk Prediction.
Evaluates multiple models, selects the best performer, and serializes the pipeline.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline


FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal"
]


def generate_sample_data(filepath: str, n_samples: int = 300) -> pd.DataFrame:
    """Generates synthetic patient data matching the UCI Cleveland Heart Disease schema."""
    np.random.seed(42)
    ages = np.random.randint(29, 77, n_samples)
    sexes = np.random.choice([0, 1], n_samples, p=[0.32, 0.68])
    cps = np.random.choice([0, 1, 2, 3], n_samples, p=[0.48, 0.17, 0.28, 0.07])
    trestbps = np.random.randint(94, 200, n_samples)
    chol = np.random.randint(126, 564, n_samples)
    fbs = np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
    restecg = np.random.choice([0, 1, 2], n_samples, p=[0.49, 0.48, 0.03])
    thalach = np.random.randint(71, 202, n_samples)
    exang = np.random.choice([0, 1], n_samples, p=[0.67, 0.33])
    oldpeak = np.round(np.random.uniform(0.0, 6.2, n_samples), 1)
    slope = np.random.choice([0, 1, 2], n_samples, p=[0.07, 0.46, 0.47])
    ca = np.random.choice([0, 1, 2, 3], n_samples, p=[0.58, 0.22, 0.13, 0.07])
    thal = np.random.choice([1, 2, 3], n_samples, p=[0.06, 0.55, 0.39])

    # Realistic correlation for target risk score
    risk_score = (
        (ages > 55).astype(int) * 1.2 +
        (sexes == 1).astype(int) * 0.8 +
        (cps > 0).astype(int) * 1.5 +
        (trestbps > 140).astype(int) * 1.0 +
        (chol > 240).astype(int) * 0.9 +
        (thalach < 140).astype(int) * 1.1 +
        (exang == 1).astype(int) * 1.4 +
        (oldpeak > 1.5).astype(int) * 1.3 +
        np.random.normal(0, 1.0, n_samples)
    )
    targets = (risk_score > np.median(risk_score)).astype(int)

    df = pd.DataFrame({
        "age": ages, "sex": sexes, "cp": cps, "trestbps": trestbps,
        "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
        "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca,
        "thal": thal, "target": targets
    })

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"[+] Sample dataset saved to {filepath} ({n_samples} records).")
    return df


def load_dataset(data_path: str) -> pd.DataFrame:
    """Loads dataset from file or generates sample data if not present."""
    if not os.path.exists(data_path):
        print(f"[!] Dataset not found at {data_path}. Generating sample benchmark data...")
        return generate_sample_data(data_path)
    print(f"[+] Loading dataset from {data_path}...")
    return pd.read_csv(data_path)


def train_and_evaluate():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_path = os.path.join(project_root, "data", "sample_heart.csv")
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)

    df = load_dataset(data_path)
    X = df[FEATURE_NAMES]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(random_state=42, max_iter=1000))
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42))
        ])
    }

    print("\n" + "=" * 50)
    print(" [*] MODEL TRAINING & BENCHMARKING REPORT")
    print("=" * 50)

    best_score = 0.0
    best_model_name = None
    best_pipeline = None

    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cv_scores = cross_val_score(pipeline, X, y, cv=5)

        print(f"\nModel: {name}")
        print(f"  Accuracy:  {acc * 100:.2f}%")
        print(f"  Precision: {prec:.3f}")
        print(f"  Recall:    {rec:.3f}")
        print(f"  F1-Score:  {f1:.3f}")
        print(f"  5-Fold CV: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)")

        if f1 > best_score:
            best_score = f1
            best_model_name = name
            best_pipeline = pipeline

    print("\n" + "-" * 50)
    print(f"[+] Best Selected Model: {best_model_name} (F1: {best_score:.3f})")
    print("-" * 50)

    model_output_path = os.path.join(models_dir, "heart_disease_model.joblib")
    joblib.dump(best_pipeline, model_output_path)
    print(f"[+] Serialized model successfully saved to: {model_output_path}")

    # Detailed report on test set
    y_pred_best = best_pipeline.predict(X_test)
    print("\nClassification Matrix:")
    print(confusion_matrix(y_test, y_pred_best))
    print("\nDetailed Metrics:")
    print(classification_report(y_test, y_pred_best, target_names=["Low Risk (0)", "High Risk (1)"]))


if __name__ == "__main__":
    train_and_evaluate()
