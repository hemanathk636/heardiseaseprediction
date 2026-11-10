# 🫀 Heart Disease Risk Prediction System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end Machine Learning classification pipeline designed to predict patient cardiovascular disease risk based on clinical biomarkers.

---

## 📌 Problem Overview
Cardiovascular diseases (CVDs) are the leading cause of death globally. Early diagnosis and proactive clinical intervention can significantly reduce mortality. This project leverages supervised machine learning to classify patients as either **high risk** or **low risk** using standard physiological markers (e.g., blood pressure, cholesterol, resting ECG, max heart rate).

---

## ✨ Key Features
- **Exploratory Data Analysis (EDA):** Correlation heatmaps, distribution plots, and outlier detection.
- **Robust Preprocessing Pipeline:** Handles missing values, performs feature scaling (`StandardScaler`), and encodes categorical variables.
- **Model Benchmarking:** Evaluates multiple classifiers including Logistic Regression and Random Forest.
- **Comprehensive Evaluation:** Accuracy, Precision, Recall, F1-score, and Confusion Matrix analysis.
- **Modular Architecture:** Clean separation of data loading, training, serialization, and inference.

---

## 📊 Dataset & Features
Based on the UCI Heart Disease dataset (Cleveland Clinic).

| Feature | Description |
|---|---|
| `age` | Age of patient (years) |
| `sex` | Sex (1 = male, 0 = female) |
| `cp` | Chest pain type (0-3) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = true, 0 = false) |
| `restecg` | Resting electrocardiographic results (0-2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise induced angina (1 = yes, 0 = no) |
| `oldpeak` | ST depression induced by exercise relative to rest |
| `slope` | Slope of the peak exercise ST segment (0-2) |
| `ca` | Number of major vessels colored by fluoroscopy (0-3) |
| `thal` | Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect) |
| `target` | Diagnosis (0 = no disease, 1 = heart disease detected) |

---

## 🚀 Quickstart & Installation

### 1. Clone the repository
```bash
git clone https://github.com/hemanathk636/heart-disease-prediction.git
cd heart-disease-prediction
```

### 2. Set up virtual environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux / macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Train the model
```bash
python src/train.py
```
This generates and saves the trained pipeline into `models/heart_disease_model.joblib`.

### 4. Run an inference / prediction
```bash
python src/predict.py
```

### 5. Interactive CLI Demonstration
```bash
python app.py
```

---

## 📈 Model Performance
| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | 84.5% | 0.83 | 0.85 | 0.84 |
| **Random Forest (Best)** | **88.2%** | **0.87** | **0.89** | **0.88** |

---

## 📁 Repository Structure
```text
heart-disease-prediction/
├── data/
│   └── sample_heart.csv          # Sample dataset for demonstration
├── models/
│   └── heart_disease_model.joblib # Serialized model pipeline
├── src/
│   ├── __init__.py
│   ├── train.py                  # Training pipeline & evaluation
│   └── predict.py                # Single & batch inference module
├── app.py                        # Interactive console application
├── requirements.txt              # Project dependencies
├── .gitignore                    # Ignored artifacts
├── LICENSE                       # MIT License
└── README.md                     # Documentation
```

---

## 👤 Author
**Hemanath K**  
- GitHub: [@hemanathk636](https://github.com/hemanathk636)  
- LinkedIn: [Hemanath K](https://www.linkedin.com/in/hemanath)  
- LeetCode: [Hemanath25](https://leetcode.com/u/Hemanath25/)  
- HackerRank: [KhemanathBtech24](https://hackerrank.com/profile/KhemanathBtech24)
