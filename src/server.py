"""
Full-Stack Web Server for Heart Disease Risk Assessment.
Serves both the REST API (/api/predict) and the Frontend Web Application.
"""

import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Include current directory in path to import predict module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from predict import load_model, predict_risk

app = FastAPI(
    title="Heart Disease Risk Prediction API",
    description="Machine Learning REST API and Web Interface for clinical risk prediction",
    version="1.0.0"
)

# Enable CORS for local client development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Patient Data Schema
class PatientData(BaseModel):
    age: int = Field(..., ge=1, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="0: Female, 1: Male")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type (0-3)")
    trestbps: float = Field(..., ge=50, le=260, description="Resting blood pressure")
    chol: float = Field(..., ge=50, le=700, description="Serum cholesterol in mg/dl")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results (0-2)")
    thalach: float = Field(..., ge=40, le=250, description="Max heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina (0-1)")
    oldpeak: float = Field(..., ge=0.0, le=10.0, description="ST depression (oldpeak)")
    slope: int = Field(..., ge=0, le=2, description="Slope of peak exercise ST")
    ca: int = Field(..., ge=0, le=3, description="Number of major vessels (0-3)")
    thal: int = Field(..., ge=1, le=3, description="Thalassemia (1-3)")


# Load model at startup
try:
    ml_model = load_model()
    print("[+] Machine learning model successfully loaded into server memory.")
except Exception as e:
    print(f"[!] Warning: Failed to load model: {e}")
    ml_model = None


@app.get("/api/health")
def health_check():
    return {"status": "online", "model_loaded": ml_model is not None}


@app.post("/api/predict")
def api_predict_risk(patient: PatientData):
    if ml_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded on server.")
    
    patient_dict = patient.model_dump()
    pred, label, proba = predict_risk(patient_dict, ml_model)
    
    recommendation = (
        "Immediate clinical follow-up is recommended with a cardiologist for comprehensive ECG and stress-test evaluation."
        if pred == 1 else
        "Clinical biomarkers are within expected baseline ranges. Regular preventative screenings recommended."
    )
    
    return {
        "prediction": pred,
        "label": label,
        "probability": proba,
        "recommendation": recommendation
    }


# Mount the frontend directory so HTML/CSS/JS are served directly
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
frontend_dir = os.path.join(project_root, "frontend")

if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(" 🚀 FULL-STACK SERVER STARTING")
    print(" Web App: http://localhost:8000")
    print(" API Docs: http://localhost:8000/docs")
    print("=" * 60 + "\n")
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
