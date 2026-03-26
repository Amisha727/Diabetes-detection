"""
FastAPI backend for AI-driven diabetes detection clinical decision support.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Annotated

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field
from pytorch_tabnet.tab_model import TabNetClassifier
from sqlalchemy.orm import Session

from .auth import create_access_token, get_current_user, hash_password, require_auth, verify_password
from .database import Prediction, User, get_db, init_db

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
DATA_PATH = os.path.join(PROJECT_DIR, "data", "diabetes.csv")

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

ZERO_IMPUTE_COLUMNS = ["Glucose", "BloodPressure",
                       "SkinThickness", "Insulin", "BMI"]


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(min_length=6, max_length=128)
    full_name: str = Field(default="", max_length=100)


class LoginRequest(BaseModel):
    username: str
    password: str


class PredictionInput(BaseModel):
    pregnancies: float = Field(ge=0)
    glucose: float = Field(ge=0)
    blood_pressure: float = Field(ge=0)
    skin_thickness: float = Field(ge=0)
    insulin: float = Field(ge=0)
    bmi: float = Field(ge=0)
    diabetes_pedigree: float = Field(ge=0)
    age: float = Field(ge=1)


class PredictionResponse(BaseModel):
    prediction_id: int | None
    prediction: str
    probability: float
    risk_category: str
    confidence: str
    model_used: str
    lifestyle_recommendations: list[str]
    local_explanation: dict
    timestamp: str


app = FastAPI(title="AI Driven Diabetes Detection API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Runtime artefacts
MODEL: TabNetClassifier | None = None
SCALER = None
ARTEFACTS: dict = {}
SHAP_BACKGROUND = None
SHAP_EXPLAINER = None
MEDIANS: dict[str, float] = {}


def _load_medians() -> dict[str, float]:
    if not os.path.exists(DATA_PATH):
        return {
            "Glucose": 117.0,
            "BloodPressure": 72.0,
            "SkinThickness": 23.0,
            "Insulin": 30.5,
            "BMI": 32.0,
        }

    df = pd.read_csv(DATA_PATH)
    medians: dict[str, float] = {}
    for col in ZERO_IMPUTE_COLUMNS:
        non_zero = df[col].replace(0, np.nan)
        medians[col] = float(non_zero.median())
    return medians


def _load_runtime_artefacts() -> None:
    global MODEL, SCALER, ARTEFACTS, SHAP_BACKGROUND

    model_zip = os.path.join(MODELS_DIR, "tabnet_model.zip")
    scaler_path = os.path.join(MODELS_DIR, "scaler.joblib")
    artefacts_path = os.path.join(MODELS_DIR, "artefacts.json")
    shap_bg_path = os.path.join(MODELS_DIR, "shap_background.joblib")

    if not os.path.exists(model_zip):
        raise RuntimeError(
            "Missing model file models/tabnet_model.zip. Run notebooks/train_pipeline.py first.")
    if not os.path.exists(scaler_path):
        raise RuntimeError(
            "Missing scaler file models/scaler.joblib. Run notebooks/train_pipeline.py first.")
    if not os.path.exists(artefacts_path):
        raise RuntimeError(
            "Missing artefacts file models/artefacts.json. Run notebooks/train_pipeline.py first.")

    model = TabNetClassifier()
    model.load_model(model_zip)

    MODEL = model
    SCALER = joblib.load(scaler_path)
    with open(artefacts_path, "r", encoding="utf-8") as f:
        ARTEFACTS = json.load(f)

    if os.path.exists(shap_bg_path):
        SHAP_BACKGROUND = joblib.load(shap_bg_path)
    else:
        SHAP_BACKGROUND = np.zeros((50, len(FEATURES)), dtype=np.float32)


def _feature_array(payload: PredictionInput) -> np.ndarray:
    values = {
        "Pregnancies": float(payload.pregnancies),
        "Glucose": float(payload.glucose),
        "BloodPressure": float(payload.blood_pressure),
        "SkinThickness": float(payload.skin_thickness),
        "Insulin": float(payload.insulin),
        "BMI": float(payload.bmi),
        "DiabetesPedigreeFunction": float(payload.diabetes_pedigree),
        "Age": float(payload.age),
    }

    for col in ZERO_IMPUTE_COLUMNS:
        if values[col] == 0:
            values[col] = MEDIANS.get(col, values[col])

    ordered = np.array([[values[name] for name in FEATURES]], dtype=np.float32)
    return ordered


def _risk_category(probability: float) -> str:
    if probability >= 0.7:
        return "High"
    if probability >= 0.4:
        return "Medium"
    return "Low"


def _confidence(probability: float) -> str:
    distance = abs(probability - 0.5)
    if distance >= 0.3:
        return "High"
    if distance >= 0.15:
        return "Medium"
    return "Low"


def _recommendations(probability: float) -> list[str]:
    common = [
        "Maintain regular physical activity (at least 150 minutes/week).",
        "Adopt a balanced, low-refined-sugar diet with high fiber.",
        "Track fasting glucose and schedule periodic clinical follow-up.",
    ]
    if probability >= 0.7:
        return [
            "Consult an endocrinologist for comprehensive screening soon.",
            "Begin a structured weight and glucose management plan.",
            "Discuss HbA1c, lipid profile, and blood pressure monitoring with your clinician.",
        ] + common
    if probability >= 0.4:
        return [
            "Increase moderate exercise frequency and reduce sedentary time.",
            "Prioritize carbohydrate quality and portion control.",
            "Reassess metabolic markers within 3 months.",
        ] + common
    return [
        "Continue healthy lifestyle habits and preventive screening.",
        "Sustain hydration and sleep hygiene to support metabolic health.",
    ] + common


def _get_shap_explainer():
    global SHAP_EXPLAINER
    if SHAP_EXPLAINER is None:
        if MODEL is None:
            raise RuntimeError("Model is not loaded")
        background = SHAP_BACKGROUND
        if len(background) > 100:
            background = background[:100]
        SHAP_EXPLAINER = shap.KernelExplainer(MODEL.predict_proba, background)
    return SHAP_EXPLAINER


def _local_shap(scaled_features: np.ndarray) -> dict:
    explainer = _get_shap_explainer()
    shap_values = explainer.shap_values(scaled_features, nsamples=120)

    if isinstance(shap_values, list):
        local = np.asarray(shap_values[1])[0]
    else:
        local_arr = np.asarray(shap_values)
        if local_arr.ndim == 3:
            local = local_arr[0, :, 1]
        else:
            local = local_arr[0]

    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base = float(np.asarray(base_value)[1])
    else:
        base = float(base_value)

    return {
        "features": FEATURES,
        "shap_values": [float(v) for v in local],
        "base_value": base,
    }


@app.on_event("startup")
def startup() -> None:
    global MEDIANS
    init_db()
    MEDIANS = _load_medians()
    _load_runtime_artefacts()


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/register")
def register(payload: RegisterRequest, db: Session = Depends(get_db)) -> dict:
    existing = db.query(User).filter(
        (User.username == payload.username) | (User.email == payload.email)).first()
    if existing:
        raise HTTPException(
            status_code=400, detail="Username or email already exists")

    user = User(
        username=payload.username,
        email=payload.email,
        hashed_password=hash_password(payload.password),
        full_name=payload.full_name,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token({"sub": user.username})
    return {
        "message": "Registration successful",
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "full_name": user.full_name,
        },
    }


@app.post("/login")
def login(payload: LoginRequest, db: Session = Depends(get_db)) -> dict:
    user = db.query(User).filter(User.username == payload.username).first()
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(
            status_code=401, detail="Invalid username or password")

    token = create_access_token({"sub": user.username})
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "full_name": user.full_name,
        },
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(
    payload: PredictionInput,
    db: Session = Depends(get_db),
    current_user: Annotated[User | None, Depends(get_current_user)] = None,
):
    if MODEL is None or SCALER is None:
        raise HTTPException(
            status_code=500, detail="Model artefacts are not loaded")

    raw = _feature_array(payload)
    scaled = SCALER.transform(raw)

    probability = float(MODEL.predict_proba(scaled)[0][1])
    pred_class = int(probability >= 0.5)
    local_explanation = _local_shap(scaled)

    prediction_text = "Diabetic" if pred_class == 1 else "Non-Diabetic"
    risk = _risk_category(probability)
    confidence = _confidence(probability)

    saved_id = None
    if current_user is not None:
        record = Prediction(
            user_id=current_user.id,
            pregnancies=payload.pregnancies,
            glucose=payload.glucose,
            blood_pressure=payload.blood_pressure,
            skin_thickness=payload.skin_thickness,
            insulin=payload.insulin,
            bmi=payload.bmi,
            diabetes_pedigree=payload.diabetes_pedigree,
            age=payload.age,
            prediction=prediction_text,
            probability=probability,
            risk_category=risk,
            model_used="tabnet",
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        saved_id = record.id

    return PredictionResponse(
        prediction_id=saved_id,
        prediction=prediction_text,
        probability=probability,
        risk_category=risk,
        confidence=confidence,
        model_used="tabnet",
        lifestyle_recommendations=_recommendations(probability),
        local_explanation=local_explanation,
        timestamp=datetime.utcnow().isoformat(),
    )


@app.get("/explain")
def explain(
    prediction_id: int | None = Query(default=None),
    db: Session = Depends(get_db),
    current_user: Annotated[User, Depends(require_auth)] = None,
):
    query = db.query(Prediction).filter(Prediction.user_id == current_user.id)
    if prediction_id is not None:
        record = query.filter(Prediction.id == prediction_id).first()
    else:
        record = query.order_by(Prediction.created_at.desc()).first()

    if record is None:
        raise HTTPException(
            status_code=404, detail="No prediction found for explanation")

    payload = PredictionInput(
        pregnancies=record.pregnancies,
        glucose=record.glucose,
        blood_pressure=record.blood_pressure,
        skin_thickness=record.skin_thickness,
        insulin=record.insulin,
        bmi=record.bmi,
        diabetes_pedigree=record.diabetes_pedigree,
        age=record.age,
    )

    raw = _feature_array(payload)
    scaled = SCALER.transform(raw)
    local = _local_shap(scaled)

    return {
        "prediction_id": record.id,
        "features": local["features"],
        "feature_values": raw[0].tolist(),
        "shap_values": local["shap_values"],
        "base_value": local["base_value"],
        "model_used": record.model_used,
        "created_at": record.created_at.isoformat(),
    }


@app.get("/feature-importance")
def feature_importance() -> dict:
    fi = ARTEFACTS.get("feature_importance", {})
    return {
        "tabnet_shap": fi.get("tabnet_shap", {}),
        "tabnet_attention": fi.get("tabnet_attention", {}),
    }


@app.get("/metrics")
def metrics() -> dict:
    return ARTEFACTS.get("metrics", {})


@app.get("/roc")
def roc() -> dict:
    return ARTEFACTS.get("roc", {})


@app.get("/history")
def history(
    db: Session = Depends(get_db),
    current_user: Annotated[User, Depends(require_auth)] = None,
):
    rows = (
        db.query(Prediction)
        .filter(Prediction.user_id == current_user.id)
        .order_by(Prediction.created_at.desc())
        .all()
    )

    history_items = [
        {
            "id": r.id,
            "prediction": r.prediction,
            "probability": r.probability,
            "risk_category": r.risk_category,
            "model_used": r.model_used,
            "created_at": r.created_at.isoformat(),
            "inputs": {
                "pregnancies": r.pregnancies,
                "glucose": r.glucose,
                "blood_pressure": r.blood_pressure,
                "skin_thickness": r.skin_thickness,
                "insulin": r.insulin,
                "bmi": r.bmi,
                "diabetes_pedigree": r.diabetes_pedigree,
                "age": r.age,
            },
        }
        for r in rows
    ]

    trend = [
        {"timestamp": item["created_at"], "probability": item["probability"]}
        for item in reversed(history_items)
    ]

    return {
        "count": len(history_items),
        "history": history_items,
        "trend": trend,
    }


@app.get("/profile")
def profile(current_user: Annotated[User, Depends(require_auth)] = None) -> dict:
    return {
        "id": current_user.id,
        "username": current_user.username,
        "email": current_user.email,
        "full_name": current_user.full_name,
        "created_at": current_user.created_at.isoformat(),
    }
