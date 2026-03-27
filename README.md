# AI-Driven Diabetes Detection - Clinical Decision Support

Implementation of the research paper "AI Driven Approaches for Diabetes Detection in Healthcare".

This repository contains a full-stack web application for diabetes risk prediction with explainable AI. The currently implemented model pipeline is **TabNet** (not RandomForest/XGBoost) and includes SHAP-based local/global explanation support.

---

## What This Project Does

1. Accepts patient clinical features from a web UI.
2. Predicts diabetes probability and class (Diabetic / Non-Diabetic).
3. Assigns risk category and confidence level.
4. Provides local SHAP explanation per prediction.
5. Shows global feature importance, ROC data, and model metrics on dashboard.
6. Supports authentication, user profile, and per-user prediction history.

---

## Current System Architecture

### 1. Frontend (React + Vite + Tailwind)

- Multi-page UI: Predict, Dashboard, History, Profile.
- API client uses Axios and JWT from localStorage.
- Dev proxy forwards `/api/*` to backend (`http://localhost:8000`).

### 2. Backend (FastAPI)

- Loads model/runtime artefacts on startup.
- Handles prediction, explainability, analytics, auth, and history endpoints.
- Performs zero-value imputation using dataset medians for physiological columns.

### 3. ML Pipeline (Training)

- Dataset: Pima Indians Diabetes Dataset.
- Preprocessing: median imputation + StandardScaler.
- Class balancing: SMOTE.
- Model: TabNetClassifier.
- Validation: Stratified K-Fold CV.
- Tuning: Optuna.
- Explainability: SHAP KernelExplainer + TabNet attention importance.

### 4. Storage

- Dataset CSV in `data/diabetes.csv`.
- Model artefacts in `models/`.
- User/prediction records in SQL database (SQLite fallback by default).

---

## Project Structure

```text
Diabetes-detection/
  backend/
    __init__.py
    auth.py
    database.py
    main.py
    requirements.txt
  data/
    diabetes.csv
    download_dataset.py
  frontend/
    index.html
    package.json
    vite.config.js
    src/
      App.jsx
      main.jsx
      components/
        PredictionResult.jsx
        ShapExplanation.jsx
      pages/
        About.jsx
        Dashboard.jsx
        History.jsx
        PatientForm.jsx
        Profile.jsx
      services/
        api.js
  models/
    artefacts.json
  notebooks/
    train_pipeline.py
  README.md
```

---

## Data and Features

The model uses 8 features from the Pima Indians Diabetes dataset:

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age

Target column: `Outcome` (0 = non-diabetic, 1 = diabetic).

Dataset source is downloaded by `data/download_dataset.py` and stored as `data/diabetes.csv`.

---

## Model Performance (Current Artefact)

From `models/artefacts.json` currently in this repository:

- Accuracy: `0.7422`
- Precision: `0.6044`
- Recall: `0.7651`
- F1 score: `0.6744`
- ROC-AUC: `0.8186`

Confusion matrix (aggregated from CV predictions):

```text
[[365, 135],
 [ 63, 205]]
```

Note: metrics can change after retraining with different Optuna trials/epochs.

---

## Setup and Run

### Prerequisites

- Python 3.10+
- Node.js 18+
- pip and npm

### 1) Install backend dependencies

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
# source venv/bin/activate

pip install -r backend/requirements.txt
```

### 2) Prepare dataset and model artefacts

```bash
python data/download_dataset.py
python notebooks/train_pipeline.py
```

Training outputs expected by backend:

- `models/tabnet_model.zip`
- `models/scaler.joblib`
- `models/shap_background.joblib`
- `models/artefacts.json`

### 3) Start backend

Run from project root:

```bash
uvicorn backend.main:app --reload --port 8000
```

Backend URLs:

- API base: `http://localhost:8000`
- Swagger docs: `http://localhost:8000/docs`

### 4) Start frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend URL:

- `http://localhost:3000`

---

## API Endpoints (Implemented)

| Method | Endpoint | Auth | Purpose |
|---|---|---|---|
| GET | `/health` | No | Service and model load status |
| POST | `/register` | No | Create user account |
| POST | `/login` | No | Get JWT access token |
| POST | `/predict` | Optional | Predict risk (saved to history only if logged in) |
| GET | `/explain` | Yes | SHAP explanation for a saved prediction |
| GET | `/feature-importance` | No | Global SHAP and TabNet attention importance |
| GET | `/metrics` | No | Stored model evaluation metrics |
| GET | `/roc` | No | ROC curve points |
| GET | `/history` | Yes | Logged-in user's prediction history and trend |
| GET | `/profile` | Yes | Logged-in user profile |

---

## Sample Prediction Request

```json
{
  "pregnancies": 6,
  "glucose": 148,
  "blood_pressure": 72,
  "skin_thickness": 35,
  "insulin": 0,
  "bmi": 33.6,
  "diabetes_pedigree": 0.627,
  "age": 50
}
```

Example response shape:

```json
{
  "prediction_id": 12,
  "prediction": "Diabetic",
  "probability": 0.82,
  "risk_category": "High",
  "confidence": "High",
  "model_used": "tabnet",
  "lifestyle_recommendations": [
    "Consult an endocrinologist for comprehensive screening soon.",
    "Begin a structured weight and glucose management plan."
  ],
  "local_explanation": {
    "features": ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"],
    "shap_values": [0.01, 0.15, -0.02, 0.01, 0.00, 0.08, 0.03, 0.04],
    "base_value": 0.42
  },
  "timestamp": "2026-03-27T12:34:56.000000"
}
```

---

## Frontend Workflow Mapping

- `frontend/src/pages/PatientForm.jsx` -> `POST /predict`
- `frontend/src/components/PredictionResult.jsx` renders prediction/risk/confidence/recommendations.
- `frontend/src/components/ShapExplanation.jsx` renders local SHAP chart/table.
- `frontend/src/pages/Dashboard.jsx` -> `/metrics`, `/roc`, `/feature-importance`
- `frontend/src/pages/Profile.jsx` -> `/register`, `/login`, `/profile`
- `frontend/src/pages/History.jsx` -> `/history`

---

## Troubleshooting

1. **Backend startup fails with missing model file**

Run training pipeline first:

```bash
python notebooks/train_pipeline.py
```

2. **Dashboard empty or prediction errors**

- Ensure backend is running on port 8000.
- Ensure frontend runs on port 3000 (Vite proxy is configured for this).

3. **History/Profile returns 401**

- Login/Register from Profile page first.
- Confirm JWT token exists in browser localStorage.

---

## Disclaimer

This application is for educational and research purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.
