# AI-Driven Diabetes Detection — Clinical Decision Support

Implementation of the research paper **"AI Driven Approaches for Diabetes Detection in Healthcare"**.

A full-stack web application that predicts diabetes risk using Random Forest and XGBoost classifiers, with SHAP-based explainability.

---

## Project Structure

```
diabetes-detection/
├── backend/              # FastAPI REST API
│   ├── main.py
│   └── requirements.txt
├── frontend/             # React + TailwindCSS UI
│   ├── src/
│   │   ├── pages/        # PatientForm, Dashboard, About
│   │   ├── components/   # PredictionResult, ShapExplanation
│   │   └── services/     # API client
│   └── package.json
├── models/               # Saved models & artefacts (generated)
├── data/                 # Dataset & downloader
│   └── download_dataset.py
├── notebooks/            # Training pipeline
│   └── train_pipeline.py
└── README.md
```

---

## Quick Start

### Prerequisites

- **Python 3.10+**
- **Node.js 18+**
- pip / npm

### 1. Set up Python environment

```bash
cd diabetes-detection

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt
```

### 2. Download dataset & train models

```bash
# Download Pima Indians Diabetes Dataset
python data/download_dataset.py

# Train Random Forest + XGBoost, compute SHAP, save models
python notebooks/train_pipeline.py
```

This will:
- Download the dataset to `data/diabetes.csv`
- Train both models with cross-validation
- Compute evaluation metrics (accuracy, precision, recall, F1, ROC-AUC)
- Generate SHAP feature importance values
- Save models to `models/` directory

### 3. Start the backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

API available at: **http://localhost:8000**

API docs at: **http://localhost:8000/docs**

### 4. Start the frontend

```bash
cd frontend
npm install
npm run dev
```

App available at: **http://localhost:3000**

---

## API Endpoints

| Method | Endpoint              | Description                              |
|--------|-----------------------|------------------------------------------|
| POST   | `/predict`            | Predict diabetes risk for a patient      |
| POST   | `/explain`            | SHAP explanation for a prediction        |
| GET    | `/feature-importance` | Global SHAP feature importance           |
| GET    | `/metrics`            | Model evaluation metrics                 |
| GET    | `/roc`                | ROC curve data                           |
| GET    | `/health`             | Health check                             |

### Sample Predict Request

```json
POST /predict
{
  "pregnancies": 6,
  "glucose": 148,
  "blood_pressure": 72,
  "skin_thickness": 35,
  "insulin": 0,
  "bmi": 33.6,
  "diabetes_pedigree": 0.627,
  "age": 50,
  "model": "xgboost"
}
```

### Sample Response

```json
{
  "prediction": "Diabetic",
  "probability": 0.8234,
  "confidence": "High",
  "risk_category": "High",
  "model_used": "xgboost"
}
```

---

## Features

- **Dual Model Support** — Switch between Random Forest and XGBoost
- **Risk Categories** — Low / Medium / High based on probability thresholds
- **Confidence Score** — How confident the model is in its prediction
- **SHAP Explanations** — Per-patient feature contribution analysis
- **Global Feature Importance** — Which features matter most overall
- **ROC Curves** — Visual model performance comparison
- **Model Comparison Table** — Side-by-side metrics for both models
- **Sample Data** — One-click fill for testing

---

## Methodology

1. **Dataset**: Pima Indians Diabetes Dataset (768 samples, 8 features)
2. **Preprocessing**: Zero → NaN replacement, median imputation, StandardScaler
3. **Models**: RandomForestClassifier (200 trees), XGBClassifier (200 rounds)
4. **Evaluation**: Accuracy, Precision, Recall, F1, ROC-AUC
5. **Explainability**: SHAP TreeExplainer for both global and local explanations

---

## Tech Stack

| Layer    | Technology                        |
|----------|-----------------------------------|
| ML       | scikit-learn, XGBoost, SHAP       |
| Backend  | Python, FastAPI, Uvicorn          |
| Frontend | React 18, TailwindCSS, Chart.js   |
| Data     | pandas, NumPy                     |

---

## Disclaimer

This application is for **educational and research purposes only**. It is not a substitute for professional medical advice, diagnosis, or treatment.
