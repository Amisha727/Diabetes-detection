# AI-Driven Diabetes Detection - Clinical Decision Support System

**A full-stack web application for diabetes risk prediction with explainable AI (SHAP + TabNet attention-based explanations)**

This repository implements a production-ready diabetes detection system based on machine learning research. The system provides:
- Real-time diabetes risk predictions with confidence scores
- Local (per-prediction) and global (model-level) explainability
- User authentication and prediction history tracking
- Dashboard with model performance metrics and feature importance visualizations
- Research-grade model evaluation with 5-fold cross-validation

**Implementation Status:** ✅ TabNet model with SHAP explainability (fully functional)

---

## 📋 Project Overview

### What This System Does

1. **Accept** patient clinical features (8 inputs) through an interactive web UI
2. **Predict** diabetes probability, risk category, and confidence level
3. **Explain** predictions using SHAP values (local) and attention-based feature importance (global)
4. **Authenticate** users with JWT tokens and track prediction history
5. **Visualize** model performance metrics, ROC curves, calibration plots, and feature importance
6. **Generate** publication-quality research plots for academic papers

### Key Features

- 🔐 User authentication with JWT tokens and password hashing (PBKDF2)
- 📊 Interactive dashboard with model metrics, ROC, and SHAP charts
- 💾 Prediction history with trend analysis
- 🧠 Explainable AI via SHAP KernelExplainer and TabNet attention masks
- 🎯 Risk stratification with lifestyle recommendations
- 📈 Hyperparameter tuning with Optuna
- ⚖️ Class balancing with SMOTE for imbalanced dataset
- 🔍 Health check endpoints for monitoring

---

## 🏗️ System Architecture (5-Tier)

```
┌─────────────────────────────────────────────────────────┐
│  TIER 1: Presentation Layer (Browser)                   │
│  ├─ React 18 SPA with Vite                              │
│  └─ TailwindCSS + Chart.js + React Icons               │
└────────────┬────────────────────────────────────────────┘
             │ API Calls (JSON)
┌────────────▼────────────────────────────────────────────┐
│  TIER 2: API Gateway (Vite Dev Proxy)                   │
│  ├─ Rewrites /api/* → http://localhost:8000/*           │
│  └─ Includes Authorization headers                      │
└────────────┬────────────────────────────────────────────┘
             │ HTTP/REST
┌────────────▼────────────────────────────────────────────┐
│  TIER 3: Application Server (FastAPI)                   │
│  ├─ RESTful endpoints (/register, /predict, etc.)       │
│  ├─ JWT authentication & authorization                  │
│  ├─ Request validation & error handling                 │
│  └─ Port: 8000 (configurable)                           │
└────────────┬────────────────────────────────────────────┘
             │ Python imports
┌────────────▼────────────────────────────────────────────┐
│  TIER 4: ML Runtime Layer                               │
│  ├─ TabNet model inference                              │
│  ├─ SHAP explanation engine                             │
│  ├─ Feature scaling & preprocessing                     │
│  └─ Model artefacts: tabnet_model.zip, scaler.joblib    │
└────────────┬────────────────────────────────────────────┘
             │ SQL/File I/O
┌────────────▼────────────────────────────────────────────┐
│  TIER 5: Data Layer                                      │
│  ├─ SQLite (default): diabetes_app.db                   │
│  ├─ PostgreSQL (production): configurable via env       │
│  ├─ Data files: diabetes.csv, model weights             │
│  └─ Predictions & User records persisted here           │
└─────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

### **Frontend**
- **React 18** - UI framework with hooks
- **Vite** - Build tool with dev server proxy support
- **TailwindCSS** - Utility-first CSS framework
- **Axios** - HTTP client with interceptors for JWT
- **Chart.js** - Data visualization (ROC, SHAP bar charts)
- **React Icons** - SVG icon library

### **Backend**
- **FastAPI** - Modern async Python web framework
- **Uvicorn** - ASGI server (with auto-reload in dev)
- **SQLAlchemy** - ORM for database abstraction
- **Pydantic** - Request/response validation
- **python-dotenv** - Environment configuration

### **Machine Learning**
- **PyTorch TabNet** - Attention-based deep learning for tabular data
- **scikit-learn** - Preprocessing, metrics, cross-validation
- **SHAP** - KernelExplainer for model interpretation
- **Imbalanced-learn** - SMOTE for class balancing
- **Optuna** - Hyperparameter optimization
- **pandas, numpy** - Data manipulation
- **joblib** - Model serialization

### **Data Visualization (Research)**
- **matplotlib** - Publication-quality plots (300 DPI)
- **seaborn** - Statistical visualization
- **scikit-learn** - Confusion matrix, calibration curves

### **Database**
- **SQLite** - Development default (file-based)
- **PostgreSQL** - Production option (via DATABASE_URL env)
- **SQLAlchemy ORM** - Database abstraction layer

---

## 📁 Project Structure

```
Diabetes-detection/
│
├── frontend/                          # React + Vite application
│   ├── index.html                     # Entry point
│   ├── package.json                   # npm dependencies
│   ├── vite.config.js                 # Vite config with proxy setup
│   ├── tailwind.config.js             # TailwindCSS config
│   ├── postcss.config.js              # PostCSS plugins
│   ├── .env.example                   # Environment template
│   └── src/
│       ├── main.jsx                   # React entry
│       ├── App.jsx                    # Main component
│       ├── index.css                  # Global styles
│       ├── components/
│       │   ├── PredictionResult.jsx   # Display prediction output
│       │   └── ShapExplanation.jsx    # SHAP visualization
│       ├── pages/
│       │   ├── PatientForm.jsx        # Input form & prediction
│       │   ├── Dashboard.jsx          # Metrics & charts
│       │   ├── History.jsx            # User prediction history
│       │   ├── Profile.jsx            # Auth & user profile
│       │   └── About.jsx              # Project info
│       └── services/
│           └── api.js                 # Axios client & endpoints
│
├── backend/                           # FastAPI application
│   ├── main.py                        # FastAPI app & endpoints
│   ├── auth.py                        # JWT, password hashing
│   ├── database.py                    # SQLAlchemy models
│   ├── requirements.txt               # Python dependencies
│   ├── __init__.py                    # Package marker
│   └── .env.example                   # Environment template
│
├── notebooks/                         # ML pipeline & utilities
│   ├── train_pipeline.py              # Model training & evaluation
│   └── generate_plots.py              # Research visualization script
│
├── data/                              # Dataset
│   ├── diabetes.csv                   # Pima Indians Diabetes (768 samples)
│   └── download_dataset.py            # Dataset download utility
│
├── models/                            # Trained artifacts
│   ├── tabnet_model.zip               # Serialized TabNet model
│   ├── scaler.joblib                  # StandardScaler for features
│   ├── artefacts.json                 # Metrics, importance, fold data
│   └── shap_background.joblib         # SHAP background samples
│
├── plots/                             # Generated research plots (output)
│   ├── 1_class_distribution.png       # Before/after SMOTE
│   ├── 2_confusion_matrix.png         # TN/FP/FN/TP with metrics
│   ├── 3_roc_curve.png                # ROC with AUC = 0.8265
│   ├── 4_calibration_curve.png        # Model reliability
│   ├── 5_fold_metrics_comparison.png  # 5 subplots: accuracy, precision, recall, F1, AUC
│   ├── 6_model_metrics_table.png      # Summary table
│   ├── 7_shap_feature_importance.png  # Global importance (Glucose: 0.1498)
│   ├── 8_hyperparameters_table.png    # Optuna best params
│   └── 9_tabnet_attention_importance.png  # Attention masks
│
├── venv/                              # Python virtual environment (created on setup)
├── .env                               # Environment variables (local, not tracked)
├── README.md                          # This file
├── build-output.txt                   # Build logs
└── .gitignore                         # Git ignore rules
```

---

## 📊 Data & Features

### Dataset: Pima Indians Diabetes Dataset
- **Source:** UCI Machine Learning Repository
- **Samples:** 768 (500 non-diabetic, 268 diabetic) → **balanced to 500:500 after SMOTE**
- **Features:** 8 clinical/anthropometric measurements
- **Target:** Binary classification (0 = non-diabetic, 1 = diabetic)

### Input Features (8)

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| **Pregnancies** | Integer | 0-17 | Number of times pregnant |
| **Glucose** | Integer | 0-199 | 2-hour plasma glucose concentration (mg/dL) |
| **BloodPressure** | Integer | 0-122 | Diastolic blood pressure (mmHg) |
| **SkinThickness** | Integer | 0-99 | Triceps skin fold thickness (mm) |
| **Insulin** | Integer | 0-846 | 2-hour serum insulin (mu U/ml) |
| **BMI** | Float | 0.0-67.1 | Body mass index (weight in kg/(height in m)²) |
| **DiabetesPedigreeFunction** | Float | 0.078-2.42 | Genetic risk score |
| **Age** | Integer | 21-81 | Age in years |

### Data Preprocessing

1. **Median Imputation** - Physiological zeroes (Glucose, BloodPressure, SkinThickness, Insulin, BMI) replaced with dataset median
2. **StandardScaler Normalization** - All features scaled to mean=0, std=1
3. **SMOTE Balancing** - Minority class oversampled 1.86x to achieve 1:1 ratio
4. **Train/Test Split** - 5-fold stratified cross-validation (no hold-out test set)

---

## 🤖 Model Performance

### Current Model: TabNet (Optuna-Tuned)

From `models/artefacts.json`:

| Metric | Value |
|--------|-------|
| Accuracy | 0.7226 |
| Precision | 0.5781 |
| Recall | 0.7648 |
| F1 Score | 0.6579 |
| ROC-AUC | 0.8265 |

### Confusion Matrix (Aggregated 5-Fold CV)

```
                    Predicted
                Non-Diabetic  Diabetic
Actual Non-Diabetic     350        150
Actual Diabetic          63        205
```

**Sensitivity (Recall):** 0.765 | **Specificity:** 0.700

### Cross-Validation Scores (Per Fold)

| Fold | Accuracy | Precision | Recall | F1    | ROC-AUC |
|------|----------|-----------|--------|-------|---------|
| 1    | 0.7266   | 0.6050    | 0.8000 | 0.6897 | 0.8299 |
| 2    | 0.7273   | 0.5763    | 0.7586 | 0.6547 | 0.8265 |
| 3    | 0.7208   | 0.5833    | 0.7742 | 0.6667 | 0.8294 |
| 4    | 0.7143   | 0.5625    | 0.7586 | 0.6471 | 0.8163 |
| 5    | 0.7195   | 0.5714    | 0.7419 | 0.6452 | 0.8256 |
| **Mean** | **0.7217** | **0.5797** | **0.7747** | **0.6607** | **0.8255** |

### Best Hyperparameters (Optuna)

| Parameter | Value |
|-----------|-------|
| n_d (feature dimension) | 9 |
| n_a (attention dimension) | 8 |
| n_steps (decision steps) | 4 |
| gamma (relaxation) | 1.137 |
| lambda_sparse (sparsity reg) | 0.0007 |
| momentum | 0.359 |
| mask_type | entmax |
| learning_rate | 0.0089 |
| batch_size | 256 |

### Feature Importance

**SHAP Global Importance (Mean |SHAP|):**
1. **Glucose** - 0.1498 (most important)
2. **BMI** - 0.0669
3. **Age** - 0.0506
4. **DiabetesPedigreeFunction** - 0.0454
5. **SkinThickness** - 0.0380
6. **Pregnancies** - 0.0352
7. **BloodPressure** - 0.0258
8. **Insulin** - 0.0247

**TabNet Attention Importance (Global):**
1. **Glucose** - 1.0107
2. **Insulin** - 0.9535
3. **BMI** - 0.8104
4. **DiabetesPedigreeFunction** - 0.7028
5. **Pregnancies** - 0.6948
6. **SkinThickness** - 0.6789
7. **BloodPressure** - 0.6287
8. **Age** - 0.6040

---

## 🚀 Setup & Installation

### Prerequisites

- **Python** 3.10+ (3.11, 3.12 recommended)
- **Node.js** 18+ with npm
- **pip** and package manager
- **Git** (for cloning)

### Step 1: Clone & Navigate

```bash
cd C:\Users\40109124\Downloads\Amisha\AI\Diabetes-detection
```

### Step 2: Create Python Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Backend Dependencies

```bash
pip install -r backend/requirements.txt
```

**Key packages installed:**
- fastapi, uvicorn, sqlalchemy, pydantic
- pytorch-tabnet, torch
- scikit-learn, pandas, numpy, joblib
- shap, imbalanced-learn, optuna
- python-dotenv, pytest

### Step 4: Download Dataset & Train Model

```bash
# Download dataset from UCI repository
python data/download_dataset.py

# OR use existing diabetes.csv if already present
```

**Output:** `data/diabetes.csv` (768 rows × 9 cols)

```bash
# Train TabNet model with Optuna tuning
python notebooks/train_pipeline.py
```

**Training outputs (in `models/` folder):**
- `tabnet_model.zip` - Serialized TabNet classifier
- `scaler.joblib` - StandardScaler for preprocessing
- `artefacts.json` - Metrics, feature importance, fold results
- `shap_background.joblib` - Background samples for SHAP

**Expected training time:** 3-5 minutes (5-fold CV + 100 Optuna trials)

### Step 5: Generate Research Plots (Optional)

```bash
python notebooks/generate_plots.py
```

**Output:** 9 publication-quality PNG plots in `plots/` folder (300 DPI, seaborn styled):
1. Class distribution (before/after SMOTE)
2. Confusion matrix with metrics
3. ROC curve (AUC = 0.8265)
4. Calibration curve
5. 5-fold CV metrics comparison
6. Model performance table
7. SHAP feature importance
8. Hyperparameters table
9. TabNet attention importance

---

## 🏃 Running the Application

### Option A: Run Frontend & Backend Separately (Recommended for Development)

#### **Start Backend**

From project root:

```bash
# Activate venv first (if not already)
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# Run FastAPI server with auto-reload
uvicorn backend.main:app --reload --port 8000
```

**Expected output:**
```
Uvicorn running on http://127.0.0.1:8000
Application startup complete
INFO: Loaded model: TabNetClassifier
INFO: Database initialized
```

**API Documentation:** http://localhost:8000/docs (Swagger UI)

#### **Start Frontend** (in a new terminal)

```bash
cd frontend
npm install  # First time only
npm run dev
```

**Expected output:**
```
VITE v4.3.9
➜  Local:   http://localhost:3000
➜  press h to show help
```

**Application URL:** http://localhost:3000

> **Note:** If ports 3000, 3001, 3002 are busy, Vite picks the next available port. Check console output for actual port.

---

### Option B: Run with Python Backend Only (Headless/Testing)

```bash
# Backend running on http://localhost:8000
uvicorn backend.main:app --reload --port 8000

# Test endpoints with curl
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"pregnancies":6,"glucose":148,"blood_pressure":72,"skin_thickness":35,"insulin":0,"bmi":33.6,"diabetes_pedigree":0.627,"age":50}'
```

---

## 📡 API Endpoints Reference

### **Authentication & Health**

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `GET` | `/health` | ❌ | Service status & model load info |
| `POST` | `/register` | ❌ | Create user account |
| `POST` | `/login` | ❌ | Get JWT access token (24hr expiry) |
| `GET` | `/profile` | ✅ | Get logged-in user profile |

### **Prediction & Explanation**

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `POST` | `/predict` | Optional | Diabetes prediction with SHAP explanation |
| `GET` | `/explain` | ✅ | Detailed SHAP explanation for saved prediction |
| `GET` | `/feature-importance` | ❌ | Global SHAP + TabNet attention importance |
| `GET` | `/metrics` | ❌ | Model performance metrics (accuracy, F1, etc.) |
| `GET` | `/roc` | ❌ | ROC curve data (fpr, tpr, AUC) |

### **User History**

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `GET` | `/history` | ✅ | User's prediction history & trend |

---

### Example: Health Check

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_error": null,
  "timestamp": "2026-03-27T19:38:30.462701"
}
```

---

### Example: Prediction Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "pregnancies": 6,
    "glucose": 148,
    "blood_pressure": 72,
    "skin_thickness": 35,
    "insulin": 0,
    "bmi": 33.6,
    "diabetes_pedigree": 0.627,
    "age": 50
  }'
```

**Response:**
```json
{
  "prediction_id": 42,
  "prediction": "Diabetic",
  "probability": 0.8303,
  "risk_category": "High",
  "confidence": "High",
  "model_used": "tabnet",
  "lifestyle_recommendations": [
    "Consult an endocrinologist for comprehensive screening soon.",
    "Begin a structured weight and glucose management plan.",
    "Increase physical activity: aim for 150 min/week moderate intensity."
  ],
  "local_explanation": {
    "features": ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"],
    "shap_values": [0.0041, 0.3201, -0.0321, 0.0089, 0.0000, 0.1544, 0.0821, 0.0634],
    "base_value": 0.4219
  },
  "timestamp": "2026-03-27T19:45:12.123456"
}
```

---

### Example: Global Feature Importance

```bash
curl http://localhost:8000/feature-importance
```

**Response:**
```json
{
  "shap_importance": {
    "Glucose": 0.1498,
    "BMI": 0.0669,
    "Age": 0.0506,
    ...
  },
  "tabnet_attention_importance": {
    "Glucose": 1.0107,
    "Insulin": 0.9535,
    ...
  }
}
```

---

## 🔧 Environment Configuration

### Backend Configuration (`.env`)

Create `backend/.env` in the backend folder:

```env
# Database (SQLite default, PostgreSQL optional)
DATABASE_URL=sqlite:///diabetes_app.db
# DATABASE_URL=postgresql://user:password@localhost/diabetes_db (production)

# JWT Secret Key (generate a strong random key!)
SECRET_KEY=your-super-secret-jwt-key-change-this-in-production

# Optional: Use SQLite fallback if PostgreSQL unavailable
USE_SQLITE_FALLBACK=true

# Optional: Server port
PORT=8000

# Optional: Optuna trials (training only)
OPTUNA_TRIALS=100
```

**Generate a secure SECRET_KEY:**
```python
import secrets
print(secrets.token_urlsafe(32))
```

### Frontend Configuration (`.env`)

Create `frontend/.env` in the frontend folder:

```env
# API base URL (relative path uses Vite proxy)
VITE_API_BASE_URL=/api

# Backend URL for Vite proxy (dev only)
VITE_BACKEND_URL=http://127.0.0.1:8000

# Optional: Frontend port
# (Vite auto-picks 3000, 3001, 3002 based on availability)
```

---

## 🔄 Frontend → Backend Workflow

```
User Input (PatientForm.jsx)
    ↓
POST /predict (via api.js Axios client)
    ↓
Backend Preprocessing (zero-imputation, scaling)
    ↓
TabNet Inference
    ↓
SHAP Explanation (KernelExplainer)
    ↓
Response JSON with prediction + SHAP values
    ↓
Frontend Rendering
    ├─ PredictionResult.jsx (display risk/confidence)
    ├─ ShapExplanation.jsx (SHAP chart)
    └─ Save to localStorage + /history API call
```

### Component Mapping

| Frontend Component | API Endpoint | Purpose |
|-------------------|--------------|---------|
| `PatientForm.jsx` | `POST /predict` | Patient input → prediction |
| `PredictionResult.jsx` | - | Display prediction & recommendations |
| `ShapExplanation.jsx` | - | Render SHAP values as bar chart |
| `Dashboard.jsx` | `GET /metrics`, `/roc`, `/feature-importance` | Model performance dashboard |
| `History.jsx` | `GET /history` | User's predictions over time |
| `Profile.jsx` | `POST /register`, `/login`, `GET /profile` | Authentication & profile |
| `About.jsx` | - | Project information |

---

## 🐛 Troubleshooting

### 1. Backend Startup Fails: "Model file not found"

```
RuntimeError: models/tabnet_model.zip not found
```

**Solution:** Train the model first:
```bash
python notebooks/train_pipeline.py
```

**Alternative:** Backend gracefully degrades to `model_loaded=false` but `/predict` will return HTTP 503 until model is trained.

---

### 2. Frontend "Not Found" or CORS Errors

```
Error: Cannot POST /api/predict (404)
```

**Solutions:**
1. Ensure backend is running on port 8000:
   ```bash
   uvicorn backend.main:app --reload --port 8000
   ```
2. Verify Vite proxy is configured (check `frontend/vite.config.js`)
3. Check frontend `.env` has correct `VITE_BACKEND_URL=http://127.0.0.1:8000`

---

### 3. Authentication Error: History/Profile Returns 401

```
Error: {"detail": "Not authenticated"}
```

**Solution:**
1. Register/Login from Profile page first
2. Verify JWT token in browser localStorage: `localStorage.getItem('access_token')`
3. Ensure token is not expired (24-hour expiry)

---

### 4. Database Error: "Unable to connect to PostgreSQL"

```
sqlalchemy.exc.OperationalError: (psycopg2.OperationalError) could not connect to server
```

**Solution:** Use SQLite fallback (default behavior):
```env
USE_SQLITE_FALLBACK=true
DATABASE_URL=sqlite:///diabetes_app.db
```

For production PostgreSQL, ensure:
- PostgreSQL server is running
- DATABASE_URL syntax is correct: `postgresql://user:password@localhost:5432/diabetes_db`
- User has sufficient privileges

---

### 5. Frontend Port Already in Use

```
VITE v4.3.9  ready in 500ms
✗ error when starting dev server:
listen EADDRINUSE: address already in use :::3000
```

**Solution:** Vite auto-selects next available port (3001, 3002, etc.). Check console output for actual URL.

To manually specify port:
```bash
npm run dev -- --port 3003
```

---

### 6. Plot Generation Fails: "matplotlib backend issue"

```
RuntimeError: No GPU found or matplotlib backend error
```

**Solution:** Ensure matplotlib is installed:
```bash
pip install matplotlib seaborn -U
python notebooks/generate_plots.py
```

---

## 📚 Commands Quick Reference

### **Dataset & Training**

| Command | Purpose |
|---------|---------|
| `python data/download_dataset.py` | Download Pima dataset from UCI |
| `python notebooks/train_pipeline.py` | Train TabNet with Optuna tuning |
| `python notebooks/generate_plots.py` | Generate 9 research plots |

### **Backend**

| Command | Purpose |
|---------|---------|
| `venv\Scripts\activate` | Activate virtual environment (Windows) |
| `pip install -r backend/requirements.txt` | Install Python dependencies |
| `uvicorn backend.main:app --reload --port 8000` | Start FastAPI server |
| `uvicorn backend.main:app --port 8000` | Run without auto-reload (production) |

### **Frontend**

| Command | Purpose |
|---------|---------|
| `cd frontend && npm install` | Install npm dependencies |
| `npm run dev` | Start Vite dev server (hot-reload) |
| `npm run build` | Build for production (dist/ folder) |
| `npm run preview` | Preview production build |

### **Testing**

| Command | Purpose |
|---------|---------|
| `curl http://localhost:8000/health` | Test backend health |
| `curl http://localhost:3000` | Test frontend |

---

## 🧪 Testing the Integration

### Manual End-to-End Test

1. **Start backend:**
   ```bash
   uvicorn backend.main:app --reload --port 8000
   ```

2. **Start frontend:**
   ```bash
   cd frontend && npm run dev
   ```

3. **Visit dashboard:**
   - Open http://localhost:3000 in browser
   - Navigate to "Patient Form" page
   - Enter sample values:
     ```
     Pregnancies: 6
     Glucose: 148
     Blood Pressure: 72
     Skin Thickness: 35
     Insulin: 0
     BMI: 33.6
     Diabetes Pedigree: 0.627
     Age: 50
     ```
   - Click "Predict"
   - Verify prediction appears with SHAP chart

4. **Test Dashboard:**
   - Click "Dashboard" tab
   - Verify charts load (ROC, SHAP, fold metrics)

5. **Test Auth:**
   - Click "Profile" tab
   - Register new account
   - Login
   - Click "History" tab (requires auth)
   - Verify prediction history displays

---

## 📊 Research Artifacts

### Generated Plots (Publication-Quality)

Located in `plots/` folder after running `python notebooks/generate_plots.py`:

| File | Content | Used In |
|------|---------|---------|
| `1_class_distribution.png` | Original vs SMOTE-balanced classes | Methods section |
| `2_confusion_matrix.png` | TP/FP/FN/TN with sensitivity/specificity | Results section |
| `3_roc_curve.png` | ROC with AUC = 0.8265 | Results section |
| `4_calibration_curve.png` | Model reliability vs ideal | Results section |
| `5_fold_metrics_comparison.png` | 5 subplots of per-fold performance | Methods/Validation |
| `6_model_metrics_table.png` | Summary accuracy, precision, recall, F1, AUC | Results table |
| `7_shap_feature_importance.png` | Global importance ranking | Explainability |
| `8_hyperparameters_table.png` | Optuna best hyperparameters | Methods |
| `9_tabnet_attention_importance.png` | Alternative importance via attention masks | Explainability |

All plots are:
- **DPI:** 300 (publication standard)
- **Format:** PNG
- **Colors:** Seaborn palette (suitable for B&W printing)
- **Fonts:** Matplotlib default (readable at paper scale)

---

## 📖 Key Files Documentation

### `backend/main.py`
- FastAPI application instance
- All REST endpoints
- Model loading & prediction logic
- SHAP explanation generation
- Health check & status monitoring

### `backend/auth.py`
- JWT token creation & validation
- PBKDF2 password hashing
- User authentication decorators

### `backend/database.py`
- SQLAlchemy ORM models (User, Prediction)
- Database session management
- Schema creation

### `notebooks/train_pipeline.py`
- Dataset loading & preprocessing
- SMOTE balancing
- TabNet training with 5-fold CV
- Optuna hyperparameter tuning
- Metrics calculation & storage
- Model serialization

### `notebooks/generate_plots.py`
- Matplotlib/seaborn plot generation
- 9 publication-quality visualizations
- 300 DPI rendering
- Automatic plot saving to `plots/`

### `frontend/src/services/api.js`
- Axios HTTP client
- JWT token management
- API endpoint wrappers
- Request/response interceptors

### `frontend/vite.config.js`
- Vite dev server proxy configuration
- Rewrites `/api/*` to backend
- Environment variable loading

---

## 🔐 Security Notes

- **JWT Secret:** Change `SECRET_KEY` in backend `.env` before deployment
- **Password Hashing:** Uses PBKDF2-HMAC-SHA256 with 200k iterations
- **CORS:** Configure in production based on frontend domain
- **Database:** Use PostgreSQL + SSL in production (SQLite only for dev)
- **API Keys:** Never commit `.env` files to version control

---

## 🚢 Deployment

### Docker (Optional)

Backend Dockerfile:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install -r requirements.txt
COPY backend/ .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Frontend Dockerfile:
```dockerfile
FROM node:18-alpine AS build
WORKDIR /app
COPY frontend/package*.json .
RUN npm ci
COPY frontend/src .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

---

## 📝 License & Citation

This project implements research findings from the paper:  
**"AI Driven Approaches for Diabetes Detection in Healthcare"**

### Citation Format
```bibtex
@software{diabetes_detection_2026,
  title={AI-Driven Diabetes Detection System},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/diabetes-detection}
}
```

---

## ⚠️ Disclaimer

**This application is for educational and research purposes only.**

It is **NOT** a substitute for professional medical advice, diagnosis, or treatment. All health-related decisions should be made in consultation with qualified healthcare professionals. The predictions made by this system should not be relied upon for clinical decision-making without expert review.

---

## 📧 Support & Contribution

- Report issues on GitHub Issues
- Submit pull requests for improvements
- Discussion: Use GitHub Discussions for questions

---

## 🎯 Future Roadmap

- [ ] Model versioning & A/B testing
- [ ] Monitoring & retraining pipeline
- [ ] Mobile app (React Native)
- [ ] Advanced analytics & cohort analysis
- [ ] Integration with EHR systems
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Kubernetes deployment configuration
- [ ] Multi-language UI support

---

**Last Updated:** March 28, 2026  
**Status:** ✅ Fully Functional (TabNet + SHAP Explainability)
