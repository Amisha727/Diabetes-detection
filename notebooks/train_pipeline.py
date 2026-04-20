"""
ML Training Pipeline for Diabetes Detection
=============================================
Research Paper: "AI Driven Approaches for Diabetes Detection in Healthcare"

Implements:
  - Data preprocessing (median imputation, StandardScaler)
  - SMOTE for class imbalance handling
  - TabNet deep learning model
  - Stratified K-Fold cross validation (k=5)
  - Hold-out validation (80/20 split)
  - External dataset validation (Kaggle)
  - Robustness analysis (multi-seed)
  - Optuna hyperparameter tuning
  - SHAP explainability
  - Model persistence
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
)
from imblearn.over_sampling import SMOTE
from pytorch_tabnet.tab_model import TabNetClassifier
import optuna
import shap
import joblib

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

OPTUNA_TRIALS = int(os.getenv("OPTUNA_TRIALS", "3"))
TUNING_EPOCHS = int(os.getenv("TUNING_EPOCHS", "8"))
FINAL_EPOCHS = int(os.getenv("FINAL_EPOCHS", "12"))
SHAP_BACKGROUND_SIZE = int(os.getenv("SHAP_BACKGROUND_SIZE", "50"))
SHAP_SAMPLE_SIZE = int(os.getenv("SHAP_SAMPLE_SIZE", "60"))
SHAP_NSAMPLES = int(os.getenv("SHAP_NSAMPLES", "80"))
ROBUSTNESS_SEEDS = [42, 123, 256, 512, 1024]

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

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

# ──────────────────────────────────────────────
# 1. Load & Preprocess
# ──────────────────────────────────────────────


def load_data():
    csv_path = os.path.join(DATA_DIR, "diabetes.csv")
    if not os.path.exists(csv_path):
        sys.path.insert(0, DATA_DIR)
        from download_dataset import download
        download()
    df = pd.read_csv(csv_path)
    return df


def preprocess(df: pd.DataFrame, scaler=None):
    """Replace zero values with median for physiological columns, then standardize."""
    zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in zero_cols:
        df[col] = df[col].replace(0, np.nan)
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    X = df[FEATURES].values
    y = df["Outcome"].values

    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    return X_scaled, y, scaler


# ──────────────────────────────────────────────
# 1b. Load External (Kaggle) Dataset
# ──────────────────────────────────────────────

# Common features between Pima and external Kaggle dataset
COMMON_FEATURES = ["Glucose", "BMI", "Age"]

# Mapping from external dataset column names to Pima column names
EXTERNAL_COL_MAP = {
    "blood_glucose_level": "Glucose",
    "bmi": "BMI",
    "age": "Age",
    "diabetes": "Outcome",
}


def load_external_dataset():
    """Load the Kaggle diabetes prediction dataset (iammustafatz) for external validation.

    This dataset has different features than Pima. We align on the common
    subset: Glucose (blood_glucose_level), BMI (bmi), Age (age).
    """
    csv_path = os.path.join(DATA_DIR, "diabetes_prediction_dataset.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"External dataset not found at {csv_path}. "
            "Place diabetes_prediction_dataset.csv in the data/ folder."
        )

    df_ext = pd.read_csv(csv_path)
    print(f"  External dataset loaded: {df_ext.shape[0]} samples, {df_ext.shape[1]} columns")

    # Rename columns to match Pima naming
    df_ext = df_ext.rename(columns=EXTERNAL_COL_MAP)

    # Keep only the common features + target
    keep_cols = COMMON_FEATURES + ["Outcome"]
    missing = [c for c in keep_cols if c not in df_ext.columns]
    if missing:
        raise ValueError(f"External dataset missing columns after mapping: {missing}")

    df_ext = df_ext[keep_cols].copy()
    print(f"  Common features used: {COMMON_FEATURES}")
    print(f"  External class distribution: "
          f"{dict(zip(*np.unique(df_ext['Outcome'], return_counts=True)))}")
    return df_ext


def preprocess_external(df_ext: pd.DataFrame, scaler: StandardScaler):
    """Preprocess external dataset using the TRAINING scaler on common features only.

    Fills remaining Pima features (not present in external data) with the
    training-set median (i.e. scaler.mean_ from StandardScaler) so they
    become zero after scaling and don't affect the prediction much.
    """
    # Build a full-feature DataFrame with training-set medians as defaults
    n = len(df_ext)
    full_df = pd.DataFrame(
        np.tile(scaler.mean_, (n, 1)),
        columns=FEATURES,
    )

    # Overwrite the common features with actual external data
    for feat in COMMON_FEATURES:
        full_df[feat] = df_ext[feat].values

    # Replace zeros with NaN then median for physiological columns
    zero_cols = [c for c in ["Glucose", "BMI"] if c in COMMON_FEATURES]
    for col in zero_cols:
        full_df[col] = full_df[col].replace(0, np.nan)
        median_val = full_df[col].median()
        full_df[col] = full_df[col].fillna(median_val)

    X = full_df[FEATURES].values
    y = df_ext["Outcome"].values

    X_scaled = scaler.transform(X)
    return X_scaled, y


# ──────────────────────────────────────────────
# 2. SMOTE for class imbalance
# ──────────────────────────────────────────────


def apply_smote(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"    SMOTE: {len(y_train)} -> {len(y_res)} samples "
          f"(class 0: {sum(y_res == 0)}, class 1: {sum(y_res == 1)})")
    return X_res, y_res


# ──────────────────────────────────────────────
# 3. Optuna hyperparameter tuning
# ──────────────────────────────────────────────


def objective(trial, X, y):
    """Optuna objective: train TabNet with Stratified 3-Fold CV and return mean AUC."""
    params = {
        "n_d": trial.suggest_int("n_d", 8, 16),
        "n_a": trial.suggest_int("n_a", 8, 16),
        "n_steps": trial.suggest_int("n_steps", 3, 5),
        "gamma": trial.suggest_float("gamma", 1.0, 2.0),
        "lambda_sparse": trial.suggest_float("lambda_sparse", 1e-6, 1e-3, log=True),
        "momentum": trial.suggest_float("momentum", 0.01, 0.4),
        "mask_type": trial.suggest_categorical("mask_type", ["sparsemax", "entmax"]),
    }
    lr = trial.suggest_float("learning_rate", 0.005, 0.05, log=True)
    batch_size = trial.suggest_categorical("batch_size", [128, 256])

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    aucs = []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # SMOTE on train fold only
        smote = SMOTE(random_state=42)
        X_tr, y_tr = smote.fit_resample(X_tr, y_tr)

        model = TabNetClassifier(
            **params,
            optimizer_params={"lr": lr},
            scheduler_params={"step_size": 10, "gamma": 0.9},
            scheduler_fn=__import__("torch").optim.lr_scheduler.StepLR,
            verbose=0,
            seed=42,
        )
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric=["auc"],
            max_epochs=TUNING_EPOCHS,
            patience=15,
            batch_size=batch_size,
        )
        y_prob = model.predict_proba(X_val)[:, 1]
        aucs.append(roc_auc_score(y_val, y_prob))

    return np.mean(aucs)


def tune_hyperparameters(X, y, n_trials=OPTUNA_TRIALS):
    print(f"  Optuna tuning ({n_trials} trials, 3-fold CV each)...")
    study = optuna.create_study(
        direction="maximize", study_name="tabnet_diabetes")
    study.optimize(lambda trial: objective(trial, X, y),
                   n_trials=n_trials, show_progress_bar=True)

    print(f"    Best AUC: {study.best_value:.4f}")
    print(f"    Best params: {study.best_params}")
    return study.best_params


# ──────────────────────────────────────────────
# 4. Train TabNet with Stratified K-Fold
# ──────────────────────────────────────────────


def train_tabnet_kfold(X, y, best_params, n_splits=5):
    """Train TabNet with 5-fold Stratified CV, SMOTE per fold. Returns best model & aggregated metrics."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    lr = best_params.pop("learning_rate", 0.02)
    batch_size = best_params.pop("batch_size", 128)

    # Remove non-TabNet params if any
    tabnet_params = {k: v for k, v in best_params.items()
                     if k in ("n_d", "n_a", "n_steps", "gamma", "lambda_sparse",
                              "momentum", "mask_type")}

    fold_metrics = []
    best_model = None
    best_auc = 0
    all_y_true = []
    all_y_prob = []

    print(f"\n  Training {n_splits}-Fold Stratified CV with SMOTE...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # SMOTE on train fold only
        X_tr_res, y_tr_res = apply_smote(X_tr, y_tr)

        model = TabNetClassifier(
            **tabnet_params,
            optimizer_params={"lr": lr},
            scheduler_params={"step_size": 10, "gamma": 0.9},
            scheduler_fn=__import__("torch").optim.lr_scheduler.StepLR,
            verbose=0,
            seed=42,
        )
        model.fit(
            X_tr_res, y_tr_res,
            eval_set=[(X_val, y_val)],
            eval_metric=["auc"],
            max_epochs=FINAL_EPOCHS,
            patience=20,
            batch_size=batch_size,
        )

        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred)
        rec = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_prob)

        fold_metrics.append({
            "fold": fold, "accuracy": acc, "precision": prec,
            "recall": rec, "f1_score": f1, "roc_auc": auc,
        })

        print(f"    Fold {fold}: ACC={acc:.4f}  PREC={prec:.4f}  "
              f"REC={rec:.4f}  F1={f1:.4f}  AUC={auc:.4f}")

        all_y_true.extend(y_val.tolist())
        all_y_prob.extend(y_prob.tolist())

        if auc > best_auc:
            best_auc = auc
            best_model = model

    # Aggregate metrics
    avg = {
        "model": "TabNet",
        "accuracy": round(np.mean([m["accuracy"] for m in fold_metrics]), 4),
        "precision": round(np.mean([m["precision"] for m in fold_metrics]), 4),
        "recall": round(np.mean([m["recall"] for m in fold_metrics]), 4),
        "f1_score": round(np.mean([m["f1_score"] for m in fold_metrics]), 4),
        "roc_auc": round(np.mean([m["roc_auc"] for m in fold_metrics]), 4),
    }

    fpr, tpr, _ = roc_curve(all_y_true, all_y_prob)
    roc_data = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}

    # Precision-Recall curve
    pr_precision, pr_recall, _ = precision_recall_curve(all_y_true, all_y_prob)
    pr_auc = round(float(average_precision_score(all_y_true, all_y_prob)), 4)
    pr_data = {"precision": pr_precision.tolist(), "recall": pr_recall.tolist()}
    avg["pr_auc"] = pr_auc

    cm = confusion_matrix(
        all_y_true, [1 if p >= 0.5 else 0 for p in all_y_prob]
    ).tolist()

    print(f"\n  {'='*50}")
    print(f"  TabNet Average Results ({n_splits}-Fold CV)")
    print(f"  {'='*50}")
    for k, v in avg.items():
        if k != "model":
            print(f"    {k:>12}: {v}")

    return best_model, avg, roc_data, pr_data, fold_metrics, cm


# ──────────────────────────────────────────────
# 4b. Hold-out Evaluation
# ──────────────────────────────────────────────


def evaluate_holdout(model, X_test, y_test):
    """Evaluate trained model on strictly unseen hold-out test set."""
    print("\n  Evaluating on hold-out test set...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
    }

    cm = confusion_matrix(y_test, y_pred).tolist()
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_data = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}

    print(f"    Hold-out Results:")
    for k, v in metrics.items():
        print(f"      {k:>12}: {v}")

    return metrics, cm, roc_data


# ──────────────────────────────────────────────
# 4c. External Dataset Validation
# ──────────────────────────────────────────────


def evaluate_external(model, X_ext, y_ext):
    """Evaluate model trained on Pima against external Kaggle dataset."""
    print("\n  Evaluating on external (Kaggle) dataset...")
    y_pred = model.predict(X_ext)
    y_prob = model.predict_proba(X_ext)[:, 1]

    metrics = {
        "accuracy": round(float(accuracy_score(y_ext, y_pred)), 4),
        "precision": round(float(precision_score(y_ext, y_pred)), 4),
        "recall": round(float(recall_score(y_ext, y_pred)), 4),
        "f1_score": round(float(f1_score(y_ext, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_ext, y_prob)), 4),
    }

    cm = confusion_matrix(y_ext, y_pred).tolist()
    fpr, tpr, _ = roc_curve(y_ext, y_prob)
    roc_data = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}

    print(f"    External Validation Results:")
    for k, v in metrics.items():
        print(f"      {k:>12}: {v}")

    return metrics, cm, roc_data


# ──────────────────────────────────────────────
# 4d. Robustness Analysis (Multi-Seed)
# ──────────────────────────────────────────────


def robustness_analysis(X_train, y_train, X_test, y_test, best_params, seeds=None):
    """Run training multiple times with different random seeds to assess stability."""
    if seeds is None:
        seeds = ROBUSTNESS_SEEDS

    print(f"\n  Robustness analysis ({len(seeds)} seeds)...")

    lr = best_params.get("learning_rate", 0.02)
    batch_size = best_params.get("batch_size", 128)
    tabnet_params = {k: v for k, v in best_params.items()
                     if k in ("n_d", "n_a", "n_steps", "gamma", "lambda_sparse",
                              "momentum", "mask_type")}

    all_run_metrics = []

    for i, seed in enumerate(seeds, 1):
        smote = SMOTE(random_state=seed)
        X_tr_res, y_tr_res = smote.fit_resample(X_train, y_train)

        model = TabNetClassifier(
            **tabnet_params,
            optimizer_params={"lr": lr},
            scheduler_params={"step_size": 10, "gamma": 0.9},
            scheduler_fn=__import__("torch").optim.lr_scheduler.StepLR,
            verbose=0,
            seed=seed,
        )
        model.fit(
            X_tr_res, y_tr_res,
            eval_set=[(X_test, y_test)],
            eval_metric=["auc"],
            max_epochs=FINAL_EPOCHS,
            patience=20,
            batch_size=batch_size,
        )

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        run_metrics = {
            "seed": seed,
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "f1_score": float(f1_score(y_test, y_pred)),
            "roc_auc": float(roc_auc_score(y_test, y_prob)),
        }
        all_run_metrics.append(run_metrics)

        print(f"    Seed {seed}: ACC={run_metrics['accuracy']:.4f}  "
              f"F1={run_metrics['f1_score']:.4f}  AUC={run_metrics['roc_auc']:.4f}")

    # Compute mean and std
    metric_keys = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    summary = {}
    for key in metric_keys:
        vals = [m[key] for m in all_run_metrics]
        summary[key] = {
            "mean": round(float(np.mean(vals)), 4),
            "std": round(float(np.std(vals)), 4),
        }

    print(f"\n    Robustness Summary (mean ± std):")
    for k, v in summary.items():
        print(f"      {k:>12}: {v['mean']:.4f} ± {v['std']:.4f}")

    return {"runs": all_run_metrics, "summary": summary}


# ──────────────────────────────────────────────
# 5. SHAP Explainability
# ──────────────────────────────────────────────


def compute_shap(model, X, n_background=SHAP_BACKGROUND_SIZE):
    """Compute SHAP values using KernelExplainer (model-agnostic)."""
    print("\n  Computing SHAP values (KernelExplainer)...")

    # Use a subset as background for efficiency
    np.random.seed(42)
    bg_idx = np.random.choice(len(X), min(n_background, len(X)), replace=False)
    background = X[bg_idx]

    explainer = shap.KernelExplainer(model.predict_proba, background)
    # Compute on a sample for global importance
    sample_idx = np.random.choice(len(X), min(
        SHAP_SAMPLE_SIZE, len(X)), replace=False)
    shap_values = explainer.shap_values(X[sample_idx], nsamples=SHAP_NSAMPLES)

    # Normalize SHAP output shape across versions:
    # - list[class][sample, feature]
    # - ndarray[sample, feature]
    # - ndarray[sample, feature, class]
    if isinstance(shap_values, list):
        shap_vals = np.asarray(shap_values[1])
    else:
        arr = np.asarray(shap_values)
        if arr.ndim == 3 and arr.shape[-1] == 2:
            shap_vals = arr[:, :, 1]
        else:
            shap_vals = arr

    global_importance = np.abs(shap_vals).mean(axis=0)
    importance_dict = {
        feat: round(float(imp), 6)
        for feat, imp in zip(FEATURES, global_importance)
    }

    print(f"  SHAP feature importance:")
    for feat, imp in sorted(importance_dict.items(), key=lambda x: -x[1]):
        print(f"    {feat:>28}: {imp:.6f}")

    return importance_dict, background


# ──────────────────────────────────────────────
# 6. TabNet Attention Masks
# ──────────────────────────────────────────────


def get_tabnet_attention(model, X):
    """Extract TabNet's built-in attention/feature importance masks."""
    explain_matrix, masks = model.explain(X)
    attention_importance = {
        feat: round(float(val), 6)
        for feat, val in zip(FEATURES, explain_matrix.mean(axis=0))
    }
    print(f"\n  TabNet Attention Feature Importance:")
    for feat, imp in sorted(attention_importance.items(), key=lambda x: -x[1]):
        print(f"    {feat:>28}: {imp:.6f}")
    return attention_importance


# ──────────────────────────────────────────────
# 7. Save artefacts
# ──────────────────────────────────────────────


def save_artefacts(model, scaler, metrics, roc_data, pr_data, fold_metrics,
                   confusion, shap_importance, attention_importance,
                   background, best_params,
                   holdout_metrics=None, holdout_cm=None, holdout_roc=None,
                   external_metrics=None, external_cm=None, external_roc=None,
                   robustness_results=None):
    # Save TabNet model
    model_path = os.path.join(MODELS_DIR, "tabnet_model")
    model.save_model(model_path)

    # Save scaler
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))

    # Save SHAP background data for inference-time explanations
    joblib.dump(background, os.path.join(MODELS_DIR, "shap_background.joblib"))

    artefacts = {
        "model_type": "TabNet",
        "methodology": {
            "preprocessing": "Median imputation + StandardScaler",
            "class_balancing": "SMOTE",
            "cross_validation": "Stratified 5-Fold",
            "holdout_split": "80% train / 20% test (stratified)",
            "external_validation": "Kaggle Diabetes Prediction (iammustafatz) — common features: Glucose, BMI, Age",
            "robustness_analysis": f"Multi-seed ({len(ROBUSTNESS_SEEDS)} seeds)",
            "hyperparameter_tuning": f"Optuna ({OPTUNA_TRIALS} trials)",
            "explainability": "SHAP (KernelExplainer) + TabNet Attention Masks",
        },
        "best_hyperparameters": best_params,
        "metrics": {"tabnet": metrics},
        "fold_metrics": fold_metrics,
        "confusion_matrix": confusion,
        "roc": {"tabnet": roc_data},
        "pr": {"tabnet": pr_data},
        "feature_importance": {
            "tabnet_shap": shap_importance,
            "tabnet_attention": attention_importance,
        },
        "features": FEATURES,
    }

    # Hold-out validation results
    if holdout_metrics is not None:
        artefacts["holdout_validation"] = {
            "metrics": holdout_metrics,
            "confusion_matrix": holdout_cm,
            "roc": holdout_roc,
        }

    # External validation results
    if external_metrics is not None:
        artefacts["external_validation"] = {
            "metrics": external_metrics,
            "confusion_matrix": external_cm,
            "roc": external_roc,
        }

    # Robustness analysis results
    if robustness_results is not None:
        artefacts["robustness_analysis"] = robustness_results

    with open(os.path.join(MODELS_DIR, "artefacts.json"), "w") as f:
        json.dump(artefacts, f, indent=2)

    print(f"\n  Models & artefacts saved to {MODELS_DIR}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────


def main():
    print("=" * 60)
    print("  AI-Driven Diabetes Detection Training Pipeline")
    print("  TabNet + SMOTE + Stratified K-Fold + Optuna + SHAP")
    print("  + Hold-Out + External Validation + Robustness")
    print("=" * 60)

    print("\n[1/9] Loading dataset...")
    df = load_data()
    print(f"  Dataset: {df.shape[0]} samples, {df.shape[1]} columns")
    print(
        f"  Class distribution: {dict(zip(*np.unique(df['Outcome'], return_counts=True)))}")

    print("\n[2/9] Preprocessing & Hold-Out Split...")
    X, y, scaler = preprocess(df.copy())

    # ── Hold-out split: 80% train, 20% test (STRICTLY unseen) ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"  Train set: {X_train.shape[0]} samples")
    print(f"  Test set (hold-out): {X_test.shape[0]} samples")

    print("\n[3/9] Hyperparameter tuning with Optuna (on train set only)...")
    best_params = tune_hyperparameters(X_train, y_train, n_trials=OPTUNA_TRIALS)

    print("\n[4/9] Training TabNet with Stratified 5-Fold CV (on train set only)...")
    model, metrics, roc_data, pr_data, fold_metrics, cv_confusion = train_tabnet_kfold(
        X_train, y_train, best_params.copy(), n_splits=5
    )

    print("\n[5/9] Hold-out evaluation (strictly unseen test set)...")
    holdout_metrics, holdout_cm, holdout_roc = evaluate_holdout(model, X_test, y_test)

    print("\n[6/9] External dataset validation...")
    external_metrics, external_cm, external_roc = None, None, None
    try:
        df_ext = load_external_dataset()
        X_ext, y_ext = preprocess_external(df_ext.copy(), scaler)
        external_metrics, external_cm, external_roc = evaluate_external(
            model, X_ext, y_ext
        )
    except Exception as e:
        print(f"  WARNING: External validation skipped: {e}")

    print("\n[7/9] Robustness analysis (multi-seed)...")
    robustness_results = robustness_analysis(
        X_train, y_train, X_test, y_test, best_params.copy()
    )

    print("\n[8/9] Computing explainability...")
    shap_importance, background = compute_shap(model, X_train)
    attention_importance = get_tabnet_attention(model, X_train)

    print("\n[9/9] Saving artefacts...")
    save_artefacts(
        model, scaler, metrics, roc_data, pr_data, fold_metrics,
        cv_confusion, shap_importance, attention_importance,
        background, best_params,
        holdout_metrics=holdout_metrics,
        holdout_cm=holdout_cm,
        holdout_roc=holdout_roc,
        external_metrics=external_metrics,
        external_cm=external_cm,
        external_roc=external_roc,
        robustness_results=robustness_results,
    )

    print("\n" + "=" * 60)
    print("  Training pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
