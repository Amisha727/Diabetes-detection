import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix
import shap

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
DATA_DIR = os.path.join(PROJECT_DIR, "data")
PLOTS_DIR = os.path.join(PROJECT_DIR, "plots")

# Create plots directory
os.makedirs(PLOTS_DIR, exist_ok=True)

# Set style for publication quality
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Configure figure defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

# Load artefacts
with open(os.path.join(MODELS_DIR, "artefacts.json"), "r") as f:
    artefacts = json.load(f)

fold_metrics = artefacts["fold_metrics"]
metrics_all = artefacts.get("metrics", {})
roc_all = artefacts.get("roc", {})
pr_all = artefacts.get("pr", {})

model_display_names = {
    "tabnet": "TabNet",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}

print("Generating publication-quality plots...\n")

# ============================================
# 1. CLASS DISTRIBUTION: Before/After SMOTE
# ============================================
print("1️⃣  Generating: 1_class_distribution.png")
df = pd.read_csv(os.path.join(DATA_DIR, "diabetes.csv"))

before = df["Outcome"].value_counts().sort_index()

from imblearn.over_sampling import SMOTE
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_res, y_res = SMOTE(random_state=42).fit_resample(X, y)
after = pd.Series(y_res).value_counts().sort_index()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Before SMOTE
ax = axes[0]
colors_before = ['#2ecc71', '#e74c3c']
bars1 = ax.bar(['Non-Diabetic (0)', 'Diabetic (1)'], before.values, color=colors_before, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Count', fontweight='bold')
ax.set_title('Class Distribution (Before SMOTE)', fontweight='bold', fontsize=12)
ax.grid(axis='y', alpha=0.3)
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}', ha='center', va='bottom', fontweight='bold')

# After SMOTE
ax = axes[1]
bars2 = ax.bar(['Non-Diabetic (0)', 'Diabetic (1)'], after.values, color=colors_before, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Count', fontweight='bold')
ax.set_title('Class Distribution (After SMOTE)', fontweight='bold', fontsize=12)
ax.grid(axis='y', alpha=0.3)
for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "1_class_distribution.png"), bbox_inches='tight', dpi=300)
plt.close()
print(f"  ✓ Saved to plots/1_class_distribution.png\n")


# ============================================
# 2. CONFUSION MATRIX
# ============================================
print("2️⃣  Generating: 2_confusion_matrix.png")
cm = np.array(artefacts["confusion_matrix"])

fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=ax,
            xticklabels=['Non-Diabetic', 'Diabetic'],
            yticklabels=['Non-Diabetic', 'Diabetic'],
            annot_kws={'size': 14, 'weight': 'bold'},
            cbar_kws={'label': 'Count'})
ax.set_ylabel('True Label', fontweight='bold', fontsize=12)
ax.set_xlabel('Predicted Label', fontweight='bold', fontsize=12)
ax.set_title('Confusion Matrix (Aggregated CV Predictions)', fontweight='bold', fontsize=13)

# Add metrics
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
plt.text(0.5, -0.15, f'Sensitivity: {sensitivity:.3f} | Specificity: {specificity:.3f}',
         ha='center', transform=ax.transAxes, fontsize=10, style='italic')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "2_confusion_matrix.png"), bbox_inches='tight', dpi=300)
plt.close()
print(f"  ✓ Saved to plots/2_confusion_matrix.png\n")


# ============================================
# 3. ROC CURVE
# ============================================
print("3️⃣  Generating: 3_roc_curve.png")
fig, ax = plt.subplots(figsize=(8, 7))
colors = {
    "tabnet": "#3498db",
    "random_forest": "#2ecc71",
    "xgboost": "#e67e22",
}

for model_key, roc_data in roc_all.items():
    fpr = roc_data["fpr"]
    tpr = roc_data["tpr"]
    auc_score = metrics_all.get(model_key, {}).get("roc_auc", 0.0)
    label = model_display_names.get(model_key, model_key)
    ax.plot(
        fpr,
        tpr,
        color=colors.get(model_key, "#34495e"),
        lw=2.5,
        label=f"{label} (AUC = {auc_score:.4f})",
    )

ax.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Random Classifier', alpha=0.7)
ax.set_xlabel('False Positive Rate', fontweight='bold', fontsize=12)
ax.set_ylabel('True Positive Rate', fontweight='bold', fontsize=12)
ax.set_title('ROC Curve Comparison', fontweight='bold', fontsize=13)
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "3_roc_curve.png"), bbox_inches='tight', dpi=300)
plt.close()
print(f"  ✓ Saved to plots/3_roc_curve.png\n")


# ============================================
# 4. PRECISION-RECALL CURVE
# ============================================
print("4️⃣  Generating: 4_pr_auc_curve.png")

fig, ax = plt.subplots(figsize=(8, 7))
for model_key, pr_data in pr_all.items():
    recall = pr_data["recall"]
    precision = pr_data["precision"]
    pr_auc = metrics_all.get(model_key, {}).get("pr_auc", 0.0)
    label = model_display_names.get(model_key, model_key)
    ax.plot(
        recall,
        precision,
        lw=2.5,
        color=colors.get(model_key, "#34495e"),
        label=f"{label} (PR AUC = {pr_auc:.4f})",
    )

positive_rate = y.mean()
ax.hlines(
    y=positive_rate,
    xmin=0,
    xmax=1,
    colors='gray',
    linestyles='--',
    linewidth=1.5,
    label=f'Baseline (Pos Rate = {positive_rate:.4f})',
)
ax.set_xlabel('Recall', fontweight='bold', fontsize=12)
ax.set_ylabel('Precision', fontweight='bold', fontsize=12)
ax.set_title('Precision-Recall Curve Comparison', fontweight='bold', fontsize=13)
ax.legend(loc='lower left', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "4_pr_auc_curve.png"), bbox_inches='tight', dpi=300)
plt.close()
print(f"  ✓ Saved to plots/4_pr_auc_curve.png\n")


# ============================================
# 5. FOLD-WISE METRICS
# ============================================
print("5️⃣  Generating: 5_fold_metrics_comparison.png")
fold_data = pd.DataFrame(fold_metrics)
metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(16, 4))

colors_fold = plt.cm.Set2(np.linspace(0, 1, len(fold_data)))

for idx, metric in enumerate(metrics_to_plot):
    ax = axes[idx]
    values = fold_data[metric].values
    folds = fold_data['fold'].values
    
    bars = ax.bar(folds, values, color=colors_fold, edgecolor='black', linewidth=1.5, alpha=0.8)
    ax.axhline(y=values.mean(), color='red', linestyle='--', lw=2, label=f'μ={values.mean():.3f}')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_xlabel('Fold', fontweight='bold')
    ax.set_title(metric.replace('_', ' ').title(), fontweight='bold', fontsize=11)
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=8)

plt.suptitle('5-Fold CV Performance Metrics', fontweight='bold', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "5_fold_metrics_comparison.png"), bbox_inches='tight', dpi=300)
plt.close()
print(f"  ✓ Saved to plots/5_fold_metrics_comparison.png\n")


# ============================================
# 6. MODEL METRICS TABLE
# ============================================
print("6️⃣  Generating: 6_model_metrics_table.png")
rows = []
for model_key, model_metrics in metrics_all.items():
    rows.append({
        "Model": model_display_names.get(model_key, model_key),
        "Accuracy": f"{model_metrics.get('accuracy', 0.0):.4f}",
        "Precision": f"{model_metrics.get('precision', 0.0):.4f}",
        "Recall": f"{model_metrics.get('recall', 0.0):.4f}",
        "F1 Score": f"{model_metrics.get('f1_score', 0.0):.4f}",
        "ROC AUC": f"{model_metrics.get('roc_auc', 0.0):.4f}",
        "PR AUC": f"{model_metrics.get('pr_auc', 0.0):.4f}",
        "Brier Score": f"{model_metrics.get('brier_score', 0.0):.4f}",
    })

metrics_df = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(15, 3.8))
ax.axis('off')

table = ax.table(cellText=metrics_df.values, colLabels=metrics_df.columns, loc='center',
                cellLoc='center', colWidths=[0.18, 0.11, 0.11, 0.11, 0.1, 0.1, 0.1, 0.12])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.1)

for i in range(len(metrics_df.columns)):
    table[(0, i)].set_facecolor('#3498db')
    table[(0, i)].set_text_props(weight='bold', color='white')

for i in range(1, len(metrics_df) + 1):
    for j in range(len(metrics_df.columns)):
        table[(i, j)].set_facecolor('#ecf0f1' if i % 2 == 0 else '#ffffff')

plt.title('Model Performance Metrics', fontweight='bold', fontsize=13, pad=20)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "6_model_metrics_table.png"), bbox_inches='tight', dpi=300)
plt.close()
print(f"  ✓ Saved to plots/6_model_metrics_table.png\n")

print("PR AUC Scores:")
for model_key, model_metrics in metrics_all.items():
    print(
        f"  {model_display_names.get(model_key, model_key)}: "
        f"{model_metrics.get('pr_auc', 0.0):.4f}"
    )
print()


# ============================================
# 7. SHAP FEATURE IMPORTANCE
# ============================================
print("7️⃣  Generating: 7_shap_feature_importance.png")
importance = artefacts["feature_importance"]["tabnet_shap"]
features = list(importance.keys())
values = list(importance.values())

sorted_idx = np.argsort(values)[::-1]
features_sorted = [features[i] for i in sorted_idx]
values_sorted = [values[i] for i in sorted_idx]

fig, ax = plt.subplots(figsize=(10, 6))
colors_shap = plt.cm.RdYlGn_r(np.linspace(0.3, 0.7, len(features_sorted)))
bars = ax.barh(features_sorted, values_sorted, color=colors_shap, edgecolor='black', linewidth=1.5)

ax.set_xlabel('Mean |SHAP Value|', fontweight='bold', fontsize=12)
ax.set_title('SHAP Global Feature Importance', fontweight='bold', fontsize=13)
ax.grid(axis='x', alpha=0.3)

for bar in bars:
    width = bar.get_width()
    ax.text(width, bar.get_y() + bar.get_height()/2., f' {width:.4f}', 
           ha='left', va='center', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "7_shap_feature_importance.png"), bbox_inches='tight', dpi=300)
plt.close()
print(f"  ✓ Saved to plots/7_shap_feature_importance.png\n")


# ============================================
# 8. HYPERPARAMETERS TABLE
# ============================================
print("8️⃣  Generating: 8_hyperparameters_table.png")
params = artefacts["best_hyperparameters"]

hp_df = pd.DataFrame({
    "Parameter": list(params.keys()),
    "Value": [f"{v:.6f}" if isinstance(v, float) else str(v) for v in params.values()]
})

fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')

table = ax.table(cellText=hp_df.values, colLabels=hp_df.columns, loc='center',
                cellLoc='left', colWidths=[0.5, 0.5])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.2)

for i in range(len(hp_df.columns)):
    table[(0, i)].set_facecolor('#9b59b6')
    table[(0, i)].set_text_props(weight='bold', color='white')

for i in range(1, len(hp_df) + 1):
    for j in range(len(hp_df.columns)):
        table[(i, j)].set_facecolor('#ecf0f1' if i % 2 == 0 else '#ffffff')

plt.title('Best Hyperparameters (Optuna)', fontweight='bold', fontsize=13, pad=20)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "8_hyperparameters_table.png"), bbox_inches='tight', dpi=300)
plt.close()
print(f"  ✓ Saved to plots/8_hyperparameters_table.png\n")


# ============================================
# 9. TABNET ATTENTION IMPORTANCE
# ============================================
print("9️⃣  Generating: 9_tabnet_attention_importance.png")
if "tabnet_attention" in artefacts["feature_importance"]:
    attention = artefacts["feature_importance"]["tabnet_attention"]
    features_att = list(attention.keys())
    values_att = list(attention.values())
    
    sorted_idx_att = np.argsort(values_att)[::-1]
    features_sorted_att = [features_att[i] for i in sorted_idx_att]
    values_sorted_att = [values_att[i] for i in sorted_idx_att]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors_att = plt.cm.viridis(np.linspace(0.2, 0.8, len(features_sorted_att)))
    bars = ax.barh(features_sorted_att, values_sorted_att, color=colors_att, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('TabNet Attention Score', fontweight='bold', fontsize=12)
    ax.set_title('TabNet Attention-based Feature Importance', fontweight='bold', fontsize=13)
    ax.grid(axis='x', alpha=0.3)
    
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2., f' {width:.4f}', 
               ha='left', va='center', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "9_tabnet_attention_importance.png"), bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  ✓ Saved to plots/9_tabnet_attention_importance.png\n")

print("="*60)
print("✅ All plots generated successfully!")
print(f"📁 Location: {PLOTS_DIR}")
print("="*60)
for f in sorted(os.listdir(PLOTS_DIR)):
    print(f"  ✓ {f}")


# ============================================
# 10. HOLD-OUT CONFUSION MATRIX
# ============================================
if "holdout_validation" in artefacts:
    print("\n🔟  Generating: holdout_confusion_matrix.png")
    ho = artefacts["holdout_validation"]
    ho_cm = np.array(ho["confusion_matrix"])

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(ho_cm, annot=True, fmt='d', cmap='Oranges', cbar=True, ax=ax,
                xticklabels=['Non-Diabetic', 'Diabetic'],
                yticklabels=['Non-Diabetic', 'Diabetic'],
                annot_kws={'size': 14, 'weight': 'bold'},
                cbar_kws={'label': 'Count'})
    ax.set_ylabel('True Label', fontweight='bold', fontsize=12)
    ax.set_xlabel('Predicted Label', fontweight='bold', fontsize=12)
    ax.set_title('Hold-Out Test Set Confusion Matrix', fontweight='bold', fontsize=13)

    tn, fp, fn, tp = ho_cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    plt.text(0.5, -0.15, f'Sensitivity: {sens:.3f} | Specificity: {spec:.3f}',
             ha='center', transform=ax.transAxes, fontsize=10, style='italic')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "holdout_confusion_matrix.png"), bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  ✓ Saved to plots/holdout_confusion_matrix.png\n")
else:
    print("\n⚠️  Skipping holdout_confusion_matrix.png (no holdout data in artefacts)\n")


# ============================================
# 11. EXTERNAL VALIDATION ROC CURVE
# ============================================
if "external_validation" in artefacts:
    print("1️⃣1️⃣  Generating: external_validation_roc.png")
    ev = artefacts["external_validation"]
    ev_roc = ev["roc"]
    ev_auc = ev["metrics"]["roc_auc"]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(ev_roc["fpr"], ev_roc["tpr"], color='#e74c3c', lw=2.5,
            label=f'External Validation (AUC = {ev_auc:.4f})')

    # Overlay Pima CV ROC for comparison
    if "tabnet" in roc_all:
        pima_auc = metrics_all.get("tabnet", {}).get("roc_auc", 0.0)
        ax.plot(roc_all["tabnet"]["fpr"], roc_all["tabnet"]["tpr"],
                color='#3498db', lw=2.5, linestyle='--',
                label=f'Cross-Validation (AUC = {pima_auc:.4f})')

    # Overlay hold-out ROC
    if "holdout_validation" in artefacts:
        ho_roc = artefacts["holdout_validation"]["roc"]
        ho_auc = artefacts["holdout_validation"]["metrics"]["roc_auc"]
        ax.plot(ho_roc["fpr"], ho_roc["tpr"],
                color='#2ecc71', lw=2.5, linestyle='-.',
                label=f'Hold-Out (AUC = {ho_auc:.4f})')

    ax.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--',
            label='Random Classifier', alpha=0.7)
    ax.set_xlabel('False Positive Rate', fontweight='bold', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontweight='bold', fontsize=12)
    ax.set_title('External Validation ROC Curve', fontweight='bold', fontsize=13)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "external_validation_roc.png"), bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  ✓ Saved to plots/external_validation_roc.png\n")
else:
    print("⚠️  Skipping external_validation_roc.png (no external data in artefacts)\n")


# ============================================
# 12. ROBUSTNESS METRICS (Mean ± Std)
# ============================================
if "robustness_analysis" in artefacts:
    print("1️⃣2️⃣  Generating: robustness_metrics.png")
    rob = artefacts["robustness_analysis"]
    rob_summary = rob["summary"]
    metric_names = list(rob_summary.keys())
    means = [rob_summary[m]["mean"] for m in metric_names]
    stds = [rob_summary[m]["std"] for m in metric_names]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metric_names))
    colors_rob = ['#3498db', '#2ecc71', '#e67e22', '#9b59b6', '#e74c3c']
    bars = ax.bar(x, means, yerr=stds, capsize=8, color=colors_rob[:len(metric_names)],
                  edgecolor='black', linewidth=1.5, alpha=0.85, error_kw={'linewidth': 2})

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metric_names],
                       fontweight='bold', fontsize=10)
    ax.set_ylabel('Score', fontweight='bold', fontsize=12)
    ax.set_title(f'Robustness Analysis (Mean ± Std over {len(rob["runs"])} Seeds)',
                 fontweight='bold', fontsize=13)
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)

    for i, (bar, m, s) in enumerate(zip(bars, means, stds)):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + s + 0.02,
                f'{m:.4f}\n±{s:.4f}', ha='center', va='bottom',
                fontweight='bold', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "robustness_metrics.png"), bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  ✓ Saved to plots/robustness_metrics.png\n")
else:
    print("⚠️  Skipping robustness_metrics.png (no robustness data in artefacts)\n")


# ============================================
# 13. COMPARISON TABLE (CV vs Hold-out vs External)
# ============================================
print("1️⃣3️⃣  Generating: comparison_table.png")

comp_rows = []

# Cross-Validation metrics
if "tabnet" in metrics_all:
    cv_m = metrics_all["tabnet"]
    comp_rows.append({
        "Validation": "5-Fold CV",
        "Accuracy": f"{cv_m.get('accuracy', 0):.4f}",
        "Precision": f"{cv_m.get('precision', 0):.4f}",
        "Recall": f"{cv_m.get('recall', 0):.4f}",
        "F1 Score": f"{cv_m.get('f1_score', 0):.4f}",
        "ROC AUC": f"{cv_m.get('roc_auc', 0):.4f}",
    })

# Hold-out metrics
if "holdout_validation" in artefacts:
    ho_m = artefacts["holdout_validation"]["metrics"]
    comp_rows.append({
        "Validation": "Hold-Out (20%)",
        "Accuracy": f"{ho_m.get('accuracy', 0):.4f}",
        "Precision": f"{ho_m.get('precision', 0):.4f}",
        "Recall": f"{ho_m.get('recall', 0):.4f}",
        "F1 Score": f"{ho_m.get('f1_score', 0):.4f}",
        "ROC AUC": f"{ho_m.get('roc_auc', 0):.4f}",
    })

# External validation metrics
if "external_validation" in artefacts:
    ex_m = artefacts["external_validation"]["metrics"]
    comp_rows.append({
        "Validation": "External (Kaggle)",
        "Accuracy": f"{ex_m.get('accuracy', 0):.4f}",
        "Precision": f"{ex_m.get('precision', 0):.4f}",
        "Recall": f"{ex_m.get('recall', 0):.4f}",
        "F1 Score": f"{ex_m.get('f1_score', 0):.4f}",
        "ROC AUC": f"{ex_m.get('roc_auc', 0):.4f}",
    })

# Robustness mean ± std
if "robustness_analysis" in artefacts:
    rb = artefacts["robustness_analysis"]["summary"]
    comp_rows.append({
        "Validation": "Robustness (mean±std)",
        "Accuracy": f"{rb['accuracy']['mean']:.4f}±{rb['accuracy']['std']:.4f}",
        "Precision": f"{rb['precision']['mean']:.4f}±{rb['precision']['std']:.4f}",
        "Recall": f"{rb['recall']['mean']:.4f}±{rb['recall']['std']:.4f}",
        "F1 Score": f"{rb['f1_score']['mean']:.4f}±{rb['f1_score']['std']:.4f}",
        "ROC AUC": f"{rb['roc_auc']['mean']:.4f}±{rb['roc_auc']['std']:.4f}",
    })

comp_df = pd.DataFrame(comp_rows)

fig, ax = plt.subplots(figsize=(15, 2.5 + 0.6 * len(comp_rows)))
ax.axis('off')

table = ax.table(cellText=comp_df.values, colLabels=comp_df.columns, loc='center',
                 cellLoc='center',
                 colWidths=[0.22, 0.13, 0.13, 0.13, 0.13, 0.13])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.2)

for j in range(len(comp_df.columns)):
    table[(0, j)].set_facecolor('#2c3e50')
    table[(0, j)].set_text_props(weight='bold', color='white')

row_colors = ['#dfe6e9', '#ffffff', '#dfe6e9', '#ffffff']
for i in range(1, len(comp_df) + 1):
    for j in range(len(comp_df.columns)):
        table[(i, j)].set_facecolor(row_colors[(i - 1) % len(row_colors)])

plt.title('Validation Metrics Comparison (TabNet)', fontweight='bold', fontsize=13, pad=20)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "comparison_table.png"), bbox_inches='tight', dpi=300)
plt.close()
print(f"  ✓ Saved to plots/comparison_table.png\n")


# ============================================
# FINAL SUMMARY
# ============================================
print("=" * 60)
print("✅ All publication plots generated!")
print(f"📁 Location: {PLOTS_DIR}")
print("=" * 60)
for f in sorted(os.listdir(PLOTS_DIR)):
    print(f"  ✓ {f}")
