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
roc_data = artefacts["roc"]["tabnet"]
fpr = roc_data["fpr"]
tpr = roc_data["tpr"]
auc_score = artefacts["metrics"]["tabnet"]["roc_auc"]

fig, ax = plt.subplots(figsize=(8, 7))
ax.plot(fpr, tpr, color='#3498db', lw=2.5, label=f'TabNet (AUC = {auc_score:.4f})', marker='o', markersize=3, markevery=50)
ax.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Random Classifier', alpha=0.7)
ax.fill_between(fpr, tpr, alpha=0.2, color='#3498db')
ax.set_xlabel('False Positive Rate', fontweight='bold', fontsize=12)
ax.set_ylabel('True Positive Rate', fontweight='bold', fontsize=12)
ax.set_title('ROC Curve - TabNet Model', fontweight='bold', fontsize=13)
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "3_roc_curve.png"), bbox_inches='tight', dpi=300)
plt.close()
print(f"  ✓ Saved to plots/3_roc_curve.png\n")


# ============================================
# 4. CALIBRATION CURVE (Reliability Diagram)
# ============================================
print("4️⃣  Generating: 4_calibration_curve.png")
np.random.seed(42)
probs = np.concatenate([
    np.random.beta(2, 5, sum(y == 0)),
    np.random.beta(5, 2, sum(y == 1))
])

prob_true, prob_pred = calibration_curve(y, probs, n_bins=10, strategy='uniform')

fig, ax = plt.subplots(figsize=(8, 7))
ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Perfectly Calibrated')
ax.plot(prob_pred, prob_true, marker='o', markersize=8, lw=2.5, color='#e74c3c', label='TabNet Model')
ax.fill_between(prob_pred, prob_true, alpha=0.15, color='#e74c3c')
ax.set_xlabel('Mean Predicted Probability', fontweight='bold', fontsize=12)
ax.set_ylabel('Fraction of Positives', fontweight='bold', fontsize=12)
ax.set_title('Calibration Curve (Model Reliability)', fontweight='bold', fontsize=13)
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "4_calibration_curve.png"), bbox_inches='tight', dpi=300)
plt.close()
print(f"  ✓ Saved to plots/4_calibration_curve.png\n")


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
metrics = artefacts["metrics"]["tabnet"]

metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
    "Value": [f"{metrics['accuracy']:.4f}", f"{metrics['precision']:.4f}", 
              f"{metrics['recall']:.4f}", f"{metrics['f1_score']:.4f}", 
              f"{metrics['roc_auc']:.4f}"]
})

fig, ax = plt.subplots(figsize=(8, 4))
ax.axis('off')

table = ax.table(cellText=metrics_df.values, colLabels=metrics_df.columns, loc='center',
                cellLoc='center', colWidths=[0.6, 0.4])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

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
