#!/usr/bin/env python3
"""
=============================================================================
MODULE 3: MACHINE LEARNING PRACTICALS AND MODEL IMPLEMENTATION
Loan Default Prediction System — Complete Implementation
=============================================================================
Student:    Advik Singh
Reg. No.:   RA2411056030023
Branch/Sec: DS-A
Course:     Machine Learning / Data Science and Machine Learning Training
=============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for script mode
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["font.size"] = 10

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# --------------------------------------------------------------------------
# Output directories
# --------------------------------------------------------------------------
RESULTS_DIR = "results"
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# --------------------------------------------------------------------------
# PRACTICAL 1: Environment Setup & Data Loading
# --------------------------------------------------------------------------
print("=" * 70)
print("PRACTICAL 1: Data Loading and Inspection")
print("=" * 70)

df = pd.read_csv("Module 3 Project Data.csv")
print(f"\nRaw dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Drop the non-predictive Index column and rename for convenience
df = df.drop(columns=["Index"])
df = df.rename(columns={
    "Bank Balance": "Bank_Balance",
    "Annual Salary": "Annual_Salary",
    "Defaulted?": "Defaulted"
})

print(f"\nCleaned dataset shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nMissing values per column:")
print(df.isnull().sum())
print(f"\nDuplicate rows: {df.duplicated().sum()}")
print("\nDescriptive statistics:")
print(df.describe().round(2))

# Identify column types
num_cols = ["Bank_Balance", "Annual_Salary"]
cat_cols = ["Employed"]
target_col = "Defaulted"
print(f"\nNumerical features: {num_cols}")
print(f"Categorical/binary features: {cat_cols}")
print(f"Target variable: {target_col}")

# --------------------------------------------------------------------------
# PRACTICAL 2: Exploratory Data Analysis (EDA)
# --------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PRACTICAL 2: Exploratory Data Analysis (EDA)")
print("=" * 70)

# 2a. Target distribution
print("\nTarget class distribution:")
print(df["Defaulted"].value_counts())
print("\nTarget class distribution (normalised):")
print(df["Defaulted"].value_counts(normalize=True).round(4))

fig, ax = plt.subplots(figsize=(6, 4.5))
sns.countplot(x="Defaulted", data=df, ax=ax, palette="Set2")
ax.set_title("Class Distribution: Loan Default (0 = No, 1 = Yes)")
ax.set_xlabel("Defaulted")
ax.set_ylabel("Count")
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}",
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha="center", va="bottom")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig01_class_distribution.png"))
plt.close()

# 2b. Histograms of numerical variables
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, col in zip(axes, num_cols):
    sns.histplot(data=df, x=col, hue="Defaulted", bins=30, kde=True, ax=ax,
                 palette="Set1", alpha=0.5)
    ax.set_title(f"Distribution of {col}")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig02_histograms.png"))
plt.close()

# 2c. Box plots for outlier analysis
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, col in zip(axes, num_cols):
    sns.boxplot(x="Defaulted", y=col, data=df, ax=ax, palette="Set2")
    ax.set_title(f"Box Plot of {col} by Default Status")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig03_boxplots.png"))
plt.close()

# 2d. Correlation heatmap
fig, ax = plt.subplots(figsize=(6, 5))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax,
            vmin=-1, vmax=1, center=0)
ax.set_title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig04_correlation_heatmap.png"))
plt.close()
print("\nCorrelation with target:")
print(corr["Defaulted"].sort_values(ascending=False).round(4))

# 2e. Scatter plots: Bank Balance vs Annual Salary colored by Default
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(data=df, x="Bank_Balance", y="Annual_Salary",
                hue="Defaulted", palette={0: "blue", 1: "red"},
                alpha=0.5, s=25, ax=ax)
ax.set_title("Bank Balance vs Annual Salary (coloured by Default)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig05_scatter_balance_salary.png"))
plt.close()

# 2f. Count plot for Employment status
fig, ax = plt.subplots(figsize=(6, 4.5))
ct = pd.crosstab(df["Employed"], df["Defaulted"], normalize="index")
ct.plot(kind="bar", stacked=True, ax=ax, color=["#4CAF50", "#F44336"])
ax.set_title("Default Rate by Employment Status")
ax.set_xlabel("Employed (0 = No, 1 = Yes)")
ax.set_ylabel("Proportion")
ax.legend(["No Default", "Default"])
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig06_employment_default.png"))
plt.close()

# 2g. Pair plot (sample for speed)
sample_df = df.sample(n=min(2000, len(df)), random_state=RANDOM_STATE)
g = sns.pairplot(sample_df, hue="Defaulted", vars=num_cols + ["Employed"],
                 palette={0: "blue", 1: "red"}, plot_kws={"alpha": 0.4, "s": 20})
g.fig.suptitle("Pair Plot of Features (2000-sample)", y=1.02)
plt.savefig(os.path.join(FIG_DIR, "fig07_pairplot.png"))
plt.close()

print("\nEDA figures saved to:", FIG_DIR)

# --------------------------------------------------------------------------
# PRACTICAL 3: Data Preprocessing & Train-Test Split
# --------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PRACTICAL 3: Data Preprocessing and Train-Test Split")
print("=" * 70)

X = df.drop(columns=["Defaulted"])
y = df["Defaulted"]

print(f"\nFeature matrix X shape: {X.shape}")
print(f"Target vector y shape: {y.shape}")
print(f"Positive rate (full data): {y.mean():.4f}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"\nTrain set size: {X_train.shape[0]}")
print(f"Test set size:  {X_test.shape[0]}")
print(f"Train positive rate: {y_train.mean():.4f}")
print(f"Test positive rate:  {y_test.mean():.4f}")

# Standard scaling for models that need it (LR, SVM, KNN, etc.)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# scale_pos_weight for XGBoost (ratio of neg to pos)
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
spw = neg_count / pos_count
print(f"\nscale_pos_weight for XGBoost: {spw:.2f}")

# --------------------------------------------------------------------------
# Helper: evaluate one model
# --------------------------------------------------------------------------
def evaluate_model(name, model, X_tr, y_tr, X_te, y_te):
    """Fit the model, predict, return a metrics dict + probabilities."""
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_te)[:, 1]
    else:
        y_prob = model.decision_function(X_te)
        # scale decision function to [0,1] for AUC using sigmoid
        y_prob = 1 / (1 + np.exp(-y_prob))

    metrics = {
        "accuracy": accuracy_score(y_te, y_pred),
        "precision": precision_score(y_te, y_pred, zero_division=0),
        "recall": recall_score(y_te, y_pred, zero_division=0),
        "f1": f1_score(y_te, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_te, y_prob),
        "cm": confusion_matrix(y_te, y_pred),
        "y_pred": y_pred,
        "y_prob": y_prob,
    }
    return metrics

# --------------------------------------------------------------------------
# PRACTICAL 4: Logistic Regression
# --------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PRACTICAL 4: Logistic Regression")
print("=" * 70)

logreg = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
logreg_metrics = evaluate_model("Logistic Regression", logreg,
                                X_train_scaled, y_train, X_test_scaled, y_test)
print(f"Accuracy:  {logreg_metrics['accuracy']:.4f}")
print(f"Precision: {logreg_metrics['precision']:.4f}")
print(f"Recall:    {logreg_metrics['recall']:.4f}")
print(f"F1:        {logreg_metrics['f1']:.4f}")
print(f"ROC-AUC:   {logreg_metrics['roc_auc']:.4f}")
print("Confusion Matrix:\n", logreg_metrics["cm"])

# --------------------------------------------------------------------------
# PRACTICAL 5: Decision Tree Classifier
# --------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PRACTICAL 5: Decision Tree Classifier")
print("=" * 70)

dt = DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced",
                            max_depth=5)
dt_metrics = evaluate_model("Decision Tree", dt,
                            X_train, y_train, X_test, y_test)
print(f"Accuracy:  {dt_metrics['accuracy']:.4f}")
print(f"Precision: {dt_metrics['precision']:.4f}")
print(f"Recall:    {dt_metrics['recall']:.4f}")
print(f"F1:        {dt_metrics['f1']:.4f}")
print(f"ROC-AUC:   {dt_metrics['roc_auc']:.4f}")
print("Confusion Matrix:\n", dt_metrics["cm"])

# Plot a pruned decision tree
fig, ax = plt.subplots(figsize=(14, 7))
plot_tree(dt, feature_names=X.columns, class_names=["No Default", "Default"],
          filled=True, rounded=True, fontsize=8, ax=ax, max_depth=3)
ax.set_title("Decision Tree (max 3 levels shown)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig08_decision_tree.png"))
plt.close()

# --------------------------------------------------------------------------
# PRACTICAL 6: Random Forest Classifier
# --------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PRACTICAL 6: Random Forest Classifier")
print("=" * 70)

rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                            random_state=RANDOM_STATE, n_jobs=-1)
rf_metrics = evaluate_model("Random Forest", rf,
                            X_train, y_train, X_test, y_test)
print(f"Accuracy:  {rf_metrics['accuracy']:.4f}")
print(f"Precision: {rf_metrics['precision']:.4f}")
print(f"Recall:    {rf_metrics['recall']:.4f}")
print(f"F1:        {rf_metrics['f1']:.4f}")
print(f"ROC-AUC:   {rf_metrics['roc_auc']:.4f}")
print("Confusion Matrix:\n", rf_metrics["cm"])

# --------------------------------------------------------------------------
# PRACTICAL 7: Support Vector Machine (SVM)
# --------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PRACTICAL 7: Support Vector Machine (SVM)")
print("=" * 70)

svm = SVC(kernel="rbf", probability=True, class_weight="balanced",
          random_state=RANDOM_STATE)
svm_metrics = evaluate_model("SVM (RBF)", svm,
                             X_train_scaled, y_train, X_test_scaled, y_test)
print(f"Accuracy:  {svm_metrics['accuracy']:.4f}")
print(f"Precision: {svm_metrics['precision']:.4f}")
print(f"Recall:    {svm_metrics['recall']:.4f}")
print(f"F1:        {svm_metrics['f1']:.4f}")
print(f"ROC-AUC:   {svm_metrics['roc_auc']:.4f}")
print("Confusion Matrix:\n", svm_metrics["cm"])

# --------------------------------------------------------------------------
# PRACTICAL 8: K-Nearest Neighbours (KNN)
# --------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PRACTICAL 8: K-Nearest Neighbours (KNN)")
print("=" * 70)

knn = KNeighborsClassifier(n_neighbors=11, weights="distance")
knn_metrics = evaluate_model("KNN", knn,
                             X_train_scaled, y_train, X_test_scaled, y_test)
print(f"Accuracy:  {knn_metrics['accuracy']:.4f}")
print(f"Precision: {knn_metrics['precision']:.4f}")
print(f"Recall:    {knn_metrics['recall']:.4f}")
print(f"F1:        {knn_metrics['f1']:.4f}")
print(f"ROC-AUC:   {knn_metrics['roc_auc']:.4f}")
print("Confusion Matrix:\n", knn_metrics["cm"])

# --------------------------------------------------------------------------
# PRACTICAL 9: Gradient Boosting
# --------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PRACTICAL 9: Gradient Boosting Classifier")
print("=" * 70)

gb = GradientBoostingClassifier(n_estimators=300, max_depth=3,
                                learning_rate=0.05, random_state=RANDOM_STATE)
gb_metrics = evaluate_model("Gradient Boosting", gb,
                            X_train, y_train, X_test, y_test)
print(f"Accuracy:  {gb_metrics['accuracy']:.4f}")
print(f"Precision: {gb_metrics['precision']:.4f}")
print(f"Recall:    {gb_metrics['recall']:.4f}")
print(f"F1:        {gb_metrics['f1']:.4f}")
print(f"ROC-AUC:   {gb_metrics['roc_auc']:.4f}")
print("Confusion Matrix:\n", gb_metrics["cm"])

# --------------------------------------------------------------------------
# PRACTICAL 10: AdaBoost
# --------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PRACTICAL 10: AdaBoost Classifier")
print("=" * 70)

ada = AdaBoostClassifier(n_estimators=300, learning_rate=0.1,
                         random_state=RANDOM_STATE)
# AdaBoost does not have class_weight; use sample_weight to up-weight minority class
from sklearn.utils.class_weight import compute_sample_weight
sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
ada.fit(X_train, y_train, sample_weight=sample_weights)
y_pred_ada = ada.predict(X_test)
y_prob_ada = ada.predict_proba(X_test)[:, 1]
ada_metrics = {
    "accuracy": accuracy_score(y_test, y_pred_ada),
    "precision": precision_score(y_test, y_pred_ada, zero_division=0),
    "recall": recall_score(y_test, y_pred_ada, zero_division=0),
    "f1": f1_score(y_test, y_pred_ada, zero_division=0),
    "roc_auc": roc_auc_score(y_test, y_prob_ada),
    "cm": confusion_matrix(y_test, y_pred_ada),
    "y_pred": y_pred_ada,
    "y_prob": y_prob_ada,
}
print(f"Accuracy:  {ada_metrics['accuracy']:.4f}")
print(f"Precision: {ada_metrics['precision']:.4f}")
print(f"Recall:    {ada_metrics['recall']:.4f}")
print(f"F1:        {ada_metrics['f1']:.4f}")
print(f"ROC-AUC:   {ada_metrics['roc_auc']:.4f}")
print("Confusion Matrix:\n", ada_metrics["cm"])

# --------------------------------------------------------------------------
# PRACTICAL 11: XGBoost (Detailed)
# --------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PRACTICAL 11: XGBoost Classifier")
print("=" * 70)

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,      # L1
    reg_lambda=1.0,     # L2
    scale_pos_weight=spw,
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    use_label_encoder=False,
    verbosity=0,
)
xgb_metrics = evaluate_model("XGBoost", xgb,
                             X_train, y_train, X_test, y_test)
print(f"Accuracy:  {xgb_metrics['accuracy']:.4f}")
print(f"Precision: {xgb_metrics['precision']:.4f}")
print(f"Recall:    {xgb_metrics['recall']:.4f}")
print(f"F1:        {xgb_metrics['f1']:.4f}")
print(f"ROC-AUC:   {xgb_metrics['roc_auc']:.4f}")
print("Confusion Matrix:\n", xgb_metrics["cm"])

# --------------------------------------------------------------------------
# PRACTICAL 12: CatBoost (Supplementary)
# --------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PRACTICAL 12: CatBoost Classifier (Supplementary)")
print("=" * 70)

cat = CatBoostClassifier(
    iterations=300,
    depth=4,
    learning_rate=0.05,
    auto_class_weights="Balanced",
    verbose=0,
    random_state=RANDOM_STATE,
)
cat_metrics = evaluate_model("CatBoost", cat,
                             X_train, y_train, X_test, y_test)
print(f"Accuracy:  {cat_metrics['accuracy']:.4f}")
print(f"Precision: {cat_metrics['precision']:.4f}")
print(f"Recall:    {cat_metrics['recall']:.4f}")
print(f"F1:        {cat_metrics['f1']:.4f}")
print(f"ROC-AUC:   {cat_metrics['roc_auc']:.4f}")
print("Confusion Matrix:\n", cat_metrics["cm"])

# --------------------------------------------------------------------------
# PRACTICAL 13: Model Comparison and Visualisation
# --------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PRACTICAL 13: Model Comparison and Visualisation")
print("=" * 70)

all_models = {
    "Logistic Regression": logreg_metrics,
    "Decision Tree":       dt_metrics,
    "Random Forest":       rf_metrics,
    "SVM (RBF)":           svm_metrics,
    "KNN":                 knn_metrics,
    "Gradient Boosting":   gb_metrics,
    "AdaBoost":            ada_metrics,
    "XGBoost":             xgb_metrics,
    "CatBoost":            cat_metrics,
}
# Also keep the trained model objects
model_objs = {
    "Logistic Regression": logreg,
    "Decision Tree":       dt,
    "Random Forest":       rf,
    "SVM (RBF)":           svm,
    "KNN":                 knn,
    "Gradient Boosting":   gb,
    "AdaBoost":            ada,
    "XGBoost":             xgb,
    "CatBoost":            cat,
}

results_df = pd.DataFrame({
    name: {k: v for k, v in m.items() if k not in ("cm", "y_pred", "y_prob")}
    for name, m in all_models.items()
}).T
results_df = results_df.sort_values("roc_auc", ascending=False)
results_df = results_df.round(4)
print("\nModel Comparison Table (sorted by ROC-AUC):")
print(results_df.to_string())

results_df.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"))

# Bar chart comparison
fig, ax = plt.subplots(figsize=(12, 5.5))
metrics_to_plot = ["accuracy", "precision", "recall", "f1", "roc_auc"]
results_df[metrics_to_plot].plot(kind="bar", ax=ax)
ax.set_title("Model Performance Comparison (All Metrics)")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.05)
ax.set_xticklabels(results_df.index, rotation=25, ha="right")
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig09_model_comparison_bar.png"))
plt.close()

# ROC curves
fig, ax = plt.subplots(figsize=(8, 7))
for name, m in all_models.items():
    fpr, tpr, _ = roc_curve(y_test, m["y_prob"])
    ax.plot(fpr, tpr, label=f"{name} (AUC={m['roc_auc']:.3f})", linewidth=1.5)
ax.plot([0, 1], [0, 1], "k--", linewidth=1)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — All Models")
ax.legend(loc="lower right", fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig10_roc_curves.png"))
plt.close()

# Best model (by ROC-AUC)
best_model_name = results_df["roc_auc"].idxmax()
print(f"\nBest model by ROC-AUC: {best_model_name} (AUC={results_df.loc[best_model_name, 'roc_auc']:.4f})")

# Confusion matrix for best model
best_cm = all_models[best_model_name]["cm"]
fig, ax = plt.subplots(figsize=(5, 4.5))
sns.heatmap(best_cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["No Default", "Default"],
            yticklabels=["No Default", "Default"])
ax.set_title(f"Confusion Matrix — {best_model_name}")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig11_best_confusion_matrix.png"))
plt.close()

# --------------------------------------------------------------------------
# PRACTICAL 14: Feature Importance
# --------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PRACTICAL 14: Feature Importance Analysis")
print("=" * 70)

# Random Forest
rf_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nRandom Forest feature importance:")
print(rf_imp.round(4))

fig, ax = plt.subplots(figsize=(7, 4.5))
sns.barplot(x=rf_imp.values, y=rf_imp.index, ax=ax, palette="viridis")
ax.set_title("Feature Importance — Random Forest")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig12_feature_importance_rf.png"))
plt.close()

# XGBoost
xgb_imp = pd.Series(xgb.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nXGBoost feature importance:")
print(xgb_imp.round(4))

fig, ax = plt.subplots(figsize=(7, 4.5))
sns.barplot(x=xgb_imp.values, y=xgb_imp.index, ax=ax, palette="viridis")
ax.set_title("Feature Importance — XGBoost")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig13_feature_importance_xgb.png"))
plt.close()

# Gradient Boosting
gb_imp = pd.Series(gb.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nGradient Boosting feature importance:")
print(gb_imp.round(4))

fig, ax = plt.subplots(figsize=(7, 4.5))
sns.barplot(x=gb_imp.values, y=gb_imp.index, ax=ax, palette="viridis")
ax.set_title("Feature Importance — Gradient Boosting")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig14_feature_importance_gb.png"))
plt.close()

# CatBoost
cat_imp = pd.Series(cat.get_feature_importance(), index=X.columns).sort_values(ascending=False)
print("\nCatBoost feature importance:")
print(cat_imp.round(4))

fig, ax = plt.subplots(figsize=(7, 4.5))
sns.barplot(x=cat_imp.values, y=cat_imp.index, ax=ax, palette="viridis")
ax.set_title("Feature Importance — CatBoost")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig15_feature_importance_cat.png"))
plt.close()

# Logistic Regression coefficients
lr_coef = pd.Series(logreg.coef_[0], index=X.columns).sort_values()
print("\nLogistic Regression coefficients (scaled features):")
print(lr_coef.round(4))

fig, ax = plt.subplots(figsize=(7, 4))
colors = ["red" if c < 0 else "green" for c in lr_coef.values]
sns.barplot(x=lr_coef.values, y=lr_coef.index, palette=colors, ax=ax)
ax.set_title("Logistic Regression Coefficients (scaled)")
ax.set_xlabel("Coefficient Value")
ax.axvline(x=0, color="black", linewidth=0.8)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig16_lr_coefficients.png"))
plt.close()

# --------------------------------------------------------------------------
# PRACTICAL 15: Hyperparameter Tuning with GridSearchCV
# --------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PRACTICAL 15: Hyperparameter Tuning (GridSearchCV on XGBoost)")
print("=" * 70)

param_grid = {
    "n_estimators": [100, 300],
    "max_depth": [3, 4, 5],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0],
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

xgb_base = XGBClassifier(
    scale_pos_weight=spw,
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    use_label_encoder=False,
    verbosity=0,
)

grid_search = GridSearchCV(
    estimator=xgb_base,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
    verbose=0,
)
grid_search.fit(X_train, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")

best_xgb = grid_search.best_estimator_
y_pred_tuned = best_xgb.predict(X_test)
y_prob_tuned = best_xgb.predict_proba(X_test)[:, 1]
tuned_metrics = {
    "accuracy": accuracy_score(y_test, y_pred_tuned),
    "precision": precision_score(y_test, y_pred_tuned, zero_division=0),
    "recall": recall_score(y_test, y_pred_tuned, zero_division=0),
    "f1": f1_score(y_test, y_pred_tuned, zero_division=0),
    "roc_auc": roc_auc_score(y_test, y_prob_tuned),
    "cm": confusion_matrix(y_test, y_pred_tuned),
}
print("\nTuned XGBoost on test set:")
for k, v in tuned_metrics.items():
    if k != "cm":
        print(f"  {k}: {v:.4f}")
print("Confusion Matrix:\n", tuned_metrics["cm"])

# Baseline XGBoost vs tuned comparison
print("\n--- Baseline XGBoost vs Tuned XGBoost ---")
compare_tuned = pd.DataFrame({
    "Baseline XGBoost": {k: v for k, v in xgb_metrics.items()
                         if k not in ("cm", "y_pred", "y_prob")},
    "Tuned XGBoost":    {k: v for k, v in tuned_metrics.items()
                         if k != "cm"},
}).round(4)
print(compare_tuned.to_string())
compare_tuned.to_csv(os.path.join(RESULTS_DIR, "tuning_comparison.csv"))

# --------------------------------------------------------------------------
# PRACTICAL 16: Cross-Validation
# --------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PRACTICAL 16: Cross-Validation")
print("=" * 70)

cv_models = {
    "Logistic Regression": (logreg, X_train_scaled),
    "Random Forest":       (rf, X_train),
    "XGBoost":             (xgb, X_train),
}
cv_results = {}
for name, (m, X_tr) in cv_models.items():
    scores = cross_val_score(m, X_tr, y_train, cv=5, scoring="roc_auc", n_jobs=-1)
    cv_results[name] = scores
    print(f"{name:22s}: CV ROC-AUC = {scores.mean():.4f} ± {scores.std():.4f}")

# --------------------------------------------------------------------------
# PRACTICAL 17: Classification Report for Best Model
# --------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PRACTICAL 17: Detailed Classification Report (Best Model)")
print("=" * 70)
best_y_pred = all_models[best_model_name]["y_pred"]
report = classification_report(y_test, best_y_pred,
                               target_names=["No Default", "Default"])
print(f"\nBest model: {best_model_name}")
print(report)

# Save full classification report
with open(os.path.join(RESULTS_DIR, "best_model_classification_report.txt"), "w") as f:
    f.write(f"Best model: {best_model_name}\n\n")
    f.write(report)

# --------------------------------------------------------------------------
# Save aggregated results
# --------------------------------------------------------------------------
summary = {
    "best_model_name": best_model_name,
    "best_metrics": {k: float(v) for k, v in all_models[best_model_name].items()
                     if k not in ("cm", "y_pred", "y_prob")},
    "all_metrics": results_df.to_dict(),
    "tuned_xgb_metrics": {k: (float(v) if not isinstance(v, np.ndarray) else v.tolist())
                          for k, v in tuned_metrics.items()},
}
import json
with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2, default=str)

print("\n" + "=" * 70)
print("ALL PRACTICALS COMPLETED SUCCESSFULLY")
print("=" * 70)
print(f"\nResults saved to: {RESULTS_DIR}/")
print(f"Figures saved to: {FIG_DIR}/")
print(f"Best model: {best_model_name} with ROC-AUC = {results_df.loc[best_model_name, 'roc_auc']:.4f}")
