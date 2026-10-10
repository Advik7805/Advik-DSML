#!/usr/bin/env python3
"""Generate the PDF report using ReportLab Platypus."""
import os
import json
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, KeepTogether, Preformatted
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

RESULTS_DIR = "results"
FIG_DIR = os.path.join(RESULTS_DIR, "figures")

results_df = pd.read_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"), index_col=0)
with open(os.path.join(RESULTS_DIR, "summary.json")) as f:
    summary = json.load(f)
best_name = summary["best_model_name"]
best_m = summary["best_metrics"]

# ---------- Styles ----------
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="TitleBig", parent=styles["Title"], fontSize=26,
                          textColor=colors.HexColor("#1A237E"), alignment=TA_CENTER,
                          spaceAfter=20))
styles.add(ParagraphStyle(name="TitleMed", parent=styles["Title"], fontSize=20,
                          textColor=colors.HexColor("#303F9F"), alignment=TA_CENTER,
                          spaceAfter=12))
styles.add(ParagraphStyle(name="SubTitle", parent=styles["Normal"], fontSize=14,
                          textColor=colors.HexColor("#555555"), alignment=TA_CENTER,
                          spaceAfter=30, fontName="Helvetica-Oblique"))
styles.add(ParagraphStyle(name="H1", parent=styles["Heading1"], fontSize=18,
                          textColor=colors.HexColor("#1A237E"), spaceBefore=16, spaceAfter=10))
styles.add(ParagraphStyle(name="H2", parent=styles["Heading2"], fontSize=14,
                          textColor=colors.HexColor("#303F9F"), spaceBefore=12, spaceAfter=6))
styles.add(ParagraphStyle(name="Body", parent=styles["Normal"], fontSize=10.5,
                          leading=14, alignment=TA_JUSTIFY, spaceAfter=6))
styles.add(ParagraphStyle(name="Caption", parent=styles["Normal"], fontSize=9,
                          textColor=colors.HexColor("#666666"), alignment=TA_CENTER,
                          fontName="Helvetica-Oblique", spaceAfter=10))
styles.add(ParagraphStyle(name="CodeStyle", parent=styles["Code"], fontSize=8,
                          leading=10, backColor=colors.HexColor("#F2F2F2"),
                          leftIndent=10, rightIndent=10, spaceAfter=8))
styles.add(ParagraphStyle(name="MyBullet", parent=styles["Normal"], fontSize=10.5,
                          leading=14, leftIndent=20, bulletIndent=8, spaceAfter=3))

# Page number callback
def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(colors.HexColor("#777777"))
    canvas.drawRightString(A4[0] - 2*cm, 1.5*cm, f"Page {doc.page}")
    canvas.drawString(2*cm, 1.5*cm, "Module 3: Machine Learning Practicals — Advik Singh (RA2411056030023)")
    canvas.restoreState()

def bullet(text):
    return Paragraph(f"• {text}", styles["MyBullet"])

def code(text):
    return Preformatted(text, styles["CodeStyle"])

def fig(path, w=12*cm, caption=""):
    elements = []
    if os.path.exists(path):
        elements.append(Image(path, width=w, height=w*0.75))
        elements[-1].hAlign = "CENTER"
    if caption:
        elements.append(Paragraph(caption, styles["Caption"]))
    return elements

def make_table(df, col_widths=None):
    data = [["Model"] + [c.replace("_", " ").title() for c in df.columns]]
    for idx, row in df.iterrows():
        data.append([str(idx)] + [f"{float(v):.4f}" for v in row])
    t = Table(data, colWidths=col_widths or [3.5*cm] + [2.2*cm]*len(df.columns))
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1A237E")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("ALIGN", (1,1), (-1,-1), "CENTER"),
        ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#AAAAAA")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F5F5FF")]),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ]))
    return t

# ---------- Build Story ----------
doc = SimpleDocTemplate("Advik_Singh_Module_3_ML_Report.pdf", pagesize=A4,
                        leftMargin=2.5*cm, rightMargin=2.5*cm,
                        topMargin=2*cm, bottomMargin=2*cm)
story = []

# Title page
story.append(Spacer(1, 3*cm))
story.append(Paragraph("MODULE 3", styles["TitleBig"]))
story.append(Paragraph("MACHINE LEARNING PRACTICALS<br/>AND MODEL IMPLEMENTATION", styles["TitleMed"]))
story.append(Spacer(1, 1*cm))
story.append(Paragraph("Loan Default Prediction System<br/>A Comparative Study of Classification Algorithms", styles["SubTitle"]))
story.append(Spacer(1, 3*cm))

info_data = [
    ["Submitted by:", "Advik Singh"],
    ["Registration Number:", "RA2411056030023"],
    ["Branch / Section:", "DS-A"],
    ["Course:", "Machine Learning / Data Science and Machine Learning Training"],
]
info_t = Table(info_data, colWidths=[5*cm, 9*cm])
info_t.setStyle(TableStyle([
    ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,-1), 11),
    ("BOTTOMPADDING", (0,0), (-1,-1), 8),
    ("TEXTCOLOR", (0,0), (0,-1), colors.HexColor("#1A237E")),
]))
story.append(info_t)
story.append(PageBreak())

# Abstract
story.append(Paragraph("Abstract", styles["H1"]))
story.append(Paragraph(
    "Banks approve loans every day without knowing for certain who will repay and who will default. "
    "This project builds a predictive system that estimates the risk of loan default from three customer "
    "attributes — employment status, bank balance, and annual salary — using a banking dataset of 10,000 "
    "customers. Nine classification algorithms (Logistic Regression, Decision Tree, Random Forest, Support "
    "Vector Machine, K-Nearest Neighbours, Gradient Boosting, AdaBoost, XGBoost, and CatBoost) are trained "
    "with appropriate handling of class imbalance, and compared using accuracy, precision, recall, F1-score, "
    "and ROC-AUC. Hyperparameter tuning via GridSearchCV is demonstrated on XGBoost, feature importance is "
    "analysed across tree-based models, and cross-validation is used to obtain stable performance estimates. "
    "Logistic Regression emerges as the best performer by ROC-AUC (0.95), achieving the highest recall for "
    "catching defaulters. Bank balance is found to be the dominant predictive feature.",
    styles["Body"]))
story.append(PageBreak())

# TOC
story.append(Paragraph("Table of Contents", styles["H1"]))
toc_items = [
    "1. Introduction to Machine Learning",
    "2. Project Objectives",
    "3. Original Module 3 Requirements",
    "4. Dataset Description",
    "5. Software and Libraries Used",
    "6. Machine Learning Theory",
    "7. Data Loading and Inspection",
    "8. Exploratory Data Analysis",
    "9. Data Cleaning and Preprocessing",
    "10. Feature Engineering and Train-Test Split",
    "11. Implementation of Required Algorithms",
    "12. Detailed XGBoost Implementation",
    "13. CatBoost (Supplementary)",
    "14. Hyperparameter Tuning",
    "15. Model Evaluation Metrics",
    "16. Results and Model Comparison",
    "17. Visualizations and Interpretation",
    "18. Discussion of Findings",
    "19. Challenges and Solutions",
    "20. Limitations",
    "21. Conclusion",
    "22. Future Scope",
    "23. References",
    "Appendix A: Complete Python Code",
    "Appendix B: Installation and Execution Instructions",
    "Appendix C: Viva Questions",
]
for item in toc_items:
    story.append(Paragraph(item, styles["Body"]))
story.append(PageBreak())

# 1. Introduction
story.append(Paragraph("1. Introduction to Machine Learning", styles["H1"]))
story.append(Paragraph(
    "Machine Learning (ML) is a subfield of artificial intelligence that enables systems to learn patterns "
    "from data and make predictions or decisions without being explicitly programmed for every scenario. "
    "In supervised learning — the focus of Module 3 — a model learns a mapping from input features X to a "
    "known target variable y using labelled training examples. When the target is categorical (e.g. default "
    "vs. no default), the task is called classification.", styles["Body"]))
story.append(Paragraph(
    "Classification algorithms are foundational to many real-world applications: fraud detection, medical "
    "diagnosis, spam filtering, credit risk assessment, and — the subject of this project — loan default "
    "prediction. Banks use ML models to estimate the probability that an applicant will fail to repay, "
    "enabling them to price risk appropriately.", styles["Body"]))
story.append(Paragraph(
    "This module covers: linear models (Logistic Regression), tree-based models (Decision Tree, Random Forest, "
    "Gradient Boosting, AdaBoost, XGBoost, CatBoost), kernel methods (Support Vector Machines), instance-based "
    "methods (KNN), data preprocessing, feature scaling, class imbalance handling, cross-validation, "
    "hyperparameter tuning, evaluation metrics, and feature importance.", styles["Body"]))

# 2. Objectives
story.append(Paragraph("2. Project Objectives", styles["H1"]))
for obj in [
    "Load, clean, and explore a real-world banking dataset (10,000 customer records).",
    "Perform comprehensive Exploratory Data Analysis (EDA) with visualizations.",
    "Preprocess the data, scale features, and split into train/test sets while preventing data leakage.",
    "Implement nine classification algorithms: Logistic Regression, Decision Tree, Random Forest, SVM, KNN, Gradient Boosting, AdaBoost, XGBoost, and CatBoost.",
    "Address class imbalance using class weights, scale_pos_weight, and sample weighting.",
    "Evaluate models using accuracy, precision, recall, F1-score, and ROC-AUC.",
    "Compare models visually and select the best.",
    "Analyze feature importance and interpret logistic regression coefficients.",
    "Demonstrate GridSearchCV hyperparameter tuning and k-fold cross-validation.",
    "Provide complete, documented, reproducible Python code for every practical.",
]:
    story.append(bullet(obj))

# 3. Original Requirements
story.append(Paragraph("3. Original Module 3 Requirements", styles["H1"]))
story.append(Paragraph("Per the AI_ML Trainer Syllabus and the original Module 3 notebook/report, students are required to:", styles["Body"]))
for r in [
    "Implement Linear & Logistic Regression.",
    "Implement Decision Trees & Random Forests.",
    "Implement Support Vector Machines (SVM).",
    "Implement K-Nearest Neighbours (KNN).",
    "Implement Gradient Boosting — XGBoost.",
    "Apply model evaluation metrics: accuracy, F1, ROC-AUC.",
    "Perform cross-validation and hyperparameter tuning (GridSearchCV).",
    "Complete the Mini Project: Loan Default Prediction System — build, compare, and evaluate classifiers (LR, RF, XGBoost, SVM) on the banking dataset, select the best model, and explain results using feature importance.",
]:
    story.append(bullet(r))
story.append(Paragraph("CatBoost is included as a supplementary comparison, clearly identified as such.", styles["Body"]))

# 4. Dataset Description
story.append(Paragraph("4. Dataset Description", styles["H1"]))
story.append(Paragraph(
    "The dataset (Module 3 Project Data.csv) contains 10,000 customer records. The Index column was dropped as "
    "it carries no predictive information. The dataset has no missing values and no duplicate rows.", styles["Body"]))

ds_data = [
    ["Column", "Type", "Description"],
    ["Employed", "Binary (0/1)", "1 if employed, 0 otherwise"],
    ["Bank Balance", "Continuous", "Current bank balance [0, 31851.84]"],
    ["Annual Salary", "Continuous", "Annual income [9263.64, 882650.76]"],
    ["Defaulted", "Binary (0/1)", "Target: 1 = default, 0 = no default"],
]
ds_t = Table(ds_data, colWidths=[3.5*cm, 3*cm, 9*cm])
ds_t.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1A237E")),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,-1), 9.5),
    ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#AAAAAA")),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F5F5FF")]),
]))
story.append(ds_t)
story.append(Spacer(1, 0.3*cm))
story.append(Paragraph(
    "Class balance: Only 3.33% of customers (333 out of 10,000) defaulted, confirming a highly imbalanced dataset.", styles["Body"]))

# 5. Software
story.append(Paragraph("5. Software and Libraries Used", styles["H1"]))
for lib, purp in [
    ("Python 3.11", "Core language"),
    ("NumPy", "Numerical computing"),
    ("Pandas", "DataFrames and data manipulation"),
    ("Matplotlib", "Plotting"),
    ("Seaborn", "Statistical visualization"),
    ("scikit-learn", "ML algorithms, preprocessing, metrics"),
    ("XGBoost", "Extreme Gradient Boosting"),
    ("CatBoost", "Categorical Boosting"),
    ("Jupyter", "Interactive notebook environment"),
]:
    story.append(bullet(f"<b>{lib}:</b> {purp}"))

# 6. Theory
story.append(PageBreak())
story.append(Paragraph("6. Machine Learning Theory", styles["H1"]))

theory_sections = [
    ("6.1 Logistic Regression",
     "Models P(y=1|x) via the sigmoid σ(z) = 1/(1+e⁻ᶻ), where z = β₀+β₁x₁+…+βₙxₙ. "
     "Learns coefficients by minimizing log-loss. Fast, interpretable, outputs calibrated probabilities. "
     "Sensitive to feature scale and outliers."),
    ("6.2 Decision Tree",
     "Recursively partitions feature space using threshold splits to minimize Gini/entropy impurity. "
     "Interpretable, scale-invariant, but prone to overfitting; controlled via max_depth, min_samples_split."),
    ("6.3 Random Forest (Bagging)",
     "Ensemble of many decision trees trained on bootstrap samples (bagging), with random feature subsets at each split. "
     "Majority voting reduces variance."),
    ("6.4 Support Vector Machine",
     "Finds a maximum-margin separating hyperplane. The kernel trick (RBF) handles non-linear boundaries. "
     "Requires scaling; C and γ are key parameters."),
    ("6.5 K-Nearest Neighbours",
     "Lazy instance-based method; classifies by majority vote of k nearest training points (Euclidean distance). "
     "Requires scaling."),
    ("6.6 Gradient Boosting & AdaBoost (Boosting)",
     "Sequential ensemble where each new learner corrects predecessor errors. AdaBoost re-weights misclassified samples; "
     "Gradient Boosting fits trees to negative gradients of the loss."),
    ("6.7 XGBoost",
     "Optimized regularized gradient boosting: adds L1/L2 penalties on leaf weights, column/row subsampling, "
     "parallelized split finding, built-in missing value handling, and scale_pos_weight for class imbalance. "
     "Objective: Obj = ΣL(y,ŷ) + Σ(γTₖ + ½λ||wₖ||²)."),
    ("6.8 CatBoost",
     "Handles categorical features natively via ordered target encoding; uses ordered boosting to reduce prediction shift."),
    ("6.9 Class Imbalance Handling",
     "We used class_weight='balanced' (LR, DT, RF, SVM), scale_pos_weight≈29 (XGBoost), "
     "auto_class_weights='Balanced' (CatBoost), compute_sample_weight='balanced' (AdaBoost), and stratified splits."),
    ("6.10 Evaluation Metrics",
     "Accuracy=(TP+TN)/Total; Precision=TP/(TP+FP); Recall=TP/(TP+FN); "
     "F1=2·P·R/(P+R); ROC-AUC measures ranking ability across thresholds and is robust to imbalance."),
]
for title, text in theory_sections:
    story.append(Paragraph(title, styles["H2"]))
    story.append(Paragraph(text, styles["Body"]))

# 7. Data Loading
story.append(PageBreak())
story.append(Paragraph("7. Data Loading and Inspection", styles["H1"]))
story.append(Paragraph("The dataset is loaded with Pandas; the Index column is dropped and columns are renamed.", styles["Body"]))
story.append(code("""import pandas as pd
df = pd.read_csv("Module 3 Project Data.csv")
df = df.drop(columns=["Index"])
df = df.rename(columns={"Bank Balance": "Bank_Balance",
                         "Annual Salary": "Annual_Salary",
                         "Defaulted?": "Defaulted"})
print(df.shape)        # (10000, 4)
print(df.isnull().sum())  # no missing values
print(df.duplicated().sum())  # 0 duplicates"""))
for obs in [
    "Cleaned dataset: 10,000 rows × 4 columns (3 features + target).",
    "No missing values; no duplicate rows.",
    "Employed: binary (70.56% employed).",
    "Bank Balance: mean ≈ 10,024, range [0, 31,852].",
    "Annual Salary: mean ≈ 402,204, range [9,264, 882,651].",
    "Target: 3.33% defaults, 96.67% non-defaults (heavily imbalanced).",
]:
    story.append(bullet(obs))

# 8. EDA
story.append(Paragraph("8. Exploratory Data Analysis", styles["H1"]))
story.append(Paragraph("8.1 Target Distribution", styles["H2"]))
story.extend(fig(os.path.join(FIG_DIR, "fig01_class_distribution.png"), 9*cm,
                 "Figure 1: Class distribution — severe imbalance."))
story.append(Paragraph("8.2 Feature Distributions", styles["H2"]))
story.extend(fig(os.path.join(FIG_DIR, "fig02_histograms.png"), 14*cm,
                 "Figure 2: Histograms of numerical features by default status."))
story.append(Paragraph("8.3 Outlier Analysis (Box Plots)", styles["H2"]))
story.extend(fig(os.path.join(FIG_DIR, "fig03_boxplots.png"), 14*cm,
                 "Figure 3: Box plots by default status."))
story.append(Paragraph("8.4 Correlation Heatmap", styles["H2"]))
story.extend(fig(os.path.join(FIG_DIR, "fig04_correlation_heatmap.png"), 9*cm,
                 "Figure 4: Correlation heatmap. Bank Balance has the strongest positive correlation with Default (r≈0.35)."))
story.append(Paragraph("8.5 Scatter Plot", styles["H2"]))
story.extend(fig(os.path.join(FIG_DIR, "fig05_scatter_balance_salary.png"), 12*cm,
                 "Figure 5: Bank Balance vs Annual Salary. Defaulters (red) cluster at higher balances."))
story.append(Paragraph("8.6 Employment vs Default", styles["H2"]))
story.extend(fig(os.path.join(FIG_DIR, "fig06_employment_default.png"), 10*cm,
                 "Figure 6: Default rates are similar for employed and unemployed customers."))

# 9. Preprocessing
story.append(PageBreak())
story.append(Paragraph("9. Data Cleaning and Preprocessing", styles["H1"]))
for s in [
    "Dropped the non-predictive Index column.",
    "Renamed columns for easier handling.",
    "Zero missing values → no imputation needed.",
    "Zero duplicates → no deduplication.",
    "Outliers preserved — they may represent genuine at-risk customers.",
    "Employed already numeric (0/1), so no encoding needed.",
    "StandardScaler fit on training set only; applied to LR, SVM, KNN. Tree models do not require scaling.",
]:
    story.append(bullet(s))

# 10. Split
story.append(Paragraph("10. Feature Engineering and Train-Test Split", styles["H1"]))
story.append(code("""from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df.drop(columns=["Defaulted"])
y = df["Defaulted"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)"""))
story.append(Paragraph(
    "80/20 stratified split preserves the 3.33% default rate in both sets. The scaler is fit only on training "
    "data to prevent leakage. scale_pos_weight ≈ 29.08 is computed from the training set for XGBoost.", styles["Body"]))

# 11. Algorithms
story.append(PageBreak())
story.append(Paragraph("11. Implementation of Required Algorithms", styles["H1"]))

# Helper to add a practical section
def practical(num, title, aim, code_text, metrics_dict=None):
    story.append(Paragraph(f"11.{num} Practical {num+3} — {title}", styles["H2"]))
    story.append(Paragraph(f"<b>Aim:</b> {aim}", styles["Body"]))
    story.append(code(code_text))
    if metrics_dict:
        story.append(Paragraph(
            f"<b>Result:</b> Accuracy={metrics_dict['accuracy']:.4f}, "
            f"Precision={metrics_dict['precision']:.4f}, Recall={metrics_dict['recall']:.4f}, "
            f"F1={metrics_dict['f1']:.4f}, ROC-AUC={metrics_dict['roc_auc']:.4f}", styles["Body"]))

practical(1, "Logistic Regression",
          "Train logistic regression with balanced class weights on scaled features.",
          """from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
logreg.fit(X_train_scaled, y_train)""", best_m)

practical(2, "Decision Tree",
          "Build a decision tree with max_depth=5 and balanced class weights.",
          """from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42, class_weight="balanced", max_depth=5)
dt.fit(X_train, y_train)""", results_df.loc["Decision Tree"].to_dict())

story.extend(fig(os.path.join(FIG_DIR, "fig08_decision_tree.png"), 14*cm,
                 "Figure 7: Visualization of the trained decision tree (first 3 levels)."))

practical(3, "Random Forest",
          "Train a Random Forest of 300 trees with balanced class weights.",
          """from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                            random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)""", results_df.loc["Random Forest"].to_dict())

practical(4, "Support Vector Machine (SVM)",
          "Train RBF-kernel SVM with probability estimates on scaled features.",
          """from sklearn.svm import SVC
svm = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)
svm.fit(X_train_scaled, y_train)""", results_df.loc["SVM (RBF)"].to_dict())

practical(5, "K-Nearest Neighbours (KNN)",
          "Train KNN with k=11 and distance-based weighting on scaled features.",
          """from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=11, weights="distance")
knn.fit(X_train_scaled, y_train)""", results_df.loc["KNN"].to_dict())

practical(6, "Gradient Boosting",
          "Train Gradient Boosting (300 trees, lr=0.05, max_depth=3).",
          """from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(n_estimators=300, max_depth=3,
                                learning_rate=0.05, random_state=42)
gb.fit(X_train, y_train)""", results_df.loc["Gradient Boosting"].to_dict())

practical(7, "AdaBoost",
          "Train AdaBoost with balanced sample weights (no built-in class_weight).",
          """from sklearn.ensemble import AdaBoostClassifier
from sklearn.utils.class_weight import compute_sample_weight
ada = AdaBoostClassifier(n_estimators=300, learning_rate=0.1, random_state=42)
sw = compute_sample_weight(class_weight="balanced", y=y_train)
ada.fit(X_train, y_train, sample_weight=sw)""", results_df.loc["AdaBoost"].to_dict())

# 12. XGBoost
story.append(PageBreak())
story.append(Paragraph("12. Detailed XGBoost Implementation and Explanation", styles["H1"]))
story.append(code("""from xgboost import XGBClassifier
xgb = XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    scale_pos_weight=29.08, eval_metric="logloss",
    random_state=42, verbosity=0)
xgb.fit(X_train, y_train)"""))
xgb_concepts = [
    "<b>Boosting vs Bagging:</b> Bagging builds independent trees in parallel (reduces variance); Boosting builds sequential trees correcting errors (reduces bias).",
    "<b>Objective function:</b> Obj = ΣL(y,ŷ) + Σ(γTₖ + ½λ||wₖ||²) — loss plus complexity penalty.",
    "<b>Learning rate (eta):</b> Shrinks each tree's contribution; smaller values generalize better but need more trees.",
    "<b>n_estimators / max_depth:</b> Control ensemble size and individual tree complexity.",
    "<b>subsample / colsample_bytree:</b> Stochastic sampling to reduce overfitting.",
    "<b>L1/L2 regularization:</b> Penalty terms α (L1) and λ (L2) on leaf weights.",
    "<b>scale_pos_weight:</b> Balances classes for imbalanced datasets; set to neg/pos ≈ 29.",
    "<b>Feature importance:</b> Based on gain (split improvement), weight (split count), and cover.",
    "<b>Hyperparameter tuning:</b> Use GridSearchCV/RandomizedSearchCV with CV to optimize.",
]
for c in xgb_concepts:
    story.append(bullet(c))
story.append(Paragraph(
    f"<b>Result:</b> Accuracy={results_df.loc['XGBoost','accuracy']:.4f}, "
    f"Precision={results_df.loc['XGBoost','precision']:.4f}, "
    f"Recall={results_df.loc['XGBoost','recall']:.4f}, "
    f"F1={results_df.loc['XGBoost','f1']:.4f}, ROC-AUC={results_df.loc['XGBoost','roc_auc']:.4f}",
    styles["Body"]))

# 13. CatBoost
story.append(Paragraph("13. CatBoost (Supplementary Comparison)", styles["H1"]))
story.append(Paragraph(
    "CatBoost natively handles categorical features via ordered target encoding and uses ordered boosting to "
    "reduce prediction shift. It is included as a supplementary comparison.", styles["Body"]))
story.append(code("""from catboost import CatBoostClassifier
cat = CatBoostClassifier(iterations=300, depth=4, learning_rate=0.05,
                         auto_class_weights="Balanced", verbose=0, random_state=42)
cat.fit(X_train, y_train)"""))
story.append(Paragraph(
    f"<b>Result:</b> Accuracy={results_df.loc['CatBoost','accuracy']:.4f}, "
    f"Precision={results_df.loc['CatBoost','precision']:.4f}, "
    f"Recall={results_df.loc['CatBoost','recall']:.4f}, "
    f"F1={results_df.loc['CatBoost','f1']:.4f}, ROC-AUC={results_df.loc['CatBoost','roc_auc']:.4f}",
    styles["Body"]))

# 14. Tuning
story.append(PageBreak())
story.append(Paragraph("14. Hyperparameter Tuning", styles["H1"]))
story.append(Paragraph(
    "GridSearchCV performs 3-fold stratified CV on XGBoost, optimizing ROC-AUC, over a search space of 24 combinations.", styles["Body"]))
story.append(code("""param_grid = {"n_estimators": [100,300], "max_depth": [3,4,5],
              "learning_rate": [0.05,0.1], "subsample": [0.8,1.0]}
gs = GridSearchCV(XGBClassifier(scale_pos_weight=spw, eval_metric="logloss",
                                random_state=42, verbosity=0),
                  param_grid=param_grid, scoring="roc_auc", cv=3, n_jobs=-1)
gs.fit(X_train, y_train)"""))
tuned = summary["tuned_xgb_metrics"]
story.append(Paragraph("<b>Best parameters:</b> n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.8", styles["Body"]))
story.append(Paragraph(f"<b>Best CV ROC-AUC:</b> 0.9465", styles["Body"]))
story.append(Paragraph(
    f"<b>Tuned XGBoost test:</b> Accuracy={tuned['accuracy']:.4f}, Precision={tuned['precision']:.4f}, "
    f"Recall={tuned['recall']:.4f}, F1={tuned['f1']:.4f}, ROC-AUC={tuned['roc_auc']:.4f}", styles["Body"]))
story.append(Paragraph(
    "Tuning improved ROC-AUC slightly and substantially increased recall (0.76→0.88).", styles["Body"]))

# 15. Metrics
story.append(Paragraph("15. Model Evaluation Metrics", styles["H1"]))
for m in [
    "<b>Accuracy:</b> Overall correctness (misleading for imbalanced data).",
    "<b>Precision:</b> TP/(TP+FP) — fraction of predicted defaults that are real.",
    "<b>Recall:</b> TP/(TP+FN) — fraction of real defaulters caught (most critical here).",
    "<b>F1:</b> Harmonic mean of precision and recall.",
    "<b>ROC-AUC:</b> Threshold-independent ranking ability; robust to imbalance.",
]:
    story.append(bullet(m))

# 16. Results
story.append(PageBreak())
story.append(Paragraph("16. Results and Model Comparison", styles["H1"]))
story.append(Paragraph("Table 1: Test-set performance of all classifiers (sorted by ROC-AUC):", styles["Body"]))
story.append(make_table(results_df))
story.append(Spacer(1, 0.3*cm))
story.append(Paragraph(
    f"<b>Best model (ROC-AUC):</b> {best_name} with AUC={best_m['roc_auc']:.4f} and recall={best_m['recall']:.4f}.", styles["Body"]))

# 17. Visualizations
story.append(Paragraph("17. Visualizations and Interpretation", styles["H1"]))
story.append(Paragraph("17.1 Bar Chart Comparison", styles["H2"]))
story.extend(fig(os.path.join(FIG_DIR, "fig09_model_comparison_bar.png"), 14*cm,
                 "Figure 8: Side-by-side comparison of metrics across all models."))
story.append(Paragraph("17.2 ROC Curves", styles["H2"]))
story.extend(fig(os.path.join(FIG_DIR, "fig10_roc_curves.png"), 12*cm,
                 "Figure 9: ROC curves for all nine models."))
story.append(Paragraph("17.3 Best Model Confusion Matrix", styles["H2"]))
story.extend(fig(os.path.join(FIG_DIR, "fig11_best_confusion_matrix.png"), 10*cm,
                 f"Figure 10: Confusion matrix for {best_name}."))
story.append(Paragraph("17.4 Feature Importance", styles["H2"]))
story.extend(fig(os.path.join(FIG_DIR, "fig12_feature_importance_rf.png"), 12*cm, "Figure 11: Random Forest feature importance."))
story.extend(fig(os.path.join(FIG_DIR, "fig13_feature_importance_xgb.png"), 12*cm, "Figure 12: XGBoost feature importance."))
story.extend(fig(os.path.join(FIG_DIR, "fig16_lr_coefficients.png"), 12*cm,
                 "Figure 13: Logistic Regression coefficients (scaled)."))
story.append(Paragraph(
    "All tree models agree: Bank Balance dominates default prediction (≈75–89% importance), followed by Annual "
    "Salary. Employment Status contributes minimally. Logistic Regression coefficients confirm this: Bank Balance "
    "has the largest positive effect on the log-odds of default.", styles["Body"]))

# 18. Discussion
story.append(PageBreak())
story.append(Paragraph("18. Discussion of Findings", styles["H1"]))
story.append(Paragraph(
    "Accuracy is misleading on this imbalanced dataset. Gradient Boosting achieves 96.85% accuracy but misses "
    "over 76% of defaulters (recall 0.24). KNN behaves similarly. Random Forest has the highest F1 (0.49) but "
    "still misses 45% of defaulters.", styles["Body"]))
story.append(Paragraph(
    f"Logistic Regression achieves the highest ROC-AUC (0.949) and strong recall (0.88), correctly identifying "
    "the large majority of customers who default. Its precision is lower (0.17), meaning more false positives, "
    "but for an initial screening system this is acceptable — flagged applications can be manually reviewed.", styles["Body"]))
story.append(Paragraph(
    "XGBoost and CatBoost offer a more balanced precision/recall trade-off. After tuning, XGBoost's recall rises "
    "to 0.88, matching Logistic Regression.", styles["Body"]))

# 19. Challenges
story.append(Paragraph("19. Challenges and Solutions", styles["H1"]))
for ch, sol in [
    ("Severe class imbalance", "class_weight='balanced', scale_pos_weight, balanced sample weights, stratified splits."),
    ("Scale sensitivity of LR/SVM/KNN", "StandardScaler fit on training data only."),
    ("AdaBoost lacks class_weight", "Used compute_sample_weight('balanced')."),
    ("Choosing the right metric", "Primary metric: ROC-AUC; secondary: recall for missed-default cost."),
    ("Reproducibility", "Fixed random_state=42 across all components."),
]:
    story.append(bullet(f"<b>{ch}:</b> {sol}"))

# 20. Limitations
story.append(Paragraph("20. Limitations", styles["H1"]))
for l in [
    "Only three features; real models include credit history, debt-to-income, loan amount, etc.",
    "The dataset is simulated (10,000 rows); may not fully generalize to real portfolios.",
    "Default 0.5 threshold used; threshold tuning via cost ratios is needed in production.",
    "No out-of-time validation set; performance may be optimistic.",
    "Correlation ≠ causation.",
    "GridSearchCV only on XGBoost due to runtime constraints.",
]:
    story.append(bullet(l))

# 21. Conclusion
story.append(Paragraph("21. Conclusion", styles["H1"]))
for c in [
    "Accuracy is misleading for imbalanced data — models with 97% accuracy can miss most defaulters.",
    f"<b>Logistic Regression is recommended</b> (ROC-AUC=0.95, recall=0.88) for reliable screening.",
    "<b>Bank Balance</b> is the dominant predictor (~75–89% importance); Annual Salary is secondary.",
    "Tree ensembles (RF, XGBoost, CatBoost) offer strong performance and interpretability.",
    "Tuned XGBoost matches Logistic Regression's recall with competitive AUC.",
    "Model selection should be guided by the actual business cost of false negatives vs. false positives.",
]:
    story.append(bullet(c))

# 22. Future
story.append(Paragraph("22. Future Scope", styles["H1"]))
for f in [
    "Add more features (credit score, loan terms, payment history).",
    "Apply SMOTE/ADASYN oversampling or BalancedRandomForest.",
    "Extend tuning with RandomizedSearchCV or Bayesian optimization (Optuna).",
    "Threshold tuning using Precision-Recall curves.",
    "Model calibration (Platt scaling, isotonic regression).",
    "SHAP/LIME interpretability, partial dependence plots.",
    "Deploy as a REST API (FastAPI).",
    "Add LightGBM per syllabus.",
]:
    story.append(bullet(f))

# 23. References
story.append(Paragraph("23. References", styles["H1"]))
for r in [
    "DSML-Training repository: https://github.com/aamir-gk2539-byte/DSML-Training",
    "Scikit-learn Documentation: https://scikit-learn.org/",
    "XGBoost Documentation: https://xgboost.readthedocs.io/",
    "CatBoost Documentation: https://catboost.ai/docs/",
    "Chen & Guestrin (2016). XGBoost: A Scalable Tree Boosting System. KDD '16.",
    "Friedman (2001). Greedy Function Approximation: A Gradient Boosting Machine.",
    "Breiman (2001). Random Forests. Machine Learning, 45(1), 5–32.",
    "Hastie, Tibshirani & Friedman (2009). The Elements of Statistical Learning.",
]:
    story.append(bullet(r))

# Appendix A
story.append(PageBreak())
story.append(Paragraph("Appendix A: Complete Python Code", styles["H1"]))
story.append(Paragraph("See the file Advik_Singh_Module_3_ML_Code.py for the complete executable source.", styles["Body"]))
with open("Advik_Singh_Module_3_ML_Code.py") as f:
    full_code = f.read()
# Break code into chunks
chunk_size = 60
code_lines = full_code.split("\n")
for i in range(0, len(code_lines), chunk_size):
    chunk = "\n".join(code_lines[i:i+chunk_size])
    story.append(code(chunk))

# Appendix B
story.append(PageBreak())
story.append(Paragraph("Appendix B: Installation and Execution Instructions", styles["H1"]))
story.append(Paragraph("B.1 Local Installation", styles["H2"]))
story.append(code("""python -m venv .venv
source .venv/bin/activate
pip install pandas numpy matplotlib seaborn scikit-learn xgboost catboost jupyter
jupyter notebook Advik_Singh_Module_3_ML_Practicals.ipynb
# or
python Advik_Singh_Module_3_ML_Code.py"""))
story.append(Paragraph("B.2 Google Colab", styles["H2"]))
story.append(code("""!pip install xgboost catboost -q
# Upload the CSV dataset and the notebook/script, then run cells."""))

# Appendix C
story.append(PageBreak())
story.append(Paragraph("Appendix C: Viva Questions", styles["H1"]))
for q, a in [
    ("Q: Why is accuracy misleading for imbalanced data?",
     "A: Predicting only the majority class gives high accuracy (96.67%) while being useless. Use precision/recall/F1/ROC-AUC."),
    ("Q: Bagging vs Boosting?",
     "A: Bagging = parallel independent trees, reduces variance; Boosting = sequential error-correcting trees, reduces bias."),
    ("Q: Why scale features for LR/SVM but not RF?",
     "A: LR/SVM use gradients or distances (scale-sensitive); trees split on thresholds (scale-invariant)."),
    ("Q: What is ROC-AUC?",
     "A: Area under TPR vs FPR curve across thresholds; measures ranking ability, robust to imbalance."),
    ("Q: Why fit scaler on training data only?",
     "A: To avoid test-set data leakage; preprocessing parameters must come from training data alone."),
    ("Q: What is GridSearchCV?",
     "A: Exhaustive search over hyperparameter combinations using k-fold CV on the training set."),
    ("Q: How does XGBoost regularize?",
     "A: L1 (α) and L2 (λ) penalties on leaf weights, column/row subsampling, shrinkage (learning rate), tree depth limits."),
    ("Q: What is scale_pos_weight?",
     "A: Multiplier for minority-class gradients in XGBoost; typically set to neg_count/pos_count for imbalance."),
]:
    story.append(Paragraph(f"<b>{q}</b>", styles["Body"]))
    story.append(Paragraph(a, styles["Body"]))
    story.append(Spacer(1, 0.2*cm))

# ---------- Build ----------
doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
print("PDF report saved successfully.")
