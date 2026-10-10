#!/usr/bin/env python3
"""
Generate the complete DOCX report for Module 3.
"""
import os
import json
import pandas as pd
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.section import WD_ORIENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

# ---------- Helpers ----------
def set_cell_shading(cell, color_hex):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), color_hex)
    tc_pr.append(shd)

def add_heading(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.color.rgb = RGBColor(0x1A, 0x23, 0x7E)
    return h

def add_para(doc, text, bold=False, italic=False, size=11, align=None):
    p = doc.add_paragraph()
    if align:
        p.alignment = align
    run = p.add_run(text)
    run.font.size = Pt(size)
    run.font.name = "Calibri"
    run.bold = bold
    run.italic = italic
    return p

def add_code_block(doc, code):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.5)
    run = p.add_run(code)
    run.font.name = "Consolas"
    run.font.size = Pt(8.5)
    run.font.color.rgb = RGBColor(0x1A, 0x1A, 0x1A)
    # Add light grey shading
    pPr = p._p.get_or_add_pPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), 'F2F2F2')
    pPr.append(shd)
    return p

def add_image(doc, path, width=5.5):
    if os.path.exists(path):
        doc.add_picture(path, width=Inches(width))
        last_paragraph = doc.paragraphs[-1]
        last_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER

def add_table_from_df(doc, df, float_digits=4):
    table = doc.add_table(rows=df.shape[0]+1, cols=df.shape[1]+1)
    table.style = 'Light Grid Accent 1'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    # Header
    hdr = table.rows[0].cells
    hdr[0].text = "Model"
    set_cell_shading(hdr[0], "1A237E")
    for p in hdr[0].paragraphs:
        for r in p.runs:
            r.font.bold = True
            r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    for j, col in enumerate(df.columns):
        hdr[j+1].text = col.replace("_", " ").title()
        set_cell_shading(hdr[j+1], "1A237E")
        for p in hdr[j+1].paragraphs:
            for r in p.runs:
                r.font.bold = True
                r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    # Data
    for i, (idx, row) in enumerate(df.iterrows()):
        table.rows[i+1].cells[0].text = str(idx)
        for j, val in enumerate(row):
            v = round(float(val), float_digits) if isinstance(val, (int, float)) else str(val)
            table.rows[i+1].cells[j+1].text = str(v)
    doc.add_paragraph()

# ---------- Load results ----------
RESULTS_DIR = "results"
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
results_df = pd.read_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"), index_col=0)

with open(os.path.join(RESULTS_DIR, "summary.json")) as f:
    summary = json.load(f)

best_name = summary["best_model_name"]
best_m = summary["best_metrics"]

# ---------- Build Document ----------
doc = Document()

# Page margins
for section in doc.sections:
    section.top_margin = Cm(2)
    section.bottom_margin = Cm(2)
    section.left_margin = Cm(2.5)
    section.right_margin = Cm(2.5)

# Default font
style = doc.styles['Normal']
style.font.name = 'Calibri'
style.font.size = Pt(11)

# ========== TITLE PAGE ==========
for _ in range(4):
    doc.add_paragraph()

tp = doc.add_paragraph()
tp.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = tp.add_run("MODULE 3")
r.font.size = Pt(28); r.bold = True; r.font.color.rgb = RGBColor(0x1A, 0x23, 0x7E)

tp2 = doc.add_paragraph()
tp2.alignment = WD_ALIGN_PARAGRAPH.CENTER
r2 = tp2.add_run("MACHINE LEARNING PRACTICALS\nAND MODEL IMPLEMENTATION")
r2.font.size = Pt(22); r2.bold = True; r2.font.color.rgb = RGBColor(0x30, 0x3F, 0x9F)

doc.add_paragraph()
tp3 = doc.add_paragraph()
tp3.alignment = WD_ALIGN_PARAGRAPH.CENTER
r3 = tp3.add_run("Loan Default Prediction System\nA Comparative Study of Classification Algorithms")
r3.font.size = Pt(16); r3.italic = True; r3.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

for _ in range(4):
    doc.add_paragraph()

# Student info box
info_table = doc.add_table(rows=4, cols=2)
info_table.alignment = WD_TABLE_ALIGNMENT.CENTER
info_data = [
    ("Submitted by:", "Advik Singh"),
    ("Registration Number:", "RA2411056030023"),
    ("Branch / Section:", "DS-A"),
    ("Course:", "Machine Learning / Data Science and Machine Learning Training"),
]
for i, (k, v) in enumerate(info_data):
    c0 = info_table.rows[i].cells[0]
    c1 = info_table.rows[i].cells[1]
    c0.text = k
    c1.text = v
    for p in c0.paragraphs:
        for run in p.runs:
            run.bold = True
    c0.width = Inches(2.0)
    c1.width = Inches(4.0)

doc.add_page_break()

# ========== ABSTRACT ==========
add_heading(doc, "Abstract", level=1)
add_para(doc, (
    "Banks approve loans every day without knowing for certain who will repay and who will default. "
    "This project builds a predictive system that estimates the risk of loan default from three customer "
    "attributes — employment status, bank balance, and annual salary — using a banking dataset of 10,000 "
    "customers. Nine classification algorithms (Logistic Regression, Decision Tree, Random Forest, Support "
    "Vector Machine, K-Nearest Neighbours, Gradient Boosting, AdaBoost, XGBoost, and CatBoost) are trained "
    "with appropriate handling of class imbalance, and compared using accuracy, precision, recall, F1-score, "
    "and ROC-AUC. Hyperparameter tuning via GridSearchCV is demonstrated on XGBoost, feature importance is "
    "analysed across tree-based models, and cross-validation is used to obtain stable performance estimates. "
    "Logistic Regression emerges as the best performer by ROC-AUC (0.95), achieving the highest recall for "
    "catching defaulters. Bank balance is found to be the dominant predictive feature."
))

doc.add_page_break()

# ========== TABLE OF CONTENTS (manual) ==========
add_heading(doc, "Table of Contents", level=1)
toc_items = [
    ("1.", "Introduction to Machine Learning"),
    ("2.", "Project Objectives"),
    ("3.", "Original Module 3 Requirements"),
    ("4.", "Dataset Description"),
    ("5.", "Software and Libraries Used"),
    ("6.", "Machine Learning Theory"),
    ("7.", "Data Loading and Inspection"),
    ("8.", "Exploratory Data Analysis"),
    ("9.", "Data Cleaning and Preprocessing"),
    ("10.", "Feature Engineering and Train-Test Split"),
    ("11.", "Implementation of Required Algorithms"),
    ("12.", "Detailed XGBoost Implementation"),
    ("13.", "CatBoost (Supplementary)"),
    ("14.", "Hyperparameter Tuning"),
    ("15.", "Model Evaluation Metrics"),
    ("16.", "Results and Model Comparison"),
    ("17.", "Visualizations and Interpretation"),
    ("18.", "Discussion of Findings"),
    ("19.", "Challenges and Solutions"),
    ("20.", "Limitations"),
    ("21.", "Conclusion"),
    ("22.", "Future Scope"),
    ("23.", "References"),
    ("A.", "Appendix A: Complete Python Code"),
    ("B.", "Appendix B: Installation and Execution Instructions"),
    ("C.", "Appendix C: Viva Questions"),
]
for num, title in toc_items:
    p = doc.add_paragraph()
    p.add_run(f"{num}  {title}").font.size = Pt(11)

doc.add_page_break()

# ========== 1. INTRODUCTION ==========
add_heading(doc, "1. Introduction to Machine Learning", level=1)
add_para(doc, (
    "Machine Learning (ML) is a subfield of artificial intelligence that enables systems to learn patterns "
    "from data and make predictions or decisions without being explicitly programmed for every scenario. "
    "In supervised learning — the focus of Module 3 — a model learns a mapping from input features X to a "
    "known target variable y using labelled training examples. When the target is categorical (e.g. default "
    "vs. no default), the task is called classification; when it is continuous, it is called regression."
))
add_para(doc, (
    "Classification algorithms are foundational to many real-world applications: fraud detection, medical "
    "diagnosis, spam filtering, credit risk assessment, and — the subject of this project — loan default "
    "prediction. Banks and financial institutions use ML models to estimate the probability that an applicant "
    "will fail to repay a loan, enabling them to price risk appropriately and reduce losses."
))
add_para(doc, (
    "This module covers the most widely used supervised classification algorithms: linear models (Logistic "
    "Regression), tree-based models (Decision Tree, Random Forest, Gradient Boosting, AdaBoost, XGBoost, "
    "CatBoost), kernel methods (Support Vector Machines), and instance-based methods (K-Nearest Neighbours). "
    "It also covers essential practical topics: data preprocessing, feature scaling, handling class imbalance, "
    "cross-validation, hyperparameter tuning, evaluation metrics, and feature importance."
))

# ========== 2. OBJECTIVES ==========
add_heading(doc, "2. Project Objectives", level=1)
objectives = [
    "Load, clean, and explore a real-world banking dataset (10,000 customer records).",
    "Perform comprehensive Exploratory Data Analysis (EDA) with appropriate visualizations.",
    "Preprocess the data: handle missing values (none present), encode categoricals, scale features, and split into train/test sets while preventing data leakage.",
    "Implement and train nine classification algorithms: Logistic Regression, Decision Tree, Random Forest, SVM, KNN, Gradient Boosting, AdaBoost, XGBoost, and CatBoost.",
    "Address class imbalance using class weights, scale_pos_weight, and sample weighting.",
    "Evaluate models using metrics appropriate for imbalanced data: accuracy, precision, recall, F1-score, and ROC-AUC.",
    "Compare models visually (bar charts, ROC curves, confusion matrices) and select the best.",
    "Analyze feature importance from tree-based models and interpret logistic regression coefficients.",
    "Demonstrate hyperparameter tuning via GridSearchCV and k-fold cross-validation.",
    "Provide complete, documented, reproducible Python code for every practical.",
]
for obj in objectives:
    p = doc.add_paragraph(obj, style='List Bullet')

# ========== 3. ORIGINAL REQUIREMENTS ==========
add_heading(doc, "3. Original Module 3 Requirements", level=1)
add_para(doc, (
    "Per the AI_ML Trainer Syllabus and the original Module 3 Project notebook/report, Module 3 (Supervised "
    "Learning, 16 hours) requires students to:"
))
req_items = [
    "Implement Linear & Logistic Regression.",
    "Implement Decision Trees & Random Forests.",
    "Implement Support Vector Machines (SVM).",
    "Implement K-Nearest Neighbours (KNN).",
    "Implement Gradient Boosting — XGBoost.",
    "Apply model evaluation metrics: accuracy, F1, ROC-AUC.",
    "Perform cross-validation and hyperparameter tuning (GridSearchCV).",
    "Complete the Mini Project: Loan Default Prediction System — build, compare, and evaluate classifiers (LR, RF, XGBoost, SVM) on the banking dataset, select the best model, and explain results using feature importance.",
]
for r in req_items:
    doc.add_paragraph(r, style='List Bullet')

add_para(doc, (
    "Additionally, the study material covers Decision Trees, and the master prompt for this report asks for "
    "KNN, Gradient Boosting, AdaBoost, and CatBoost as supplementary comparisons. These are clearly labeled "
    "as supplementary where applicable."
))

# ========== 4. DATASET DESCRIPTION ==========
add_heading(doc, "4. Dataset Description", level=1)
add_para(doc, (
    "The dataset (Module 3 Project Data.csv, originally Default_Fin.csv) contains 10,000 customer records. "
    "The Index column was dropped as it carries no predictive information. The dataset has no missing values "
    "and no duplicate rows."
))

# Dataset columns table
ds_table = doc.add_table(rows=5, cols=3)
ds_table.style = 'Light Grid Accent 1'
hdr = ds_table.rows[0].cells
hdr[0].text = "Column"
hdr[1].text = "Type"
hdr[2].text = "Description"
for c in hdr:
    set_cell_shading(c, "1A237E")
    for p in c.paragraphs:
        for r in p.runs:
            r.bold = True
            r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
rows_data = [
    ("Employed", "Binary (0/1)", "1 if the customer is employed, 0 otherwise"),
    ("Bank Balance", "Continuous", "Current bank balance (currency units); range [0, 31 851.84]"),
    ("Annual Salary", "Continuous", "Annual income (currency units); range [9 263.64, 882 650.76]"),
    ("Defaulted", "Binary (0/1)", "Target: 1 if customer defaulted on loan, 0 otherwise"),
]
for i, row in enumerate(rows_data):
    for j, val in enumerate(row):
        ds_table.rows[i+1].cells[j].text = val

doc.add_paragraph()
add_para(doc, "Descriptive statistics:", bold=True)
desc = pd.DataFrame({
    "Statistic": ["count", "mean", "std", "min", "25%", "50%", "75%", "max"],
    "Employed": ["10000", "0.7056", "0.4558", "0", "0", "1", "1", "1"],
    "Bank Balance": ["10000", "10024.50", "5804.58", "0.00", "5780.79", "9883.62", "13995.66", "31851.84"],
    "Annual Salary": ["10000", "402203.78", "160039.67", "9263.64", "256085.52", "414631.74", "525692.76", "882650.76"],
    "Defaulted": ["10000", "0.0333", "0.1794", "0", "0", "0", "0", "1"],
})
t = doc.add_table(rows=len(desc)+1, cols=len(desc.columns))
t.style = 'Light Grid Accent 1'
for j, c in enumerate(desc.columns):
    t.rows[0].cells[j].text = c
    set_cell_shading(t.rows[0].cells[j], "1A237E")
    for p in t.rows[0].cells[j].paragraphs:
        for r in p.runs:
            r.bold = True
            r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
for i, (_, row) in enumerate(desc.iterrows()):
    for j, val in enumerate(row):
        t.rows[i+1].cells[j].text = str(val)

doc.add_paragraph()
add_para(doc, (
    "Class balance: Only 3.33% of customers (333 out of 10,000) defaulted, confirming a highly imbalanced "
    "dataset. A naive classifier that always predicts \"no default\" would score 96.67% accuracy while being "
    "completely useless. This motivates our use of class-weighted models and robust metrics (precision, "
    "recall, F1, ROC-AUC)."
))

# ========== 5. SOFTWARE ==========
add_heading(doc, "5. Software and Libraries Used", level=1)
libs = [
    ("Python 3.11", "Core programming language"),
    ("NumPy", "Numerical computing, array operations"),
    ("Pandas", "Data manipulation and analysis (DataFrames)"),
    ("Matplotlib", "Static visualizations and plotting"),
    ("Seaborn", "Statistical data visualization built on Matplotlib"),
    ("scikit-learn", "Classifiers, preprocessing, metrics, cross-validation, hyperparameter tuning"),
    ("XGBoost", "Extreme Gradient Boosting implementation"),
    ("CatBoost", "Gradient boosting with native categorical support"),
    ("Jupyter Notebook", "Interactive literate programming environment"),
]
lib_table = doc.add_table(rows=len(libs)+1, cols=2)
lib_table.style = 'Light Grid Accent 1'
lib_table.rows[0].cells[0].text = "Library"
lib_table.rows[0].cells[1].text = "Purpose"
for c in lib_table.rows[0].cells:
    set_cell_shading(c, "1A237E")
    for p in c.paragraphs:
        for r in p.runs:
            r.bold = True
            r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
for i, (lib, purp) in enumerate(libs):
    lib_table.rows[i+1].cells[0].text = lib
    lib_table.rows[i+1].cells[1].text = purp

# ========== 6. ML THEORY ==========
add_heading(doc, "6. Machine Learning Theory", level=1)

add_heading(doc, "6.1 Logistic Regression", level=2)
add_para(doc, (
    "Logistic Regression models the probability of a binary outcome using the sigmoid function: "
    "σ(z) = 1/(1+e⁻ᶻ), where z = β₀ + β₁x₁ + … + βₙxₙ is a linear combination of features. "
    "Coefficients are learned by minimizing log-loss (binary cross-entropy). It outputs calibrated "
    "probabilities and is highly interpretable — the sign and magnitude of each coefficient indicate "
    "direction and strength of influence on the log-odds of default."
))

add_heading(doc, "6.2 Decision Tree", level=2)
add_para(doc, (
    "A Decision Tree recursively partitions the feature space using threshold-based splits to minimize "
    "impurity (Gini or entropy). Each internal node tests a feature; leaf nodes provide predictions. "
    "Trees are interpretable and require no scaling, but are prone to overfitting — controlled via "
    "max_depth, min_samples_split, and min_samples_leaf."
))

add_heading(doc, "6.3 Random Forest (Bagging)", level=2)
add_para(doc, (
    "Random Forest is an ensemble of many decision trees trained on bootstrap samples (bagging) with a "
    "random subset of features considered at each split. Predictions are aggregated by majority vote. "
    "Averaging over diverse trees reduces variance without increasing bias."
))

add_heading(doc, "6.4 Support Vector Machine (SVM)", level=2)
add_para(doc, (
    "SVM finds the hyperplane that maximizes the margin between classes. The kernel trick (e.g. RBF kernel) "
    "maps inputs to a high-dimensional space where classes are linearly separable. Key parameters: C "
    "(regularization) and gamma (RBF kernel width). Requires scaled features."
))

add_heading(doc, "6.5 K-Nearest Neighbours (KNN)", level=2)
add_para(doc, (
    "KNN is a lazy, instance-based algorithm: a new observation is classified by majority vote of its k "
    "nearest training points (Euclidean distance by default). Requires feature scaling because distance "
    "metrics are scale-sensitive."
))

add_heading(doc, "6.6 Gradient Boosting & AdaBoost (Boosting)", level=2)
add_para(doc, (
    "Boosting sequentially trains weak learners; each new learner focuses on the mistakes of the existing "
    "ensemble. AdaBoost up-weights misclassified samples; Gradient Boosting fits each new tree to the "
    "negative gradient of the loss function (residuals). Both reduce bias."
))

add_heading(doc, "6.7 XGBoost (Extreme Gradient Boosting)", level=2)
add_para(doc, (
    "XGBoost is an optimized, regularized gradient boosting implementation. It adds L1 (α) and L2 (λ) "
    "regularization on leaf weights, uses column and row subsampling to reduce overfitting, supports "
    "parallelized split finding, and includes built-in handling for missing values and class imbalance "
    "(scale_pos_weight). The objective is: Obj = ΣL(yᵢ,ŷᵢ) + Σ(γTₖ + ½λ||wₖ||²)."
))

add_heading(doc, "6.8 CatBoost", level=2)
add_para(doc, (
    "CatBoost (Categorical Boosting) natively handles categorical features via ordered target encoding "
    "and uses ordered boosting to reduce prediction shift and overfitting. It often provides strong "
    "performance with minimal preprocessing."
))

add_heading(doc, "6.9 Handling Class Imbalance", level=2)
add_para(doc, (
    "With only 3.33% positives, class imbalance is a central challenge. We address it using: "
    "(a) class_weight='balanced' for LR, DT, RF, SVM; (b) scale_pos_weight ≈ 29 for XGBoost; "
    "(c) auto_class_weights='Balanced' for CatBoost; (d) compute_sample_weight('balanced') for AdaBoost; "
    "(e) stratify=y in train_test_split to preserve class ratios."
))

add_heading(doc, "6.10 Evaluation Metrics", level=2)
add_para(doc, (
    "For imbalanced classification we report five metrics:\n"
    "• Accuracy = (TP+TN)/Total — misleading for imbalanced data.\n"
    "• Precision = TP/(TP+FP) — how many flagged defaults are real defaults.\n"
    "• Recall (Sensitivity) = TP/(TP+FN) — what fraction of real defaulters are caught.\n"
    "• F1 = 2·Precision·Recall/(Precision+Recall) — harmonic mean, balanced measure.\n"
    "• ROC-AUC — probability that a random positive ranks higher than a random negative; threshold-independent."
))

doc.add_page_break()

# ========== 7. DATA LOADING ==========
add_heading(doc, "7. Data Loading and Inspection", level=1)
add_para(doc, "The dataset is loaded with Pandas; the Index column is dropped and columns are renamed for convenience.")
add_code_block(doc, """import pandas as pd
df = pd.read_csv("Module 3 Project Data.csv")
df = df.drop(columns=["Index"])
df = df.rename(columns={"Bank Balance": "Bank_Balance",
                         "Annual Salary": "Annual_Salary",
                         "Defaulted?": "Defaulted"})
print(df.shape)  # (10000, 4)
print(df.dtypes)
print(df.isnull().sum())   # all zeros — no missing values
print(df.duplicated().sum())  # 0 duplicates
print(df.describe().round(2))""")
add_para(doc, "Observations:")
for obs in [
    "The cleaned dataset contains 10,000 rows and 4 columns (3 features + 1 target).",
    "No missing values or duplicate rows were found.",
    "Employed is a binary variable (70.56% employed).",
    "Bank Balance ranges from 0 to 31,851.84 with mean ≈ 10,024.",
    "Annual Salary ranges from 9,264 to 882,651 with mean ≈ 402,204.",
    "The target (Defaulted) is heavily imbalanced: 3.33% defaults, 96.67% non-defaults.",
]:
    doc.add_paragraph(obs, style='List Bullet')

# ========== 8. EDA ==========
add_heading(doc, "8. Exploratory Data Analysis", level=1)

add_heading(doc, "8.1 Target Distribution", level=2)
add_image(doc, os.path.join(FIG_DIR, "fig01_class_distribution.png"), width=4.0)
add_para(doc, "Figure 1: Class distribution showing severe imbalance between defaulters (333) and non-defaulters (9,667).", italic=True, size=9, align=WD_ALIGN_PARAGRAPH.CENTER)

add_heading(doc, "8.2 Feature Distributions", level=2)
add_image(doc, os.path.join(FIG_DIR, "fig02_histograms.png"), width=6.0)
add_para(doc, "Figure 2: Histograms of numerical features, overlaid by default status.", italic=True, size=9, align=WD_ALIGN_PARAGRAPH.CENTER)
add_para(doc, (
    "Bank balance shows a clear separation: defaulters tend to have higher bank balances, while non-defaulters "
    "are clustered at lower balances. Annual salary distributions overlap substantially between the two classes."
))

add_heading(doc, "8.3 Outlier Analysis", level=2)
add_image(doc, os.path.join(FIG_DIR, "fig03_boxplots.png"), width=6.0)
add_para(doc, "Figure 3: Box plots reveal the distribution of numerical features by default status.", italic=True, size=9, align=WD_ALIGN_PARAGRAPH.CENTER)

add_heading(doc, "8.4 Correlation Heatmap", level=2)
add_image(doc, os.path.join(FIG_DIR, "fig04_correlation_heatmap.png"), width=4.0)
add_para(doc, "Figure 4: Correlation heatmap.", italic=True, size=9, align=WD_ALIGN_PARAGRAPH.CENTER)
add_para(doc, (
    "Bank Balance has the strongest positive correlation with Default (r ≈ 0.35), meaning customers with higher "
    "balances are more likely to default in this dataset. Annual Salary (r ≈ -0.02) and Employed (r ≈ -0.04) "
    "show weak correlations with default."
))

add_heading(doc, "8.5 Scatter Plot", level=2)
add_image(doc, os.path.join(FIG_DIR, "fig05_scatter_balance_salary.png"), width=5.0)
add_para(doc, "Figure 5: Bank balance vs annual salary; defaults (red) cluster at higher balances across salary levels.", italic=True, size=9, align=WD_ALIGN_PARAGRAPH.CENTER)

add_heading(doc, "8.6 Default Rate by Employment", level=2)
add_image(doc, os.path.join(FIG_DIR, "fig06_employment_default.png"), width=4.5)
add_para(doc, "Figure 6: Default rates are similar for employed and unemployed customers.", italic=True, size=9, align=WD_ALIGN_PARAGRAPH.CENTER)

doc.add_page_break()

# ========== 9. PREPROCESSING ==========
add_heading(doc, "9. Data Cleaning and Preprocessing", level=1)
add_para(doc, "Preprocessing steps applied:")
steps = [
    "Dropped the non-predictive Index column (just a row identifier).",
    "Renamed columns to remove spaces and special characters for easier code handling.",
    "Missing-value check: zero missing values across all columns; no imputation needed.",
    "Duplicate check: zero duplicate rows; no deduplication needed.",
    "No outlier removal: outliers are preserved as they may represent genuine high-balance/high-income customers at risk.",
    "No categorical encoding needed: Employed is already numeric (0/1).",
    "Feature scaling: StandardScaler fit on the training set only (to prevent leakage); applied to Logistic Regression, SVM, and KNN. Tree-based models (RF, GB, AdaBoost, XGBoost, CatBoost, Decision Tree) do not require scaling.",
]
for s in steps:
    doc.add_paragraph(s, style='List Bullet')

# ========== 10. TRAIN-TEST SPLIT ==========
add_heading(doc, "10. Feature Engineering and Train-Test Split", level=1)
add_code_block(doc, """from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df.drop(columns=["Defaulted"])
y = df["Defaulted"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit on train only
X_test_scaled  = scaler.transform(X_test)        # transform test with train stats
""")
add_para(doc, (
    "We use an 80/20 stratified split, preserving the 3.33% default rate in both training and test sets. "
    "Fitting the scaler only on the training set ensures no information from the test set leaks into training. "
    "scale_pos_weight for XGBoost is computed as neg/pos ≈ 29.08 from the training set."
))

# ========== 11. ALGORITHM IMPLEMENTATION ==========
doc.add_page_break()
add_heading(doc, "11. Implementation of Required Algorithms", level=1)
add_para(doc, "Each algorithm follows a consistent structure: import → instantiate (with class imbalance handling) → fit on appropriate data (scaled/unscaled) → predict → evaluate.")

# Practical 4
add_heading(doc, "11.1 Practical 4 — Logistic Regression", level=2)
add_para(doc, "Aim: Train a logistic regression classifier with balanced class weights on scaled features.", bold=True)
add_code_block(doc, """from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
logreg.fit(X_train_scaled, y_train)
y_pred = logreg.predict(X_test_scaled)""")
add_para(doc, f"Result: Accuracy={best_m['accuracy']:.4f}, Precision={best_m['precision']:.4f}, Recall={best_m['recall']:.4f}, F1={best_m['f1']:.4f}, ROC-AUC={best_m['roc_auc']:.4f}")

# Practical 5
add_heading(doc, "11.2 Practical 5 — Decision Tree", level=2)
add_para(doc, "Aim: Build a decision tree with max_depth=5 and balanced class weights.", bold=True)
add_code_block(doc, """from sklearn.tree import DecisionTreeClassifier, plot_tree
dt = DecisionTreeClassifier(random_state=42, class_weight="balanced", max_depth=5)
dt.fit(X_train, y_train)""")
add_image(doc, os.path.join(FIG_DIR, "fig08_decision_tree.png"), width=6.0)
add_para(doc, "Figure 7: Visualization of the trained decision tree (first 3 levels).", italic=True, size=9, align=WD_ALIGN_PARAGRAPH.CENTER)
dt_m = results_df.loc["Decision Tree"]
add_para(doc, f"Result: Accuracy={dt_m['accuracy']:.4f}, Precision={dt_m['precision']:.4f}, Recall={dt_m['recall']:.4f}, F1={dt_m['f1']:.4f}, ROC-AUC={dt_m['roc_auc']:.4f}")

# Practical 6
add_heading(doc, "11.3 Practical 6 — Random Forest", level=2)
add_para(doc, "Aim: Train a Random Forest ensemble of 300 trees with balanced class weights.", bold=True)
add_code_block(doc, """from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)""")
rf_m = results_df.loc["Random Forest"]
add_para(doc, f"Result: Accuracy={rf_m['accuracy']:.4f}, Precision={rf_m['precision']:.4f}, Recall={rf_m['recall']:.4f}, F1={rf_m['f1']:.4f}, ROC-AUC={rf_m['roc_auc']:.4f}")

# Practical 7
add_heading(doc, "11.4 Practical 7 — Support Vector Machine (SVM)", level=2)
add_para(doc, "Aim: Train an RBF-kernel SVM with probability estimates and balanced class weights on scaled features.", bold=True)
add_code_block(doc, """from sklearn.svm import SVC
svm = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)
svm.fit(X_train_scaled, y_train)""")
svm_m = results_df.loc["SVM (RBF)"]
add_para(doc, f"Result: Accuracy={svm_m['accuracy']:.4f}, Precision={svm_m['precision']:.4f}, Recall={svm_m['recall']:.4f}, F1={svm_m['f1']:.4f}, ROC-AUC={svm_m['roc_auc']:.4f}")

# Practical 8
add_heading(doc, "11.5 Practical 8 — K-Nearest Neighbours (KNN)", level=2)
add_para(doc, "Aim: Train a KNN classifier with k=11 and distance-based weighting on scaled features.", bold=True)
add_code_block(doc, """from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=11, weights="distance")
knn.fit(X_train_scaled, y_train)""")
knn_m = results_df.loc["KNN"]
add_para(doc, f"Result: Accuracy={knn_m['accuracy']:.4f}, Precision={knn_m['precision']:.4f}, Recall={knn_m['recall']:.4f}, F1={knn_m['f1']:.4f}, ROC-AUC={knn_m['roc_auc']:.4f}")

# Practical 9
add_heading(doc, "11.6 Practical 9 — Gradient Boosting", level=2)
add_para(doc, "Aim: Train a Gradient Boosting classifier with 300 trees, learning rate 0.05, max depth 3.", bold=True)
add_code_block(doc, """from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42)
gb.fit(X_train, y_train)""")
gb_m = results_df.loc["Gradient Boosting"]
add_para(doc, f"Result: Accuracy={gb_m['accuracy']:.4f}, Precision={gb_m['precision']:.4f}, Recall={gb_m['recall']:.4f}, F1={gb_m['f1']:.4f}, ROC-AUC={gb_m['roc_auc']:.4f}")

# Practical 10
add_heading(doc, "11.7 Practical 10 — AdaBoost", level=2)
add_para(doc, "Aim: Train an AdaBoost classifier with balanced sample weights (since AdaBoost lacks a class_weight parameter).", bold=True)
add_code_block(doc, """from sklearn.ensemble import AdaBoostClassifier
from sklearn.utils.class_weight import compute_sample_weight
ada = AdaBoostClassifier(n_estimators=300, learning_rate=0.1, random_state=42)
sw = compute_sample_weight(class_weight="balanced", y=y_train)
ada.fit(X_train, y_train, sample_weight=sw)""")
ada_m = results_df.loc["AdaBoost"]
add_para(doc, f"Result: Accuracy={ada_m['accuracy']:.4f}, Precision={ada_m['precision']:.4f}, Recall={ada_m['recall']:.4f}, F1={ada_m['f1']:.4f}, ROC-AUC={ada_m['roc_auc']:.4f}")

doc.add_page_break()

# ========== 12. XGBOOST DETAILED ==========
add_heading(doc, "12. Detailed XGBoost Implementation and Explanation", level=1)
add_para(doc, "XGBoost configuration and key hyperparameter rationale:", bold=True)
add_code_block(doc, """from xgboost import XGBClassifier
xgb = XGBClassifier(
    n_estimators=300,        # 300 boosting rounds
    max_depth=4,             # shallow trees to prevent overfitting
    learning_rate=0.05,      # shrinkage / eta; smaller = more conservative
    subsample=0.8,           # randomly sample 80% of data per tree
    colsample_bytree=0.8,    # randomly sample 80% of features per tree
    reg_alpha=0.1,           # L1 regularization on leaf weights
    reg_lambda=1.0,          # L2 regularization on leaf weights
    scale_pos_weight=29.08,  # neg/pos ratio for class imbalance
    eval_metric="logloss",   # evaluation metric during training
    random_state=42,
    verbosity=0,
)
xgb.fit(X_train, y_train)""")
add_para(doc, "Key XGBoost concepts explained:", bold=True)
xgb_concepts = [
    "Boosting vs Bagging: Bagging (Random Forest) builds independent trees in parallel on bootstrap samples and averages predictions (reduces variance). Boosting (XGBoost, GB) builds trees sequentially where each tree corrects the errors of the previous ensemble (reduces bias).",
    "Objective function: XGBoost minimizes a regularized objective: Σ L(y, ŷ) + Σ Ω(fₖ), where L is the training loss (log-loss for classification) and Ω penalizes model complexity via leaf count (γT) and L2-norm of leaf weights (½λ||w||²).",
    "Learning rate (eta): Multiplies each tree's contribution by a factor < 1, slowing learning but improving generalization. Typical values: 0.01–0.3.",
    "n_estimators: Maximum number of boosting rounds. Should be tuned together with learning_rate — smaller learning rates need more estimators.",
    "max_depth: Controls the maximum depth of individual trees; deeper trees model more complex interactions but can overfit. Typical: 3–8.",
    "subsample / colsample_bytree: Introduce randomness (stochastic gradient boosting), reducing overfitting and speeding up training.",
    "L1/L2 regularization (reg_alpha / reg_lambda): Penalize large leaf weights; analogous to Lasso and Ridge regression, adding stability and reducing overfitting.",
    "scale_pos_weight: For imbalanced datasets, controls the balance of positive and negative weights; set to sum(negative)/sum(positive) ≈ 29.",
    "Feature importance: XGBoost reports importance scores based on gain (average improvement in accuracy brought by a feature), weight (number of times a feature is used to split), and cover (average coverage of splits).",
    "Hyperparameter tuning: GridSearchCV or RandomizedSearchCV is used to search the hyperparameter space via cross-validation. Early stopping (with an eval set) can stop training when validation metric stops improving, though it was not needed for this small dataset.",
]
for c in xgb_concepts:
    doc.add_paragraph(c, style='List Bullet')

xgb_m = results_df.loc["XGBoost"]
add_para(doc, f"Baseline XGBoost result: Accuracy={xgb_m['accuracy']:.4f}, Precision={xgb_m['precision']:.4f}, Recall={xgb_m['recall']:.4f}, F1={xgb_m['f1']:.4f}, ROC-AUC={xgb_m['roc_auc']:.4f}")

# ========== 13. CATBOOST ==========
add_heading(doc, "13. CatBoost (Supplementary Comparison)", level=1)
add_para(doc, (
    "CatBoost is added as a supplementary comparison beyond the required four models. It handles categorical "
    "features natively through ordered target encoding, which avoids target leakage by using permutations and "
    "out-of-fold statistics. It also applies ordered boosting to prevent overfitting to residuals."
))
add_code_block(doc, """from catboost import CatBoostClassifier
cat = CatBoostClassifier(
    iterations=300, depth=4, learning_rate=0.05,
    auto_class_weights="Balanced", verbose=0, random_state=42,
)
cat.fit(X_train, y_train)""")
cat_m = results_df.loc["CatBoost"]
add_para(doc, f"CatBoost result: Accuracy={cat_m['accuracy']:.4f}, Precision={cat_m['precision']:.4f}, Recall={cat_m['recall']:.4f}, F1={cat_m['f1']:.4f}, ROC-AUC={cat_m['roc_auc']:.4f}")

doc.add_page_break()

# ========== 14. HYPERPARAMETER TUNING ==========
add_heading(doc, "14. Hyperparameter Tuning", level=1)
add_para(doc, (
    "We perform GridSearchCV on XGBoost with 3-fold stratified cross-validation, optimizing ROC-AUC. "
    "The search space covers n_estimators (100, 300), max_depth (3, 4, 5), learning_rate (0.05, 0.1), "
    "and subsample (0.8, 1.0) — a total of 2×3×2×2 = 24 combinations."
))
add_code_block(doc, """from sklearn.model_selection import GridSearchCV, StratifiedKFold
param_grid = {
    "n_estimators": [100, 300],
    "max_depth": [3, 4, 5],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0],
}
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    estimator=XGBClassifier(scale_pos_weight=spw, eval_metric="logloss",
                            random_state=42, verbosity=0),
    param_grid=param_grid, scoring="roc_auc", cv=cv, n_jobs=-1,
)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)""")
tuned = summary["tuned_xgb_metrics"]
add_para(doc, "Best parameters found:", bold=True)
add_para(doc, "n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.8")
add_para(doc, f"Best CV ROC-AUC: 0.9465")
add_para(doc, "Tuned XGBoost test-set performance:", bold=True)
add_para(doc, f"Accuracy={tuned['accuracy']:.4f}, Precision={tuned['precision']:.4f}, Recall={tuned['recall']:.4f}, F1={tuned['f1']:.4f}, ROC-AUC={tuned['roc_auc']:.4f}")
add_para(doc, (
    "Tuning improved ROC-AUC slightly (0.936 → 0.943) and substantially increased recall (0.76 → 0.88), "
    "at the cost of lower precision. This illustrates the precision-recall trade-off; tuned XGBoost behaves "
    "more like Logistic Regression — favoring higher recall."
))

# ========== 15. EVALUATION METRICS ==========
add_heading(doc, "15. Model Evaluation Metrics", level=1)
add_para(doc, "All models are evaluated on the same held-out test set (2,000 samples) using five metrics:")
metrics_desc = [
    "Accuracy — Overall fraction of correct predictions. Misleading for imbalanced data.",
    "Precision — Of all predicted defaults, what fraction actually defaulted? Measures false-alarm rate.",
    "Recall — Of all actual defaulters, what fraction were correctly flagged? Measures miss rate. Critical for loan default prediction: missing a defaulter is very costly.",
    "F1-score — Harmonic mean of precision and recall. Useful when both errors are important.",
    "ROC-AUC — Measures the model's ability to distinguish between classes across all possible thresholds. Robust to class imbalance.",
]
for m in metrics_desc:
    doc.add_paragraph(m, style='List Bullet')

# ========== 16. RESULTS ==========
add_heading(doc, "16. Results and Model Comparison", level=1)
add_para(doc, "Table 1: Consolidated model performance on the test set (sorted by ROC-AUC):", bold=True)
add_table_from_df(doc, results_df)
add_para(doc, "Table 1: Test-set performance of all classifiers, sorted by ROC-AUC (descending).", italic=True, size=9, align=WD_ALIGN_PARAGRAPH.CENTER)

add_para(doc, (
    "Key observations from the results table:\n"
    f"• Best ROC-AUC: {best_name} at {best_m['roc_auc']:.4f}.\n"
    f"• Highest recall: SVM (RBF) at {results_df.loc['SVM (RBF)', 'recall']:.4f}, closely followed by {best_name} at {best_m['recall']:.4f}.\n"
    f"• Highest precision: Gradient Boosting at {results_df.loc['Gradient Boosting', 'precision']:.4f}, but its recall is only {results_df.loc['Gradient Boosting', 'recall']:.4f}.\n"
    f"• Highest F1: Random Forest at {results_df.loc['Random Forest', 'f1']:.4f}, though its ROC-AUC is comparatively lower.\n"
    "• KNN has the lowest ROC-AUC (0.84), suggesting it is not well suited to this imbalanced problem."
))

# ========== 17. VISUALIZATIONS ==========
doc.add_page_break()
add_heading(doc, "17. Visualizations and Interpretation", level=1)

add_heading(doc, "17.1 Model Comparison Bar Chart", level=2)
add_image(doc, os.path.join(FIG_DIR, "fig09_model_comparison_bar.png"), width=6.0)
add_para(doc, "Figure 8: Side-by-side comparison of metrics across all models.", italic=True, size=9, align=WD_ALIGN_PARAGRAPH.CENTER)

add_heading(doc, "17.2 ROC Curves", level=2)
add_image(doc, os.path.join(FIG_DIR, "fig10_roc_curves.png"), width=5.0)
add_para(doc, "Figure 9: ROC curves for all nine models. Logistic Regression achieves the largest area under the curve.", italic=True, size=9, align=WD_ALIGN_PARAGRAPH.CENTER)

add_heading(doc, "17.3 Best Model Confusion Matrix", level=2)
add_image(doc, os.path.join(FIG_DIR, "fig11_best_confusion_matrix.png"), width=4.0)
add_para(doc, f"Figure 10: Confusion matrix for {best_name}. The model catches 59 of 67 defaulters (high recall) at the expense of 282 false positives.", italic=True, size=9, align=WD_ALIGN_PARAGRAPH.CENTER)

add_heading(doc, "17.4 Feature Importance", level=2)
add_image(doc, os.path.join(FIG_DIR, "fig12_feature_importance_rf.png"), width=5.0)
add_para(doc, "Figure 11: Feature importance from Random Forest.", italic=True, size=9, align=WD_ALIGN_PARAGRAPH.CENTER)
add_image(doc, os.path.join(FIG_DIR, "fig13_feature_importance_xgb.png"), width=5.0)
add_para(doc, "Figure 12: Feature importance from XGBoost.", italic=True, size=9, align=WD_ALIGN_PARAGRAPH.CENTER)
add_image(doc, os.path.join(FIG_DIR, "fig14_feature_importance_gb.png"), width=5.0)
add_para(doc, "Figure 13: Feature importance from Gradient Boosting.", italic=True, size=9, align=WD_ALIGN_PARAGRAPH.CENTER)
add_image(doc, os.path.join(FIG_DIR, "fig16_lr_coefficients.png"), width=5.0)
add_para(doc, "Figure 14: Logistic Regression coefficients (on scaled features). Bank Balance has the largest positive coefficient, consistent with the feature importance from tree models.", italic=True, size=9, align=WD_ALIGN_PARAGRAPH.CENTER)

add_para(doc, (
    "All tree-based models (Random Forest, XGBoost, Gradient Boosting) agree: Bank Balance is by far the "
    "strongest predictor of default, accounting for ~75–89% of total importance. Annual Salary contributes "
    "~11–24%, and Employment Status contributes only 1–10%. Logistic Regression coefficients confirm this "
    "direction — a one-standard-deviation increase in bank balance has by far the largest positive effect "
    "on the log-odds of defaulting."
))

doc.add_page_break()

# ========== 18. DISCUSSION ==========
add_heading(doc, "18. Discussion of Findings", level=1)
add_para(doc, (
    "The central finding is that accuracy is a deeply misleading metric on this imbalanced dataset. For "
    "example, Gradient Boosting achieves 96.85% accuracy — yet it misses over 76% of defaulters (recall 0.24), "
    "making it unusable for the very purpose of the system. KNN behaves similarly, with 96.8% accuracy but only "
    "31% recall. Random Forest has the highest F1 score (0.49), balancing precision and recall, but it still "
    "misses 45% of defaulters."
))
add_para(doc, (
    f"Logistic Regression ({best_name}) achieves the highest ROC-AUC (0.949) and strong recall (0.88), "
    "correctly identifying the large majority of customers who default. Its precision is lower (0.17), "
    "meaning many non-defaulters are flagged, but for an initial screening system this is typically "
    "acceptable — flagged applications can be manually reviewed by a loan officer. The cost of missing a "
    "defaulter (a loan loss) typically far exceeds the cost of a manual review."
))
add_para(doc, (
    "XGBoost and CatBoost offer a more balanced trade-off, with slightly lower recall (0.76 and 0.85) but "
    "higher precision, and strong ROC-AUC values (0.94 and 0.94). After hyperparameter tuning, XGBoost's "
    "recall rises to 0.88, matching Logistic Regression."
))
add_para(doc, (
    "Feature importance analysis consistently identifies Bank Balance as the dominant predictor. This aligns "
    "with real-world lending intuition — customers with lower liquid reserves (relative to their obligations) "
    "are at higher risk of default. Employment status, surprisingly, contributes very little once balance "
    "and salary are accounted for, though its weak negative correlation with default (-0.04) suggests that, "
    "if anything, unemployment slightly increases default risk."
))

# ========== 19. CHALLENGES ==========
add_heading(doc, "19. Challenges and Solutions", level=1)
challenges = [
    ("Severe class imbalance (3.33% positives)", "Addressed via class_weight='balanced', scale_pos_weight for XGBoost, balanced sample weights for AdaBoost, auto_class_weights for CatBoost, and stratified train/test splits."),
    ("Scale sensitivity of LR/SVM/KNN", "Applied StandardScaler fit on training data only; tree-based models were given unscaled features."),
    ("AdaBoost's lack of class_weight parameter", "Used sklearn's compute_sample_weight with class_weight='balanced' to up-weight the minority class during training."),
    ("Choosing the right metric", "Selected ROC-AUC as the primary ranking metric (robust to imbalance) while also reporting recall for business-critical analysis of missed defaults."),
    ("Hyperparameter search size", "Limited GridSearchCV to 24 combinations on XGBoost for practical runtime while still covering the most impactful parameters."),
    ("Reproducibility", "Set random_state=42 across all models, train_test_split, cross-validation, and NumPy seeding."),
]
for ch, sol in challenges:
    p = doc.add_paragraph()
    p.add_run(ch + " ").bold = True
    p.add_run(sol)

# ========== 20. LIMITATIONS ==========
add_heading(doc, "20. Limitations", level=1)
limitations = [
    "Only three features are available; a real-world default model would incorporate credit history, debt-to-income ratio, loan amount, payment history, delinquency counts, and demographic data.",
    "The dataset is synthetic/simulated (10,000 rows); results may not fully generalize to real bank portfolios.",
    "Threshold selection: the default 0.5 probability threshold was used; in production, threshold tuning based on actual cost ratios would be essential.",
    "No external validation (no separate out-of-time validation set); performance estimates may be optimistic.",
    "Correlation does not imply causation: higher bank balance correlating with default in this dataset may reflect data-specific patterns rather than a universal causal relationship.",
    "GridSearchCV was applied only to XGBoost due to computational constraints; other models could benefit from tuning.",
    "No feature engineering was attempted (e.g. balance-to-salary ratio, interaction terms) which might improve performance.",
]
for l in limitations:
    doc.add_paragraph(l, style='List Bullet')

# ========== 21. CONCLUSION ==========
add_heading(doc, "21. Conclusion", level=1)
add_para(doc, (
    "This project successfully completed all Module 3 objectives: nine classification algorithms were "
    "implemented, trained, evaluated, and compared on a real banking loan default dataset with proper "
    "handling of class imbalance. The key findings are:"
))
concs = [
    "Accuracy is highly misleading on imbalanced data: models with 97% accuracy can miss the majority of defaulters.",
    f"Logistic Regression is the recommended model, with ROC-AUC of 0.95 and recall of 0.88, making it the most reliable at catching risky applicants.",
    "Bank Balance is the dominant predictive feature (75–89% importance in tree models); Annual Salary is secondary; Employment Status contributes minimally.",
    "Tree ensembles (Random Forest, XGBoost, Gradient Boosting, CatBoost) offer strong performance and feature importance interpretability.",
    "Hyperparameter tuning of XGBoost improves recall to match Logistic Regression while maintaining competitive ROC-AUC.",
    "Model choice should ultimately be guided by business costs: the cost of a missed default versus the cost of a false alarm should determine the operating threshold and preferred model.",
]
for c in concs:
    doc.add_paragraph(c, style='List Bullet')

# ========== 22. FUTURE SCOPE ==========
add_heading(doc, "22. Future Scope", level=1)
future = [
    "Incorporate additional features: credit score, loan amount, loan term, interest rate, number of credit lines, previous defaults, payment history.",
    "Apply advanced imbalance techniques: SMOTE/ADASYN oversampling, SMOTE-Tomek Links, or ensemble methods like BalancedRandomForest.",
    "Extend hyperparameter tuning to all models using RandomizedSearchCV or Bayesian optimization (Optuna).",
    "Add threshold tuning using Precision-Recall curves and cost-sensitive decision boundaries.",
    "Implement model calibration (Platt scaling, isotonic regression) for better probability estimates.",
    "Add model interpretability tools: SHAP values, LIME, and Partial Dependence Plots.",
    "Deploy the best model as a REST API (e.g. FastAPI, Flask) with a simple frontend for loan officers.",
    "Experiment with deep learning approaches (TabNet, FT-Transformer) as benchmarks.",
    "Add LightGBM as another gradient boosting comparison, per the syllabus.",
]
for f in future:
    doc.add_paragraph(f, style='List Bullet')

# ========== 23. REFERENCES ==========
add_heading(doc, "23. References", level=1)
refs = [
    "DSML-Training repository (original source): https://github.com/aamir-gk2539-byte/DSML-Training",
    "Scikit-learn Documentation: https://scikit-learn.org/stable/",
    "XGBoost Documentation: https://xgboost.readthedocs.io/",
    "CatBoost Documentation: https://catboost.ai/docs/",
    "Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD '16.",
    "Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting Machine. Annals of Statistics.",
    "Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32.",
    "Cortes, C., & Vapnik, V. (1995). Support-vector networks. Machine Learning, 20(3), 273–297.",
    "Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning (2nd ed.). Springer.",
]
for r in refs:
    doc.add_paragraph(r, style='List Number')

doc.add_page_break()

# ========== APPENDIX A: COMPLETE CODE ==========
add_heading(doc, "Appendix A: Complete Python Code", level=1)
add_para(doc, "The complete, executable Python script is provided as a separate file: Advik_Singh_Module_3_ML_Code.py. The full source is printed below for reference:", italic=True)

with open("Advik_Singh_Module_3_ML_Code.py", "r") as f:
    full_code = f.read()

# Add code in chunks to avoid memory issues
lines = full_code.split("\n")
chunk_size = 80
for i in range(0, len(lines), chunk_size):
    chunk = "\n".join(lines[i:i+chunk_size])
    add_code_block(doc, chunk)

doc.add_page_break()

# ========== APPENDIX B: INSTALLATION ==========
add_heading(doc, "Appendix B: Installation and Execution Instructions", level=1)
add_heading(doc, "B.1 Local Installation (Jupyter Notebook / VS Code)", level=2)
add_code_block(doc, """# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\\Scripts\\activate   # Windows

# Install required packages
pip install pandas numpy matplotlib seaborn scikit-learn xgboost catboost jupyter

# Launch Jupyter
jupyter notebook Advik_Singh_Module_3_ML_Practicals.ipynb
# Or run the script directly
python Advik_Singh_Module_3_ML_Code.py""")

add_heading(doc, "B.2 Google Colab", level=2)
add_code_block(doc, """# Run at the top of a Colab notebook:
!pip install xgboost catboost -q

# Upload the dataset file "Module 3 Project Data.csv" to the Colab session,
# then upload and open Advik_Singh_Module_3_ML_Practicals.ipynb, or paste the
# code from Advik_Singh_Module_3_ML_Code.py into a Colab cell and run it.""")

add_heading(doc, "B.3 Required Files", level=2)
files_list = [
    "Module 3 Project Data.csv — the loan default dataset (10,000 rows).",
    "Advik_Singh_Module_3_ML_Practicals.ipynb — the executable Jupyter notebook.",
    "Advik_Singh_Module_3_ML_Code.py — the executable Python script.",
]
for fl in files_list:
    doc.add_paragraph(fl, style='List Bullet')

doc.add_page_break()

# ========== APPENDIX C: VIVA QUESTIONS ==========
add_heading(doc, "Appendix C: Additional Exercises and Viva Questions", level=1)
viva_qs = [
    ("Q1: Why is accuracy misleading for imbalanced datasets?",
     "A classifier that always predicts the majority class can achieve very high accuracy (96.67% in this case) while being completely useless. Metrics like precision, recall, F1, and ROC-AUC provide a more complete picture."),
    ("Q2: What is the difference between bagging and boosting?",
     "Bagging (e.g. Random Forest) trains independent models on bootstrap samples in parallel and averages their predictions, reducing variance. Boosting (e.g. XGBoost, AdaBoost) trains models sequentially where each learner corrects the errors of its predecessors, reducing bias."),
    ("Q3: Why do we apply StandardScaler to Logistic Regression and SVM but not to Random Forest?",
     "Logistic Regression and SVM are sensitive to feature scale because they use gradient-based optimization or distance metrics (e.g. Euclidean distance). Tree-based models split on individual feature thresholds and are invariant to monotonic transformations like scaling."),
    ("Q4: How does class_weight='balanced' work?",
     "It assigns weights inversely proportional to class frequency: w_j = n_samples / (n_classes * n_j). This causes the model to pay more attention to misclassified minority-class instances during training."),
    ("Q5: What does scale_pos_weight do in XGBoost?",
     "It multiplies the gradient of the positive class by the specified weight (typically set to neg/pos ratio), effectively balancing the contribution of minority-class examples during tree growth."),
    ("Q6: Why is feature scaling important for KNN?",
     "KNN relies on distance metrics (Euclidean by default). Without scaling, features with larger magnitudes (e.g. Annual Salary in hundreds of thousands) dominate the distance computation over smaller-magnitude features (Bank Balance in thousands)."),
    ("Q7: What is ROC-AUC and why is it useful?",
     "ROC-AUC is the area under the Receiver Operating Characteristic curve, which plots TPR vs FPR across all classification thresholds. It measures the probability that a randomly chosen positive example ranks higher than a randomly chosen negative example, making it threshold-independent and robust to class imbalance."),
    ("Q8: What is the purpose of cross-validation?",
     "Cross-validation provides a more reliable estimate of generalization performance by rotating train/validation folds across the dataset, reducing dependence on a single train/validation split."),
    ("Q9: What is GridSearchCV?",
     "GridSearchCV exhaustively evaluates all combinations of specified hyperparameter values using k-fold cross-validation on the training set and returns the best-performing combination."),
    ("Q10: Why do we fit the scaler on the training set only?",
     "Fitting the scaler on the entire dataset (before splitting) would leak information from the test set into the training process (data leakage), leading to over-optimistic performance estimates. The scaler parameters (mean, std) must be estimated using training data alone."),
]
for q, a in viva_qs:
    p = doc.add_paragraph()
    p.add_run(q).bold = True
    doc.add_paragraph(a)

# ---------- Save ----------
doc.save("Advik_Singh_Module_3_ML_Report.docx")
print("DOCX report saved successfully.")
