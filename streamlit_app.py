
# app.py
# Streamlit App — Alzheimer's (OASIS Longitudinal) — Mirror of Notebook Metrics
# Run (locally/Colab): streamlit run app.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

st.set_page_config(page_title="Alzheimer’s — OASIS ML Mirror", layout="wide", page_icon="🧠")

# ---------------------- Sidebar: Data ----------------------
st.sidebar.header("🗂️ Data")
uploaded = st.sidebar.file_uploader("Upload oasis_longitudinal.csv", type=["csv"])

DEFAULT_CSV = "oasis_longitudinal.csv"
CSV_PATHS = [DEFAULT_CSV, "/mnt/data/oasis_longitudinal.csv"]

def load_csv():
    if uploaded is not None:
        return pd.read_csv(uploaded)
    for p in CSV_PATHS:
        if os.path.exists(p):
            return pd.read_csv(p)
    st.error("CSV not found. Please upload oasis_longitudinal.csv.")
    st.stop()

df_raw = load_csv()

# ---------------------- Pre-clean to match notebook ----------------------
df = df_raw.copy()

# 1) Map 'Converted' → 'Demented' (common notebook tweak)
if "Group" in df.columns:
    df["Group"] = df["Group"].replace({"Converted": "Demented"})

# 2) Drop obvious ID columns (kept out of feature set)
id_cols = [c for c in ["Subject ID", "MRI ID"] if c in df.columns]

# 3) Drop rows with missing values (the notebook used dropna())
df = df.dropna().reset_index(drop=True)

# ---------------------- Target & features ----------------------
TARGET = "Group"  # detected from notebook

if TARGET not in df.columns:
    st.error(f"Target column '{TARGET}' not found in CSV.")
    st.stop()

# Suggested feature set (typical OASIS columns, excluding IDs & target)
default_feature_order = [
    "Age", "EDUC", "SES", "MMSE", "CDR", "eTIV", "nWBV", "ASF", "M/F", "Hand", "Visit", "MR Delay"
]
features_default = [c for c in default_feature_order if c in df.columns and c != TARGET]

with st.sidebar.expander("🧮 Features", expanded=True):
    use_features = st.multiselect(
        "Select feature columns",
        options=[c for c in df.columns if c not in id_cols + [TARGET]],
        default=features_default if features_default else [c for c in df.columns if c not in id_cols + [TARGET]]
    )

# Keep only selected columns
df = df[id_cols + use_features + [TARGET]]

# ---------------------- Encode target (binary) ----------------------
# Ensure binary setup: "Demented" vs "Nondemented"
y = df[TARGET].astype(str)
# If more than 2 classes accidentally present, restrict to the two main ones
if set(y.unique()) - {"Demented", "Nondemented"}:
    # Map everything not "Nondemented" to "Demented"
    y = y.where(y == "Nondemented", "Demented")

# Encode y to 0/1 in a stable order
cls_order = ["Nondemented", "Demented"]
y = pd.Categorical(y, categories=cls_order, ordered=True)
y_int = y.codes  # Nondemented=0, Demented=1

# ---------------------- Encode X (match simple notebook style) ----------------------
X = df[use_features].copy()

# Simple pandas get_dummies (no scaler/CT, since not found in notebook)
X = pd.get_dummies(X, drop_first=True)

# ---------------------- Split (notebook showed test_size=0.2 and random_state≈27) ----------------------
TEST_SIZE = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, 0.01)
RANDOM_STATE = st.sidebar.number_input("Random state", min_value=0, value=27, step=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_int, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_int
)

# ---------------------- Models (detected in notebook) ----------------------
models = {
    # Defaults align with detected notebook code
    "RandomForest": RandomForestClassifier(random_state=RANDOM_STATE),
    "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    "DecisionTree": DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "SVC (RBF)": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
}

st.title("🧠 Alzheimer’s Diagnosis — OASIS (Notebook Mirror)")
st.caption("Reproduces the notebook’s classical ML baselines with the same split/settings for matching results.")

col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("Dataset Snapshot")
    st.write(df.head())

with col2:
    st.subheader("Class Balance (after cleaning)")
    st.bar_chart(pd.Series(pd.Categorical.from_codes(y_int, categories=cls_order)).value_counts())

# ---------------------- Train & Evaluate ----------------------
def eval_binary(y_true, y_pred, y_proba=None):
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        try:
            out["roc_auc"] = roc_auc_score(y_true, y_proba)
        except Exception:
            out["roc_auc"] = np.nan
    else:
        out["roc_auc"] = np.nan
    return out

st.header("📊 Results")
tabs = st.tabs(list(models.keys()))
plots = {}

for (name, model), tab in zip(models.items(), tabs):
    with tab:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        # Probability estimates (for ROC AUC)
        y_proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)
            if proba is not None and proba.shape[1] == 2:
                y_proba = proba[:, 1]

        scores = eval_binary(y_test, y_pred, y_proba)
        st.write({k: round(v, 4) if isinstance(v, (int, float, np.floating)) else v for k, v in scores.items()})
        st.text("Classification report:")
        st.code(classification_report(y_test, y_pred, target_names=cls_order, zero_division=0))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        st.write("Confusion matrix (rows=true, cols=pred):")
        st.dataframe(pd.DataFrame(cm, index=cls_order, columns=cls_order))

        # ROC curve if available
        if scores.get("roc_auc", np.nan) == scores.get("roc_auc", np.nan) and not np.isnan(scores["roc_auc"]):
            fig, ax = plt.subplots()
            RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax, name=name)
            ax.set_title(f"ROC Curve — {name} (AUC={scores['roc_auc']:.3f})")
            st.pyplot(fig)

st.success("Done. If you keep the same split (test_size=0.2, random_state=27), results should match the notebook’s classical models closely.")

# ---------------------- Repro bundle ----------------------
with st.expander("🧩 Reproducibility"):
    st.write("• Target fixed to **Group** with `'Converted' → 'Demented'` mapping, matching the notebook.")
    st.write("• Missing rows dropped to mirror `.dropna()` usage detected.")
    st.write("• Simple `get_dummies` encoding (no scaler/pipeline detected in notebook).")
    st.write("• Models & hyperparameters mirror the notebook (DT `max_depth=4`, KNN `n_neighbors=7`, SVC RBF with `probability=True`).")
    st.write("• Use `Random state = 27` and `Test size = 0.2` to align splits and metrics.")
