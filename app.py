"""
Streamlit app for ML Assignment 2
Dataset: Breast Cancer Wisconsin (Diagnostic) — binary classification
Models: Logistic Regression, Decision Tree, kNN, Naive Bayes, Random Forest
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

st.set_page_config(page_title="Breast Cancer Classifier Comparison", layout="wide")

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "kNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl",
}

@st.cache_resource
def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_scaler():
    with open("model/scaler.pkl", "rb") as f:
        return pickle.load(f)

st.title("Breast Cancer Diagnosis — Classification Model Comparison")
st.markdown(
    "This app demonstrates 5 classification models trained on the "
    "**Breast Cancer Wisconsin (Diagnostic)** dataset (569 instances, 30 features, "
    "binary target: 0 = malignant, 1 = benign)."
)

# -----------------------------------------------------------------------
# Step A: Dataset upload
# -----------------------------------------------------------------------
st.header("1. Upload test data")
uploaded_file = st.file_uploader(
    "Upload the test_data.csv (must include a 'target' column)", type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns.")
    st.dataframe(df.head())
else:
    st.info("No file uploaded yet — using the bundled test_data.csv by default.")
    df = pd.read_csv("test_data.csv")
    st.dataframe(df.head())

if "target" not in df.columns:
    st.error("Uploaded file must contain a 'target' column with the true class label.")
    st.stop()

X_test = df.drop(columns=["target"])
y_test = df["target"]

# -----------------------------------------------------------------------
# Step B: Model selection
# -----------------------------------------------------------------------
st.header("2. Select a model")
model_choice = st.selectbox("Choose a classification model", list(MODEL_FILES.keys()))

model_bundle = load_model(MODEL_FILES[model_choice])
model = model_bundle["model"]
needs_scaling = model_bundle["needs_scaling"]

# Always load the scaler — it's needed here if the selected model uses it,
# and also later in Section 5 when comparing against OTHER models that may
# need scaling even if the currently selected model doesn't.
scaler = load_scaler()

if needs_scaling:
    X_input = scaler.transform(X_test)
else:
    X_input = X_test.values

y_pred = model.predict(X_input)
y_proba = model.predict_proba(X_input)[:, 1]

# -----------------------------------------------------------------------
# Step C: Evaluation metrics
# -----------------------------------------------------------------------
st.header("3. Evaluation metrics")

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Accuracy", f"{acc:.4f}")
col2.metric("AUC", f"{auc:.4f}")
col3.metric("Precision", f"{prec:.4f}")
col4.metric("Recall", f"{rec:.4f}")
col5.metric("F1 Score", f"{f1:.4f}")
col6.metric("MCC", f"{mcc:.4f}")

# -----------------------------------------------------------------------
# Step D: Confusion matrix + classification report
# -----------------------------------------------------------------------
st.header("4. Confusion matrix & classification report")

cm_col, report_col = st.columns(2)

with cm_col:
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Malignant (0)", "Benign (1)"],
                yticklabels=["Malignant (0)", "Benign (1)"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_choice}")
    st.pyplot(fig)

with report_col:
    report = classification_report(y_test, y_pred, target_names=["Malignant", "Benign"])
    st.text("Classification Report")
    st.code(report)

# -----------------------------------------------------------------------
# Step E: Compare all models side by side
# -----------------------------------------------------------------------
st.header("5. Compare all models on this test data")
if st.checkbox("Show comparison across all 5 models"):
    rows = []
    for name, path in MODEL_FILES.items():
        bundle = load_model(path)
        m = bundle["model"]
        Xi = scaler.transform(X_test) if bundle["needs_scaling"] else X_test.values
        yp = m.predict(Xi)
        ypr = m.predict_proba(Xi)[:, 1]
        rows.append({
            "Model": name,
            "Accuracy": round(accuracy_score(y_test, yp), 4),
            "AUC": round(roc_auc_score(y_test, ypr), 4),
            "Precision": round(precision_score(y_test, yp), 4),
            "Recall": round(recall_score(y_test, yp), 4),
            "F1": round(f1_score(y_test, yp), 4),
            "MCC": round(matthews_corrcoef(y_test, yp), 4),
        })
    st.dataframe(pd.DataFrame(rows).set_index("Model"))

st.markdown("---")
st.caption("ML Assignment 2 — Streamlit deployment demo")
