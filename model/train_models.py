"""
train_models.py
----------------
Trains 5 classification models on the Breast Cancer Wisconsin (Diagnostic)
dataset, evaluates them, and saves:
  - trained model objects (model/*.pkl)
  - a fitted StandardScaler (model/scaler.pkl)
  - test_data.csv (held-out test set, features + true label, used by the
    Streamlit app for "upload test data" and metric display)
  - comparison_metrics.csv (the 6-metric table for all 5 models)

Dataset: sklearn.datasets.load_breast_cancer
  - 569 instances (> 500 required)
  - 30 numeric features (> 12 required)
  - Binary classification target: 0 = malignant, 1 = benign
"""

import pandas as pd
import numpy as np
import pickle
import os

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

RANDOM_STATE = 42
os.makedirs("model", exist_ok=True)

# ---------------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")   # 0 = malignant, 1 = benign

print(f"Dataset shape: {X.shape}, classes: {np.unique(y)}")

# ---------------------------------------------------------------------
# 2. Train / test split (stratified, 80/20)
# ---------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

# ---------------------------------------------------------------------
# 3. Scale features (fit on train only, then apply to test)
#    Needed for Logistic Regression / kNN which are scale-sensitive.
# ---------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# ---------------------------------------------------------------------
# 4. Save test data (features + true label) for the Streamlit app
# ---------------------------------------------------------------------
test_df = X_test.copy()
test_df["target"] = y_test.values
test_df.to_csv("test_data.csv", index=False)
print("Saved test_data.csv:", test_df.shape)

# ---------------------------------------------------------------------
# 5. Define models
#    Tree-based / NB models are trained on unscaled data (not required),
#    Logistic Regression & kNN trained on scaled data.
# ---------------------------------------------------------------------
models = {
    "Logistic Regression": (LogisticRegression(max_iter=5000, random_state=RANDOM_STATE), True),
    "Decision Tree":       (DecisionTreeClassifier(random_state=RANDOM_STATE), False),
    "kNN":                 (KNeighborsClassifier(n_neighbors=5), True),
    "Naive Bayes":         (GaussianNB(), False),
    "Random Forest":       (RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE), False),
}

results = []

for name, (model, needs_scaling) in models.items():
    Xtr = X_train_scaled if needs_scaling else X_train.values
    Xte = X_test_scaled if needs_scaling else X_test.values

    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    y_proba = model.predict_proba(Xte)[:, 1]

    metrics = {
        "ML Model Name": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "AUC": round(roc_auc_score(y_test, y_proba), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1": round(f1_score(y_test, y_pred), 4),
        "MCC": round(matthews_corrcoef(y_test, y_pred), 4),
    }
    results.append(metrics)

    # save model
    fname = "model/" + name.lower().replace(" ", "_") + ".pkl"
    with open(fname, "wb") as f:
        pickle.dump({"model": model, "needs_scaling": needs_scaling}, f)
    print(f"Trained {name}: {metrics}")

# ---------------------------------------------------------------------
# 6. Save comparison table
# ---------------------------------------------------------------------
results_df = pd.DataFrame(results)
results_df.to_csv("model/comparison_metrics.csv", index=False)
print("\n=== Comparison Table ===")
print(results_df.to_string(index=False))
