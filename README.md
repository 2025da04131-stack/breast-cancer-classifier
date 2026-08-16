# ML Assignment 2 — Breast Cancer Classification

## a. Problem statement

Breast cancer diagnosis is a binary classification problem: given a set of
numeric features computed from a digitized image of a breast mass (fine
needle aspirate), predict whether the mass is **malignant (0)** or
**benign (1)**. Early and accurate diagnosis directly affects treatment
decisions, so this assignment compares five standard classification
algorithms on this task using six evaluation metrics, and exposes the
results through an interactive Streamlit application.

## b. Dataset description

- **Name:** Breast Cancer Wisconsin (Diagnostic) Data Set
- **Source:** `sklearn.datasets.load_breast_cancer` (a bundled, cleaned
  version of the UCI ML Repository dataset:
  https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
- **Instances:** 569 (> 500 required ✅)
- **Features:** 30 numeric features (> 12 required ✅) — mean, standard
  error, and "worst" (largest) values of 10 cell-nuclei characteristics
  (radius, texture, perimeter, area, smoothness, compactness, concavity,
  concave points, symmetry, fractal dimension), computed from digitized
  images of fine needle aspirate (FNA) biopsies.
- **Target:** Binary — `0 = malignant`, `1 = benign`
  (212 malignant / 357 benign — a moderately imbalanced but manageable split)
- **Train/test split:** 80% train / 20% test, stratified by class, random
  state fixed at 42 for reproducibility.

## c. GitHub repository link

`https://github.com/<your-username>/<your-repo-name>`
*(Replace with your actual repo link before submission.)*

## d. Models used

All 5 models were trained on the same 80/20 train-test split of the dataset
described above. Logistic Regression and kNN were trained on
**standardized** features (via `StandardScaler`, fit on train data only);
Decision Tree, Naive Bayes, and Random Forest do not require scaling and
were trained on the raw features.

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| kNN | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Random Forest (Ensemble) | 0.9561 | 0.9931 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best overall performer across every metric (Accuracy 0.983, MCC 0.962). The dataset's classes are close to linearly separable after standardization, which suits a linear decision boundary well, and the small feature set (30) relative to sample size limits overfitting risk. |
| Decision Tree | Weakest of the five models (Accuracy 0.912, MCC 0.817). A single unpruned tree overfits the training data and produces a comparatively poor generalization; recall (0.903) is noticeably lower, meaning it missed more benign cases than the other models. |
| kNN | Strong, balanced performance (Accuracy 0.956, F1 0.966) after feature scaling — distance-based methods are sensitive to feature magnitude, and standardization was essential here. Performs almost identically to Random Forest on this test split. |
| Naive Bayes | Solid AUC (0.988) despite the model's simplifying independence assumption between features, but Accuracy/MCC (0.939 / 0.868) trail behind Logistic Regression and the ensemble — likely because several of the 30 features are correlated (e.g., radius, perimeter, area), violating Naive Bayes' core assumption. |
| Random Forest (Ensemble) | Very strong and stable performance (Accuracy 0.956, AUC 0.993) — the ensemble of 200 trees reduces the variance/overfitting problem seen in the single Decision Tree, and its AUC is the second-highest of all models, reflecting well-calibrated class probabilities. |
| **Overall winner for this dataset** | **Logistic Regression** — it achieved the top score on every single metric (Accuracy, AUC, Precision, Recall, F1, MCC), suggesting the standardized feature set is well-suited to a linear classifier for this particular diagnostic task. Random Forest is the strongest non-linear alternative and would be the recommended choice if more complex, non-linear class boundaries were expected on unseen data. |

## Repository structure

```
project-folder/
│-- app.py                  # Streamlit application
│-- requirements.txt
│-- README.md
│-- test_data.csv           # held-out test set (features + true label)
│-- model/
│   │-- train_models.py     # trains all 5 models, saves .pkl files
│   │-- logistic_regression.pkl
│   │-- decision_tree.pkl
│   │-- knn.pkl
│   │-- naive_bayes.pkl
│   │-- random_forest.pkl
│   │-- scaler.pkl
│   └-- comparison_metrics.csv
```

## How to run locally

```bash
pip install -r requirements.txt
python model/train_models.py   # regenerates model/*.pkl and test_data.csv
streamlit run app.py
```

## Streamlit app features

1. **Dataset upload** — upload a CSV (must include a `target` column); defaults
   to the bundled `test_data.csv` if nothing is uploaded.
2. **Model selection dropdown** — choose any of the 5 trained models.
3. **Evaluation metrics display** — Accuracy, AUC, Precision, Recall, F1, MCC
   shown live for the selected model on the uploaded data.
4. **Confusion matrix & classification report** — visual heatmap plus a full
   per-class precision/recall/F1 report.
5. **Side-by-side comparison** — optional toggle to see all 5 models' metrics
   on the same uploaded test data at once.
