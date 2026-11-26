# 🩺 Pediatric Diabetes Risk Prediction & Explainable AI (XAI) Studio

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML%20Pipeline-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/XAI-SHAP%20%26%20LIME-6C63FF)](https://shap.readthedocs.io/)

An end-to-end **Machine Learning & Explainable AI (XAI)** web application for pediatric diabetes risk assessment. Trained on **322,841 pediatric records**, the system combines a leakage-free scikit-learn pipeline with real-time **SHAP (SHapley Additive exPlanations)** visualizations so every individual prediction is transparent and interpretable.

> **Live Demo:** Replace the Streamlit badge link above with your deployed `https://your-app-name.streamlit.app` URL.

---

## ✨ Key Features

- **Interactive Patient Risk Predictor:** Enter 17 demographic, clinical vital, family history, and lifestyle inputs to get an instant class prediction, estimated probability, and risk tier (*Low*, *Moderate*, or *High*).
- **Real-Time Local Explainability (SHAP):** Generates a per-patient feature attribution chart highlighting exactly which factors pushed the risk score higher or lower.
- **Model Benchmarking Dashboard:** Compare 3-fold stratified cross-validation and held-out test metrics across 5 classifiers (*Random Forest, AdaBoost, Extra Trees, Decision Tree, and Logistic Regression*).
- **Batch CSV Scoring:** Upload a CSV containing patient records to score thousands of rows in seconds and download the annotated predictions.

---

## 🔬 Methodology & Data Integrity

A major pitfall in medical ML projects is **target leakage**. This pipeline enforces strict data auditing and leakage prevention:

1. **Pediatric Cohort Filtering:** Audited **323,941** raw records (`diabetes_tipe_1_child.csv`) and explicitly excluded **1,100** records with out-of-range ages outside `0–19` years, yielding **322,841** valid pediatric records (`189,097` No Diabetes | `133,744` Diabetes).
2. **Leakage Prevention:** Removed **8** post-diagnosis/work-up variables (`DiseaseName`, `Medication`, `Pancreatic`, `GADA`, `IAA`, `IA-2A`, `ZnT8A`, `MedicalHistory`) so the model relies strictly on **17 non-leakage predictors** (8 numeric, 9 categorical).
3. **Zero-Leakage Preprocessing:** Median imputation, standard scaling, and one-hot encoding are encapsulated inside scikit-learn `Pipeline` and `ColumnTransformer` objects fitted exclusively on the training split (**258,272** train / **64,569** stratified held-out test).

---

## 📊 Model Benchmark Results (Held-Out Test Set)

All models were evaluated using 3-fold stratified cross-validation and benchmarked on the **64,569-row** held-out test set:

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** *(Deployed)* | **0.8501** | **0.8500** | **0.8391** | **0.8433** | 0.8382 | **0.8340** |
| **AdaBoost** | 0.8501 | 0.8500 | 0.8391 | 0.8433 | **0.8384** | 0.8301 |
| **Extra Trees** | 0.8034 | 0.7977 | 0.7965 | 0.7971 | 0.8297 | 0.8286 |
| **Decision Tree** | 0.7281 | 0.7198 | 0.7195 | 0.7196 | 0.7195 | 0.7252 |
| **Logistic Regression** | 0.7196 | 0.7150 | 0.7206 | 0.7157 | 0.7791 | 0.7985 |

---

## 🧠 Explainability Insights (SHAP & Permutation Importance)

Both global **SHAP** analysis and **Permutation Feature Importance** independently converged on the same top three predictive drivers:

| Rank | Feature | Mean \|SHAP\| | Permutation Importance (Δ Macro-F1) |
| :---: | :--- | :---: | :---: |
| 1 | **BloodPressure** | `0.1744` | `0.1941 ± 0.0017` |
| 2 | **BMI** | `0.1621` | `0.1611 ± 0.0012` |
| 3 | **HbA1c** | `0.1008` | `0.1427 ± 0.0008` |

---

## 📁 Repository Structure

```text
Diabetes-XAI/
├── app.py                        # Streamlit web application (Predictor, SHAP, Benchmarks, Batch CSV)
├── pediatric_diabetes_rf.joblib  # Serialized best Random Forest pipeline & evaluation tables
├── requirements.txt              # Python dependencies for local & cloud deployment
└── README.md                     # Project documentation
```

---

## 🚀 Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/Mithilesh2025/Diabetes-XAI.git
   cd Diabetes-XAI
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Streamlit app**
   ```bash
   streamlit run app.py
   ```

The app will open automatically at `http://localhost:8501`.

---

## ⚠️ Disclaimer

This application is built strictly for **academic research and educational demonstration purposes**. Predictions reflect statistical patterns learned from the training dataset and do **not** constitute medical advice or a clinical diagnosis.

---

## 👤 Author

**Mithilesh R**  
GitHub: [@Mithilesh2025](https://github.com/Mithilesh2025)