import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import streamlit as st
import sklearn.compose._column_transformer as _ct


class _RemainderColsList(list):
    pass


if not hasattr(_ct, "_RemainderColsList"):
    _ct._RemainderColsList = _RemainderColsList

st.set_page_config(
    page_title="Pediatric Diabetes XAI Predictor",
    layout="wide",
)

MODEL_PATH = "pediatric_diabetes_rf.joblib"


@st.cache_resource
def load_artifact(path: str):
    if not hasattr(_ct, "_RemainderColsList"):
        _ct._RemainderColsList = _RemainderColsList
    return joblib.load(path)


@st.cache_resource
def get_explainer(_clf):
    return shap.TreeExplainer(_clf)


def extract_metadata(pipe):
    prep = pipe.named_steps["prep"]
    num_cols, cat_cols = [], []
    for name, _, cols in prep.transformers_:
        if name == "num":
            num_cols = list(cols)
        elif name == "cat":
            cat_cols = list(cols)

    medians = {}
    num_imputer = prep.named_transformers_["num"].named_steps["imputer"]
    for col, val in zip(num_cols, num_imputer.statistics_):
        medians[col] = round(float(val), 1)

    categories = {}
    cat_ohe = prep.named_transformers_["cat"].named_steps["onehot"]
    for col, cats in zip(cat_cols, cat_ohe.categories_):
        categories[col] = [str(c) for c in cats]

    return medians, categories


def transform_features(pipe, df: pd.DataFrame):
    prep = pipe.named_steps["prep"]
    blocks = []
    feat_names = []

    for name, _, cols in prep.transformers_:
        cols = list(cols)
        if name == "num" and cols:
            num_pipe = prep.named_transformers_["num"]
            num_imputer = num_pipe.named_steps["imputer"]
            num_scaler = num_pipe.named_steps["scale"]

            vals = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64, copy=True)
            for j, stat in enumerate(num_imputer.statistics_):
                mask = np.isnan(vals[:, j])
                if mask.any():
                    vals[mask, j] = float(stat)

            scale = np.where(num_scaler.scale_ == 0, 1.0, num_scaler.scale_)
            vals = (vals - num_scaler.mean_) / scale
            blocks.append(vals)
            feat_names.extend(cols)

        elif name == "cat" and cols:
            cat_pipe = prep.named_transformers_["cat"]
            cat_imputer = cat_pipe.named_steps["imputer"]
            cat_ohe = cat_pipe.named_steps["onehot"]

            for j, (col, cats) in enumerate(zip(cols, cat_ohe.categories_)):
                fill_val = str(cat_imputer.statistics_[j])
                series = df[col].fillna(fill_val).astype(str).to_numpy()
                for c in cats:
                    blocks.append((series == str(c)).astype(np.float64).reshape(-1, 1))
                    feat_names.append(f"{col}_{c}")

    return np.hstack(blocks), feat_names


def plot_local_shap(clf, X_proc, feat_names, diabetes_idx, model_name):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    try:
        explainer = get_explainer(clf)
        raw_sv = explainer.shap_values(X_proc)

        if isinstance(raw_sv, list):
            sv = np.asarray(raw_sv[diabetes_idx])[0]
        elif np.ndim(raw_sv) == 3:
            sv = raw_sv[0, :, diabetes_idx]
        else:
            sv = raw_sv[0]

        top_k = min(10, len(sv))
        idx = np.argsort(np.abs(sv))[-top_k:]
        names = [feat_names[i] for i in idx]
        vals = [sv[i] for i in idx]
        colors = ["#d9534f" if v > 0 else "#2e86de" for v in vals]

        ax.barh(names, vals, color=colors)
        ax.axvline(0, color="#333333", linewidth=0.9)
        ax.set_title(f"Top Feature Impacts on 'Diabetes' Risk ({model_name})", fontweight="bold")
        ax.set_xlabel("SHAP value (→ increases Diabetes risk | ← decreases risk)")
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        plt.tight_layout()
    except Exception as e:
        ax.text(0.5, 0.5, f"SHAP explanation unavailable:\n{e}", ha="center", va="center")
        ax.axis("off")

    return fig


def plot_model_comparison(test_results):
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    plot_df = test_results.set_index("Model")[["Accuracy", "F1_macro", "ROC_AUC", "PR_AUC"]]
    plot_df.plot(kind="bar", ax=ax, width=0.75)
    ax.set_ylim(0.6, 0.95)
    ax.set_ylabel("Score")
    ax.set_title("Held-Out Test Set Performance by Model", fontweight="bold")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="lower right")
    plt.tight_layout()
    return fig


# --- Load Model ---
try:
    artifact = load_artifact(MODEL_PATH)
except FileNotFoundError:
    st.error(
        f"Could not find `{MODEL_PATH}` in the current folder. "
        "Place the downloaded `.joblib` file next to `app.py`."
    )
    st.stop()

pipe = artifact["model"]
clf = pipe.named_steps["model"]
best_name = artifact["best_name"]
feature_cols = artifact["feature_cols"]
class_names = artifact["class_names"]
test_results = artifact["test_results"]
cv_results = artifact["cv_results"]

diabetes_idx = class_names.index("Diabetes") if "Diabetes" in class_names else 0
medians, categories = extract_metadata(pipe)

st.title("Pediatric Diabetes Risk & Explainability Studio")
st.caption(
    f"Active Pipeline: **{best_name}** | 17 Non-Leakage Features | Research demo only — not a clinical diagnosis."
)

tab1, tab2, tab3 = st.tabs([
    "Patient Predictor & SHAP",
    "Model Comparison",
    "Batch CSV Scoring",
])

# --- TAB 1: SINGLE PREDICTION ---
with tab1:
    with st.form("patient_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Demographics & Vitals")
            gender = st.radio("Gender", categories.get("Gender", ["Female", "Male"]), horizontal=True)
            age = st.slider("Age (years, 0–19)", 0.0, 19.0, float(medians.get("Age", 8.2)), 0.1)
            hba1c = st.slider("HbA1c (%)", 3.0, 10.0, float(medians.get("HbA1c", 5.0)), 0.1)
            bmi = st.slider("BMI", 10.0, 55.0, float(medians.get("BMI", 28.5)), 0.1)
            bp = st.slider("Blood Pressure", 50.0, 190.0, float(medians.get("BloodPressure", 116.4)), 0.5)
            htn_check = st.slider("Hypertension Check", 70.0, 170.0, float(medians.get("HypertensionCheck", 121.3)), 0.5)
            heart_check = st.slider("Heart Check", 40.0, 200.0, float(medians.get("HeartCheck", 94.1)), 0.5)

        with col2:
            st.subheader("Family & Health Status")
            father_t1d = st.radio("Father T1D", categories.get("Father_T1D", ["No", "Yes"]), horizontal=True)
            mother_t2d = st.radio("Mother T2D", categories.get("Mother_T2D", ["No", "Yes"]), horizontal=True)
            health_level = st.selectbox(
                "Health Level",
                categories.get("HealthLevel", ["Excellent", "Very Good", "Good", "Fair", "Poor"]),
            )
            mental_health = st.slider("Mental Health Score", -8.0, 10.5, float(medians.get("MentalHealth", 1.2)), 0.1)
            physical_health = st.slider("Physical Health Score", -11.0, 15.0, float(medians.get("PhysicalHealth", 2.0)), 0.1)
            mobility = st.radio("Mobility Difficulty", categories.get("MobilityDifficulty", ["No", "Yes"]), horizontal=True)

        with col3:
            st.subheader("Diet & Access")
            fruits = st.radio("Regular Fruits", categories.get("Fruits", ["Yes", "No"]), horizontal=True)
            vegetables = st.radio("Regular Vegetables", categories.get("Vegetables", ["Yes", "No"]), horizontal=True)
            packaged_drinks = st.radio("Packaged Drinks", categories.get("PackagedDrinks", ["No", "Yes"]), horizontal=True)
            insurance = st.radio("Health Insurance", categories.get("Insurance", ["Yes", "No"]), horizontal=True)

        submitted = st.form_submit_button("Predict & Explain", type="primary")

    if submitted:
        input_df = pd.DataFrame([{
            "Gender": gender,
            "Age": age,
            "BloodPressure": bp,
            "BMI": bmi,
            "HbA1c": hba1c,
            "HypertensionCheck": htn_check,
            "HeartCheck": heart_check,
            "Father_T1D": father_t1d,
            "Mother_T2D": mother_t2d,
            "Fruits": fruits,
            "Vegetables": vegetables,
            "PackagedDrinks": packaged_drinks,
            "Insurance": insurance,
            "HealthLevel": health_level,
            "MentalHealth": mental_health,
            "PhysicalHealth": physical_health,
            "MobilityDifficulty": mobility,
        }])[feature_cols]

        X_proc, feat_names = transform_features(pipe, input_df)
        probs = clf.predict_proba(X_proc)[0]
        diabetes_prob = float(probs[diabetes_idx])
        predicted_label = class_names[int(np.argmax(probs))]
        risk_band = (
            "High Risk" if diabetes_prob >= 0.65
            else "Moderate Risk" if diabetes_prob >= 0.35
            else "Low Risk"
        )

        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted Class", predicted_label)
        m2.metric("Diabetes Probability", f"{diabetes_prob:.1%}")
        m3.metric("Risk Tier", risk_band)

        with st.spinner("Computing SHAP explanation..."):
            fig = plot_local_shap(clf, X_proc, feat_names, diabetes_idx, best_name)
            st.pyplot(fig)
            plt.close(fig)

# --- TAB 2: MODEL BENCHMARKING ---
with tab2:
    st.subheader("Held-Out Test Set Results")
    st.dataframe(test_results.round(4), hide_index=True)

    fig_cmp = plot_model_comparison(test_results)
    st.pyplot(fig_cmp)
    plt.close(fig_cmp)

    st.subheader("3-Fold Stratified Cross-Validation Results")
    st.dataframe(cv_results.round(4), hide_index=True)

# --- TAB 3: BATCH SCORING ---
with tab3:
    st.write("Upload a CSV containing the 17 predictor columns to score multiple records at once.")
    uploaded = st.file_uploader("Upload Patient CSV", type=["csv"])

    if uploaded is not None:
        batch_df = pd.read_csv(uploaded)
        batch_df.columns = batch_df.columns.astype(str).str.strip()
        missing = [c for c in feature_cols if c not in batch_df.columns]

        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            X_batch, _ = transform_features(pipe, batch_df[feature_cols])
            preds = clf.predict(X_batch)
            probs = clf.predict_proba(X_batch)[:, diabetes_idx]

            out_df = batch_df.copy()
            out_df["Predicted_Diagnosis"] = [class_names[i] for i in preds]
            out_df["Diabetes_Probability"] = np.round(probs, 4)

            pos_count = int((out_df["Predicted_Diagnosis"] == "Diabetes").sum())
            st.success(
                f"Scored {len(out_df):,} records — {pos_count:,} predicted as Diabetes ({pos_count / len(out_df):.1%})."
            )
            st.dataframe(out_df.head(50))

            st.download_button(
                label="Download Scored CSV",
                data=out_df.to_csv(index=False).encode("utf-8"),
                file_name="diabetes_predictions.csv",
                mime="text/csv",
                type="primary",
            )