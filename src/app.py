import datetime as dt
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
import streamlit as st

from src.database import get_engine

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="wide",
)

MODEL_DIR = Path("models")
BEST_MODEL_PATH = MODEL_DIR / "best_heart_model.joblib"
META_PATH = MODEL_DIR / "model_meta.json"


@st.cache_resource
def load_model():
    if not BEST_MODEL_PATH.exists():
        st.error(
            f"Model not found at `{BEST_MODEL_PATH}`.\n\n"
            "Run `python -m src.train` first."
        )
        st.stop()
    return joblib.load(BEST_MODEL_PATH)


@st.cache_resource
def load_meta():
    if META_PATH.exists():
        with open(META_PATH) as f:
            return json.load(f)
    return {}


def log_prediction(input_df: pd.DataFrame, probability: float, label: int):
    """Best-effort log to PostgreSQL — never break the UI if the DB is down."""
    try:
        engine = get_engine()
        pd.DataFrame([{
            "timestamp": dt.datetime.now(dt.timezone.utc),
            "input_json": input_df.to_json(orient="records"),
            "probability": float(probability),
            "prediction": int(label),
        }]).to_sql("prediction_log", engine, if_exists="append", index=False)
    except Exception as exc:
        st.warning(f"Prediction made, but DB logging failed: {exc}")


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
model = load_model()
meta = load_meta()

st.title("❤️ Heart Disease Risk Predictor")
st.caption(
    "Enter patient clinical measurements to estimate heart-disease risk. "
    "This is a **student ML project** — not a medical device."
)
if meta:
    st.info(
        f"Model: **{meta.get('best_model', 'unknown')}** · "
        f"Test ROC-AUC: **{meta.get('roc_auc', 0):.3f}** · "
        f"Trained on {meta.get('trained_on_rows', '?'):,} rows"
    )

# --------------------------------------------------------------------------- #
# Input form — categories match the UCI dataset strings exactly
# --------------------------------------------------------------------------- #
with st.form("patient_form"):
    st.subheader("Patient Clinical Data")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=54)
        sex = st.selectbox("Sex", ["Male", "Female"], index=0)
        cp = st.selectbox(
            "Chest pain type",
            ["typical angina", "atypical angina", "non-anginal", "asymptomatic"],
            index=1,
        )
        trestbps = st.number_input(
            "Resting blood pressure (mm Hg)", min_value=50, max_value=250, value=130
        )
        chol = st.number_input(
            "Cholesterol (mg/dl)", min_value=50, max_value=700, value=230
        )

    with col2:
        fbs = st.selectbox("Fasting blood sugar > 120 mg/dl", ["FALSE", "TRUE"], index=0)
        restecg = st.selectbox(
            "Resting ECG",
            ["normal", "st-t abnormality", "lv hypertrophy"],
            index=0,
        )
        thalch = st.number_input(
            "Max heart rate achieved", min_value=50, max_value=220, value=150
        )
        exang = st.selectbox("Exercise-induced angina", ["FALSE", "TRUE"], index=0)
        oldpeak = st.number_input(
            "ST depression (oldpeak)", min_value=-3.0, max_value=7.0,
            value=0.0, step=0.1, format="%.1f"
        )

    with col3:
        slope = st.selectbox(
            "ST slope", ["upsloping", "flat", "downsloping"], index=1
        )
        ca = st.selectbox("Major vessels colored by fluoroscopy (ca)", [0, 1, 2, 3], index=0)
        thal = st.selectbox(
            "Thalassemia (thal)",
            ["normal", "fixed defect", "reversable defect"],
            index=0,
        )

    submitted = st.form_submit_button("Predict Risk", use_container_width=True)

# --------------------------------------------------------------------------- #
# Predict
# --------------------------------------------------------------------------- #
if submitted:
    # Build a 1-row DataFrame with column names identical to training X
    input_df = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": 1 if fbs == "TRUE" else 0,
        "restecg": restecg,
        "thalch": thalch,
        "exang": 1 if exang == "TRUE" else 0,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
    }])

    probability = float(model.predict_proba(input_df)[0, 1])
    label = int(model.predict(input_df)[0])

    # ------- result panel -------
    st.subheader("Result")

    c1, c2, c3 = st.columns([1, 1, 2])

    with c1:
        if label == 1:
            st.error("⚠️ High Risk")
        else:
            st.success("✅ Low Risk")

    with c2:
        st.metric("Risk probability", f"{probability:.1%}")

    with c3:
        st.progress(min(max(probability, 0.0), 1.0))

    if probability >= 0.65:
        st.write(
            "The model estimates a **high** probability of coronary artery disease. "
            "Recommend further diagnostic testing."
        )
    elif probability >= 0.35:
        st.write(
            "The model estimates an **intermediate** probability. "
            "Consider additional workup based on clinical context."
        )
    else:
        st.write(
            "The model estimates a **low** probability of coronary artery disease."
        )

    # ------- logging + raw input -------
    log_prediction(input_df, probability, label)

    with st.expander("Input data sent to model"):
        st.dataframe(input_df.T.rename(columns={0: "value"}))

st.divider()
st.caption(
    "Built with scikit-learn, XGBoost, PostgreSQL, SQLAlchemy and Streamlit. "
    "For educational use only — not a substitute for clinical judgment."
)