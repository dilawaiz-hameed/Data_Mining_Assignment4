"""
DS-3002 Assignment #4 — Part E: CardioAI Local Dashboard
Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="CardioAI — Heart Disease Screener",
    page_icon="🫀",
    layout="wide"
)

# ── Load model & preprocessor ─────────────────────────────────
@st.cache_resource
def load_artifacts():
    import joblib
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    model = joblib.load("model_xgb.pkl")

    # Rebuild preprocessor from scratch (avoids sklearn version mismatch)
    cat_cols  = ['cp', 'restecg', 'slope', 'thal']
    cont_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']

    preprocessor = ColumnTransformer(transformers=[
    ('ohe',   OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_cols),
    ('scale', StandardScaler(), cont_cols)
], remainder='passthrough')

    # Fit on the Cleveland data directly
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
    col_names = ['age','sex','cp','trestbps','chol','fbs','restecg',
                 'thalach','exang','oldpeak','slope','ca','thal','target']
    df = pd.read_csv(url, header=None, names=col_names, na_values='?').dropna()
    df['target'] = (df['target'] > 0).astype(int)
    X = df.drop('target', axis=1)
    for c in cat_cols:
        X[c] = X[c].astype(int).astype(str)
    preprocessor.fit(X)

    feat_names = joblib.load("feature_names.pkl")
    explainer  = shap.TreeExplainer(model)
    return model, preprocessor, feat_names, explainer
try:
    model, preprocessor, feature_names, explainer = load_artifacts()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.warning(f"⚠️ Model files not found. Run the notebook first to generate model_xgb.pkl. ({e})")

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<h1 style='text-align:center; color:#c0392b;'>🫀 CardioAI — Heart Disease Risk Screener</h1>
<p style='text-align:center; color:#666; font-size:15px;'>
DS-3002 Assignment #4 · FAST-NUCES BSDS · XGBoost model trained on UCI Cleveland dataset
</p>
<hr>
""", unsafe_allow_html=True)

# ── Sidebar: real test patient pre-populated ───────────────────
st.sidebar.header("📋 Patient Input Form")
st.sidebar.caption("Pre-populated with a real test patient from the dataset.")

def input_form():
    age      = st.sidebar.number_input("Age (20–80)",                 min_value=20, max_value=80,  value=63)
    sex = st.sidebar.selectbox("Sex",
                           [(1, "Male"), (0, "Female")],
                           format_func=lambda x: x[1])
    cp       = st.sidebar.selectbox("Chest Pain Type (cp)",
                                    [(0,"0 — Typical Angina"),
                                     (1,"1 — Atypical Angina"),
                                     (2,"2 — Non-anginal"),
                                     (3,"3 — Asymptomatic")],
                                    format_func=lambda x: x[1], index=3)
    trestbps = st.sidebar.number_input("Resting Blood Pressure mmHg (80–200)", min_value=80,  max_value=200, value=145)
    chol     = st.sidebar.number_input("Serum Cholesterol mg/dl (100–600)",    min_value=100, max_value=600, value=233)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl",
                           [(1, "Yes"), (0, "No")],
                           format_func=lambda x: x[1], index=1)
    restecg  = st.sidebar.selectbox("Resting ECG (restecg)",
                                    [(0,"0 — Normal"), (1,"1 — ST-T abnormality"), (2,"2 — LV hypertrophy")],
                                    format_func=lambda x: x[1])
    thalach  = st.sidebar.number_input("Max Heart Rate Achieved (70–210)",  min_value=70,  max_value=210, value=150)
    exang = st.sidebar.selectbox("Exercise-Induced Angina",
                             [(1, "Yes"), (0, "No")],
                             format_func=lambda x: x[1], index=1)
    oldpeak  = st.sidebar.number_input("ST Depression (oldpeak, 0.0–6.2)",  min_value=0.0, max_value=6.2,  value=2.3, step=0.1)
    slope = st.sidebar.selectbox("ST Slope (slope)",
                             [(1,"1 — Upsloping"), (2,"2 — Flat"), (3,"3 — Downsloping")],
                             format_func=lambda x: x[1])
    ca       = st.sidebar.number_input("Major Vessels (ca, 0–3)",       min_value=0, max_value=3, value=0)
    thal = st.sidebar.selectbox("Thalassemia (thal)",
                            [(1,"1 — Normal"), (2,"2 — Fixed Defect"), (3,"3 — Reversible Defect")],
                            format_func=lambda x: x[1])
    return {
    'age':      age,
    'sex':      sex[0],      # was sex[1]
    'cp':       str(cp[0]),
    'trestbps': trestbps,
    'chol':     chol,
    'fbs':      fbs[0],      # was fbs[1]
    'restecg':  str(restecg[0]),
    'thalach':  thalach,
    'exang':    exang[0],    # was exang[1]
    'oldpeak':  oldpeak,
    'slope':    str(slope[0]),
    'ca':       ca,
    'thal':     str(thal[0]),
}

patient_data = input_form()

# ── Predict button ─────────────────────────────────────────────
predict_clicked = st.sidebar.button("🔍 Predict", type="primary", use_container_width=True)

# ── Main panel ────────────────────────────────────────────────
col1, col2 = st.columns([1, 1.3])

with col1:
    st.subheader("📊 Patient Summary")
    patient_df = pd.DataFrame(list(patient_data.items()), columns=["Feature", "Value"])
    st.dataframe(patient_df, use_container_width=True, hide_index=True)

with col2:
    st.subheader("🧬 Prediction Result")

    if predict_clicked and model_loaded:
        # Build input dataframe with correct dtypes
        input_df = pd.DataFrame([patient_data])
        # Categorical columns need string dtype for OneHotEncoder
        for c in ['cp', 'restecg', 'slope', 'thal']:
            input_df[c] = input_df[c].astype(str)

        # Transform
        X_input = preprocessor.transform(input_df)

        # Predict
        proba  = model.predict_proba(X_input)[0][1]
        pred   = int(proba >= 0.5)

        # ── Result display ────────────────────────────────────
        if pred == 1:
            st.markdown(f"""
            <div style='background-color:#fdecea; border-left:6px solid #c0392b;
                        padding:16px; border-radius:8px; margin-bottom:12px;'>
                <h2 style='color:#c0392b; margin:0;'>🔴 Disease Present</h2>
                <p style='font-size:22px; margin:4px 0;'>Confidence: <b>{proba*100:.1f}%</b></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='background-color:#eafaf1; border-left:6px solid #27ae60;
                        padding:16px; border-radius:8px; margin-bottom:12px;'>
                <h2 style='color:#27ae60; margin:0;'>🟢 No Disease Detected</h2>
                <p style='font-size:22px; margin:4px 0;'>Confidence: <b>{(1-proba)*100:.1f}%</b></p>
            </div>
            """, unsafe_allow_html=True)

        # ── SHAP top 3 features ────────────────────────────────
        st.markdown("**Top 3 features driving this prediction:**")
        shap_vals = explainer.shap_values(X_input)[0]
        shap_df   = pd.DataFrame({
            'Feature': feature_names,
            'SHAP':    shap_vals
        }).sort_values('SHAP', key=abs, ascending=False).head(3)

        fig, ax = plt.subplots(figsize=(5, 2.5))
        colors = ['tomato' if v > 0 else 'steelblue' for v in shap_df['SHAP']]
        ax.barh(shap_df['Feature'], shap_df['SHAP'], color=colors)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel('SHAP value (positive = increases disease risk)')
        ax.set_title('Feature Impact on Prediction')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # ── Plain-English explanation ──────────────────────────
        top_pos = shap_df[shap_df['SHAP'] > 0].head(2)['Feature'].tolist()
        top_neg = shap_df[shap_df['SHAP'] < 0].head(1)['Feature'].tolist()

        expl_map = {
            'thalach':   'low maximum heart rate (reduced exercise capacity)',
            'oldpeak':   'elevated ST depression (myocardial stress during exercise)',
            'ca':        'number of blocked major vessels',
            'exang':     'exercise-induced angina',
            'sex':       'sex as a cardiovascular risk factor',
            'age':       'age-related cardiovascular risk',
            'fbs':       'elevated fasting blood sugar',
            'chol':      'high serum cholesterol',
            'trestbps':  'elevated resting blood pressure',
        }

        def get_expl(f):
            for k, v in expl_map.items():
                if k in f:
                    return v
            return f

        pos_parts = ' and '.join([get_expl(f) for f in top_pos]) if top_pos else ''
        risk_word = 'elevated cardiac risk' if pred == 1 else 'lower cardiac risk'

        if pos_parts:
            plain = (f"This patient's {pos_parts} are the strongest indicators of {risk_word}. "
                     "The model recommends further clinical evaluation. "
                     "Please consult the responsible cardiologist before acting on this result.")
        else:
            plain = ("This patient's features are consistent with a low-risk cardiac profile. "
                     "Routine monitoring is still advised based on clinical judgement.")

        st.info(f"📝 **For clinical staff:** {plain}")

    elif predict_clicked and not model_loaded:
        st.error("Model not loaded. Run the notebook first to generate model_xgb.pkl.")
    else:
        st.markdown("""
        <div style='background:#f8f9fa; border-radius:8px; padding:20px; text-align:center;'>
            <h3 style='color:#999;'>👈 Fill in patient details and click <b>Predict</b></h3>
            <p style='color:#aaa;'>Results and SHAP explanations will appear here.</p>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.caption("⚠️ **Disclaimer:** This tool is for educational purposes only. "
           "It is not a certified medical device and must not replace clinical judgement. "
           "Model: XGBoost trained on UCI Cleveland Heart Disease dataset (n=297).")
