import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Approval System",
    page_icon="🏦",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f0f4f8; }
    .stApp { max-width: 900px; margin: auto; }

    .header-box {
        background: linear-gradient(135deg, #1a3c6e, #2563eb);
        color: white;
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .header-box h1 { font-size: 2rem; margin: 0; }
    .header-box p  { margin: 0.4rem 0 0; opacity: 0.85; font-size: 1rem; }

    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1a3c6e;
        border-left: 4px solid #2563eb;
        padding-left: 10px;
        margin: 1.5rem 0 0.8rem;
    }

    .result-approved {
        background: linear-gradient(135deg, #d1fae5, #a7f3d0);
        border: 2px solid #10b981;
        border-radius: 14px;
        padding: 1.8rem;
        text-align: center;
        margin-top: 1.5rem;
    }
    .result-rejected {
        background: linear-gradient(135deg, #fee2e2, #fecaca);
        border: 2px solid #ef4444;
        border-radius: 14px;
        padding: 1.8rem;
        text-align: center;
        margin-top: 1.5rem;
    }
    .result-icon  { font-size: 3.5rem; margin-bottom: 0.5rem; }
    .result-title { font-size: 1.6rem; font-weight: 800; margin: 0.3rem 0; }
    .result-prob  { font-size: 1rem; color: #374151; margin-top: 0.4rem; }

    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #1a3c6e, #2563eb);
        color: white;
        font-size: 1.1rem;
        font-weight: 700;
        padding: 0.75rem 1rem;
        border: none;
        border-radius: 10px;
        cursor: pointer;
        margin-top: 1rem;
    }
    .stButton > button:hover { opacity: 0.9; }

    .stSelectbox label, .stNumberInput label, .stSlider label {
        font-weight: 600;
        color: #374151;
    }

    .info-card {
        background: #eff6ff;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
        font-size: 0.9rem;
        color: #1e40af;
    }
</style>
""", unsafe_allow_html=True)


# ── Load artifacts ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    base = "artifacts"
    if not os.path.exists(base):
        return None

    def _load(name):
        with open(os.path.join(base, name), "rb") as f:
            return pickle.load(f)

    return {
        "model":           _load("model.pkl"),
        "scaler":          _load("scaler.pkl"),
        "ohe":             _load("ohe.pkl"),
        "le_edu":          _load("le_edu.pkl"),
        "le_target":       _load("le_target.pkl"),
        "num_imp":         _load("num_imp.pkl"),
        "feature_names":   _load("feature_names.pkl"),
        "dropdown_options":_load("dropdown_options.pkl"),
    }

arts = load_artifacts()


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-box">
    <h1>🏦 Loan Approval Prediction System</h1>
    <p>Fill in the applicant details below to get an instant approval decision</p>
</div>
""", unsafe_allow_html=True)

if arts is None:
    st.error(
        "⚠️ Model artifacts not found!  \n"
        "Please run **`python train_model.py`** first to generate the model, "
        "then restart this app."
    )
    st.stop()


# ── Helper: get dropdown choices ───────────────────────────────────────────────
opts = arts["dropdown_options"]

def get_opts(col, fallback):
    return opts.get(col, fallback)


# ── Form ───────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">👤 Personal Information</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    gender = st.selectbox("Gender", get_opts("Gender", ["Male", "Female", "Other"]))
with col2:
    marital_status = st.selectbox("Marital Status", get_opts("Marital_Status", ["Single", "Married", "Divorced"]))
with col3:
    education = st.selectbox("Education Level", get_opts("Education_Level", ["Graduate", "Not Graduate", "Post Graduate"]))


st.markdown('<div class="section-title">💼 Employment & Income</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    employment_status = st.selectbox("Employment Status", get_opts("Employment_Status", ["Employed", "Self-Employed", "Unemployed"]))
with col2:
    employer_category = st.selectbox("Employer Category", get_opts("Employer_Category", ["Government", "Private", "NGO"]))
with col3:
    applicant_income = st.number_input("Applicant Income (₹)", min_value=0, value=50000, step=1000)

col1, col2 = st.columns(2)
with col1:
    co_applicant_income = st.number_input("Co-Applicant Income (₹)", min_value=0, value=0, step=1000)
with col2:
    num_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0, step=1)


st.markdown('<div class="section-title">🏠 Loan Details</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    loan_amount = st.number_input("Loan Amount (₹)", min_value=1000, value=200000, step=5000)
with col2:
    loan_term = st.number_input("Loan Term (months)", min_value=6, max_value=360, value=120, step=6)
with col3:
    loan_purpose = st.selectbox("Loan Purpose", get_opts("Loan_Purpose", ["Home", "Education", "Business", "Personal"]))

col1, col2 = st.columns(2)
with col1:
    property_area = st.selectbox("Property Area", get_opts("Property_Area", ["Urban", "Semiurban", "Rural"]))
with col2:
    previous_loan_taken = st.selectbox("Previous Loan Taken?", ["No", "Yes"])


st.markdown('<div class="section-title">📊 Financial Health</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    credit_score = st.slider("Credit Score", 300, 900, 700, step=10)
with col2:
    dti_ratio = st.slider("Debt-to-Income Ratio (DTI)", 0.0, 1.0, 0.3, step=0.01,
                           help="Total monthly debt / gross monthly income")
with col3:
    existing_loans = st.number_input("Existing Loans Count", min_value=0, max_value=10, value=0, step=1)

col1, col2 = st.columns(2)
with col1:
    loan_to_value = st.slider("Loan-to-Value Ratio (LTV)", 0.0, 1.5, 0.8, step=0.01)
with col2:
    age = st.number_input("Applicant Age", min_value=18, max_value=80, value=30, step=1)


# ── Predict ────────────────────────────────────────────────────────────────────
st.markdown("---")

if st.button("🔍 Check Loan Eligibility"):

    # Build raw input dict (mirroring original CSV structure)
    raw_input = {
        "Applicant_Income":      [applicant_income],
        "Co_Applicant_Income":   [co_applicant_income],
        "Loan_Amount":           [loan_amount],
        "Loan_Term":             [loan_term],
        "Credit_Score":          [credit_score],
        "DTI_Ratio":             [dti_ratio],
        "Age":                   [age],
        "Existing_Loans":        [existing_loans],
        "Loan_to_Value_Ratio":   [loan_to_value],
        "Number_of_Dependents":  [num_dependents],
        "Previous_Loan_Taken":   [1 if previous_loan_taken == "Yes" else 0],
        "Education_Level":       [education],
        "Employment_Status":     [employment_status],
        "Marital_Status":        [marital_status],
        "Loan_Purpose":          [loan_purpose],
        "Property_Area":         [property_area],
        "Gender":                [gender],
        "Employer_Category":     [employer_category],
    }

    input_df = pd.DataFrame(raw_input)

    try:
        # ── a) Encode Education_Level ──────────────────────────────────────────
        le_edu = arts["le_edu"]
        edu_classes = list(le_edu.classes_)
        if education not in edu_classes:
            input_df["Education_Level"] = 0          # fallback
        else:
            input_df["Education_Level"] = le_edu.transform([education])

        # ── b) One-hot encode categorical cols ─────────────────────────────────
        ohe_cols = ["Employment_Status", "Marital_Status", "Loan_Purpose",
                    "Property_Area", "Gender", "Employer_Category"]

        ohe     = arts["ohe"]
        encoded = ohe.transform(input_df[ohe_cols])
        enc_df  = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(ohe_cols))

        input_df = pd.concat(
            [input_df.drop(columns=ohe_cols).reset_index(drop=True),
             enc_df.reset_index(drop=True)],
            axis=1
        )

        # ── c) Feature engineering ─────────────────────────────────────────────
        input_df["DTI_Ratio_sq"]    = input_df["DTI_Ratio"]    ** 2
        input_df["Credit_Score_sq"] = input_df["Credit_Score"] ** 2
        input_df = input_df.drop(columns=["DTI_Ratio", "Credit_Score"], errors="ignore")

        # ── d) Align columns with training data ────────────────────────────────
        feature_names = arts["feature_names"]
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[feature_names]

        # ── e) Scale & predict ─────────────────────────────────────────────────
        scaled  = arts["scaler"].transform(input_df)
        pred    = arts["model"].predict(scaled)[0]
        prob    = arts["model"].predict_proba(scaled)[0]

        # Decode prediction
        le_target = arts["le_target"]
        label = le_target.inverse_transform([pred])[0]
        approved_prob = prob[list(le_target.classes_).index(le_target.classes_[pred])] * 100

        # ── f) Show result ─────────────────────────────────────────────────────
        if str(label).lower() in ["1", "yes", "y", "approved"]:
            st.markdown(f"""
            <div class="result-approved">
                <div class="result-icon">✅</div>
                <div class="result-title" style="color:#065f46;">LOAN APPROVED!</div>
                <div class="result-prob">Confidence: <b>{approved_prob:.1f}%</b></div>
                <p style="color:#065f46; margin-top:0.6rem;">
                    The applicant meets the eligibility criteria. Loan can be processed.
                </p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-rejected">
                <div class="result-icon">❌</div>
                <div class="result-title" style="color:#991b1b;">LOAN REJECTED</div>
                <div class="result-prob">Confidence: <b>{approved_prob:.1f}%</b></div>
                <p style="color:#991b1b; margin-top:0.6rem;">
                    The applicant does not meet the eligibility criteria at this time.
                </p>
            </div>""", unsafe_allow_html=True)

        # ── g) Key metrics summary ─────────────────────────────────────────────
        st.markdown('<div class="section-title">📋 Application Summary</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Credit Score",  credit_score)
        c2.metric("DTI Ratio",     f"{dti_ratio:.2f}")
        c3.metric("Loan Amount",   f"₹{loan_amount:,}")
        c4.metric("Income",        f"₹{applicant_income:,}")

    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.exception(e)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#9ca3af; font-size:0.85rem;'>"
    "🏦 Loan Approval Prediction System · Built with Scikit-learn & Streamlit"
    "</p>",
    unsafe_allow_html=True
)
