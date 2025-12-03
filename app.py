import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import pickle
import streamlit as st

st.set_page_config(
    page_title="CHURN OFF - Customer Churn Prediction",
    page_icon="📉",
    layout="wide"
)

st.markdown(
    """
    <style>
    .main {
        background-color: #0f172a;
        color: #e5e7eb;
    }
    .stMetric {
        background-color: #020617 !important;
        padding: 0.8rem 1rem !important;
        border-radius: 0.75rem !important;
        border: 1px solid #1f2937 !important;
    }
    .big-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #e5e7eb;
    }
    .sub-title {
        font-size: 0.95rem;
        color: #9ca3af;
    }
    .result-box {
        padding: 1.2rem 1.5rem;
        border-radius: 0.75rem;
        border: 1px solid #1f2937;
        background: #020617;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = tf.keras.models.load_model("churn_model.h5")
    with open("le_gender.pkl", "rb") as f:
        le_gender = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("ohe_geography.pkl", "rb") as f:
        ohe_geography = pickle.load(f)
    return model, le_gender, scaler, ohe_geography

model, le_gender, scaler, ohe_geography = load_artifacts()

st.markdown(
    """
    <div>
        <div class="big-title">CHURN OFF</div>
        <p class="sub-title">
            Interactive customer churn prediction. Tune customer attributes on the left and get real‑time risk scores.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")


st.sidebar.header("Customer Profile")

geo_options = list(ohe_geography.categories_[0])
gender_options = list(le_gender.classes_)

geography = st.sidebar.selectbox("Geography", geo_options)
gender = st.sidebar.selectbox("Gender", gender_options)
age = st.sidebar.slider("Age", 18, 92, 35)
tenure = st.sidebar.slider("Tenure (years)", 0, 10, 3)
credit_score = st.sidebar.number_input("Credit Score", min_value=0, value=650, step=1)
balance = st.sidebar.number_input("Balance", min_value=0.0, step=0.01, format="%.2f")
num_of_products = st.sidebar.slider("Number of Products", 1, 4, 2)
has_cr_card = st.sidebar.selectbox("Has Credit Card", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
is_active_member = st.sidebar.selectbox("Is Active Member", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
estimated_salary = st.sidebar.number_input("Estimated Salary", min_value=0.0, step=0.01, format="%.2f")

predict_button = st.sidebar.button("Predict Churn 🔮")


def build_feature_row():
    input_dict = {
        "CreditScore": [credit_score],
        "Gender": [int(le_gender.transform([gender])[0])],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_of_products],
        "HasCrCard": [1 if has_cr_card == 1 else 0],
        "IsActiveMember": [1 if is_active_member == 1 else 0],
        "EstimatedSalary": [estimated_salary],
    }
    input_df = pd.DataFrame(input_dict)

    
    geo_encoded = ohe_geography.transform([[geography]])
    if hasattr(geo_encoded, "toarray"):
        geo_encoded = geo_encoded.toarray()

    geo_cols = ohe_geography.get_feature_names_out(["Geography"])
    geo_df = pd.DataFrame(geo_encoded, columns=geo_cols)

    
    full_df = pd.concat(
        [input_df.reset_index(drop=True), geo_df.reset_index(drop=True)], axis=1
    )

    return full_df


col_left, col_right = st.columns([1.3, 1.7])

with col_left:
    st.subheader("Key Attributes")
    m1, m2, m3 = st.columns(3)
    m1.metric("Age", f"{age} yrs")
    m2.metric("Credit Score", credit_score)
    m3.metric("Balance", f"{balance:,.0f}")

    m4, m5, m6 = st.columns(3)
    m4.metric("Tenure", f"{tenure} yrs")
    m5.metric("Products", num_of_products)
    m6.metric("Active Member", "Yes" if is_active_member == 1 else "No")

with col_right:
    st.subheader("Prediction")

    if predict_button:
        features = build_feature_row()
        
        input_scaled = scaler.transform(features)
    
        pred = model.predict(input_scaled)
        predict_prob = float(pred[0][0])

        churn_prob = predict_prob
        stay_prob = 1 - predict_prob
        is_churn = churn_prob > 0.5

        status = "High Churn Risk" if is_churn else "Low Churn Risk"
        status_color = "#ef4444" if is_churn else "#22c55e"

        st.markdown(
            f"""
            <div class="result-box">
                <h3 style="margin-bottom:0.4rem;">Prediction Result</h3>
                <p style="margin:0.1rem 0 0.8rem 0; color:#9ca3af;">
                    Model‑estimated probability of this customer leaving the bank.
                </p>
                <div style="display:flex; gap:1.5rem; align-items:center; flex-wrap:wrap;">
                    <div style="flex:1;">
                        <div style="font-size:2.4rem; font-weight:700; color:{status_color};">
                            {churn_prob:.2%}
                        </div>
                        <div style="font-size:0.9rem; color:#9ca3af;">
                            Churn probability
                        </div>
                    </div>
                    <div style="flex:1;">
                        <div style="
                            padding:0.5rem 0.9rem;
                            border-radius:999px;
                            background-color:rgba(15,23,42,0.9);
                            border:1px solid #1f2937;
                            display:inline-flex;
                            align-items:center;
                            gap:0.4rem;
                            ">
                            <span style="
                                width:10px;height:10px;border-radius:999px;
                                background-color:{status_color};
                            "></span>
                            <span style="font-weight:600;color:#e5e7eb;">{status}</span>
                        </div>
                        <div style="margin-top:0.6rem;font-size:0.9rem;color:#9ca3af;">
                            Stay probability: {stay_prob:.2%}
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("View model input row"):
            st.dataframe(build_feature_row())

    else:
        st.info("Adjust the customer profile in the sidebar and click **Predict Churn 🔮** to get a prediction.")
