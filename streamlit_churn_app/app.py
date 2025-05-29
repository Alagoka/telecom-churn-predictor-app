
import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load("churn_model_rebuilt.pkl")

model = load_model()

st.title("📉 Customer Churn Prediction App")

dark_mode = st.toggle("🌙 Dark Mode")
if dark_mode:
    st.markdown("""<style>
        html, body, [class*="css"] {
            background-color: #111 !important;
            color: #eee !important;
        }
    </style>""", unsafe_allow_html=True)

with st.form("user_input_form"):
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, step=1.0)
    total_charges = st.number_input("Total Charges", min_value=0.0, step=1.0)

    submitted = st.form_submit_button("Predict Churn")

    if submitted:
        input_dict = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }
        df_input = pd.DataFrame([input_dict])

        encode_map = {
            "gender": {"Female": 0, "Male": 1},
            "Partner": {"No": 0, "Yes": 1},
            "Dependents": {"No": 0, "Yes": 1},
            "PhoneService": {"No": 0, "Yes": 1},
            "MultipleLines": {"No": 0, "Yes": 1, "No phone service": 2},
            "InternetService": {"DSL": 0, "Fiber optic": 1, "No": 2},
            "OnlineSecurity": {"No": 0, "Yes": 1, "No internet service": 2},
            "OnlineBackup": {"No": 0, "Yes": 1, "No internet service": 2},
            "DeviceProtection": {"No": 0, "Yes": 1, "No internet service": 2},
            "TechSupport": {"No": 0, "Yes": 1, "No internet service": 2},
            "StreamingTV": {"No": 0, "Yes": 1, "No internet service": 2},
            "StreamingMovies": {"No": 0, "Yes": 1, "No internet service": 2},
            "Contract": {"Month-to-month": 0, "One year": 1, "Two year": 2},
            "PaperlessBilling": {"No": 0, "Yes": 1},
            "PaymentMethod": {
                "Electronic check": 0,
                "Mailed check": 1,
                "Bank transfer (automatic)": 2,
                "Credit card (automatic)": 3
            }
        }

        for col, mapping in encode_map.items():
            df_input[col] = df_input[col].map(mapping)

        prediction = model.predict(df_input)[0]
        prob = model.predict_proba(df_input)[0][1]

        st.markdown("### Prediction: {}".format("🔴 Churn" if prediction else "🟢 No Churn"))
        st.write("**Confidence**: {:.2%}".format(prob))

        try:
            explainer = shap.Explainer(model)
            shap_values = explainer(df_input)
            st.subheader("🔍 Explanation (SHAP)")
            fig, ax = plt.subplots()
            shap.plots.waterfall(shap_values[0], show=False)
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"SHAP explanation unavailable: {e}")

        buffer = BytesIO()
        df_input["Prediction"] = prediction
        df_input["Confidence"] = prob
        df_input.to_csv(buffer, index=False)
        buffer.seek(0)
        
buffer_download = buffer.getvalue()
st.session_state["report_buffer"] = buffer_download


st.header("📥 Batch Upload for Multiple Customers")
csv_file = st.file_uploader("Upload CSV file", type=["csv"])

if csv_file:
    batch_df = pd.read_csv(csv_file)

    for col, mapping in encode_map.items():
        if col in batch_df.columns:
            batch_df[col] = batch_df[col].map(mapping)

    batch_df.fillna(0, inplace=True)
    batch_preds = model.predict(batch_df)
    batch_probs = model.predict_proba(batch_df)[:, 1]
    batch_df["Prediction"] = batch_preds
    batch_df["Confidence"] = batch_probs

    st.dataframe(batch_df)
    csv_out = batch_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Full Batch Predictions", csv_out, "batch_churn_predictions.csv", "text/csv")

if "report_buffer" in st.session_state:
    st.download_button("Download Prediction Report", st.session_state["report_buffer"], "churn_report.csv", "text/csv")
