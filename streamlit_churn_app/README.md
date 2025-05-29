
# 📊 Churn Prediction App

A Streamlit web app that predicts whether a telecom customer is likely to **stay or churn** using a machine learning model trained on customer behavior data.

---

## 🚀 Features

- 🔘 Manual customer input via interactive form  
- 📁 CSV upload for batch predictions  
- 🌙 Dark mode toggle  
- 📄 Downloadable prediction report  
- 💬 Simple explanation of each prediction  
- ⚙️ Powered by XGBoost (with class imbalance handling)

---

## 🧠 Problem It Solves

Telecom companies often lose customers without warning. This app predicts churn risk, allowing businesses to take early action to improve customer retention and reduce revenue loss.

---

## 📦 Files Included

- `app.py` — main Streamlit app  
- `churn_model_weighted.pkl` — trained XGBoost model  
- `requirements.txt` — dependencies  
- `README.md` — this file

---

## ▶️ How to Run Locally

1. Clone this repo:
   ```bash
   git clone https://github.com/Alagoka/churn-prediction-app.git
   cd churn-prediction-app
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

---

## 🧪 Example Input Fields

- Gender, Senior Citizen, Partner, Dependents  
- Monthly Charges, Total Charges  
- Contract Type, Payment Method  
- Internet and Phone Services

---

## 📄 Output

- ✅ Prediction: Will the customer stay or churn?  
- 📊 Model confidence score  
- 📃 SHAP explanation in plain language  
- 📥 Downloadable CSV report for batch uploads

---

## 📌 License

MIT License – free to use, modify, and share.

---

Built with ❤️ using Streamlit and XGBoost.
