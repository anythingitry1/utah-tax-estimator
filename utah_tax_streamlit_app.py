
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load models
model_lr = joblib.load("model_linear_regression_utah.pkl")
model_rf = joblib.load("model_random_forest_utah.pkl")
model_xgb = joblib.load("model_xgboost_utah.pkl")

st.title("🧾 Utah State Tax Estimator with Model Comparison")

st.sidebar.header("Enter Your Info")
wages = st.sidebar.number_input("Annual Wages ($)", 0, 300000, 60000)
num_children = st.sidebar.slider("Number of Children", 0, 10, 0)
pre_tax_deductions = st.sidebar.number_input("Pre-tax Deductions ($)", 0, 50000, 5000)
itemized_deductions = st.sidebar.selectbox("Itemized Deductions?", [0, 1])
raise_percent = st.sidebar.slider("Raise % (for year)", 0.0, 25.0, 3.0)
months_elapsed = st.sidebar.slider("Months Elapsed", 1, 12, 6)
pay_frequency = st.sidebar.selectbox("Pay Frequency", ["weekly", "biweekly", "monthly"])

# One-hot encode pay_frequency
freq_cols = {
    "pay_frequency_biweekly": int(pay_frequency == "biweekly"),
    "pay_frequency_monthly": int(pay_frequency == "monthly"),
    "pay_frequency_weekly": int(pay_frequency == "weekly")
}

# Prepare input for prediction
X_input = pd.DataFrame([{
    "wages": wages,
    "num_children": num_children,
    "pre_tax_deductions": pre_tax_deductions,
    "itemized_deductions": itemized_deductions,
    "raise_percent": raise_percent,
    "months_elapsed": months_elapsed,
    **freq_cols
}])

# Predict with each model
pred_lr = model_lr.predict(X_input)[0]
pred_rf = model_rf.predict(X_input)[0]
pred_xgb = model_xgb.predict(X_input)[0]

st.subheader("📊 Estimated Tax by Model")
results_df = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest", "XGBoost"],
    "Estimated Tax ($)": [pred_lr, pred_rf, pred_xgb]
})

st.dataframe(results_df.style.format({"Estimated Tax ($)": "{:,.2f}"}), use_container_width=True)

fig, ax = plt.subplots()
ax.bar(results_df["Model"], results_df["Estimated Tax ($)"], color=["skyblue", "orange", "green"])
ax.set_ylabel("Estimated Tax ($)")
ax.set_title("Comparison of Model Outputs")
st.pyplot(fig)

# Tax Bracket Chart
st.subheader("📈 Utah 2023 Tax Bracket Reference")
brackets = pd.DataFrame({
    "Taxable Income ($)": ["$0 - $5,000", "$5,001 - $10,000", "$10,001 - $20,000", "$20,001+"],
    "Effective Rate": ["2%", "3%", "4%", "4.65%"]
})
st.table(brackets)
