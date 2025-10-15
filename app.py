# 1 = good, 0 = bad

import streamlit as st
import pandas as pd
import joblib

model = joblib.load("extratree_model.pkl")

encoder = {
    col: joblib.load(f"{col}_encoder.pkl") 
    for col in ["Sex", "Housing", "Saving accounts", "Checking account"]
}

st.title("Credit Risk Prediction App")
st.write("Enter the applicant information to predict if the credit is good or bad")

age = st.number_input("Age", min_value=18, max_value=80, value=30)
sex = st.selectbox("Sex", encoder["Sex"].classes_)
job = st.number_input("Job (0-3)", min_value=0, max_value=3, value=1)
housing = st.selectbox("Housing", encoder["Housing"].classes_)
saving_account = st.selectbox("Saving account", encoder["Saving accounts"].classes_)
checking_account = st.selectbox("Checking account", encoder["Checking account"].classes_)
credit_account = st.number_input("Credit amount", min_value=0, value=1000)
duration = st.number_input("Duration (months)", min_value=1, value=12)

input_df = pd.DataFrame({
    "Age": [age],
    "Job": [job],
    "Housing": [encoder["Housing"].transform([housing])[0]],
    "Saving accounts": [encoder["Saving accounts"].transform([saving_account])[0]],
    "Credit amount": [credit_account],
    "Checking account": [encoder["Checking account"].transform([checking_account])[0]],
    "Sex": [encoder["Sex"].transform([sex])[0]],
    "Duration": [duration]
})

input_df = input_df[model.feature_names_in_]

if st.button("Predict risk"):
    pred = model.predict(input_df)[0]
    if pred == 1:
        st.balloons()
        st.success("The predicted credit is : **Good**")
    else:
        st.error("The predicted credit is : **Bad**")
