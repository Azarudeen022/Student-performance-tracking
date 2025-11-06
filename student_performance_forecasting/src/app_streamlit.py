import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("🎓 Student Performance Forecasting")
st.write("Predict whether a student will pass an exam using a trained ML model.")

@st.cache_resource
def load_model(path="models/best_model.joblib"):
    return joblib.load(path)

model = load_model()

st.subheader("Input Features")
col1, col2 = st.columns(2)
with col1:
    study_time = st.number_input("Study time (hrs/week)", min_value=0.0, max_value=40.0, value=8.0, step=0.5)
    absences = st.number_input("Absences", min_value=0, max_value=60, value=3, step=1)
    previous_score = st.number_input("Previous score (%)", min_value=0, max_value=100, value=65, step=1)
    hours_sleep = st.number_input("Hours of sleep", min_value=3.0, max_value=12.0, value=7.0, step=0.5)
    parental_education = st.selectbox("Parental education", ["none","primary","secondary","tertiary"])
with col2:
    internet = st.selectbox("Internet", ["yes","no"])
    gender = st.selectbox("Gender", ["male","female"])
    school_support = st.selectbox("School support", ["yes","no"])
    test_prep = st.selectbox("Test preparation", ["course","none"])
    lunch_type = st.selectbox("Lunch type", ["standard","free/reduced"])

row = pd.DataFrame([{
    "study_time": study_time,
    "absences": absences,
    "previous_score": previous_score,
    "parental_education": parental_education,
    "internet": internet,
    "gender": gender,
    "school_support": school_support,
    "test_prep": test_prep,
    "hours_sleep": hours_sleep,
    "lunch_type": lunch_type
}])

if st.button("Predict"):
    prob = model.predict_proba(row)[:,1][0]
    pred = int(prob >= 0.5)
    st.write(f"**Predicted class:** {'Pass' if pred==1 else 'Fail'}")
    st.write(f"**Probability of passing:** {prob:.2f}")
