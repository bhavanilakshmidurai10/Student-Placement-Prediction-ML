import streamlit as st
import joblib
import numpy as np

model = joblib.load("../saved_model/placement_model.pkl")
scaler = joblib.load("../saved_model/scaler.pkl")

st.title("🎓 Student Placement Prediction System")

cgpa = st.number_input("CGPA", 0.0, 10.0, 7.5)
skills = st.slider("Skills Score", 0, 100, 70)
intern = st.number_input("Internships", 0, 5, 1)
projects = st.number_input("Projects", 0, 10, 2)
apt = st.slider("Aptitude Score", 0, 100, 65)

if st.button("Predict Placement"):
    data = np.array([[cgpa, skills, intern, projects, apt]])
    data_scaled = scaler.transform(data)
    pred = model.predict(data_scaled)

    if pred[0] == 1:
        st.success("Likely to be Placed ✅")
    else:
        st.error("Not Likely to be Placed ❌")
