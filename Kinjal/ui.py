import streamlit as st
import pandas as pd
import cloudpickle

# --------------------------------------
# Load Trained Pipeline
# --------------------------------------
try:
    with open("sleep_pipeline.pkl", "rb") as f:
        pipeline = cloudpickle.load(f)
except Exception as e:
    st.error(f"❌ Could not load pipeline: {e}")
    st.stop()

# Class label mapping
class_labels = {
    0: "No Disorder",
    1: "Insomnia",
    2: "Sleep Apnea"
}

# --------------------------------------
# Page Configuration and Styling
# --------------------------------------
st.set_page_config(page_title="Sleep Disorder Predictor", layout="centered")

# Background and CSS styling
st.markdown(
    """
    <style>
    body {
        background-color: #f2f6fc;
    }
    .main {
        background: linear-gradient(to right, #dae2f8, #d6a4a4);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0px 4px 8px rgba(0,0,0,0.1);
    }
    h1, h3 {
        color: #003566;
    }
    .stButton>button {
        background-color: #003566;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.6em 2em;
    }
    .stSelectbox, .stSlider, .stTextInput {
        background-color: #ffffff;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------
# Title and Instructions
# --------------------------------------
st.markdown('<div class="main">', unsafe_allow_html=True)
st.title("😴 Sleep Disorder Prediction App")
st.markdown("Enter your health and lifestyle information below to check for sleep disorders.")

# --------------------------------------
# Input Fields
# --------------------------------------
st.subheader("📋 Personal & Lifestyle Details")
age = st.slider("Age", 0, 100, 25)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
occupation = st.selectbox("Occupation", [
    "Doctor", "Nurse", "Engineer", "Teacher", "Sales",
    "Software Engineer", "Lawyer", "Scientist"
])
bmi_category = st.selectbox("BMI Category", ["Normal", "Overweight", "Obese", "Underweight"])
sleep_duration = st.slider("Sleep Duration (hours)", 0.0, 12.0, 6.5)
physical_activity = st.slider("Physical Activity Level (minutes/day)", 0, 300, 30)
stress_level = st.slider("Stress Level (1–10)", 1, 10, 5)

# Blood Pressure Input (Split into high and low)
bp_input = st.text_input("Blood Pressure (Systolic/Diastolic, e.g., 120/80)", "120/80")
try:
    high_pressure, low_pressure = [int(x) for x in bp_input.strip().split("/")]
except:
    st.error("❌ Please enter blood pressure in the correct format (e.g., 120/80).")
    st.stop()

# --------------------------------------
# Data Preparation
# --------------------------------------
input_data = {
    "Age": age,
    "Gender": gender,
    "Occupation": occupation,
    "BMI Category": bmi_category,
    "Sleep Duration": sleep_duration,
    "Physical Activity Level": physical_activity,
    "Stress Level": stress_level,
    "High_pressure": high_pressure,
    "Low_pressure": low_pressure
}

# --------------------------------------
# Prediction Button and Output
# --------------------------------------
st.subheader("🔍 Prediction")
if st.button("🧠 Predict"):
    try:
        input_df = pd.DataFrame([input_data])
        prediction = pipeline.predict(input_df)[0]
        predicted_label = class_labels.get(prediction, "Unknown")
        st.success(f"🩺 **Predicted Sleep Disorder**: **{predicted_label}** (Class {prediction})")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

st.markdown('</div>', unsafe_allow_html=True)
