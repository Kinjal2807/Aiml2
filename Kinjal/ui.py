import streamlit as st
import pandas as pd
import joblib

# Load pipeline
try:
    pipeline = joblib.load("sleep_pipeline.pkl")
except Exception as e:
    st.error(f"❌ Could not load pipeline: {e}")
    st.stop()

# Class label mapping
class_labels = {0: "No Disorder", 1: "Insomnia", 2: "Sleep Apnea"}

# Load training data (to align dummy columns)
train_df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")
train_df = train_df[train_df["Sleep Disorder"].notnull()]
X_train = train_df.drop("Sleep Disorder", axis=1)
X_encoded = pd.get_dummies(X_train)
expected_columns = X_encoded.columns  # columns model expects

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("😴 Sleep Disorder Predictor")

age = st.slider("Age", 0, 100, 25)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
occupation = st.text_input("Occupation", "Engineer")
bmi_category = st.selectbox("BMI Category", ["Normal", "Overweight", "Obese", "Underweight"])
sleep_duration = st.slider("Sleep Duration (hours)", 0.0, 12.0, 6.5)
physical_activity = st.slider("Physical Activity Level (minutes/day)", 0, 300, 30)
stress_level = st.slider("Stress Level (1–10)", 1, 10, 5)
bp_input = st.text_input("Blood Pressure (e.g., 120/80)", "120/80")

try:
    high_pressure, low_pressure = [int(x) for x in bp_input.split("/")]
except:
    st.error("❌ Enter BP like 120/80")
    st.stop()

# Prepare raw input
input_data = {
    "Age": age,
    "Gender": gender,
    "Occupation": occupation,
    "BMI Category": bmi_category,
    "Sleep Duration": sleep_duration,
    "Physical Activity Level": physical_activity,
    "Stress Level": stress_level,
    "High_pressure": high_pressure,
    "Low_pressure": low_pressure,
}

input_df = pd.DataFrame([input_data])

# Apply same encoding
input_encoded = pd.get_dummies(input_df)

# Align columns with training
input_encoded = input_encoded.reindex(columns=expected_columns, fill_value=0)

# Prediction
if st.button("Predict"):
    try:
        pred = pipeline.predict(input_encoded)[0]
        st.success(f"🩺 Prediction: {class_labels[pred]}")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
