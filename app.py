import streamlit as st
import joblib
import numpy as np

model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")
target_encoder = joblib.load("target_encoder.pkl")
category_maps = joblib.load("category_maps.pkl")

category_maps["Location"] = [
    "Dhaka", "Chattogram", "Khulna", "Rajshahi",
    "Sylhet", "Barishal", "Rangpur", "Mymensingh"
]

st.title("Teen Smartphone Addiction Risk Predictor")
st.write("Enter the values below to predict addiction risk category.")

user_input = {}
for col in feature_columns:
    if col in category_maps:
        choice = st.selectbox(col, category_maps[col])
        user_input[col] = category_maps[col].index(choice)
    else:
        user_input[col] = st.number_input(col, value=0.0)

if st.button("Predict Risk"):
    features = np.array([[user_input[col] for col in feature_columns]])
    features_scaled = scaler.transform(features)
    pred_encoded = model.predict(features_scaled)[0]
    pred_label = target_encoder.inverse_transform([pred_encoded])[0]
    st.success(f"Predicted Addiction Risk: **{pred_label}**")
