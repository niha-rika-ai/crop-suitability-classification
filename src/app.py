import streamlit as st
import pandas as pd
import joblib


# ==========================================
# LOAD MODEL
# ==========================================

model = joblib.load("models/crop_model.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Crop Suitability Classification",
    page_icon="🌱",
    layout="centered"
)


# ==========================================
# TITLE
# ==========================================

st.title("🌱 Crop Suitability Classification")

st.write(
    "Enter soil and environmental conditions "
    "to predict the suitable crop category."
)


# ==========================================
# INPUTS
# ==========================================

st.subheader("Soil Parameters")

N = st.number_input(
    "Nitrogen (N)",
    min_value=0.0,
    max_value=200.0,
    value=50.0
)

P = st.number_input(
    "Phosphorus (P)",
    min_value=0.0,
    max_value=200.0,
    value=50.0
)

K = st.number_input(
    "Potassium (K)",
    min_value=0.0,
    max_value=250.0,
    value=50.0
)

ph = st.number_input(
    "Soil pH",
    min_value=0.0,
    max_value=14.0,
    value=6.5
)


st.subheader("Environmental Parameters")

temperature = st.number_input(
    "Temperature (°C)",
    min_value=-10.0,
    max_value=60.0,
    value=25.0
)

humidity = st.number_input(
    "Humidity (%)",
    min_value=0.0,
    max_value=100.0,
    value=70.0
)

rainfall = st.number_input(
    "Rainfall (mm)",
    min_value=0.0,
    max_value=500.0,
    value=100.0
)


# ==========================================
# PREDICTION
# ==========================================

if st.button("Predict Suitable Crop"):

    input_data = pd.DataFrame({

        "N": [N],

        "P": [P],

        "K": [K],

        "temperature": [temperature],

        "humidity": [humidity],

        "ph": [ph],

        "rainfall": [rainfall]

    })


    prediction = model.predict(input_data)

    predicted_crop = label_encoder.inverse_transform(
        prediction
    )[0]


    st.success(
        f"Predicted Suitable Crop: **{predicted_crop.upper()}**"
    )


    # Probability if supported
    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(
            input_data
        )[0]

        max_probability = probabilities.max()

        st.info(
            f"Model confidence: "
            f"{max_probability * 100:.2f}%"
        )