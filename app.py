import streamlit as st
import joblib
import pandas as pd
import numpy as np


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Solar Power Predictor",
    page_icon="☀️",
    layout="wide"
)


# --------------------------------------------------
# Load Models
# --------------------------------------------------

models = joblib.load("models/solar_power_models.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("☀️ Solar Power Generation Predictor")
st.markdown(
    "Predict **AC Power generation** using environmental "
    "and time-based parameters."
)

st.divider()


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.header("⚙️ Plant Configuration")

plant = st.sidebar.selectbox(
    "Select Solar Plant",
    ["Plant 1", "Plant 2"]
)


# --------------------------------------------------
# Input Parameters
# --------------------------------------------------

st.subheader("🌤️ Environmental Conditions")

col1, col2 = st.columns(2)

with col1:
    ambient_temperature = st.number_input(
        "Ambient Temperature (°C)",
        min_value=-10.0,
        max_value=60.0,
        value=25.0
    )

    module_temperature = st.number_input(
        "Module Temperature (°C)",
        min_value=-10.0,
        max_value=80.0,
        value=35.0
    )

with col2:
    irradiation = st.number_input(
        "Irradiation (W/m²)",
        min_value=0.0,
        max_value=1500.0,
        value=500.0
    )


st.subheader("🕒 Time Information")

col1, col2, col3, col4 = st.columns(4)

with col1:
    hour = st.number_input(
        "Hour",
        min_value=0,
        max_value=23,
        value=12
    )

with col2:
    day = st.number_input(
        "Day",
        min_value=1,
        max_value=31,
        value=15
    )

with col3:
    month = st.number_input(
        "Month",
        min_value=1,
        max_value=12,
        value=6
    )

with col4:
    day_of_week = st.number_input(
        "Day of Week",
        min_value=0,
        max_value=6,
        value=2
    )


# --------------------------------------------------
# Derived Feature
# --------------------------------------------------

date = pd.Timestamp(
    year=2020,
    month=int(month),
    day=min(int(day), 28)
)

day_of_year = date.dayofyear


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("🔮 Predict AC Power", use_container_width=True):

    # Create input
    input_data = pd.DataFrame([{
        "AMBIENT_TEMPERATURE": ambient_temperature,
        "MODULE_TEMPERATURE": module_temperature,
        "IRRADIATION": irradiation,
        "HOUR": hour,
        "DAY": day,
        "MONTH": month,
        "DAY_OF_YEAR": day_of_year,
        "DAY_OF_WEEK": day_of_week,
        "PLANT_Plant 1": 1 if plant == "Plant 1" else 0,
        "PLANT_Plant 2": 1 if plant == "Plant 2" else 0
    }])

    # Ensure exact training feature order
    input_data = input_data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Select appropriate model
    model = models[plant]

    # Prediction
    prediction = model.predict(input_data)[0]

    # Prevent negative power
    prediction = max(0, prediction)

    st.divider()

    st.success("Prediction generated successfully!")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Predicted AC Power",
            f"{prediction:,.2f}"
        )

    with col2:
        st.metric(
            "Solar Plant",
            plant
        )