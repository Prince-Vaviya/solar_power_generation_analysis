import streamlit as st
import pandas as pd
import joblib
from datetime import date

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Solar Power Predictor",
    page_icon="☀️",
    layout="wide"
)

# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():
    models = joblib.load("models/solar_power_models.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")
    return models, feature_columns


try:
    final_models, feature_columns = load_models()
except Exception as e:
    st.error("Could not load the trained models.")
    st.code(str(e))
    st.stop()


# ============================================================
# PLANT INFORMATION
# ============================================================

PLANT_INFO = {
    "Plant 1": {
        "inverters": 22,
        "model": "Gradient Boosting"
    },
    "Plant 2": {
        "inverters": 22,
        "model": "Random Forest"
    }
}


# ============================================================
# HEADER
# ============================================================

st.title("☀️ Solar Power Generation Predictor")
st.write(
    "Predict total plant AC power generation using environmental "
    "and time-based parameters."
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Plant Configuration")

plant = st.sidebar.selectbox(
    "Select Solar Plant",
    ["Plant 1", "Plant 2"]
)

plant_info = PLANT_INFO[plant]

st.sidebar.info(
    f"**{plant}**\n\n"
    f"🔌 Inverters: **{plant_info['inverters']}**\n\n"
    f"🤖 Model: **{plant_info['model']}**"
)


# ============================================================
# ENVIRONMENTAL CONDITIONS
# ============================================================

# Input limits are based on the actual training dataset.
# Values outside these ranges are intentionally not accepted
# because they would require the model to extrapolate beyond
# the conditions it learned from.
TRAINING_LIMITS = {
    "ambient_min": 20.39,
    "ambient_max": 39.18,
    "module_min": 18.14,
    "module_max": 66.64,
    "irradiation_min": 0.00,
    "irradiation_max": 1.22,
}

st.subheader("🌤️ Environmental Conditions")

st.caption(
    "⚠️ Input limits are based on the model's training data. "
    "The predictor does not accept values outside these ranges."
)

col1, col2 = st.columns(2)

with col1:
    ambient_temperature = st.number_input(
        "Ambient Temperature (°C)",
        min_value=TRAINING_LIMITS["ambient_min"],
        max_value=TRAINING_LIMITS["ambient_max"],
        value=25.0,
        step=0.5
    )

with col2:
    # Training data irradiation is stored in kW/m².
    irradiation = st.number_input(
        "Irradiation (kW/m²)",
        min_value=TRAINING_LIMITS["irradiation_min"],
        max_value=TRAINING_LIMITS["irradiation_max"],
        value=0.50,
        step=0.01
    )

module_temperature = st.number_input(
    "Module Temperature (°C)",
    min_value=TRAINING_LIMITS["module_min"],
    max_value=TRAINING_LIMITS["module_max"],
    value=30.0,
    step=0.5
)

with st.expander("📏 Model Input Limits"):
    st.write(
        f"• Ambient Temperature: **{TRAINING_LIMITS['ambient_min']:.2f} – {TRAINING_LIMITS['ambient_max']:.2f} °C**\n\n"
        f"• Module Temperature: **{TRAINING_LIMITS['module_min']:.2f} – {TRAINING_LIMITS['module_max']:.2f} °C**\n\n"
        f"• Irradiation: **{TRAINING_LIMITS['irradiation_min']:.2f} – {TRAINING_LIMITS['irradiation_max']:.2f} kW/m²**\n\n"
        "• Hour: **0 – 23**\n\n"
        "• Training dates: **15 May 2020 – 17 June 2020**"
    )


# ============================================================
# TIME INFORMATION
# ============================================================

st.subheader("🕒 Time Information")

time_col1, time_col2 = st.columns(2)

with time_col1:
    selected_date = st.date_input(
        "Date",
        value=date(2020, 6, 15),
        min_value=date(2020, 5, 15),
        max_value=date(2020, 6, 17)
    )

with time_col2:
    hour = st.number_input(
        "Hour",
        min_value=0,
        max_value=23,
        value=12,
        step=1
    )


# ============================================================
# DERIVED TIME FEATURES
# ============================================================

day = selected_date.day
month = selected_date.month
day_of_year = selected_date.timetuple().tm_yday

# Python: Monday=0 ... Sunday=6
day_of_week = selected_date.weekday()


# ============================================================
# PREDICTION
# ============================================================

if st.button("🔮 Predict AC Power", use_container_width=True):

    # Exact feature structure used during model training.
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

    # Force exact training column order.
    input_data = input_data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # --------------------------------------------------------
    # Physical sanity handling
    # --------------------------------------------------------
    # The training data contains many zero-output observations.
    # At night / near-zero irradiation, total solar generation
    # should be treated as zero rather than relying on an
    # extrapolation from the tree model.
    low_sunlight = irradiation <= 0.02
    nighttime = hour < 6 or hour > 18

    if low_sunlight or nighttime:
        prediction = 0.0
        prediction_note = (
            "Very low/no sunlight conditions detected. "
            "Solar AC generation is expected to be approximately zero."
        )
    else:
        model = final_models[plant]
        prediction = float(model.predict(input_data)[0])

        # Prevent tiny negative numerical predictions.
        prediction = max(0.0, prediction)

        prediction_note = (
            f"Prediction generated using the {plant_info['model']} model "
            f"trained specifically for {plant}."
        )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    st.divider()

    st.success("Prediction generated successfully!")

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:
        st.metric(
            "Predicted Total AC Power",
            f"{prediction:,.2f} W"
        )

    with result_col2:
        st.metric(
            "Solar Plant",
            plant
        )

    with result_col3:
        st.metric(
            "Number of Inverters",
            plant_info["inverters"]
        )

    st.caption(prediction_note)

    # Equivalent average only — NOT individual inverter prediction.
    if prediction > 0:
        average_equivalent = prediction / plant_info["inverters"]

        st.info(
            f"🔌 The predicted **{prediction:,.2f} W** is the estimated "
            f"**total AC output of the plant** across its "
            f"**{plant_info['inverters']} inverters**. "
            f"That corresponds to about **{average_equivalent:,.2f} W "
            f"per inverter on average**, purely as an equivalent average."
        )
    else:
        st.info(
            f"🌙 The plant has **{plant_info['inverters']} inverters**, "
            "but under the selected low-light/night conditions, "
            "expected solar AC generation is approximately zero."
        )

    # Show the exact values sent to the model for transparency.
    with st.expander("🔍 View model input"):
        st.dataframe(input_data, use_container_width=True)
