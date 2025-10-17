import streamlit as st
import numpy as np
import pandas as pd
import joblib
import pickle
import os
from tensorflow.keras.models import model_from_json

# ------------------- Streamlit Config -------------------
st.set_page_config(
    page_title="🏠 Loan Default Prediction — Project Showcase",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------- Load Model Components -------------------

@st.cache_resource(show_spinner="Loading model and scaler...")
def load_assets():
    """
    Load Keras model, scaler, feature list, and preset profiles.
    """

    base_path = os.path.join(os.path.dirname(__file__), "..", "models")

    # --- Model ---
    config_path = os.path.join(base_path, "config.json")
    weights_path = os.path.join(base_path, "model.weights.h5")
    if not (os.path.exists(config_path) and os.path.exists(weights_path)):
        st.error("❌ Model files not found in 'models' folder.")
        st.stop()

    with open(config_path, "r") as f:
        model_json = f.read()
    model = model_from_json(model_json)
    model.load_weights(weights_path)

    # --- Scaler and Features ---
    scaler_path = os.path.join(base_path, "scaler_legacy.pkl")
    features_path = os.path.join(base_path, "feature_names_legacy.pkl")

    try:
        scaler = joblib.load(scaler_path)
        with open(features_path, "rb") as f:
            feature_names = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading scaler or features: {e}")
        st.stop()

    # --- Preset Profiles ---
    high_risk_path = os.path.join(base_path, "high_risk_profile.pkl")
    low_risk_path = os.path.join(base_path, "low_risk_profile.pkl")

    try:
        with open(high_risk_path, "rb") as f:
            high_risk_profile = pickle.load(f)
        with open(low_risk_path, "rb") as f:
            low_risk_profile = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading preset profiles: {e}")
        st.stop()

    return model, scaler, feature_names, high_risk_profile, low_risk_profile


model, scaler, feature_names, high_risk_profile, low_risk_profile = load_assets()


# ------------------- Sidebar Presets -------------------
st.sidebar.header("User Input Settings")
preset_option = st.sidebar.selectbox(
    "Choose a preset profile",
    ["Custom Input", "High Risk", "Low Risk"]
)

# ------------------- Top 15 Features -------------------
top_features = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
    "REGION_RATING_CLIENT",
    "REGION_RATING_CLIENT_W_CITY",
    "HOUR_APPR_PROCESS_START",
    "FLAG_OWN_REALTY_Y"
]

st.title("🏠 Home Loan Default Risk Predictor")
st.write("Provide applicant details below to assess their default risk")

user_data = {}

st.markdown("### Applicant Details (Top 15 Features)")

for feat in top_features:
    # Pick default based on preset
    if preset_option == "High Risk":
        default_val = high_risk_profile.get(feat, 0)
    elif preset_option == "Low Risk":
        default_val = low_risk_profile.get(feat, 0)
    else:
        default_val = 0.0 if "flag" not in feat.lower() else 0

    # Float features
    if feat.lower().startswith("amt") or "days" in feat.lower() or "ext_source" in feat.lower():
        user_data[feat] = st.number_input(
            f"{feat}",
            value=float(default_val),
            step=1000.0 if "amt" in feat.lower() else 0.01
        )
    # Integer features (counts, hour)
    elif "cnt" in feat.lower() or "hour" in feat.lower():
        user_data[feat] = st.number_input(
            f"{feat}",
            value=int(default_val),
            step=1
        )
    # Flags
    elif "flag" in feat.lower():
        user_data[feat] = int(st.selectbox(f"{feat}", [0, 1], index=int(default_val)))
    else:
        user_data[feat] = st.number_input(
            f"{feat}",
            value=float(default_val),
            step=1.0
        )

# Fill remaining features with preset full profile if available
preset_profile = None
if preset_option == "High Risk":
    preset_profile = high_risk_profile
elif preset_option == "Low Risk":
    preset_profile = low_risk_profile

for feat in feature_names:
    if feat not in user_data:
        if preset_profile:
            user_data[feat] = preset_profile.get(feat, 0)
        else:
            user_data[feat] = 0

input_df = pd.DataFrame([user_data])

# ---------- Preprocessing ----------
try:
    input_df = input_df.reindex(columns=feature_names, fill_value=0)
    X_scaled = scaler.transform(input_df)
except Exception as e:
    st.error(f"Error during preprocessing: {e}")
    st.stop()

# ---------- Prediction ----------
if st.button("🔍 Predict Default Risk"):
    prob = model.predict(X_scaled)[0][0]
    prediction = "⚠️ Defaulter" if prob > 0.5 else "✅ Payer"

    st.subheader(f"Prediction: {prediction}")
    st.metric(label="Default Probability", value=f"{prob:.2%}")

    if prob > 0.7:
        st.warning("High Risk: This applicant has a strong likelihood of defaulting.")
    elif prob > 0.4:
        st.info("Moderate Risk: Review applicant details carefully.")
    else:
        st.success("Low Risk: Applicant seems safe for approval.")

# ------------------- Explainability (SHAP) -------------------
if st.button("📊 Show Feature Impact (SHAP)"):
    try:
        import shap
        explainer = shap.Explainer(model, X_scaled)
        shap_values = explainer(X_scaled)
        st.set_option("deprecation.showPyplotGlobalUse", False)
        import matplotlib.pyplot as plt
        st.pyplot(shap.plots.waterfall(shap_values[0], show=False))
    except Exception as e:
        st.error(f"SHAP explanation failed: {e}")
