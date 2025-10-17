import streamlit as st
import numpy as np
import pandas as pd
import joblib
import pickle
import shap
from tensorflow.keras.models import model_from_json

# ------------------- Streamlit Config -------------------
st.set_page_config(
    page_title="🏠 Loan Default Prediction — Project Showcase",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------- Load Model Components -------------------

@st.cache_resource(show_spinner="Loading model and scaler...")
def load_assets(path_prefix="loan_default_model_legacy"):
    """
    Load the trained model, scaler, and feature list.
    Uses the lightweight .weights.h5 + config.json format for compatibility.
    """

    # --- Load model architecture ---
    config_path = f"{path_prefix}/config.json"
    weights_path = f"{path_prefix}/model.weights.h5"

    if not (os.path.exists(config_path) and os.path.exists(weights_path)):
        st.error("❌ Model files not found. Please check the paths.")
        st.stop()

    with open(config_path, "r") as f:
        model_json = f.read()
    model = model_from_json(model_json)
    model.load_weights(weights_path)

    # --- Load scaler and features ---
    try:
        scaler = joblib.load("scaler_legacy.pkl")
        with open("feature_names_legacy.pkl", "rb") as f:
            feature_names = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading scaler or feature names: {e}")
        st.stop()

    return model, scaler, feature_names


model, scaler, feature_names = load_assets()

# ------------------- UI Layout -------------------
st.title("🏠 Home Loan Default Risk Predictor")
st.write("Provide applicant details below to assess their likelihood of defaulting.")

st.sidebar.header("User Input Settings")

# Collect user inputs
user_data = {}
st.markdown("### Applicant Details")

for feat in feature_names:
    feat_lower = feat.lower()

    if feat_lower.startswith("amt") or "days" in feat_lower or "cnt" in feat_lower:
        user_data[feat] = st.number_input(f"{feat}", value=0.0, step=1000.0)
    elif "flag" in feat_lower:
        user_data[feat] = int(st.selectbox(f"{feat}", ["0", "1"]))
    else:
        # Fallback: numeric input for simplicity; text inputs can break scaling
        user_data[feat] = st.number_input(f"{feat}", value=0.0)

input_df = pd.DataFrame([user_data])

# ------------------- Preprocessing -------------------
try:
    # Ensure correct order and handle missing features
    input_df = input_df.reindex(columns=feature_names, fill_value=0)
    X_scaled = scaler.transform(input_df)
except Exception as e:
    st.error(f"Error during preprocessing: {e}")
    st.stop()

# ------------------- Prediction -------------------
if st.button("🔍 Predict Default Risk"):
    try:
        prob = float(model.predict(X_scaled)[0][0])
        prediction = "⚠️ Defaulter" if prob > 0.5 else "✅ Payer"

        st.subheader(f"Prediction: {prediction}")
        st.metric(label="Default Probability", value=f"{prob:.2%}")

        # Add intuitive feedback
        if prob > 0.7:
            st.warning("High Risk: This applicant has a strong likelihood of defaulting.")
        elif prob > 0.4:
            st.info("Moderate Risk: Review applicant details carefully.")
        else:
            st.success("Low Risk: Applicant seems safe for approval.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ------------------- Explainability (SHAP) -------------------
if st.button("📊 Show Feature Impact (SHAP)"):
    try:
        # For performance, explain only one instance
        explainer = shap.Explainer(model, X_scaled)
        shap_values = explainer(X_scaled)
        
        # Use Streamlit-native rendering
        st.set_option("deprecation.showPyplotGlobalUse", False)
        st.pyplot(shap.plots.waterfall(shap_values[0], show=False))
    except Exception as e:
        st.error(f"SHAP explanation failed: {e}")
