import streamlit as st
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import io
import joblib
import os
import shap
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import model_from_json
import pickle

st.set_page_config(
    page_title="Loan Default Prediction — Project Showcase",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Load model components ---- #

@st.cache_resource
# def load_assets():
#     model = load_model('loan_default_model_legacy.h5', compile=False) # Keras model
#     scaler = joblib.load('scaler_legacy.pkl') # StandardScaler
#     feature_names = joblib.load('feature_names_legacy.pkl') # Saved feature names list
#     return model, scaler, feature_names

# model, scaler, feature_names = load_assets()

def load_keras_model_legacy(path_prefix):
    # Load model architecture from config.json
    with open(f'{path_prefix}/config.json', 'r') as f:
        config_json = f.read()

    model = model_from_json(config_json)

    # Load weights into the model
    model.load_weights(f'{path_prefix}/model.weights.h5')

    return model

def load_assets():
    # Load Keras model from the specified folder
    model = load_keras_model_legacy('loan_default_model_legacy')
    
    # Load scaler from current folder
    scaler = joblib.load('scaler_legacy.pkl')
    
    # Load feature names from current folder
    with open('feature_names_legacy.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    
    return model, scaler, feature_names

model, scaler, feature_names = load_assets()

# ---- Streamlit UI ---- #

st.title("🏠 Home Loan Default Risk Predictor")
st.write("Provide applicant details below to assess their default risk")

#Divide input features for simplicity
st.sidebar.header("User Input Settings")

user_data = {}
st.markdown("### Applicant Details")

for feat in feature_names:
    # Detect numerical vs categorical features using naming heuristics
    if feat.lower().startswith('amt') or 'days' in feat.lower() or 'cnt' in feat.lower():
        user_data[feat] = st.number_input(f"{feat}", value=0.0, step=1000.0)
    elif 'flag' in feat.lower():
        user_data[feat] = st.selectbox(f"{feat}", ['0', '1'])
    else:
        user_data[feat] = st.text_input(f"{feat}", value="")

# Create dataframe
input_df = pd.DataFrame([user_data])

# ---------- Preprocessing ----------
try:
    # Ensure correct column order and missing columns
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

# ---------- Explainability (optional) ----------
if st.button("📊 Show Feature Impact (SHAP)"):
    import shap
    explainer = shap.Explainer(model)
    shap_values = explainer(X_scaled)
    st.pyplot(shap.plots.waterfall(shap_values[0]))
