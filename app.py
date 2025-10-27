
import streamlit as st
import joblib

# ===============================================================
# Load Model and Encoder
# ===============================================================
model = joblib.load("C://Users//Komal//Desktop//Unemployment predictor//unemployment_model_state.pkl")
encoder = joblib.load("C://Users//Komal//Desktop//Unemployment predictor//state_encoder.pkl")

# ===============================================================
# Streamlit Web App
# ===============================================================
st.set_page_config(page_title="Unemployment Risk Predictor", page_icon="📊")

st.title("📊 U.S. Unemployment Likelihood Predictor")
st.write("Predict the likelihood of being unemployed based on your state of residence.")

# Dropdown for selecting a state
state = st.selectbox(
    "Select your state:",
    options=sorted(encoder.classes_.tolist())
)

# Predict
if st.button("Predict Likelihood"):
    state_encoded = encoder.transform([state])
    prediction = model.predict_proba([[state_encoded[0]]])[0][1] * 100
    st.success(f"Estimated unemployment likelihood in {state}: {prediction:.2f}%")

