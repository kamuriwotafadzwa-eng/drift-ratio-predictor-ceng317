import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

st.set_page_config(
    page_title="RC Bridge Pier Drift Ratio Prediction",
    page_icon="🌉",
    layout="wide"
)

# Load model and scaler
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model("dnn_drift_ratio_tuned.keras")
    scaler = joblib.load("scaler_tuned.pkl")
    return model, scaler

try:
    model, scaler = load_assets()
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

st.title("🌉 Drift Ratio Prediction in RC Bridge Piers")
st.markdown("### Research & Innovation Skills Project | Group 13")
st.write("Input the structural and material properties below to predict the **Collapse Drift Ratio (%)**.")

st.sidebar.header("Pier Input Parameters")

def user_input_features():
    fc = st.sidebar.number_input("Concrete Compressive Strength (fc') [MPa]", min_value=10.0, max_value=120.0, value=35.0, step=0.5)
    fyl = st.sidebar.number_input("Yield Strength of Long. Steel (fyl) [MPa]", min_value=200.0, max_value=700.0, value=420.0, step=5.0)
    rho_l = st.sidebar.number_input("Longitudinal Reinforcement Ratio (ρl)", min_value=0.001, max_value=0.100, value=0.020, step=0.001, format="%.3f")
    vol_rho_t = st.sidebar.number_input("Transverse Volumetric Ratio (Vol.ρt)", min_value=0.000, max_value=0.150, value=0.010, step=0.001, format="%.3f")
    h = st.sidebar.number_input("Section Depth (h) [mm]", min_value=50.0, max_value=2000.0, value=400.0, step=10.0)
    L = st.sidebar.number_input("Pier Length (L) [mm]", min_value=100.0, max_value=10000.0, value=1500.0, step=50.0)
    cover = st.sidebar.number_input("Concrete Cover [mm]", min_value=0.0, max_value=100.0, value=25.0, step=1.0)
    axial_ratio = st.sidebar.number_input("Axial Load Ratio", min_value=0.00, max_value=1.00, value=0.15, step=0.01)
    
    data = {
        "fc' (MPa)": fc,
        "fyl (MPa)": fyl,
        "ρl": rho_l,
        "Vol.ρt": vol_rho_t,
        "h (mm) Depth": h,
        "L (mm)": L,
        "cover (mm)": cover,
        "Axial load ratio": axial_ratio
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Specified Input Features")
    st.dataframe(input_df, use_container_width=True)

with col2:
    st.subheader("Prediction")
    if st.button("Predict Drift Ratio", type="primary"):
        scaled_input = scaler.transform(input_df.values)
        prediction = model.predict(scaled_input)
        predicted_drift = float(prediction[0][0])
        
        st.metric(
            label="Predicted Collapse Drift Ratio",
            value=f"{predicted_drift:.2f} %"
        )
        
        if predicted_drift < 2.0:
            st.warning("Low Drift Capacity: High risk under seismic loading.")
        elif predicted_drift < 5.0:
            st.info("Moderate Drift Capacity: Moderate seismic resistance.")
        else:
            st.success("High Drift Capacity: Excellent seismic performance.")