import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Machine Failure Predictor", page_icon="⚙️")
st.title("⚙️ Industrial Machine Failure Predictor")
st.caption("Learn Depth Academy Track 1 Capstone — Problem 21")

try:
    model = joblib.load("artifacts/machine_failure_model.joblib")
except FileNotFoundError:
    st.error("Model not found. Run `python train.py` first.")
    st.stop()

st.write("Enter machine operating conditions to estimate the failure class.")

product_type = st.selectbox("Product type", ["L", "M", "H"])
air_temperature = st.number_input("Air temperature [K]", value=300.0, step=0.1)
process_temperature = st.number_input("Process temperature [K]", value=310.0, step=0.1)
rotational_speed = st.number_input("Rotational speed [rpm]", value=1500.0, step=10.0)
torque = st.number_input("Torque [Nm]", value=40.0, step=0.5)
tool_wear = st.number_input("Tool wear [min]", value=100.0, step=1.0)

if st.button("Predict"):
    row = pd.DataFrame([{
    "product_type": product_type,
    "Air temperature": air_temperature,
    "Process temperature": process_temperature,
    "Rotational speed": rotational_speed,
    "Torque": torque,
    "Tool wear": tool_wear
}])

    pred = int(model.predict(row)[0])
    prob = float(model.predict_proba(row)[0, 1])

    if pred == 1:
        st.error(f"Predicted class: MACHINE FAILURE RISK (probability {prob:.2%})")
    else:
        st.success(f"Predicted class: NO MACHINE FAILURE (probability {prob:.2%})")

st.info("This is an educational prototype, not a production safety system.")
