import streamlit as st
import joblib
import numpy as np

# Load the trained model
with open('RF_model.pkl', 'rb') as f:
    model = joblib.load(f)

st.set_page_config(page_title="PVL Risk Prediction", page_icon="🧠", layout="centered")

st.title("🧠 PVL (Periventricular Leukomalacia) Risk Prediction System")
st.write("Please enter the newborn's clinical and imaging information. The system will predict the risk of developing PVL.")

# Input features
col1, col2 = st.columns(2)

with col1:
    GA = st.number_input('Gestational Age (days)', min_value=140, max_value=330, step=1)
    BW = st.number_input('Birth Weight (kg)', min_value=0.3, max_value=6.0, step=0.01)
    Age = st.number_input('Current Age (months)', min_value=0, max_value=24, step=1)

with col2:
    NIH = st.selectbox('Neonatal Intracranial Hemorrhage (NIH)', ['No', 'Yes'])
    MNI = st.selectbox('Maternal or Neonatal Infection (MNI)', ['No', 'Yes'])
    ELV = st.selectbox('Enlarged Lateral Ventricles (ELV)', ['No', 'Yes'])
    DM = st.selectbox('Delayed Myelination (DM)', ['No', 'Yes'])
    ASALV = st.selectbox('Abnormal Signal Around Lateral Ventricles (ASALV)', ['No', 'Yes'])

# Convert Yes/No to binary
def yes_no_to_binary(x):
    return 1 if x == 'Yes' else 0

NIH_bin = yes_no_to_binary(NIH)
MNI_bin = yes_no_to_binary(MNI)
ELV_bin = yes_no_to_binary(ELV)
DM_bin = yes_no_to_binary(DM)
ASALV_bin = yes_no_to_binary(ASALV)

# Prediction button
if st.button('🔍 Predict PVL Risk'):
    features = np.array([[GA, BW, NIH_bin, MNI_bin, Age, ELV_bin, DM_bin, ASALV_bin]])
    prob = model.predict_proba(features)[0, 1]  # Get the probability of PVL
    st.subheader(f"🩺 Predicted Risk of Developing PVL: {prob*100:.2f}%")

    if prob >= 0.5:
        st.error("⚠️ High risk! Further evaluation and intervention are recommended.")
    else:
        st.success("✅ Low risk. Please continue regular follow-up.")
