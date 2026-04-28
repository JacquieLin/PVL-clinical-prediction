import streamlit as st
import joblib
import numpy as np
import pandas as pd
from joblib import load
import json

# 直接加载模型
model = load('RF_model.joblib')

# 加载特征名列表
with open('feature_names.json', 'r') as f:
    feature_names = json.load(f)

st.set_page_config(page_title="PVL Risk Prediction", page_icon="🧠", layout="centered")

st.title("🧠 White Matter Injury（WMI） Risk Prediction System")
st.write("Please enter the newborn's clinical and imaging information. The system will predict the risk of developing WMI.")

# Input features
col1, col2 = st.columns(2)

with col1:
    #GA = st.number_input('Gestational Age (days)', min_value=140, max_value=330, step=1)
    st.markdown("**Gestational Age (GA)**")

col_ga1, col_ga2 = st.columns(2)

with col_ga1:
    GA_weeks = st.number_input('Weeks', min_value=20, max_value=45, step=1)

with col_ga2:
    GA_days = st.number_input('Days', min_value=0, max_value=6, step=1)
    Age = st.number_input('Current Age (months)', min_value=0, max_value=24, step=1)

with col2:
    NH = st.selectbox('Neonatal hypoglycaemia(NH)', ['No', 'Yes'])
    DSH = st.selectbox('Definitive subventricular haemorrhage (DSH)', ['No', 'Yes'])
    DM = st.selectbox('Delayed Myelination (DM)', ['No', 'Yes'])
    ASALV = st.selectbox('Abnormal Signal Around Lateral Ventricles (ASALV)', ['No', 'Yes'])

# Convert Yes/No to binary
def yes_no_to_binary(x):
    return 1 if x == 'Yes' else 0

NH_bin = yes_no_to_binary(NH)
DSH_bin = yes_no_to_binary(DSH)
DM_bin = yes_no_to_binary(DM)
ASALV_bin = yes_no_to_binary(ASALV)
GA_total_days = GA_weeks * 7 + GA_days

# Prediction button
if st.button('🔍 Predict WMI Risk'):
    # 创建特征数组
    features = np.array([[GA_total_days, NH_bin, DSH_bin, Age, DM_bin, ASALV_bin]])
    

    # 预测
    prob = model.predict_proba(features)[0, 1]  # 获取 PVL 的概率
    st.subheader(f"🩺 Predicted Risk of Developing WMI: {prob*100:.2f}%")

    # 根据概率显示不同信息
    if prob >= 0.5:
        st.error("⚠️ High risk! Further evaluation and intervention are recommended.")
    else:
        st.success("✅ Low risk. Please continue regular follow-up.")
