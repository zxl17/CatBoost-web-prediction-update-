import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import shap
import matplotlib.pyplot as plt


st.set_page_config(page_title="Severe pneumonia in children", layout="wide")

st.markdown("""
    <style>
        @font-face {
            font-family: 'Times New Roman';
        }
        /* 全局字体设置 */
        html, body, [class*="css"], div, p, h1, h2, h3, h4, h5, h6, 
        .stMarkdown, .stButton, .stTable, .stMetric, .stAlert, .stInfo, 
        .stSuccess, .stWarning, .stError, .stSelectbox, .stMultiSelect, 
        .stTextInput, .stNumberInput, .stDateInput, .stTimeInput, 
        .stHeader, .stSidebar, .stTabs, .stTab, .stDataFrame,
        div[data-testid="stMetricValue"], div[data-testid="stMetricLabel"],
        div[data-baseweb], button, input, select, textarea {
            font-family: 'Times New Roman', serif !important;
        }
        /* 确保表格内容也使用Times New Roman */
        .dataframe td, .dataframe th {
            font-family: 'Times New Roman', serif !important;
        }
    </style>
""", unsafe_allow_html=True)

plt.rcParams['font.family'] = 'Times New Roman'


@st.cache_resource
def load_model():
    model = load('model.pkl') 
    return model

model = load_model()


st.title("Severe pneumonia in children")
st.write("Please input patient's clinical indicators:")


col1, col2, col3 = st.columns(3)

with col1:
    cl = st.number_input('Chloride (Cl) (mmol/L)', value=None, format="%.2f")
    glu = st.number_input('Glucose (GLU) (mmol/L)', value=None, format="%.2f")
    dbil = st.number_input('Direct Bilirubin (DBil) (μmol/L)', value=None, format="%.2f")
    ldh = st.number_input('Lactate Dehydrogenase (LDH) (IU/L)', value=None, format="%.2f")

with col2:
    bun_scr = st.number_input('BUN/SCr Ratio', value=None, format="%.2f")
    che = st.number_input('Cholinesterase (CHE) (IU/L)', value=None, format="%.2f")
    ibil = st.number_input('Indirect Bilirubin (IBil) (μmol/L)', value=None, format="%.2f")
    ua = st.number_input('Uric Acid (UA) (μmol/L)', value=None, format="%.2f")

with col3:
    pdw = st.number_input('Platelet Distribution Width (PDW) (fL)', value=None, format="%.2f")
    ggt = st.number_input('Gamma - Glutamyl Transferase (GGT) (IU/L)', value=None, format="%.2f")
    ly_pct = st.number_input('Lymphocyte Ratio (LY%) (%)', value=None, format="%.2f")

if st.button('Predict'):

    input_data = pd.DataFrame([[cl, glu, dbil, ldh, bun_scr, che, ibil, ua, pdw, ggt, ly_pct]], 
                              columns=['Cl', 'GLU', 'DBil', 'LDH', 'BUN/SCr', 'CHE', 'IBil', 'UA', 'PDW', 'GGT', 'LY%'])

    prediction = model.predict_proba(input_data)[0]
    
    st.write("---")
    st.subheader("Prediction Results")
    
    st.metric(
        label="Risk of Severe Pneumonia",
        value=f"{prediction[1]:.1%}",
        delta=None
    )
    
    risk_level = "High Risk" if prediction[1] > 0.5 else "Low Risk"
    st.info(f"Risk Level: {risk_level}")

    st.write("---")
    st.subheader("Model Interpretation")
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)
    
    plt.figure(figsize=(15, 6))
    shap.force_plot(
        explainer.expected_value,
        shap_values[0],
        input_data.iloc[0],
        matplotlib=True,
        show=False,
        text_rotation=0
    )
    plt.gcf().set_facecolor('white')
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)  
    st.pyplot(plt)
    plt.close()

    st.write("---")
    st.subheader("Feature Contribution Analysis")
    
    feature_importance = pd.DataFrame({
        'No.': range(1, 12),  
        'Feature': input_data.columns,
        'SHAP Value': np.abs(shap_values[0])
    }).sort_values('SHAP Value', ascending=False)
    
    feature_importance = feature_importance.reset_index(drop=True)
    feature_importance['No.'] = range(1, 12)
    
    st.table(feature_importance.set_index('No.', drop=True))

st.write("---")
st.markdown("""
### Instructions:
1. Enter the children's clinical indicators in the input fields above.
2. Click the "Predict" button to get results.
3. The system will display the risk of severe pneumonia and risk level.
4. SHAP values show how each feature contributes to the prediction.
""")


st.sidebar.title("Model Information")
st.sidebar.info("""
- Model Type: CatBoost Classifier
- Training Data: Clinical Data
- Target Variable: Severe Pneumonia
- Number of Features: 11 Clinical Indicators
""")


st.sidebar.title("Feature Description")
st.sidebar.markdown("""
- Cl: 60-150 (mmol/L)
- GLU: 1-24 (mmol/L)
- DBil: 0-140 (μmol/L)
- LDH: 40-5800 (IU/L)
- BUN/SCr: 4-7600
- CHE: 900-27000 (IU/L)
- IBil: 0-390 (μmol/L)
- UA: 5-1300 (μmol/L)
- PDW: 6-26 (fL)
- GGT:1-1200 (IU/L)
- LY%: 0-100 (%)
""")