import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(page_title="Salary Predictor", page_icon="💰", layout="centered")

st.title("💰 Salary Predictor")
st.markdown("### Machine Learning based Salary Prediction App")

# Load the model
@st.cache_resource
def load_model():
    return joblib.load('models/salary_model.pkl')

model = load_model()

# Input fields
col1, col2 = st.columns(2)

with col1:
    experience = st.number_input("Years of Experience", min_value=0.0, max_value=30.0, value=5.0, step=0.5)
    education_level = st.selectbox("Education Level", 
                                 options=[1, 2, 3],
                                 format_func=lambda x: "Bachelor" if x==1 else "Master" if x==2 else "PhD or Higher")

with col2:
    city_tier = st.selectbox("City Tier", 
                           options=[1, 2, 3],
                           format_func=lambda x: "Tier 1 (Metro)" if x==1 else "Tier 2" if x==2 else "Tier 3 (Small City)")

# Prediction button
if st.button("🔮 Predict Salary", type="primary"):
    input_data = pd.DataFrame({
        'experience': [experience],
        'education_level': [education_level],
        'city_tier': [city_tier]
    })
    
    prediction = model.predict(input_data)[0]
    
    st.success(f"**Predicted Monthly Salary: ₹{prediction:,.2f}**")
    
    # Additional insights
    st.info(f"""
    **Breakdown:**
    - Experience: {experience} years
    - Education: {"Bachelor" if education_level==1 else "Master" if education_level==2 else "PhD"}
    - City Tier: {city_tier}
    """)

# Footer
st.markdown("---")
st.caption("Built as a beginner Machine Learning project | Salary Predictor using Linear Regression")