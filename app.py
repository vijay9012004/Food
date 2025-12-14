import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Food Waste Prediction", layout="centered")
st.title("🍽️ Food Waste Prediction App")

# Upload dataset dynamically
uploaded_file = st.file_uploader("Upload CSV file", type="csv")
if uploaded_file is not None:
    dia = pd.read_csv(uploaded_file)
    
    # Encode categorical columns
    le_country = LabelEncoder()
    le_food = LabelEncoder()
    
    dia['Country_enc'] = le_country.fit_transform(dia['Country'])
    dia['Food_enc'] = le_food.fit_transform(dia['Food Category'])
    
    # Load pre-trained model
    uploaded_model = st.file_uploader("Upload trained model (food.pkl)", type="pkl")
    if uploaded_model is not None:
        model = pickle.load(uploaded_model)
        
        st.subheader("📊 Enter Details")
        country = st.selectbox("Select Country", le_country.classes_)
        year = st.number_input("Enter Year", min_value=2000, max_value=2100, value=2020)
        food = st.selectbox("Select Food Category", le_food.classes_)
        
        country_enc = le_country.transform([country])[0]
        food_enc = le_food.transform([food])[0]
        
        if st.button("Predict Food Waste"):
            prediction = model.predict([[country_enc, year, food_enc]])
            st.success(f"Predicted Total Waste: **{prediction[0]:.2f} Tons**")
else:
    st.info("Please upload the CSV file to continue.")

st.markdown("---")
st.caption("Model: Linear Regression | Dataset: Global Food Wastage")
