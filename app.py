import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Food Waste Prediction", layout="centered")

st.title("🍽️ Food Waste Prediction App")

# Absolute path
BASE_DIR = os.path.dirname(__file__)

# Load dataset (for dropdowns)
@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(BASE_DIR, "global_food_wastage_dataset 1.csv"))

dia = load_data()

# Encode categorical columns
le_country = LabelEncoder()
le_food = LabelEncoder()

dia['Country_enc'] = le_country.fit_transform(dia['Country'])
dia['Food_enc'] = le_food.fit_transform(dia['Food Category'])

# Load trained model
@st.cache_resource
def load_model():
    with open(os.path.join(BASE_DIR, "food.pkl"), "rb") as f:
        return pickle.load(f)

model = load_model()

st.subheader("📊 Enter Details")

country = st.selectbox("Select Country", le_country.classes_)
year = st.number_input("Enter Year", min_value=2000, max_value=2100, value=2020)
food = st.selectbox("Select Food Category", le_food.classes_)

country_enc = le_country.transform([country])[0]
food_enc = le_food.transform([food])[0]

if st.button("Predict Food Waste"):
    prediction = model.predict([[country_enc, year, food_enc]])
    st.success(f"Predicted Total Waste: **{prediction[0]:.2f} Tons**")

st.markdown("---")
st.caption("Model: Linear Regression | Dataset: Global Food Wastage")
