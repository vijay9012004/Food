import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Food Waste Prediction", layout="centered")

st.title("🍽️ Food Waste Prediction App")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("global_food_wastage_dataset 1.csv")

dia = load_data()

# Encode categorical columns
le_country = LabelEncoder()
le_food = LabelEncoder()

dia['Country_enc'] = le_country.fit_transform(dia['Country'])
dia['Food_enc'] = le_food.fit_transform(dia['Food Category'])

# Independent & dependent variables
X = dia[['Country_enc', 'Year', 'Food_enc']]
y = dia['Total Waste (Tons)']

# Train model
model = LinearRegression()
model.fit(X, y)

st.subheader("📊 Enter Details")

# Streamlit inputs
country = st.selectbox("Select Country", le_country.classes_)
year = st.number_input("Enter Year", min_value=2000, max_value=2100, value=2020)
food = st.selectbox("Select Food Category", le_food.classes_)

# Encode user input
country_enc = le_country.transform([country])[0]
food_enc = le_food.transform([food])[0]

# Predict button
if st.button("Predict Food Waste"):
    prediction = model.predict([[country_enc, year, food_enc]])
    st.success(f"Predicted Total Waste: {prediction[0]:.2f} Tons")

st.markdown("---")
st.caption("Model: Linear Regression | Dataset: Global Food Wastage")
