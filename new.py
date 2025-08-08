import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
import matplotlib.pyplot as plt

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("crop_yield.csv")  # Ensure this file is in the same folder

# -----------------------------
# PAGE TITLE
# -----------------------------
st.title("🌾 AI-Powered Crop Yield Prediction")
st.markdown("""
An AI solution to predict crop yields and optimize farming practices 
based on **weather patterns, soil health, and crop management techniques**.  
**Goal:** Increase agricultural productivity & reduce hunger in rural communities.
""")

# -----------------------------
# DATA CLEANING
# -----------------------------
df = df.dropna().drop_duplicates()
df["yield"] = df["Production"] / df["Area"]

# Remove outliers
df_cleaned = df[(np.abs(stats.zscore(
    df[['yield', 'Area', 'Production', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']])) < 3).all(axis=1)]

# Encode categorical columns
df_encoded = df_cleaned.copy()
label_encoders = {}
for col in ["State", "Season", "Crop"]:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le

# Add Year_month (if needed for time-related analysis)
df_encoded["Year_month"] = df_encoded["Crop_Year"]

# -----------------------------
# MODEL TRAINING
# -----------------------------
# Drop 'yield' column (target) and any extras like 'Yield' if exists
X = df_encoded.drop(columns=["yield", "Yield"], errors="ignore")
y = df_encoded["yield"]

# Standardize numeric features
numeric_features = ['Area', 'Production', 'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Crop_Year']
scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[numeric_features] = scaler.fit_transform(X_scaled[numeric_features])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# -----------------------------
# MODEL PERFORMANCE
# -----------------------------
st.subheader("📊 Model Performance")
st.write(f"**Mean Squared Error (MSE):** {mean_squared_error(y_test, y_pred):.2f}")
st.write(f"**R² Score:** {r2_score(y_test, y_pred):.2f}")

# -----------------------------
# VISUALIZATION
# -----------------------------
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, alpha=0.5, color='blue')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax.set_xlabel("Actual Yield")
ax.set_ylabel("Predicted Yield")
ax.set_title("Linear Regression: Actual vs Predicted Yield")
st.pyplot(fig)

# -----------------------------
# FUTURE PREDICTION FORM
# -----------------------------
st.subheader("🌱 Predict Future Crop Yield")
st.markdown("Enter expected farming conditions:")

col1, col2 = st.columns(2)

with col1:
    state = st.selectbox("State", df_cleaned["State"].unique())
    season = st.selectbox("Season", df_cleaned["Season"].unique())
    crop = st.selectbox("Crop", df_cleaned["Crop"].unique())
    crop_year = st.number_input("Crop Year", min_value=2025, max_value=2100, step=1)

with col2:
    area = st.number_input("Area (hectares)", min_value=0.1, step=0.1)
    production = st.number_input("Production (tonnes)", min_value=0.1, step=0.1)
    annual_rainfall = st.number_input("Annual Rainfall (mm)", min_value=0.0, step=0.1)
    fertilizer = st.number_input("Fertilizer Usage (kg/ha)", min_value=0.0, step=0.1)
    pesticide = st.number_input("Pesticide Usage (kg/ha)", min_value=0.0, step=0.1)

if st.button("🔍 Predict Yield"):
    # Encode inputs
    state_enc = label_encoders["State"].transform([state])[0]
    season_enc = label_encoders["Season"].transform([season])[0]
    crop_enc = label_encoders["Crop"].transform([crop])[0]

    # Create input matching X.columns
    input_data = pd.DataFrame([[
        crop_enc, crop_year, season_enc, state_enc, area,
        production, annual_rainfall, fertilizer, pesticide, crop_year
    ]], columns=X.columns)

    # Scale numeric features
    input_data[numeric_features] = scaler.transform(input_data[numeric_features])

    # Predict
    predicted_yield = model.predict(input_data)[0]
    st.success(f"🌾 Predicted Yield: **{predicted_yield:.2f} tonnes/hectare**")
