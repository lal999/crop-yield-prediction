import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
import matplotlib.pyplot as plt

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Crop Yield Prediction", layout="wide")
st.title("🌾 Crop Yield Prediction")
# Sidebar
st.sidebar.header("📊 Navigation")
page = st.sidebar.radio("Go to:", ["Model Insights", "Predict New Yield"])
# Sidebar - extra info
st.sidebar.markdown("---")
st.sidebar.subheader("📜 How to Use")
st.sidebar.info("""
1. **Go to 'Model Insights'** → See which features affect yield the most.  
2. **Go to 'Predict New Yield'** → Enter crop, state, year, and inputs.  
3. **Get prediction instantly** with our AI model.  
💡 Tip: Adjust values to see how yield changes!
""")

st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ About This App")
st.sidebar.write("""
This AI tool uses **Random Forest Regression** trained on historical crop yield data.  
It predicts yield based on **weather, crop type, and input usage**.
""")

st.sidebar.markdown("---")
st.sidebar.subheader("📂 Dataset Details")
st.sidebar.write("""
- **Source:** Historical agricultural data  
- **Features:** Rainfall, fertilizer, pesticide, crop type, season, state  
- **Target:** Yield (tonnes/hectare)
""")

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Model Specs")
st.sidebar.write("""
- Algorithm: RandomForestRegressor  
- Trees: 600  
- Max Depth: 22  
- Optimized for speed and accuracy
""")

st.sidebar.markdown("---")
st.sidebar.success("🌱 Try changing values in 'Predict New Yield' to see trends!")



# -----------------------------
# LOAD DATA (CACHE)
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("crop_yield.csv")
    df = df.dropna().drop_duplicates()
    df["yield"] = df["Production"] / df["Area"]
    # Keep slightly more data (outlier threshold 3.5 instead of 3)
    df_cleaned = df[(np.abs(stats.zscore(
        df[['yield', 'Area', 'Production', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']])) < 3.5).all(axis=1)]
    return df_cleaned

df_cleaned = load_data()

# -----------------------------
# ENCODE + FEATURE ENGINEERING
# -----------------------------
@st.cache_data
def preprocess_data(df_cleaned):
    df_encoded = df_cleaned.copy()
    label_encoders = {}
    for col in ["State", "Season", "Crop"]:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le

    # Remove leakage
    X = df_encoded.drop(columns=["yield", "Yield", "Production", "Area"], errors="ignore")
    y = df_encoded["yield"]

    # Log-transform
    for col in ['Annual_Rainfall', 'Fertilizer', 'Pesticide']:
        X[col] = np.log1p(X[col])

    # Feature engineering
    X["Rainfall_Fertilizer"] = X["Annual_Rainfall"] * X["Fertilizer"]
    X["Rainfall2"] = X["Annual_Rainfall"] ** 2
    X["Fertilizer2"] = X["Fertilizer"] ** 2

    # Scale
    numeric_features = ['Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Crop_Year',
                        'Rainfall_Fertilizer', 'Rainfall2', 'Fertilizer2']
    scaler = StandardScaler()
    X[numeric_features] = scaler.fit_transform(X[numeric_features])

    return X, y, label_encoders, scaler, numeric_features

X, y, label_encoders, scaler, numeric_features = preprocess_data(df_cleaned)

# -----------------------------
# TRAIN MODEL (CACHE)
# -----------------------------
@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf_model = RandomForestRegressor(
        n_estimators=600,
        max_depth=22,
        min_samples_split=3,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        max_samples=0.9,  # faster without losing much accuracy
        random_state=42,
        n_jobs=-1
    )

    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)

    return rf_model, X_test, y_test, y_pred

rf_model, X_test, y_test, y_pred = train_model(X, y)

# -----------------------------
# PAGE TITLE
# -----------------------------
st.title("🌾 AI-Powered Crop Yield Prediction (Optimized & Fast)")
st.markdown("""
An AI tool to predict crop yields based on **weather, soil, and crop management data**.  
Optimized for speed ⚡ and accuracy 📈.
""")

# -----------------------------
# PERFORMANCE METRICS
# -----------------------------
st.subheader("📊 Model Performance")
st.write(f"**MSE:** {mean_squared_error(y_test, y_pred):.4f}")
st.write(f"**R² Score:** {r2_score(y_test, y_pred):.4f}")

# =========================
# PAGE 1: Model Insights
# =========================
if page == "Model Insights":
    st.subheader("📈 Model Insights")

    # Create side-by-side columns for graphs
    col1, col2 = st.columns(2)

    # ---- Feature Importance ----
    with col1:
        st.markdown("### Feature Importance")
        feature_importance = pd.Series(rf_model.feature_importances_, index=X_test.columns)
        fig1, ax1 = plt.subplots()
        feature_importance.sort_values(ascending=False).plot(kind='bar', ax=ax1)
        ax1.set_ylabel("Importance")
        ax1.set_xlabel("Features")
        st.pyplot(fig1)

    # ---- Scatter Plot: Predicted vs Actual ----
    with col2:
        st.markdown("### Predicted vs Actual Yield")
        fig2, ax2 = plt.subplots()
        ax2.scatter(y_test, y_pred, alpha=0.6, color="blue")
        ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax2.set_xlabel("Actual Yield")
        ax2.set_ylabel("Predicted Yield")
        ax2.set_title("Predicted vs Actual")
        st.pyplot(fig2)

# -----------------------------
# PREDICTION SCENARIO
# -----------------------------
st.subheader("🌱 Predict Future Crop Yield")
col1, col2 = st.columns(2)
with col1:
    state = st.selectbox("State", df_cleaned["State"].unique())
    season = st.selectbox("Season", df_cleaned["Season"].unique())
    crop = st.selectbox("Crop", df_cleaned["Crop"].unique())
    crop_year = st.number_input("Crop Year", min_value=2025, max_value=2100, step=1)
with col2:
    annual_rainfall = st.number_input("Annual Rainfall (mm)", min_value=0.0, step=0.1)
    fertilizer = st.number_input("Fertilizer Usage (kg/ha)", min_value=0.0, step=0.1)
    pesticide = st.number_input("Pesticide Usage (kg/ha)", min_value=0.0, step=0.1)

if st.button("🔍 Predict Yield"):
    state_enc = label_encoders["State"].transform([state])[0]
    season_enc = label_encoders["Season"].transform([season])[0]
    crop_enc = label_encoders["Crop"].transform([crop])[0]

    input_data = pd.DataFrame([[
        crop_enc, crop_year, season_enc, state_enc,
        np.log1p(annual_rainfall), np.log1p(fertilizer), np.log1p(pesticide),
        np.log1p(annual_rainfall) * np.log1p(fertilizer),
        np.log1p(annual_rainfall) ** 2,
        np.log1p(fertilizer) ** 2
    ]], columns=X.columns)

    input_data[numeric_features] = scaler.transform(input_data[numeric_features])

    predicted_yield = rf_model.predict(input_data)[0]
    st.success(f"🌾 Predicted Yield: **{predicted_yield:.2f} tonnes/hectare**")

    