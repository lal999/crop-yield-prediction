import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# =========================
# Load Pretrained Model
# =========================
@st.cache_resource
def load_trained_model():
    return joblib.load("crop_model.pkl")

model_data = load_trained_model()
rf_model = model_data["model"]
label_encoders = model_data["label_encoders"]
scaler = model_data["scaler"]
numeric_features = model_data["numeric_features"]
X_test = model_data["X_test"]
y_test = model_data["y_test"]
y_pred = model_data["y_pred"]

# =========================
# Streamlit App
# =========================
st.set_page_config(page_title="Crop Yield Prediction", layout="wide")
st.title("🌾 Crop Yield Prediction")

# Sidebar
st.sidebar.header("📊 Navigation")
page = st.sidebar.radio("Go to:", ["Model Insights", "Predict New Yield"])

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

# =========================
# PAGE 2: Predict New Yield
# =========================
elif page == "Predict New Yield":
    st.subheader("🌱 Enter Farming Details")

    # User inputs
    state = st.selectbox("State", label_encoders['State_Name'].classes_)
    season = st.selectbox("Season", label_encoders['Season'].classes_)
    crop = st.selectbox("Crop", label_encoders['Crop'].classes_)
    area = st.number_input("Area (hectares)", min_value=0.0, step=0.1)
    production = st.number_input("Production (tons)", min_value=0.0, step=0.1)
    annual_rainfall = st.number_input("Annual Rainfall (mm)", min_value=0.0, step=0.1)
    fertilizer = st.number_input("Fertilizer Used (kg)", min_value=0.0, step=0.1)
    pesticide = st.number_input("Pesticide Used (kg)", min_value=0.0, step=0.1)
    crop_year = st.number_input("Crop Year", min_value=1900, max_value=2100, step=1)

    if st.button("🔍 Predict Yield"):
        # Encode categorical
        state_enc = label_encoders['State_Name'].transform([state])[0]
        season_enc = label_encoders['Season'].transform([season])[0]
        crop_enc = label_encoders['Crop'].transform([crop])[0]

        # Create input DataFrame
        input_df = pd.DataFrame([[state_enc, season_enc, crop_enc, area, production,
                                  annual_rainfall, fertilizer, pesticide, crop_year]],
                                columns=['State_Name', 'Season', 'Crop', 'Area', 'Production',
                                         'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Crop_Year'])

        # Scale numeric
        input_df[numeric_features] = scaler.transform(input_df[numeric_features])

        # Predict
        prediction = rf_model.predict(input_df)[0]
        st.success(f"✅ Predicted Crop Yield: **{prediction:.2f} tons/hectare**")
