import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
import itertools

df=pd.read_csv("crop_yield.csv")

df=df.dropna()
df=df.drop_duplicates()
print(df.isnull().sum())
print(df)

df['yield']=df['Production'] / df['Area']

df_cleaned = df[(np.abs(stats.zscore(df[['yield', 'Area', 'Production', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']])) < 3).all(axis=1)]
print(f"Original shape: {df.shape}, After removing outliers: {df_cleaned.shape}")

print("yield correct")
print((df['yield'] == df['Yield']).sum())

print(df.info())

df_encoded=df.copy()
le=LabelEncoder()
for col in ["State","Season","Crop"]:
    df_encoded[col]=le.fit_transform(df_encoded[col])


df["Year_month"]=df["Crop_Year"]


X=df_encoded.drop(["yield"],axis=1)
y = df_encoded["yield"]

numeric_features = ['Area', 'Production', 'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Crop_Year']
scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[numeric_features] = scaler.fit_transform(X_scaled[numeric_features])


X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.8,random_state=42
)

model =LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

import matplotlib.pyplot as plt

# Plot for Linear Regression
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")
plt.title("Linear Regression: Actual vs Predicted Yield")
plt.grid(True)
plt.show()

