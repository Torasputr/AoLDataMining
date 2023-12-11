import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
path = "combined_datasett.csv"
df = pd.read_csv(path)

# Replace '---' with NaN
df.replace('---', pd.NA, inplace=True)

# Convert selected columns to numeric (integer)
columns_to_convert = ['pm10', 'pm25', 'so2', 'co', 'o3', 'no2']
for column in columns_to_convert:
    df[column] = pd.to_numeric(df[column], errors='coerce').astype('Int64')

# encode label
label_encoder = LabelEncoder()
df['categori_encoded'] = label_encoder.fit_transform(df['categori'])

# Print mapping label
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Label Mapping:", label_mapping)

# Select relevant data
df = df[['pm10', 'pm25', 'so2', 'co', 'o3', 'no2', 'categori_encoded']]

# Preprocess - Input missing data
imputer = SimpleImputer(strategy='mean')
df[['pm10', 'pm25', 'so2', 'co', 'o3', 'no2']] = imputer.fit_transform(df[['pm10', 'pm25', 'so2', 'co', 'o3', 'no2']])

# Drop duplicates
df.drop_duplicates(inplace=True)

# Split features (X) and target (y)
X = df.drop(columns=['categori_encoded'])
y = df['categori_encoded']

# Scaling
standard_scaler = StandardScaler()
X_scaled = standard_scaler.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Balancing data with oversampling SMOTE
smote_balancing = SMOTE(random_state=0)
X_balanced, y_balanced = smote_balancing.fit_resample(X_train, y_train)

# Initialize RF
rf_classifier = RandomForestClassifier(random_state=42, oob_score=True)

# Train
rf_classifier.fit(X_balanced, y_balanced)

# Streamlit UI
st.title("Air Quality Category Prediction")
st.sidebar.header("Input New Data")

# Input for new data
pm10 = st.sidebar.number_input("Enter PM10 level", value=20)
pm25 = st.sidebar.number_input("Enter PM2.5 level", value=10)
so2 = st.sidebar.number_input("Enter SO2 level", value=1)
co = st.sidebar.number_input("Enter CO level", value=0.5)
o3 = st.sidebar.number_input("Enter O3 level", value=30)
no2 = st.sidebar.number_input("Enter NO2 level", value=10)

# Create a dataframe for the new data
new_data = pd.DataFrame({'pm10': [pm10], 'pm25': [pm25], 'so2': [so2], 'co': [co], 'o3': [o3], 'no2': [no2]})

# Scaling for the new data
new_data_scaled = standard_scaler.transform(new_data)

# Prediction
prediction = rf_classifier.predict(new_data_scaled)
prediction_label = label_encoder.inverse_transform(prediction)[0]

# Display the result
st.subheader("Prediction:")
st.write(f"The predicted air quality category is: {prediction_label}")
