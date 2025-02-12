import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# Load dataset
@st.cache_data
def load_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df, data

df, data = load_data()

# Preprocess the data
selected_features = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
    'area error', 'worst radius', 'worst texture', 'worst perimeter', 'worst area'
]
X = df[selected_features]
y = df['target']

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train ANN Model
model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.001, learning_rate='adaptive', max_iter=500, random_state=42)
model.fit(X_train, y_train)

# Streamlit App Interface
st.title("Breast Cancer Prediction App")
st.write("This app uses a trained **Artificial Neural Network (ANN)** model to predict whether a tumor is **Benign** or **Malignant**.")

# Sidebar for user input
st.sidebar.header("Enter Feature Values")

def get_user_input():
    user_data = {}
    for feature in selected_features:
        user_data[feature] = st.sidebar.slider(feature, float(df[feature].min()), float(df[feature].max()), float(df[feature].mean()))
    
    return pd.DataFrame([user_data])

user_df = get_user_input()

# Normalize user input
user_df_scaled = scaler.transform(user_df)

# Make Prediction
if st.sidebar.button("Predict"):
    prediction = model.predict(user_df_scaled)[0]
    prediction_proba = model.predict_proba(user_df_scaled)[0]

    if prediction == 1:
        st.success(f"**Prediction: Malignant Tumor** 🩸 (Probability: {prediction_proba[1]:.2f})")
    else:
        st.success(f"**Prediction: Benign Tumor** ✅ (Probability: {prediction_proba[0]:.2f})")

# Show dataset sample
st.subheader("Dataset Sample")
st.write(df.head())

# Show model accuracy
st.subheader("Model Information")
st.write(f"Model Accuracy on Training Data: **{model.score(X_train, y_train) * 100:.2f}%**")
st.write(f"Model Accuracy on Test Data: **{model.score(X_test, y_test) * 100:.2f}%**")
