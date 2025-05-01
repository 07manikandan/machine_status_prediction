import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import streamlit as st

# Streamlit UI setup
st.title("Machine Status Prediction App")
st.sidebar.header("Options")

# File upload section
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
if uploaded_file is not None:
    # Load the dataset
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.write(df.head())

    # Dataset information
    st.write("### Dataset Information")
    st.write(df.info())

    # Statistical summary of the dataset
    st.write("### Statistical Summary")
    st.write(df.describe())

    # Check for missing values
    st.write("### Missing Values")
    st.write(df.isnull().sum())

    # Data Preprocessing
    if st.sidebar.button("Preprocess Data"):
        # Drop empty columns and handle missing data
        df = df.dropna(axis=1, how='all')
        df.dropna(inplace=True)
        df.fillna(df.mean(numeric_only=True), inplace=True)

        # Label encode 'machine_status'
        le = LabelEncoder()
        df['machine_status'] = le.fit_transform(df['machine_status'])
        df['machine_status'] = df['machine_status'].apply(lambda x: 0 if x == 1 else 1)
        st.write("Machine Status distribution after encoding:")
        st.write(df['machine_status'].value_counts())

        # Features and target
        X = df.drop(columns=['machine_status', 'timestamp'])
        y = df['machine_status']

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Train models
        models = {
            'Random Forest': RandomForestClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'SVM': SVC(),
            'KNN': KNeighborsClassifier()
        }

        results = {}

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            results[name] = acc
            st.write(f"\n{name} Accuracy: {acc:.4f}")
            st.text(classification_report(y_test, y_pred))

        # Display all model accuracies
        st.write("### All Model Accuracies:")
        for model_name, acc in results.items():
            st.write(f"{model_name}: {acc:.4f}")

        # Best model selection
        best_model_name = max(results, key=results.get)
        best_model = models[best_model_name]
        st.write(f"Best model is: {best_model_name}")

        # Save the best model
        joblib.dump(best_model, 'best_model.pkl')
        st.write("Model saved successfully!")

        # Load and predict using the best model
        loaded_model = joblib.load('best_model.pkl')
        y_pred = loaded_model.predict(X_test)

        # Prediction section
        st.sidebar.header("Prediction")
        st.sidebar.write("Upload your new sensor data for prediction:")

        # User input prediction
        uploaded_input_file = st.sidebar.file_uploader("Upload CSV for Prediction", type=["csv"])
        if uploaded_input_file is not None:
            new_data = pd.read_csv(uploaded_input_file)
            new_data_scaled = scaler.transform(new_data)
            prediction = loaded_model.predict(new_data_scaled)
            st.write(f"Prediction: {'Anomaly' if prediction[0] == 1 else 'Normal'}")

    # Visualization
    st.write("### Data Visualization")
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Before encoding
    sns.countplot(x='machine_status', data=df, ax=ax[0])
    ax[0].set_title('Machine Status Distribution Before Encoding')

    # After encoding
    sns.countplot(x='machine_status', data=df, ax=ax[1])
    ax[1].set_title('Machine Status Distribution After Encoding')
    ax[1].set_xticklabels(['Normal', 'Anomaly'])

    st.pyplot(fig)

else:
    st.write("Please upload a CSV file to start.")
