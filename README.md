# machine_status_prediction

This Streamlit application allows users to upload sensor data in CSV format, preprocess the data, and train machine learning models (Random Forest, Logistic Regression, SVM, KNN) to predict machine status as either 'Normal' or 'Anomaly'. It also provides a way to make predictions on new sensor data using the best-performing model.

## Features
- Upload a CSV file for training the model.
- Preprocess data (handling missing values, encoding labels).
- Train multiple machine learning models.
- Evaluate model performance and select the best model.
- Save the best model for future use.
- Predict machine status for new sensor data.

## Requirements
- Python 3.7+
- The following libraries:
  - pandas
  - numpy
  - scikit-learn
  - seaborn
  - matplotlib
  - joblib
  - streamlit

You can install all dependencies with the following command:
```bash
pip install -r requirements.txt