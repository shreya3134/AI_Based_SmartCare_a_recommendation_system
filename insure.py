import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

# Load and preprocess data
insurance_dataset = pd.read_csv("insurance.csv")
insurance_dataset.replace({'sex': {'male': 0, 'female': 1}}, inplace=True)
insurance_dataset.replace({'smoker': {'yes': 0, 'no': 1}}, inplace=True)
insurance_dataset.replace({'region': {'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3}}, inplace=True)

X = insurance_dataset.drop(columns='charges', axis=1)
y = insurance_dataset['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Train model
reg = LinearRegression()
reg.fit(X_train, y_train)

# Save model
joblib.dump(reg, 'insurance_model.pkl')
