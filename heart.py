import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Load dataset
heart_dataset = pd.read_csv("heart.csv")

# Separate categorical and continuous columns
cate_val = []
cont_val = []

for column in heart_dataset.columns:
    if heart_dataset[column].nunique() <= 10:
        cate_val.append(column)
    else:
        cont_val.append(column)

# Scale continuous values
scaler = StandardScaler()
heart_dataset[cont_val] = scaler.fit_transform(heart_dataset[cont_val])

# Split data
X = heart_dataset.drop('target', axis=1)
y = heart_dataset['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Train model
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# Predict and evaluate
y_pred = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model with .pkl extension
joblib.dump(rf, 'model_joblib_heart.pkl')
