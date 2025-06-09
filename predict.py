import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Step 1: Load the dataset
df = pd.read_csv('data.csv')

# Define expected symptom columns
expected_symptoms = ['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4', 'Symptom_5', 'Symptom_6']

# Drop completely empty columns
df = df.dropna(axis=1, how='all')

# Fill missing values with most frequent value
imputer = SimpleImputer(strategy='most_frequent')
df_imputed = imputer.fit_transform(df)
df_cleaned = pd.DataFrame(df_imputed, columns=df.columns)

# Convert symptom columns to lowercase
for col in expected_symptoms:
    if col in df_cleaned.columns:
        df_cleaned[col] = df_cleaned[col].str.lower()

# Encode target column 'Disease'
label_encoder = LabelEncoder()
df_cleaned['Disease'] = label_encoder.fit_transform(df_cleaned['Disease'])

# Fit LabelEncoders on all symptom values combined
all_symptoms = pd.Series(dtype=str)
for col in expected_symptoms:
    if col in df_cleaned.columns:
        all_symptoms = pd.concat([all_symptoms, df_cleaned[col]], ignore_index=True)

symptom_encoders = {}
for col in expected_symptoms:
    if col in df_cleaned.columns:
        le = LabelEncoder()
        le.fit(all_symptoms)
        df_cleaned[col] = le.transform(df_cleaned[col])
        symptom_encoders[col] = le

# Features and label
X = df_cleaned[expected_symptoms]
y = df_cleaned['Disease']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Save model and encoders
pickle.dump(model, open('pred.pkl', 'wb'))
pickle.dump(label_encoder, open('disease_encoder.pkl', 'wb'))
pickle.dump(symptom_encoders, open('symptom_encoders.pkl', 'wb'))


# Function to predict disease given symptoms input from web form
def predict_disease(input_symptoms: dict) -> str:
    """
    input_symptoms: dict with keys = symptom column names,
                    values = symptom strings (already lowercase)
    Returns predicted disease string.
    """

    encoded_input = []

    for col in expected_symptoms:
        le = symptom_encoders.get(col)
        if not le:
            # If encoder not found, assign default 0
            encoded_input.append(0)
            continue

        symptom_value = input_symptoms.get(col, "").lower()

        if symptom_value in le.classes_:
            encoded_value = le.transform([symptom_value])[0]
        else:
            # If symptom not found in encoder classes, assign default 0
            encoded_value = 0
        encoded_input.append(encoded_value)

    input_array = np.array(encoded_input).reshape(1, -1)

    pred_label = model.predict(input_array)[0]
    pred_disease = label_encoder.inverse_transform([pred_label])[0]

    return pred_disease
