from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Load models and encoders
disease_model = pickle.load(open('pred.pkl', 'rb'))
disease_encoder = pickle.load(open('disease_encoder.pkl', 'rb'))
symptom_encoders = pickle.load(open('symptom_encoders.pkl', 'rb'))
heart_model = joblib.load('model_joblib_heart.pkl')
insurance_model = joblib.load('insurance_model.pkl')  # renamed to avoid conflict

# Prepare data and pipelines for treatment
df = pd.read_csv("detailed_exercise_disease_dataset.csv")
df.fillna('', inplace=True)
df['combined_text'] = df['Disease'] + ' ' + df['Past_History'] + ' ' + df['BP_Level']

text_feature = 'combined_text'
numeric_features = ['Age', 'Heart_Beats', 'Oxygen_Level']
output_labels = ['Medicine_1', 'Medicine_2', 'Medicine_3', 'Medicine_4',
                 'Exercise 1', 'Exercise 2', 'Exercise 3']
regression_target = 'Recovery_Days'

preprocessor = ColumnTransformer([
    ('text', TfidfVectorizer(), text_feature),
    ('num', StandardScaler(), numeric_features)
])

X = df[[text_feature] + numeric_features]
y_classification = df[output_labels]
y_regression = df[regression_target]

clf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', MultiOutputClassifier(RandomForestClassifier(random_state=42)))
])
clf_pipeline.fit(X, y_classification)

reg_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])
reg_pipeline.fit(X, y_regression)

disease_list = sorted(df['Disease'].dropna().unique())
users = {}

def categorize_bp(bp_input):
    try:
        parts = bp_input.split('/')
        systolic = int(parts[0].strip())
        diastolic = int(parts[1].strip()) if len(parts) == 2 else 0
    except:
        return 'Unknown'
    if systolic < 90 or diastolic < 60:
        return 'Low'
    elif 90 <= systolic <= 120 and 60 <= diastolic <= 80:
        return 'Normal'
    elif systolic > 120 or diastolic > 80:
        return 'High'
    return 'Unknown'

@app.before_request
def before_request():
    session.permanent = True

@app.route('/')
def root():
    return redirect(url_for('signup'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password'].strip()
        if username in users:
            return render_template('signup.html', error='Username already exists')
        users[username] = password
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password'].strip()
        if users.get(username) == password:
            session['user'] = username
            return redirect(url_for('home'))
        return render_template('login.html', error="Invalid Credentials")
    return render_template('login.html')

@app.route('/home')
def home():
    if 'user' not in session:
        return redirect(url_for('login'))
    symptoms_list = sorted(set.union(*(set(le.classes_) for le in symptom_encoders.values())))
    return render_template('home.html', symptoms=symptoms_list)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    symptoms = data.get('symptoms')
    if not symptoms or len(symptoms) != 6:
        return jsonify({'error': 'Exactly 6 symptoms are required'}), 400
    try:
        encoded = []
        for i, symptom in enumerate(symptoms):
            col = f'Symptom_{i+1}'
            le = symptom_encoders.get(col)
            if le and symptom in le.classes_:
                encoded.append(le.transform([symptom])[0])
            else:
                return jsonify({'error': f'Invalid symptom: {symptom}'}), 400
        prediction = disease_model.predict([encoded])[0]
        disease = disease_encoder.inverse_transform([prediction])[0]
        return jsonify({'disease': disease})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/treatment', methods=["GET", "POST"])
def treatment():
    if 'user' not in session:
        return redirect(url_for('login'))
    selected_disease = request.args.get('disease', '')
    if request.method == "POST":
        try:
            age = float(request.form['age'])
            disease = request.form['disease']
            past_history = request.form.get('past_history', '')
            raw_bp = request.form['bp']
            heart_beats = float(request.form['heart_beats'])
            oxygen_level = float(request.form['oxygen_level'])

            bp_category = categorize_bp(raw_bp)
            combined_text = disease + ' ' + past_history + ' ' + bp_category
            input_df = pd.DataFrame({
                'combined_text': [combined_text],
                'Age': [age],
                'Heart_Beats': [heart_beats],
                'Oxygen_Level': [oxygen_level]
            })

            pred_class = clf_pipeline.predict(input_df)[0]
            pred_reg = reg_pipeline.predict(input_df)[0]

            medicines = [m for m in pred_class[:4] if isinstance(m, str) and m.strip().lower() != 'nan' and m.strip() != '']
            exercises = [e for e in pred_class[4:7] if isinstance(e, str) and e.strip().lower() != 'nan' and e.strip() != '']

            treatment_plan = {
                "BP_Category": bp_category,
                "Recovery_Days": round(pred_reg, 2),
                "Medicines": medicines,
                "Exercises": exercises
            }

            session['treatment_plan'] = treatment_plan
            return redirect(url_for('result'))

        except Exception as e:
            return render_template("index.html", error=str(e), disease=selected_disease)

    return render_template("index.html", disease=selected_disease)

@app.route("/result")
def result():
    treatment_plan = session.get('treatment_plan', None)
    if treatment_plan is None:
        return redirect(url_for('treatment'))
    return render_template("result.html", treatment=treatment_plan)

@app.route("/diseases")
def diseases():
    return render_template("lists.html", diseases=disease_list)

@app.route("/heart", methods=['GET', 'POST'])
def heart():
    if 'user' not in session:
        return redirect(url_for('login'))
    result = error = None
    if request.method == 'POST':
        try:
            input_features = [float(request.form.get(col)) for col in [
                'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
            ]]
            prediction = heart_model.predict([input_features])[0]
            result = 'Positive' if prediction == 1 else 'Negative'
        except Exception as e:
            error = f"Error processing input: {e}"
    return render_template('heart.html', result=result, error=error)

@app.route("/insurance", methods=['GET', 'POST'])
def insurance():
    if 'user' not in session:
        return redirect(url_for('login'))

    prediction = None
    if request.method == 'POST':
        try:
            age = float(request.form['age'])
            sex = int(request.form['sex'])
            bmi = float(request.form['bmi'])
            children = int(request.form['children'])  # NEW
            smoker = int(request.form['smoker'])
            region = int(request.form['region'])

            input_data = [[age, sex, bmi, children, smoker, region]]
            prediction = insurance_model.predict(input_data)[0]
            prediction = round(prediction, 2)

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template('insurance.html', prediction=prediction)

@app.route('/logout')
def logout():
    session.pop('user', None)
    session.pop('treatment_plan', None)
    return redirect(url_for('login'))


if __name__ == "__main__":
    app.run(debug=True)
