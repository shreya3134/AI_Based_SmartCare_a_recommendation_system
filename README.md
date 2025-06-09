# AI_Based_SmartCare_a_recommendation_system

Medical Recommendation System using Machine Learning
=====================================================

Overview
--------
This project is a full-stack machine learning application designed to provide personalized medical recommendations based on patient data. It predicts appropriate medicines, exercises, and expected recovery duration using a combination of classification and regression techniques.

Key Features
------------
- Multi-label classification using Random Forest with MultiOutputClassifier to recommend multiple medicines and exercises.
- Regression model to estimate patient recovery days.
- Text data (Disease, Past History, BP Level) processed using TF-IDF Vectorization.
- Numerical vitals (Age, Heart Beats, Oxygen Level) normalized using StandardScaler.
- End-to-end ML pipeline developed using scikit-learn.
- Model evaluation through accuracy scores, classification reports, and RMSE.

Tech Stack
----------
- Python
- scikit-learn
- pandas
- numpy
- NLP=TF-IDF
- gunicorn
- Werkzeug
- Flask (for web integration)
- HTML/CSS/JavaScript (frontend)

Usage
-----
1. Load and preprocess the dataset (`detailed_exercise_disease_dataset.csv`).
2. Train the classification and regression models.
3. Deploy the backend using Flask.
4. Interact with the web interface to input symptoms and receive medical recommendations.

Future Improvements
-------------------
- Add support for more features such as lab results or imaging data.
- Implement user authentication and history tracking.

Author
------
Shreya Gade

