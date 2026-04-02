import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from flask import Flask, request, jsonify, send_from_directory
import joblib
import pandas as pd
import random
import hashlib

app = Flask(__name__)

# Load ML models
try:
    scaler = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))
    model = joblib.load(os.path.join(BASE_DIR, 'best_model.pkl'))
    models_loaded = True
except Exception as e:
    models_loaded = False
    model_error = str(e)


@app.route('/', methods=['GET'])
def index():
    return send_from_directory(os.path.join(BASE_DIR, 'public'), 'index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if not models_loaded:
        return jsonify({'error': f'Model load error: {model_error}'}), 500

    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    try:
        gender_encoded = 1 if data['gender'].lower() == 'male' else 0
        diabetic_encoded = 1 if data['diabetic'].lower() == 'yes' else 0
        smoker_encoded = 1 if data['smoker'].lower() == 'yes' else 0

        input_df = pd.DataFrame({
            'age': [int(data['age'])],
            'gender': [gender_encoded],
            'bmi': [float(data['bmi'])],
            'bloodpressure': [int(data['bloodpressure'])],
            'diabetic': [diabetic_encoded],
            'children': [int(data['children'])],
            'smoker': [smoker_encoded]
        })

        numeric_cols = ['age', 'bmi', 'bloodpressure', 'children']
        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

        prediction = model.predict(input_df)[0]
        pred_value = float(prediction)

        insurance_companies = [
            'HDFC ERGO Health Insurance',
            'Star Health & Allied Insurance',
            'Care Health Insurance',
            'Niva Bupa Health Insurance',
            'ICICI Lombard Health Insurance'
        ]

        prediction_hash = int(hashlib.md5(str(round(pred_value, 2)).encode()).hexdigest(), 16)
        random.seed(prediction_hash)
        shuffled = insurance_companies.copy()
        random.shuffle(shuffled)

        return jsonify({
            'prediction': round(pred_value, 2),
            'companies': shuffled
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
