from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from flask_cors import CORS
import joblib
import pandas as pd
from flask import jsonify

app = Flask(__name__)
CORS(app)

# Load model and scaler
model = load_model('gold_price_model.h5')
scaler = joblib.load('scaler.save') 

@app.route('/history', methods=['GET'])
def get_gold_history():
    df = pd.read_csv('gp.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    # Clean numeric columns
    for col in ['Price', 'Open', 'High', 'Low']:
        df[col] = df[col].astype(str).str.replace(',', '')
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['Price'])  # remove NaNs
    last_60 = df.tail(60)['Price'].tolist()

    return jsonify({'prices': last_60})

@app.route('/predictN', methods=['POST'])
def predict_n():
    data = request.get_json(force=True)
    input_data = data['input']  # list of raw prices
    num_predictions = int(data.get('days', 1))

    if len(input_data) < 60:
        return jsonify({'error': 'Need at least 60 price values to make a prediction'}), 400

    input_scaled = scaler.transform(np.array(input_data).reshape(-1, 1)).flatten().tolist()
    predictions = []

    for _ in range(num_predictions):
        window = np.array(input_scaled[-60:]).reshape(1, 60, 1)
        pred_scaled = model.predict(window, verbose=0)[0][0]
        pred_actual = scaler.inverse_transform([[pred_scaled]])[0][0]
        predictions.append(float(pred_actual))
        input_scaled.append(pred_scaled)

    return jsonify({'predictions': predictions})

if __name__ == '__main__':
    app.run(debug=True, port=3001)
