# server/app.py
from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow requests from frontend (localhost)

# Load your trained model
model = load_model('gold_price_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    input_data = data['input']  # expected: list of 60 numbers
    input_array = np.array(input_data).reshape(1, 60, 1)
    prediction = model.predict(input_array)
    return jsonify({'prediction': float(prediction[0][0])})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
