from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), 'models', 'RandomForest.pkl')
crop_model = joblib.load(model_path)

# Initialize Flask application
app = Flask(__name__)

# Function to make crop recommendation prediction
def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    input_data = pd.DataFrame({
        'N': [N],
        'P': [P],
        'K': [K],
        'temperature': [temperature],
        'humidity': [humidity],
        'ph': [ph],
        'rainfall': [rainfall]
    })
    prediction = crop_model.predict(input_data)
    return prediction[0]

# Route to handle prediction as an API
@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        # Retrieve JSON data from request
        data = request.get_json()
        
        # Extract variables from the incoming JSON
        N = float(data['N'])
        P = float(data['P'])
        K = float(data['K'])
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        ph = float(data['ph'])
        rainfall = float(data['rainfall'])
        
        # Make prediction
        prediction = predict_crop(N, P, K, temperature, humidity, ph, rainfall)
        
        # Return the prediction as a JSON response
        return jsonify({'prediction': prediction})
    
    except Exception as e:
        # Handle errors and return a message
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run()
