import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np

app = Flask(__name__)

# Load the model and scaler
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    # Ensure the request has 'application/json' Content-Type
    if request.content_type != 'application/json':
        return jsonify({"error": "Content-Type must be application/json"}), 400
    
    # Parse the incoming JSON data
    data = request.json['data']
    
    try:
        # Convert data to NumPy array and reshape it for prediction
        data_values = np.array(list(data.values())).reshape(1, -1)
        
        # Apply scaling to the data
        new_data = scaler.transform(data_values)
        
        # Make prediction using the model
        prediction = regmodel.predict(new_data)
        
        # Return the prediction
        return jsonify({"prediction": prediction[0]})
    
    except Exception as e:
        # Handle any exceptions that may occur
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
