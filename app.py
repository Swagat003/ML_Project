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
    data = request.json.get('data', {})
    
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

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract and convert form data to float
        data = [float(x) for x in request.form.values()]
        
        # Reshape and scale the data
        final_input = scaler.transform(np.array(data).reshape(1, -1))
        
        # Make prediction using the model
        output = regmodel.predict(final_input)[0]
        
        # Render the result page with the prediction
        return render_template('prediction.html', prediction_text=f'{output}')
    
    except ValueError:
        # Handle invalid form data
        return render_template('prediction.html', prediction_text='Invalid input. Please check your data.')
    except Exception as e:
        # Handle any other exceptions
        return render_template('prediction.html', prediction_text=f'An error occurred: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
