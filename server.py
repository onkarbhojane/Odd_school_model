from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Initialize the Flask application
app = Flask(__name__)

# Load the model and scaler using joblib
model = joblib.load('random_forest_oddschool_model.pkl')
scaler = joblib.load('scaler.pkl')  # Load the scaler

# Define the expected feature names
Features = {'school_type', 'lowclass', 'highclass', 'classrooms_in_good_condition',
             'total_boys_func_toilet', 'total_girls_func_toilet', 'urinal_boys', 
             'urinal_girls', 'drinking_water_functional', 'library_availability',
             'playground_available', 'desktop', 'total_teacher', 'total_students'}

# Define a route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)  # Get the data from the request
        
        # Validate features in the received data
        NotContains = set(Features) - set(data.keys())
        
        if NotContains:
            return jsonify({'NotContains': list(NotContains)}), 400
        
        # Handle the input data, ensuring we handle both single values and lists
        input_data = {key: (value[0] if isinstance(value, list) and len(value) == 1 else value)
                      for key, value in data.items()}
        
        # Create a DataFrame from the processed input data
        input_df = pd.DataFrame([input_data])
        
        # Scale the input data using the fitted scaler
        input_data_scaled = scaler.transform(input_df)

        # Make a prediction using the trained model
        prediction = model.predict(input_data_scaled)

        # Determine the prediction result
        result = "School follows Odd structure" if prediction[0] == 0 else "School follows Standard structure"
        
        # Return the result as a JSON response
        return jsonify({'prediction': result})
    
    except Exception as e:
        # Handle unexpected errors
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
