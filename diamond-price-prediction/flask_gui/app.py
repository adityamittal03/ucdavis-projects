from flask import Flask, render_template, request
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

app = Flask(__name__)

# Load your trained models and scaler
models = {
    'linear_regression': joblib.load('linear_regression_model.joblib'),
    'random_forest': joblib.load('random_forest_regressor_model.joblib'),
    'XGBoost': joblib.load('xgboost_regressor_model.joblib')
}

# Load the polynomial regression model
poly_model = joblib.load('polynomial_regression_model.joblib')

# Load the TensorFlow model architecture from the JSON file
try:
    with open("model.json", "r") as json_file:
        loaded_model_json = json_file.read()
        neural_network_model = model_from_json(loaded_model_json)
    neural_network_model.load_weights("model_weights.h5")
except Exception as e:
    print("Error loading TensorFlow model:", e)
    neural_network_model = None

# Load the scaler
scaler = joblib.load('scaler.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def predict():
    try:
        # Get the input values from the form
        carat = float(request.form['carat'])
        depth = float(request.form['depth'])
        table = float(request.form['table'])
        x = float(request.form['x'])
        y = float(request.form['y'])
        z = float(request.form['z'])
        cut_encoded = int(request.form['cut_encoded'])
        color_encoded = int(request.form['color_encoded'])
        clarity_encoded = int(request.form['clarity_encoded'])
        
        # Prepare input data for prediction
        input_data = np.array([[carat, depth, table, x, y, z, cut_encoded, color_encoded, clarity_encoded]])
        
        # Scale the input data
        input_data_scaled = scaler.transform(input_data)
        
        # Preprocess input data for polynomial regression model
        poly = PolynomialFeatures(2)
        input_data_poly = poly.fit_transform(input_data_scaled)
   
        # Make predictions using all models
        predictions = {}
        for model_name, model in models.items():
            prediction = model.predict(input_data_scaled)
            predictions[model_name] = prediction[0]
        
        # Make prediction using the polynomial regression model
        poly_prediction = poly_model.predict(input_data_poly)
        predictions['polynomial_regression'] = poly_prediction[0]
        
        # Make prediction using the neural network model
        if neural_network_model:
            neural_network_prediction = neural_network_model.predict(input_data_scaled).item()
            predictions['neural_network'] = neural_network_prediction
        else:
            predictions['neural_network'] = "Model not available"
        
        # Return the predicted results to the user
        return render_template("results.html", predictions=predictions)
    except Exception as e:
        print("Error:", e)
        return render_template("error.html")

if __name__ == '__main__':
    app.run(debug=True)
