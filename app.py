from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return "Diabetes Prediction API"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    # Expected input example:
    # {
    #   "Pregnancies": 2,
    #   "Glucose": 120,
    #   "BloodPressure": 70,
    #   "SkinThickness": 20,
    #   "Insulin": 79,
    #   "BMI": 25.0,
    #   "DiabetesPedigreeFunction": 0.5,
    #   "Age": 33
    # }
    
    try:
        features = [data["Pregnancies"], data["Glucose"], data["BloodPressure"], data["SkinThickness"],
                    data["Insulin"], data["BMI"], data["DiabetesPedigreeFunction"], data["Age"]]
    except KeyError as e:
        return jsonify({"error": f"Missing key {str(e)}"}), 400

    features_np = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features_np)
    
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]

    return jsonify({
        "prediction": int(prediction),
        "probability": float(probability)
    })

if __name__ == '__main__':
    app.run(debug=True)
