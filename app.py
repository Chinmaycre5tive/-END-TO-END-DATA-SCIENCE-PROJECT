# app.py
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("iris_model.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    # API POST
    if request.is_json:
        payload = request.get_json()
        features = payload["features"]
    else:
        # Form submission
        features = [
            float(request.form["sepal_length"]),
            float(request.form["sepal_width"]),
            float(request.form["petal_length"]),
            float(request.form["petal_width"])
        ]
    pred = model.predict([features])[0]
    species = ["setosa", "versicolor", "virginica"][pred]
    if request.is_json:
        return jsonify({"species": species})
    return render_template("index.html", species=species)

if __name__ == '__main__':
    app.run(debug=True)  # use waitress or gunicorn in production :contentReference[oaicite:3]{index=3}
