# train.py
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
print("Test accuracy:", model.score(X_test, y_test))

# Save model
joblib.dump(model, "iris_model.pkl")

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

