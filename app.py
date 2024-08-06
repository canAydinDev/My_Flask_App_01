import numpy as np
from flask import Flask, request, jsonify
import pickle
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Bu satır CORS izinlerini ekler

# Modeli yükleyin
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return "<h1>Welcome to the Prediction Model API</h1>"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        int_features = [int(x) for x in request.form.values()]
        final_features = np.array([int_features])
        prediction = model.predict(final_features)
        output = round(prediction[0], 2)
        return jsonify({"prediction": output})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
