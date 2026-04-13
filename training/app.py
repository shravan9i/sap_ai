from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

MODEL_PATH = '/app/model/model.pkl'

# =========================
# Load Model
# =========================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model file not found at /app/model/model.pkl")

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

print("Model loaded successfully!")

# =========================
# Routes
# =========================
@app.route("/", methods=["GET"])
def home():
    return "Model API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if "features" not in data:
            return jsonify({"error": "Missing 'features' key"}), 400

        features = np.array(data["features"]).reshape(1, -1)

        prediction = model.predict(features)

        return jsonify({
            "prediction": prediction.tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =========================
# Run App
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)