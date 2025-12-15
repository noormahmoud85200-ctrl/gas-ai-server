from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

MODEL_PATH = os.path.join("model", "gas_model.pkl")

try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Error loading model:", e)
    model = None


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "Gas AI Server is running",
        "model_loaded": model is not None
    })


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json()

        mq2 = float(data["mq2"])
        temp = float(data["temp"])
        hum = float(data["hum"])

        X = np.array([[mq2, temp, hum]])
        prediction = model.predict(X)[0]

        return jsonify({"danger": int(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
