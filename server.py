from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# مسار المودل
MODEL_PATH = "model/gas_model.pkl"

# تحميل المودل
if not os.path.exists(MODEL_PATH):
    print("❌ Model file not found!")
    model = None
else:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully")


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "Gas AI Server Running",
        "endpoints": ["/predict"]
    })


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json()

        mq2 = float(data.get("mq2"))
        temp = float(data.get("temp"))
        hum = float(data.get("hum"))

        # شكل الداتا للمودل
        X = np.array([[mq2, temp, hum]])

        # التنبؤ
        prediction = model.predict(X)[0]

        return jsonify({
            "danger": int(prediction)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
