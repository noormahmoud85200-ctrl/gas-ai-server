from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os
from datetime import datetime

app = Flask(__name__)

# ==============================
# Paths
# ==============================
DATA_PATH = "data/data.csv"
MODEL_PATH = "model/gas_model.pkl"

# ==============================
# Load ML Model
# ==============================
model = joblib.load(MODEL_PATH)
print("🧠 ML Model Loaded")

# ==============================
# Ensure data file exists
# ==============================
if not os.path.exists("data"):
    os.makedirs("data")

if not os.path.exists(DATA_PATH):
    df = pd.DataFrame(columns=["time", "mq2", "temp", "hum"])
    df.to_csv(DATA_PATH, index=False)

# ==============================
# Routes
# ==============================

@app.route("/", methods=["GET"])
def home():
    return "🚀 Gas AI Server is Running"

@app.route("/data", methods=["POST"])
def receive_data():
    try:
        data = request.get_json()

        mq2 = float(data["mq2"])
        temp = float(data["temp"])
        hum = float(data["hum"])

        # ==============================
        # ML Prediction
        # ==============================
        X = pd.DataFrame([[mq2, temp, hum]], columns=["mq2", "temp", "hum"])
        danger = int(model.predict(X)[0])

        # ==============================
        # Save data
        # ==============================
        new_row = {
            "time": datetime.now(),
            "mq2": mq2,
            "temp": temp,
            "hum": hum
        }

        df = pd.read_csv(DATA_PATH)
        df.loc[len(df)] = new_row
        df.to_csv(DATA_PATH, index=False)

        print(f"📥 DATA: {mq2} {temp} {hum} | 🧠 ML = {danger}")

        return jsonify({"danger": danger})

    except Exception as e:
        print("❌ Error:", e)
        return jsonify({"error": str(e)}), 500

# ==============================
# Run Server (LOCAL + RENDER)
# ==============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
