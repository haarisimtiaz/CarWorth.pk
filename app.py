# =====================================================================
# FLASK API + FRONTEND — serves index.html AND /predict from one app
# =====================================================================
# Local run:  python app.py
# First time only:
#   pip install flask flask-cors pandas numpy scikit-learn==1.6.1 joblib gunicorn
# =====================================================================

import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)  # harmless to keep even though frontend now shares the same origin

# Load the trained model once when the server starts
model = joblib.load("car_price_model_v2.pkl")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    required_fields = ["make", "model_name", "city", "transmission",
                        "fuel_type", "year", "mileage", "engine_cc"]
    missing = [f for f in required_fields if f not in data or data[f] in (None, "")]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        current_year = datetime.now().year   # auto-updates every year, no hardcoding
        car_age = current_year - int(data["year"])
        mileage = float(data["mileage"])
        engine_cc = float(data["engine_cc"])
    except (ValueError, TypeError):
        return jsonify({"error": "year, mileage, and engine_cc must be numbers"}), 400

    variant = data.get("variant") or "Base"   # optional field, defaults like training data did

    sample = pd.DataFrame([{
        "Make": data["make"],
        "Model": data["model_name"],
        "Variant": variant,
        "City": data["city"],
        "Fuel Type": data["fuel_type"],
        "Transmission": data["transmission"],
        "Car_Age": car_age,
        "Mileage_Clean": mileage,
        "Engine_Clean": engine_cc,
    }])

    predicted_log_price = model.predict(sample)[0]
    predicted_price = float(np.exp(predicted_log_price))

    return jsonify({
        "predicted_price_pkr": round(predicted_price),
        "predicted_price_formatted": f"PKR {predicted_price:,.0f}"
    })


@app.route("/", methods=["GET"])
def serve_frontend():
    return send_from_directory(".", "index.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Railway sets PORT for you
    app.run(host="0.0.0.0", port=port, debug=False)
