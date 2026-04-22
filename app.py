"""
Smart Traffic AI — Flask Application
Run locally: python app.py
"""

import pickle
from flask import Flask, jsonify, render_template, request
import pandas as pd

app = Flask(__name__)

# ── Load model ──────────────────────────────────────────────
_model = None
try:
    with open("model.pkl", "rb") as f:
        _model = pickle.load(f)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"⚠️  model.pkl not found — using fallback logic. ({e})")

# ── Encoding maps ───────────────────────────────────────────
DAY_MAP = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2,
    "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6,
}
WEATHER_MAP = {"Sunny": 0, "Cloudy": 1, "Rainy": 2, "Foggy": 3}
ROAD_MAP    = {"Highway": 0, "City Road": 1, "Residential": 2}

RECOMMENDATIONS = {
    "Low":    "🟢 Roads are clear — great time to travel! Expect smooth flow and minimal delays.",
    "Medium": "🟡 Moderate congestion expected. Allow extra 10–15 minutes for your journey.",
    "High":   "🔴 Heavy traffic ahead! Consider alternate routes or travelling at a different time.",
}

# ── Routes ──────────────────────────────────────────────────
@app.route("/about")
def about():
    return render_template("about.html")
@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")
@app.route("/")
def index():
    return render_template("index.html")

# ── Prediction API ──────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # ✅ FIX: frontend sends FormData, NOT JSON — use request.form
        form = request.form

        time_val = int(form.get("time", 12))
        day_str  = form.get("day", "Monday")
        vehicles = int(form.get("vehicles", 100))
        weather_str  = form.get("weather", "Sunny")
        road_str     = form.get("road_type", "City Road")

        day_val     = DAY_MAP.get(day_str, 0)
        weather_val = WEATHER_MAP.get(weather_str, 0)
        road_val    = ROAD_MAP.get(road_str, 1)

        features = pd.DataFrame([{
            "time_of_day":   time_val,
            "day_of_week":   day_val,
            "vehicle_count": vehicles,
            "weather":       weather_val,
            "road_type":     road_val,
        }])

        if _model:
            pred  = int(_model.predict(features)[0])
            proba = _model.predict_proba(features)[0].tolist()
        else:
            # Fallback heuristic when model.pkl is missing
            if vehicles > 150:
                pred, proba = 2, [0.05, 0.20, 0.75]
            elif vehicles > 80:
                pred, proba = 1, [0.20, 0.60, 0.20]
            else:
                pred, proba = 0, [0.75, 0.20, 0.05]

        labels = ["Low", "Medium", "High"]
        result = labels[pred]

        return jsonify({
            "prediction":    result,
            "confidence":    round(max(proba) * 100, 1),
            "recommendation": RECOMMENDATIONS[result],
            "probabilities": {
                "Low":    round(proba[0] * 100, 1),
                "Medium": round(proba[1] * 100, 1),
                "High":   round(proba[2] * 100, 1),
            },
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ── Run ─────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000)