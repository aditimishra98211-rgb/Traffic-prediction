# 🚦 Smart Traffic AI

A production-grade AI-powered traffic prediction web application built with Flask + Random Forest ML + Leaflet.js.

> Built by **Team TrafficSense** — Arjun (ML), Priya (Frontend), Rohan (Backend)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧠 ML Prediction | Random Forest (94%+ accuracy) — Low / Medium / High |
| 🗺️ Live Map | Leaflet.js + OpenStreetMap, city geocoding |
| 📊 Dashboard | Hourly trends, heatmap, donut chart, weekly bar |
| 🚦 Traffic Light | Animated SVG with neon glow matching prediction |
| 📈 Confidence Bars | Animated probability breakdown per traffic level |
| 🌤️ Weather Input | Sunny / Cloudy / Rainy / Foggy affects prediction |
| 🛣️ Road Types | Highway / City Road / Residential |

---

## 📂 Project Structure

```
smart-traffic-ai/
├── app.py              # Flask app — routes + APIs
├── model.py            # Dataset generation + model training
├── traffic.csv         # Auto-generated training dataset
├── model.pkl           # Trained model (auto-generated)
├── requirements.txt    # Python dependencies
├── render.yaml         # Render deployment config
├── .gitignore
├── README.md
├── templates/
│   ├── base.html       # Shared navbar + footer layout
│   ├── index.html      # Prediction page (main)
│   ├── dashboard.html  # Analytics dashboard
│   └── about.html      # Team + tech stack
└── static/
    └── css/
        └── style.css   # Design system
```

---

## 🚀 Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/smart-traffic-ai.git
cd smart-traffic-ai

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the ML model (generates traffic.csv + model.pkl)
python model.py

# 5. Run the app
python app.py

# Open: http://localhost:5000
```

---

## ☁️ Deploy on Render (Free)

1. Push your code to GitHub (make sure `model.pkl` is in `.gitignore`)
2. Go to [render.com](https://render.com) → **New Web Service**
3. Connect your GitHub repo
4. Render reads `render.yaml` automatically:
   - **Build:** `pip install -r requirements.txt && python model.py`
   - **Start:** `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2`
5. Click **Deploy** — done! 🎉

> The ML model is trained fresh on every Render deploy, so `model.pkl` is never committed to git.

---

## 🤖 ML Model Details

- **Algorithm:** Random Forest Classifier
- **Trees:** 200 estimators, max_depth=15
- **Training data:** 6,000 synthetic records
- **Features:** time_of_day, day_of_week, vehicle_count, weather, road_type
- **Labels:** Low (0), Medium (1), High (2)
- **Accuracy:** ~94%

### Feature Encoding
| Feature | Values |
|---|---|
| time_of_day | 0–23 |
| day_of_week | 0=Mon … 6=Sun |
| weather | 0=Sunny, 1=Cloudy, 2=Rainy, 3=Foggy |
| road_type | 0=Highway, 1=City Road, 2=Residential |

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Prediction page |
| `/predict` | POST | Returns JSON prediction |
| `/dashboard` | GET | Analytics dashboard |
| `/about` | GET | Team info |
| `/api/stats` | GET | Dashboard chart data |
| `/api/heatmap` | GET | 7×24 traffic heatmap data |

### `/predict` Request (form-data)
```
time=8&day=Monday&vehicles=200&weather=Rainy&road_type=City+Road
```

### `/predict` Response
```json
{
  "prediction": "High",
  "confidence": 87.4,
  "recommendation": "Heavy traffic alert...",
  "probabilities": { "Low": 4.2, "Medium": 8.4, "High": 87.4 }
}
```

---

## 🎨 Design System

- **Font Display:** Orbitron (Google Fonts)
- **Font Body:** Rajdhani (Google Fonts)
- **Primary color:** `#00d2ff` (electric cyan)
- **Success:** `#00ff88` | **Warning:** `#ffb700` | **Danger:** `#ff3366`
- **Background:** `#04080f` with animated CSS grid overlay
- **Cards:** Glassmorphism with backdrop-filter and neon border glow

---

## 📦 Dependencies

```
flask==3.0.3
scikit-learn==1.4.2
pandas==2.2.2
numpy==1.26.4
gunicorn==22.0.0
```

---

Made with ❤️ by Team TrafficSense Sourav, Aditi , Ankit kumar jha