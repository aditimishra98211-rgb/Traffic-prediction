"""
Smart Traffic AI — Model Training
Generates synthetic traffic dataset and trains a Random Forest classifier.
Run this script once before starting the app: python model.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle


def generate_traffic_data(n_samples: int = 6000, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic synthetic traffic dataset with 5 features."""
    np.random.seed(seed)
    records = []

    for _ in range(n_samples):
        time_of_day  = np.random.randint(0, 24)   # 0–23
        day_of_week  = np.random.randint(0, 7)    # 0=Mon … 6=Sun
        weather      = np.random.randint(0, 4)    # 0=Sunny,1=Cloudy,2=Rainy,3=Foggy
        road_type    = np.random.randint(0, 3)    # 0=Highway,1=City,2=Residential

        # Realistic vehicle counts by time/day
        if   7  <= time_of_day <= 10 and day_of_week < 5:  base = np.random.randint(180, 295)
        elif 17 <= time_of_day <= 20 and day_of_week < 5:  base = np.random.randint(190, 310)
        elif 12 <= time_of_day <= 14:                       base = np.random.randint(120, 188)
        elif 0  <= time_of_day <= 5:                        base = np.random.randint(5,   42)
        elif day_of_week >= 5:                              base = np.random.randint(90,  182)
        else:                                               base = np.random.randint(55,  138)

        # Weather impact
        if   weather == 2: base = int(base * 0.82)   # Rainy  → fewer vehicles
        elif weather == 3: base = int(base * 0.70)   # Foggy  → even fewer

        vehicle_count = max(5, base + np.random.randint(-25, 25))

        # Road-type adjusted thresholds
        if   vehicle_count >= 175 or (road_type == 1 and vehicle_count >= 130): label = "High"
        elif vehicle_count >= 100 or (road_type == 1 and vehicle_count >= 75):  label = "Medium"
        else:                                                                    label = "Low"

        records.append({
            "time_of_day":   time_of_day,
            "day_of_week":   day_of_week,
            "vehicle_count": vehicle_count,
            "weather":       weather,
            "road_type":     road_type,
            "traffic_level": label,
        })

    return pd.DataFrame(records)


def train_and_save() -> float:
    print("=" * 56)
    print("  SMART TRAFFIC AI  —  Model Training Pipeline")
    print("=" * 56)

    # ── 1. Generate & save dataset ──────────────────────────
    print("\n[1/4] Generating dataset …")
    df = generate_traffic_data(6000)
    df.to_csv("traffic.csv", index=False)
    print(f"      ✅  {len(df):,} records  →  traffic.csv")
    print(f"      Distribution:\n{df['traffic_level'].value_counts().to_string()}")

    # ── 2. Encode labels ────────────────────────────────────
    print("\n[2/4] Encoding features …")
    label_map = {"Low": 0, "Medium": 1, "High": 2}
    df["target"] = df["traffic_level"].map(label_map)
    FEATURES = ["time_of_day", "day_of_week", "vehicle_count", "weather", "road_type"]
    X = df[FEATURES]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"      Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # ── 3. Train ────────────────────────────────────────────
    print("\n[3/4] Training Random Forest (200 trees) …")
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # ── 4. Evaluate & save ──────────────────────────────────
    print("\n[4/4] Evaluating model …")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"      ✅  Accuracy: {acc * 100:.2f}%\n")
    print(classification_report(y_test, y_pred, target_names=["Low", "Medium", "High"]))

    with open("model.pkl", "wb") as f:
        pickle.dump(clf, f)
    print("✅  model.pkl saved!")
    print("=" * 56)
    return acc


if __name__ == "__main__":
    train_and_save()