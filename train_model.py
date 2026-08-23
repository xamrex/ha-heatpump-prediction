#!/usr/bin/env python3
"""
Standalone script for training the NONLINEAR heat pump model.
RUN ON A PC/LAPTOP, NOT INSIDE HOME ASSISTANT!

Requirements:
    pip install pandas numpy scikit-learn joblib

The script:
1. Loads data from daily_temps.csv
2. Trains a RandomForestRegressor model
3. Saves the model to heatpump_model_rf.pkl
4. The model can be moved to HAOS or used for prediction directly in PyScript
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import joblib
import os
from datetime import datetime
import json

def train_nonlinear_model():
    print("=" * 70)
    print("TRAINING HEAT PUMP MODEL - RANDOM FOREST")
    print("=" * 70)

    # directory holding both the model and the CSV
    model_dir = "/config/pump"
    os.makedirs(model_dir, exist_ok=True)

    # paths
    csv_file = os.path.join(model_dir, "daily_temps.csv")
    model_file = os.path.join(model_dir, "heatpump_model_rf.pkl")
    metadata_file = os.path.join(model_dir, "model_metadata_rf.json")

    # check that the CSV file exists
    if not os.path.exists(csv_file):
        print(f"[ERROR] File {csv_file} does not exist!")
        return False

    df = pd.read_csv(csv_file)
    required_columns = ["avg_temp", "avg_temp_48h", "avg_humidity", "energy_consumption"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        print(f"[ERROR] Missing columns: {missing}")
        return False

    # Create the heating_degree feature
    df["heating_degree"] = np.maximum(0, 18 - df["avg_temp"])

    feature_names = ["avg_temp", "avg_temp_48h", "heating_degree", "avg_humidity"]
    X = df[feature_names]
    y = df["energy_consumption"]

    test_size = 0.2 if len(df) >= 10 else 0.1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Nonlinear model
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print(f"[OK] Model trained!")

    # Model evaluation
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    print(f"\nMODEL RESULTS:")
    print(f"  Training: MAE={mae_train:.2f}, R2={r2_train:.3f}")
    print(f"  Test:     MAE={mae_test:.2f}, RMSE={rmse_test:.2f}, R2={r2_test:.3f}")

    # Save the model
    joblib.dump(model, model_file)
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "model_type": "RandomForestRegressor",
        "mae_train_kwh": round(mae_train, 2),
        "mae_test_kwh": round(mae_test, 2),
        "rmse_test_kwh": round(rmse_test, 2),
        "r2_train": round(r2_train, 3),
        "r2_test": round(r2_test, 3),
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "features": feature_names,
    }
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[OK] Model saved: {model_file}")
    print(f"[OK] Metadata saved: {metadata_file}")

    # Prediction test
    print(f"\nPREDICTION TEST:")
    test_cases = [
        (0, -1, 18, 80),
        (-5, -6, 23, 85),
        (15, 14, 3, 70),
    ]
    for temp, temp_48h, hd, hum in test_cases:
        X_test_case = pd.DataFrame([[temp, temp_48h, hd, hum]], columns=feature_names)
        pred = model.predict(X_test_case)[0]
        print(f"temp={temp:4.1f}C, temp_48h={temp_48h:4.1f}C, hd={hd:4.1f}, hum={hum:3.0f}% -> {pred:6.2f} kWh")

    return True

if __name__ == "__main__":
    try:
        success = train_nonlinear_model()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        exit(1)
