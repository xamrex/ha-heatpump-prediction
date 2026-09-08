#!/usr/bin/env python3
"""
Heat pump energy consumption prediction script
based on the nonlinear model (Random Forest) saved in heatpump_model_rf.pkl.

Creates a chart comparing actual consumption against the model's estimate
and computes the MAE, R2, and MAPE metrics.

Requirements:
    pip install pandas numpy matplotlib scikit-learn joblib

Usage:
    python3 predict_energy_rf.py
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import json
from datetime import datetime
#show_plot = os.environ.get("SHOW_PLOT", "0") == "1"
show_plot = "1"
# =============================
# 1. Load the model
# =============================
model_dir = "/config/pump"
plots_dir = os.path.join(model_dir, "plots")
# create the plots folder if it doesn't exist
os.makedirs(plots_dir, exist_ok=True)
today = datetime.now()

model_file = os.path.join(model_dir, "heatpump_model_rf.pkl")
input_csv = os.path.join(model_dir, "daily_temps.csv")
output_csv = os.path.join(model_dir, "daily_temps_rf.csv")
plot_file = os.path.join(
    plots_dir,
    f"predicted_vs_actual_rf_{today.strftime('%d-%m-%Y')}.png"
)
if not os.path.exists(model_file):
    raise FileNotFoundError(f"[ERROR] Model file not found: {model_file}")

model = joblib.load(model_file)
print(f"[OK] Model loaded: {model_file}")

# =============================
# 2. Load data
# =============================


if not os.path.exists(input_csv):
    raise FileNotFoundError(f"[ERROR] File not found: {input_csv}")

df = pd.read_csv(input_csv)

# =============================
# 3. Compute heating_degree
# =============================

if "heating_degree" not in df.columns:
    df["heating_degree"] = np.maximum(0, 18 - df["avg_temp"])

# =============================
# 4. Prediction
# =============================

# Make sure the features are in the same order as during training
feature_names = ["avg_temp", "avg_temp_48h", "heating_degree", "avg_humidity"]

df["predicted_energy"] = model.predict(df[feature_names])

# =============================
# 5. Save results
# =============================

df.to_csv(output_csv, index=False)
print(f"[OK] Results saved to: {output_csv}")

# =============================
# 6. Compute metrics
# =============================

if "energy_consumption" in df.columns:
    mae = mean_absolute_error(df["energy_consumption"], df["predicted_energy"])
    r2 = r2_score(df["energy_consumption"], df["predicted_energy"])
    mean_error = (df["energy_consumption"] - df["predicted_energy"]).mean()

    print(f"[INFO] MAE: {mae:.2f} kWh")
    print(f"[INFO] Mean Error: {mean_error:.2f} kWh")
    print(f"[INFO] R2: {r2:.3f}")

    # MAPE
    valid_df = df[df["energy_consumption"] != 0].copy()
    if not valid_df.empty:
        mape = (np.abs(valid_df["energy_consumption"] - valid_df["predicted_energy"]) / valid_df["energy_consumption"]).mean() * 100
        print(f"[INFO] Mean absolute percentage error (MAPE): {mape:.2f} %")
    else:
        mape = None
        print("[WARNING] No data available to compute MAPE")
else:
    mae = r2 = mape = mean_error = None
    print("[WARNING] Missing 'energy_consumption' column, cannot compute MAE and R2")


# =============================
# 6b. Save metrics to JSON
# =============================

metrics_file = os.path.join(model_dir, "dane.json")
metrics_data = {
    "mae": round(mae, 4) if mae is not None else None,
    "r2": round(r2, 4) if r2 is not None else None,
    "mean_error": round(mean_error, 4) if mean_error is not None else None,
    "mape": round(mape, 2) if mape is not None else None,
    "timestamp": datetime.now().isoformat()
}

with open(metrics_file, "w") as f:
    json.dump(metrics_data, f, indent=4)
print(f"[OK] Metrics saved to file: {metrics_file}")

# =============================
# 7. Create the chart
# =============================
if show_plot:
    has_actual = "energy_consumption" in df.columns

    fig, (ax_time, ax_temp) = plt.subplots(2, 1, figsize=(10, 9))

    # ---- Panel 1: energy over time (unchanged, original chart) ----
    ax_time.plot(df["date"], df["predicted_energy"], label="Prediction (kWh)",
                 marker='o', color="tab:blue")
    if has_actual:
        ax_time.plot(df["date"], df["energy_consumption"], label="Actual consumption (kWh)",
                     marker='x', color="tab:red")

    ax_time.set_xlabel("Date")
    ax_time.set_ylabel("Energy consumption [kWh]")
    ax_time.set_title("Heat pump energy consumption - actual vs prediction")
    ax_time.tick_params(axis='x', rotation=45)
    ax_time.legend(loc="best")

    # ---- Panel 2: energy vs temperature (how the model behaves per temperature) ----
    order = df["avg_temp"].argsort()
    temp_sorted = df["avg_temp"].values[order]
    pred_sorted = df["predicted_energy"].values[order]

    ax_temp.plot(temp_sorted, pred_sorted, marker='o', color="tab:blue",
                 label="Prediction (kWh)")
    if has_actual:
        actual_sorted = df["energy_consumption"].values[order]
        ax_temp.plot(temp_sorted, actual_sorted, marker='x', color="tab:red",
                     label="Actual consumption (kWh)")
        # draw a vertical line between actual and predicted for each point to show the error
        for t, a, p in zip(temp_sorted, actual_sorted, pred_sorted):
            ax_temp.plot([t, t], [a, p], color="gray", linestyle=":", linewidth=1)

    ax_temp.set_xlabel("Avg. temperature [C]")
    ax_temp.set_ylabel("Energy consumption [kWh]")
    ax_temp.set_title("Model behavior as a function of temperature")
    ax_temp.legend(loc="best")
    ax_temp.grid(True, alpha=0.3)

    if mae is not None and r2 is not None:
        title_text = f"MAE: {mae:.2f} kWh | R2: {r2:.3f}"
        if mape is not None:
            title_text += f" | Mean error %: {mape:.2f}%"
        fig.suptitle(title_text, y=0.98, fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # Save the chart
    fig.savefig(plot_file, dpi=150)
    print(f"[OK] Chart saved to file: {plot_file}")

    # File always available for Flask
    latest_plot_file = os.path.join(plots_dir, "predicted_vs_actual_rf.png")
    fig.savefig(latest_plot_file, dpi=150)
    plt.close(fig)
