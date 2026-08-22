#!/usr/bin/env python3
"""
Skrypt predykcji zużycia energii pompy ciepła
na podstawie modelu liniowego (Linear Regression) zapisanego w heatpump_model_linear.pkl.

Tworzy wykres porównujący aktualne zużycie i estymatę modelu
i liczy metryki MAE, R² i MAPE.

Wymagania:
    pip install pandas numpy matplotlib scikit-learn joblib

Użycie:
    python3 predict_energy_linear.py
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
# 1. Wczytanie modelu
# =============================
model_dir = "/config/pump"
plots_dir = os.path.join(model_dir, "plots")
# utwórz folder plots jeśli nie istnieje
os.makedirs(plots_dir, exist_ok=True)
today = datetime.now()

model_file = os.path.join(model_dir, "heatpump_model_linear.pkl")
input_csv = os.path.join(model_dir, "daily_temps.csv")
output_csv = os.path.join(model_dir, "daily_temps_linear.csv")
plot_file = os.path.join(
    plots_dir,
    f"predicted_vs_actual_linear_{today.strftime('%d-%m-%Y')}.png"
)
if not os.path.exists(model_file):
    raise FileNotFoundError(f"[ERROR] Brak pliku modelu: {model_file}")

model = joblib.load(model_file)
print(f"[OK] Model liniowy załadowany: {model_file}")

# =============================
# 2. Wczytanie danych
# =============================
if not os.path.exists(input_csv):
    raise FileNotFoundError(f"[ERROR] Brak pliku {input_csv}")

df = pd.read_csv(input_csv)

# =============================
# 3. Obliczenie heating_degree
# =============================
if "heating_degree" not in df.columns:
    df["heating_degree"] = np.maximum(0, 18 - df["avg_temp"])

# =============================
# 4. Predykcja
# =============================
feature_names = ["avg_temp", "avg_temp_48h", "heating_degree", "avg_humidity"]

df["predicted_energy"] = model.predict(df[feature_names])

# =============================
# 5. Zapis wyników
# =============================
df.to_csv(output_csv, index=False)
print(f"[OK] Zapisano wynik do: {output_csv}")

# =============================
# 6. Obliczenie metryk
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
        print(f"[INFO] Średni błąd procentowy (MAPE): {mape:.2f} %")
    else:
        mape = None
        print("[WARNING] Brak danych do policzenia MAPE")
else:
    mae = r2 = mape = mean_error = None
    print("[WARNING] Brak kolumny 'energy_consumption', nie można policzyć MAE i R²")

# =============================
# 6b. Zapis metryk do JSON
# =============================
metrics_file = os.path.join(model_dir, "dane_linear.json")
metrics_data = {
    "mae": round(mae, 4) if mae is not None else None,
    "r2": round(r2, 4) if r2 is not None else None,
    "mean_error": round(mean_error, 4) if mean_error is not None else None,
    "mape": round(mape, 2) if mape is not None else None,
    "timestamp": datetime.now().isoformat()
}

with open(metrics_file, "w") as f:
    json.dump(metrics_data, f, indent=4)
print(f"[OK] Zapisano metryki do pliku: {metrics_file}")

# =============================
# 7. Tworzenie wykresu
# =============================
if show_plot:
    plt.figure(figsize=(10, 5))
    plt.plot(df["date"], df["predicted_energy"], label="Predykcja (kWh)", marker='o')
    if "energy_consumption" in df.columns:
        plt.plot(df["date"], df["energy_consumption"], label="Rzeczywiste zużycie (kWh)", marker='x')

    plt.xticks(rotation=45)
    plt.xlabel("Data")
    plt.ylabel("Zużycie energii [kWh]")
    plt.title("Zużycie energii pompy ciepła - rzeczywiste vs predykcja (Linear Regression)")

    if mae is not None and r2 is not None:
        title_text = f"MAE: {mae:.2f} kWh | R2: {r2:.3f}"
        if mape is not None:
            title_text += f" | Średni błąd %: {mape:.2f}%"
        plt.suptitle(title_text, y=0.92, fontsize=10)

    plt.legend()
    plt.tight_layout()

    # Zapis wykresu
    plt.savefig(plot_file, dpi=150)
    print(f"[OK] Wykres zapisany do pliku: {plot_file}")

    # Plik zawsze dostępny dla Flask
    latest_plot_file = os.path.join(plots_dir, "predicted_vs_actual_linear.png")
    plt.savefig(latest_plot_file, dpi=150)