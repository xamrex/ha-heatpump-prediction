#!/usr/bin/env python3
"""
Flask server dla addona trenowania modelu pompy ciepła.
Obsługuje modele nieliniowy (Random Forest) i liniowy (Linear Regression)
"""

from flask import Flask, jsonify, send_file, request
import subprocess
import threading
import os
import pandas as pd
import joblib
import json 
import csv
import time
import requests
from datetime import datetime, timedelta

app = Flask(__name__)

training_status = {
    "rf": {
        "is_training": False,
        "last_result": None,
        "last_error": None
    },
    "linear": {
        "is_training": False,
        "last_result": None,
        "last_error": None
    }
}

status_lock = threading.Lock()

# =============================
# Funkcje trenowania
# =============================
def run_training_rf():
    with status_lock:
        training_status["rf"]["is_training"] = True
        training_status["rf"]["last_error"] = None

    try:
        result = subprocess.run(
            ["python3", "/app/train_model.py"],
            capture_output=True,
            text=True,
            timeout=300
        )

        with status_lock:
            if result.returncode == 0:
                training_status["rf"]["last_result"] = "SUCCESS"
            else:
                training_status["rf"]["last_result"] = "FAILED"
                training_status["rf"]["last_error"] = result.stderr

    except Exception as e:
        with status_lock:
            training_status["rf"]["last_result"] = "ERROR"
            training_status["rf"]["last_error"] = str(e)

    finally:
        with status_lock:
            training_status["rf"]["is_training"] = False


def run_training_linear():
    with status_lock:
        training_status["linear"]["is_training"] = True
        training_status["linear"]["last_error"] = None

    try:
        result = subprocess.run(
            ["python3", "/app/train_model_linear.py"],
            capture_output=True,
            text=True,
            timeout=300
        )

        with status_lock:
            if result.returncode == 0:
                training_status["linear"]["last_result"] = "SUCCESS"
            else:
                training_status["linear"]["last_result"] = "FAILED"
                training_status["linear"]["last_error"] = result.stderr

    except Exception as e:
        with status_lock:
            training_status["linear"]["last_result"] = "ERROR"
            training_status["linear"]["last_error"] = str(e)

    finally:
        with status_lock:
            training_status["linear"]["is_training"] = False

# =============================
# Endpoints - info
# =============================
@app.route('/', endpoint='home')
def home_view():
    return jsonify({
        "name": "Heat Pump Training API",
        "version": "1.3",
        "Instruction": "Copy or automatically collect daily_temps.csv, Run train, and you can use prediction.",
        "endpoints": {
            "/train": "Start model RF training (POST/GET)",
            "/trainlinear": "Start linear model training (POST/GET)",
            "/status": "RF training status",
            "/statuslinear": "Linear training status",
            "/prediction?avg_temp=5&avg_temp_48h=3&avg_humidity=75": "Get RF energy prediction",
            "/predictionlinear?avg_temp=5&avg_temp_48h=3&avg_humidity=75": "Linear energy prediction",
            "/showpicresults?show_plot=1": "Run RF check script",
            "/showpicresultslinear?show_plot=0": "Run linear check script",
            "/log_now": "Force-log current sensor values to CSV immediately",
            "/log_status": "Get automatic logging daemon status"
        }
    })

# =============================
# Endpoints - trenowanie
# =============================
@app.route('/train', methods=['POST', 'GET'])
def train_rf_view():
    with status_lock:
        if training_status["rf"]["is_training"]:
            return jsonify({"status": "error", "message": "RF już się trenuje!"}), 409

    thread = threading.Thread(target=run_training_rf)
    thread.daemon = True
    thread.start()

    return jsonify({"status": "started", "message": "RF training started"}), 200


@app.route('/trainlinear', methods=['POST', 'GET'])
def train_linear_view():
    with status_lock:
        if training_status["linear"]["is_training"]:
            return jsonify({"status": "error", "message": "Linear już się trenuje!"}), 409

    thread = threading.Thread(target=run_training_linear)
    thread.daemon = True
    thread.start()

    return jsonify({"status": "started", "message": "Linear training started"}), 200


# =============================
# Endpoints - status
# =============================
@app.route('/status')
def status_rf_view():
    # bezpieczna kopia statusu (thread-safe)
    with status_lock:
        response = training_status["rf"].copy()

    metadata_file = "/config/pump/model_metadata_rf.json"

    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            response["last_model_trained"] = metadata.get("timestamp")
            response["model_metrics"] = {
                "mae_test_kwh": metadata.get("mae_test_kwh"),
                "r2_test": metadata.get("r2_test")
            }

        except Exception as e:
            response["metadata_error"] = str(e)
    else:
        response["last_model_trained"] = None
        response["model_metrics"] = None

    return jsonify(response), 200



@app.route('/statuslinear')
def status_linear_view():
    # bezpieczna kopia statusu (thread-safe)
    with status_lock:
        response = training_status["linear"].copy()

    metadata_file = "/config/pump/model_metadata_linear.json"

    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            response["last_model_trained"] = metadata.get("timestamp")
            response["model_metrics"] = {
                "mae_test_kwh": metadata.get("mae_test_kwh"),
                "r2_test": metadata.get("r2_test")
            }

        except Exception as e:
            response["metadata_error"] = str(e)
    else:
        response["last_model_trained"] = None
        response["model_metrics"] = None

    return jsonify(response), 200



# =============================
# Funkcja predykcji
# =============================
def predict_energy(model_path, avg_temp, avg_temp_48h, avg_humidity):
    if not os.path.exists(model_path):
        return {"status": "error", "message": f"Brak pliku modelu: {model_path}"}, 404
    model = joblib.load(model_path)
    heating_degree = max(0, 18 - avg_temp)
    features = pd.DataFrame([[avg_temp, avg_temp_48h, heating_degree, avg_humidity]],
                            columns=["avg_temp", "avg_temp_48h", "heating_degree", "avg_humidity"])
    prediction = model.predict(features)[0]
    return {
        "status": "success",
        "prediction": {"energy_kwh": round(prediction, 2), "unit": "kWh"},
        "input_data": {"avg_temp": avg_temp, "avg_temp_48h": avg_temp_48h,
                       "heating_degree": heating_degree, "avg_humidity": avg_humidity}
    }, 200

# =============================
# Endpoints - predykcje
# =============================
@app.route('/prediction', endpoint='predict_rf')
def prediction_rf_view():
    try:
        avg_temp = float(request.args.get("avg_temp", 0))
        avg_temp_48h = float(request.args.get("avg_temp_48h", -1))
        avg_humidity = float(request.args.get("avg_humidity", 80))
    except ValueError:
        return {"status": "error", "message": "Niepoprawny format parametrów"}, 400
    return predict_energy("/config/pump/heatpump_model_rf.pkl", avg_temp, avg_temp_48h, avg_humidity)

@app.route('/predictionlinear', endpoint='predict_linear')
def prediction_linear_view():
    try:
        avg_temp = float(request.args.get("avg_temp", 0))
        avg_temp_48h = float(request.args.get("avg_temp_48h", -1))
        avg_humidity = float(request.args.get("avg_humidity", 80))
    except ValueError:
        return {"status": "error", "message": "Niepoprawny format parametrów"}, 400
    return predict_energy("/config/pump/heatpump_model_linear.pkl", avg_temp, avg_temp_48h, avg_humidity)

# =============================
# Endpoints - weryfikacja (wykres)
# =============================
def run_check_model(script_path, plot_path, metrics_file):
    """
    Uruchamia skrypt checkModel (linear lub RF) i zwraca albo wykres, albo JSON z metrykami.
    """
    show_plot = request.args.get("show_plot", "0") == "1"
    env = os.environ.copy()
    env["SHOW_PLOT"] = "1" if show_plot else "0"

    try:
        result = subprocess.run(
            ["python3", script_path],
            capture_output=True,
            text=True,
            timeout=300,
            env=env
        )
        if result.returncode != 0:
            return jsonify({
                "status": "error",
                "message": "Script execution failed",
                "stderr": result.stderr,
                "stdout": result.stdout
            }), 500

        if show_plot:
            if os.path.exists(plot_path):
                return send_file(plot_path, mimetype="image/png")
            else:
                return jsonify({
                    "status": "error",
                    "message": f"Wykres nie znaleziony: {plot_path}"
                }), 404
        else:
            if os.path.exists(metrics_file):
                with open(metrics_file, "r") as f:
                    metrics_data = json.load(f)
                return jsonify({"status": "success", "metrics": metrics_data})
            else:
                return jsonify({
                    "status": "error",
                    "message": f"Plik z metrykami nie istnieje: {metrics_file}"
                }), 404

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/showpicresults', endpoint='show_rf')
def show_results_rf_view():
    return run_check_model(
        script_path="/app/checkModel.py",
        plot_path="/config/pump/plots/predicted_vs_actual_rf.png",
        metrics_file="/config/pump/dane.json"
    )

@app.route('/showpicresultslinear', endpoint='show_linear')
def show_results_linear_view():
    return run_check_model(
        script_path="/app/checkModel_linear.py",
        plot_path="/config/pump/plots/predicted_vs_actual_linear.png",
        metrics_file="/config/pump/dane_linear.json"
    )

# =============================
# Automatyczne logowanie danych
# =============================
OPTIONS_FILE = "/data/options.json"
FALLBACK_OPTIONS_FILE = "/config/heatpumptrain/options.json"
CSV_FILE = "/config/pump/daily_temps.csv"

logging_status = {
    "last_run": None,
    "last_status": "Idle",
    "last_error": None,
    "next_run": None
}
logging_lock = threading.Lock()

def get_addon_options():
    if os.path.exists(OPTIONS_FILE):
        try:
            with open(OPTIONS_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading {OPTIONS_FILE}: {e}")
    elif os.path.exists(FALLBACK_OPTIONS_FILE):
        try:
            with open(FALLBACK_OPTIONS_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading {FALLBACK_OPTIONS_FILE}: {e}")
    return {}

def get_sensor_names():
    opts = get_addon_options()
    return {
        "avg_temp": opts.get("avg_temp_sensor", ""),
        "avg_temp_48h": opts.get("avg_temp_48h_sensor", ""),
        "avg_humidity": opts.get("avg_humidity_sensor", ""),
        "energy_consumption": opts.get("energy_consumption_sensor", "")
    }

def get_sensor_state(entity_id):
    if not entity_id:
        raise ValueError("Sensor entity ID is empty in configuration")
    
    token = os.environ.get("SUPERVISOR_TOKEN")
    if not token:
        token = os.environ.get("HA_TOKEN")
        if not token:
            raise ValueError("SUPERVISOR_TOKEN or HA_TOKEN environment variable not set")
            
    ha_url = os.environ.get("HA_URL", "http://supervisor/core/api")
    url = f"{ha_url}/states/{entity_id}"
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        state_str = data.get("state")
        
        if state_str in (None, "unavailable", "unknown"):
            raise ValueError(f"Sensor state is '{state_str}'")
            
        return float(state_str)
    except Exception as e:
        raise RuntimeError(f"Error querying sensor {entity_id}: {e}")

def fetch_all_sensors():
    sensors = get_sensor_names()
    missing = [k for k, v in sensors.items() if not v]
    if missing:
        raise ValueError(f"Brak skonfigurowanych sensorów dla: {', '.join(missing)}")
        
    results = {}
    for name, entity_id in sensors.items():
        try:
            results[name] = get_sensor_state(entity_id)
        except Exception as e:
            raise RuntimeError(f"Błąd pobierania sensora '{name}' ({entity_id}): {e}")
    return results

def log_daily_data(date_str=None):
    if not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")
        
    data = fetch_all_sensors()
    
    row_data = {
        "date": date_str,
        "avg_temp": round(data["avg_temp"], 2),
        "avg_temp_48h": round(data["avg_temp_48h"], 2),
        "avg_humidity": round(data["avg_humidity"], 2),
        "energy_consumption": round(data["energy_consumption"], 2)
    }
    
    os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)
    
    updated = False
    rows = []
    headers = ["date", "avg_temp", "avg_temp_48h", "avg_humidity", "energy_consumption"]
    
    if os.path.exists(CSV_FILE):
        try:
            with open(CSV_FILE, mode='r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for r in reader:
                    if r.get("date") == date_str:
                        rows.append(row_data)
                        updated = True
                    else:
                        rows.append(r)
        except Exception as e:
            print(f"Error reading CSV, creating new file: {e}")
            
    if not updated:
        rows.append(row_data)
        
    with open(CSV_FILE, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "date": r.get("date"),
                "avg_temp": r.get("avg_temp"),
                "avg_temp_48h": r.get("avg_temp_48h"),
                "avg_humidity": r.get("avg_humidity"),
                "energy_consumption": r.get("energy_consumption")
            })
            
    return row_data, "updated" if updated else "appended"

def scheduler_thread():
    print("Uruchamianie wątku tła dla codziennego logowania...")
    while True:
        try:
            now = datetime.now()
            today_str = now.strftime("%Y-%m-%d")
            
            with logging_lock:
                last_run = logging_status["last_run"]
                
            next_run_str = f"{today_str} 23:59:00"
            next_run_dt = datetime.strptime(next_run_str, "%Y-%m-%d %H:%M:%S")
            if now >= next_run_dt:
                next_run_dt = next_run_dt + timedelta(days=1)
            
            with logging_lock:
                logging_status["next_run"] = next_run_dt.strftime("%Y-%m-%d %H:%M:%S")
            
            if now.hour == 23 and now.minute == 59 and last_run != today_str:
                sensors = get_sensor_names()
                if any(sensors.values()):
                    print(f"Rozpoczęcie automatycznego dobowego logowania danych o {now}...")
                    row_data, action = log_daily_data(today_str)
                    
                    with logging_lock:
                        logging_status["last_run"] = today_str
                        logging_status["last_status"] = f"SUCCESS ({action})"
                        logging_status["last_error"] = None
                    print(f"Automatyczne logowanie zakończone sukcesem: {row_data} ({action})")
                else:
                    with logging_lock:
                        logging_status["last_status"] = "Skipped (no sensors configured)"
                    
        except Exception as e:
            print(f"Błąd w wątku schedulera: {e}")
            with logging_lock:
                logging_status["last_status"] = "ERROR"
                logging_status["last_error"] = str(e)
                
        time.sleep(30)

@app.route('/log_now', methods=['POST', 'GET'])
def log_now_view():
    try:
        row_data, action = log_daily_data()
        return jsonify({
            "status": "success",
            "message": f"Data successfully logged to CSV ({action})",
            "data": row_data
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/log_status', methods=['GET'])
def log_status_view():
    with logging_lock:
        status_copy = logging_status.copy()
    
    status_copy["configured_sensors"] = get_sensor_names()
    return jsonify(status_copy), 200

# =============================
# Uruchomienie serwera
# =============================
if __name__ == '__main__':
    print("="*70)
    print("SERWER FLASK - TRENOWANIE MODELU POMPY CIEPŁA")
    print("="*70)
    
    # Uruchomienie wątku tła do automatycznego logowania
    t = threading.Thread(target=scheduler_thread)
    t.daemon = True
    t.start()
    
    app.run(host='0.0.0.0', port=8000, debug=False)
