#!/usr/bin/env python3
"""
Flask server for the heat pump model training add-on.
Handles the non-linear (Random Forest) and linear (Linear Regression) models.
"""

from flask import Flask, jsonify, send_file, request, Response
import subprocess
import threading
import os
import pandas as pd
import joblib
import json 
import csv
import time
import requests
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

try:
    import websocket  # websocket-client
except ImportError:
    websocket = None

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

RF_MAE_ENTITY_ID = "sensor.rf_mae"
LR_MAE_ENTITY_ID = "sensor.lr_mae"
RF_R2_ENTITY_ID = "sensor.rf_r2"
LR_R2_ENTITY_ID = "sensor.lr_r2"
RF_METADATA_FILE = "/config/pump/model_metadata_rf.json"
LINEAR_METADATA_FILE = "/config/pump/model_metadata_linear.json"

RF_CHECK_SCRIPT = "/app/checkModel.py"
LINEAR_CHECK_SCRIPT = "/app/checkModel_linear.py"
RF_CHECK_METRICS_FILE = "/config/pump/dane.json"
LINEAR_CHECK_METRICS_FILE = "/config/pump/dane_linear.json"

RF_VERIFICATION_MAE_ENTITY_ID = "sensor.rf_verification_mae"
RF_VERIFICATION_R2_ENTITY_ID = "sensor.rf_verification_r2"
RF_VERIFICATION_MAPE_ENTITY_ID = "sensor.rf_verification_mean_error"
LR_VERIFICATION_MAE_ENTITY_ID = "sensor.lr_verification_mae"
LR_VERIFICATION_R2_ENTITY_ID = "sensor.lr_verification_r2"
LR_VERIFICATION_MAPE_ENTITY_ID = "sensor.lr_verification_mean_error"

VERIFICATION_SENSOR_IDS = [
    RF_VERIFICATION_MAE_ENTITY_ID,
    RF_VERIFICATION_R2_ENTITY_ID,
    RF_VERIFICATION_MAPE_ENTITY_ID,
    LR_VERIFICATION_MAE_ENTITY_ID,
    LR_VERIFICATION_R2_ENTITY_ID,
    LR_VERIFICATION_MAPE_ENTITY_ID,
]

# Daily automatic retraining schedule (replaces the manual Train buttons in the UI)
AUTO_TRAIN_HOUR = 0
AUTO_TRAIN_MINUTE = 10
auto_train_status = {"last_run": None}


def get_next_auto_training_time(now=None):
    now = now or datetime.now()
    candidate = now.replace(hour=AUTO_TRAIN_HOUR, minute=AUTO_TRAIN_MINUTE, second=0, microsecond=0)
    if now >= candidate:
        candidate += timedelta(days=1)
    return candidate


REQUIRED_SENSOR_OPTIONS = [
    ("temp_sensor", "Outdoor temperature sensor"),
    ("humidity_sensor", "Outdoor humidity sensor"),
    ("weather_forecast", "Weather forecast entity"),
    ("energy_consumption_sensor", "Energy consumption sensor"),
]


def _validate_temperature_sensor(entity_id, attributes):
    unit = attributes.get("unit_of_measurement")
    device_class = attributes.get("device_class")
    if device_class == "temperature" or unit in ("°C", "°F"):
        return None
    return f"'{entity_id}' doesn't look like a temperature sensor (unit='{unit}', device_class='{device_class}')"


def _validate_humidity_sensor(entity_id, attributes):
    unit = attributes.get("unit_of_measurement")
    device_class = attributes.get("device_class")
    if device_class == "humidity" or unit == "%":
        return None
    return f"'{entity_id}' doesn't look like a humidity sensor (unit='{unit}', device_class='{device_class}')"


def _validate_weather_forecast(entity_id, attributes):
    domain = entity_id.split(".", 1)[0] if "." in entity_id else ""
    if domain != "weather":
        return f"'{entity_id}' must be a weather entity (domain 'weather.', e.g. weather.home), not a '{domain}.*' entity"

    try:
        forecast = get_hourly_forecast(entity_id)
    except Exception as e:
        return f"'{entity_id}': could not fetch its hourly forecast from Home Assistant ({e})"

    if not forecast:
        return f"'{entity_id}' returned an empty hourly forecast"

    if not any("humidity" in entry for entry in forecast):
        return (
            f"'{entity_id}' hourly forecast doesn't include humidity data. "
            f"You can use the Met.no (Meteorologisk Institutt) integration to get a humidity forecast."
        )

    return None


def _validate_energy_sensor(entity_id, attributes):
    unit = (attributes.get("unit_of_measurement") or "")
    if unit.lower() != "kwh":
        return f"'{entity_id}' must report energy consumption in kWh (found unit='{unit or 'none'}')"
    return None


SENSOR_VALIDATORS = {
    "temp_sensor": _validate_temperature_sensor,
    "humidity_sensor": _validate_humidity_sensor,
    "weather_forecast": _validate_weather_forecast,
    "energy_consumption_sensor": _validate_energy_sensor,
}


def get_sensor_config_problems():
    """Checks each required sensor option is set, exists in HA, and looks like the right kind of entity.

    A wrong-but-present entity (e.g. a battery sensor typo'd into weather_forecast) doesn't raise
    HA API errors anywhere - it just silently fails to produce sensible computed values downstream.
    So this validates domain/unit/device_class up front instead of only checking the option isn't empty.
    """
    opts = get_addon_options()
    problems = []
    for key, label in REQUIRED_SENSOR_OPTIONS:
        entity_id = (opts.get(key) or "").strip()
        if not entity_id:
            problems.append(f"{label}: not configured")
            continue

        try:
            data = get_full_sensor_state(entity_id)
        except Exception as e:
            problems.append(f"{label} ('{entity_id}'): could not read this entity from Home Assistant ({e})")
            continue

        state = data.get("state")
        if state in (None, "unknown", "unavailable"):
            problems.append(f"{label} ('{entity_id}'): current state is '{state}'")
            continue

        validator = SENSOR_VALIDATORS.get(key)
        if validator:
            error = validator(entity_id, data.get("attributes", {}) or {})
            if error:
                problems.append(f"{label}: {error}")

    return problems


def render_config_error_page(problems):
    items_html = "".join(f"<li>{problem}</li>" for problem in problems)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Heat Pump Prediction - Configuration required</title>
<style>
    body {{
        margin: 0;
        min-height: 100vh;
        display: flex;
        align-items: center;
        justify-content: center;
        background-color: #0a0b10;
        color: #f3f4f6;
        font-family: 'Segoe UI', Arial, sans-serif;
    }}
    .box {{
        max-width: 560px;
        margin: 2rem;
        padding: 2rem 2.5rem;
        background: rgba(30, 32, 48, 0.6);
        border: 1px solid rgba(239, 68, 68, 0.35);
        border-radius: 16px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.6);
    }}
    h1 {{
        margin-top: 0;
        font-size: 1.4rem;
        color: #ef4444;
    }}
    p {{
        color: #9ca3af;
        line-height: 1.5;
    }}
    ul {{
        color: #f3f4f6;
        line-height: 1.8;
    }}
    code {{
        background: rgba(255,255,255,0.08);
        padding: 0.1rem 0.4rem;
        border-radius: 4px;
    }}
</style>
</head>
<body>
    <div class="box">
        <h1>Configuration required</h1>
        <p>The Heat Pump Prediction add-on cannot start its dashboard because of the following problem(s) with the configured sensors:</p>
        <ul>{items_html}</ul>
        <p>Open <code>Settings &rarr; Add-ons &rarr; Heat Pump prediction &rarr; Configuration</code> in Home Assistant, fix the entity ID(s) above, then restart the add-on.</p>
    </div>
</body>
</html>"""

# =============================
# Training functions
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
        publish_mae_sensor(RF_MAE_ENTITY_ID, RF_METADATA_FILE, "RF Model Test MAE")
        publish_r2_sensor(RF_R2_ENTITY_ID, RF_METADATA_FILE, "RF Model Test R2")
        if run_check_model_script(RF_CHECK_SCRIPT):
            publish_verification_sensors(
                RF_CHECK_METRICS_FILE,
                RF_VERIFICATION_MAE_ENTITY_ID,
                RF_VERIFICATION_R2_ENTITY_ID,
                RF_VERIFICATION_MAPE_ENTITY_ID,
                "RF Model"
            )


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
        publish_mae_sensor(LR_MAE_ENTITY_ID, LINEAR_METADATA_FILE, "Linear Model Test MAE")
        publish_r2_sensor(LR_R2_ENTITY_ID, LINEAR_METADATA_FILE, "Linear Model Test R2")
        if run_check_model_script(LINEAR_CHECK_SCRIPT):
            publish_verification_sensors(
                LINEAR_CHECK_METRICS_FILE,
                LR_VERIFICATION_MAE_ENTITY_ID,
                LR_VERIFICATION_R2_ENTITY_ID,
                LR_VERIFICATION_MAPE_ENTITY_ID,
                "Linear Model"
            )

# =============================
# Endpoints - info
# =============================
@app.route('/', endpoint='home')
def home_view():
    problems = get_sensor_config_problems()
    if problems:
        return render_config_error_page(problems), 503

    html_path = "index.html"
    if not os.path.exists(html_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        html_path = os.path.join(script_dir, "index.html")
        
    if os.path.exists(html_path):
        return send_file(html_path)
    else:
        return jsonify({
            "status": "error",
            "message": "Dashboard HTML file index.html not found!"
        }), 404

# =============================
# Endpoints - training
# =============================
@app.route('/train', methods=['POST', 'GET'])
def train_rf_view():
    with status_lock:
        if training_status["rf"]["is_training"]:
            return jsonify({"status": "error", "message": "RF is already training!"}), 409

    thread = threading.Thread(target=run_training_rf)
    thread.daemon = True
    thread.start()

    return jsonify({"status": "started", "message": "RF training started"}), 200


@app.route('/trainlinear', methods=['POST', 'GET'])
def train_linear_view():
    with status_lock:
        if training_status["linear"]["is_training"]:
            return jsonify({"status": "error", "message": "Linear is already training!"}), 409

    thread = threading.Thread(target=run_training_linear)
    thread.daemon = True
    thread.start()

    return jsonify({"status": "started", "message": "Linear training started"}), 200


# =============================
# Endpoints - status
# =============================
@app.route('/status')
def status_rf_view():
    # safe copy of the status (thread-safe)
    with status_lock:
        response = training_status["rf"].copy()

    response["next_training"] = get_next_auto_training_time().isoformat()

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
    # safe copy of the status (thread-safe)
    with status_lock:
        response = training_status["linear"].copy()

    response["next_training"] = get_next_auto_training_time().isoformat()

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
# Prediction function
# =============================
def predict_energy(model_path, avg_temp, avg_temp_48h, avg_humidity):
    if not os.path.exists(model_path):
        return {"status": "error", "message": f"Model file not found: {model_path}"}, 404
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
# Endpoints - predictions
# =============================
@app.route('/prediction', endpoint='predict_rf')
def prediction_rf_view():
    try:
        avg_temp = float(request.args.get("avg_temp", 0))
        avg_temp_48h = float(request.args.get("avg_temp_48h", -1))
        avg_humidity = float(request.args.get("avg_humidity", 80))
    except ValueError:
        return {"status": "error", "message": "Invalid parameter format"}, 400
    return predict_energy("/config/pump/heatpump_model_rf.pkl", avg_temp, avg_temp_48h, avg_humidity)

@app.route('/predictionlinear', endpoint='predict_linear')
def prediction_linear_view():
    try:
        avg_temp = float(request.args.get("avg_temp", 0))
        avg_temp_48h = float(request.args.get("avg_temp_48h", -1))
        avg_humidity = float(request.args.get("avg_humidity", 80))
    except ValueError:
        return {"status": "error", "message": "Invalid parameter format"}, 400
    return predict_energy("/config/pump/heatpump_model_linear.pkl", avg_temp, avg_temp_48h, avg_humidity)

# =============================
# Endpoints - verification (chart)
# =============================
def get_check_model_results(plot_path, metrics_file):
    """
    Returns the existing checkModel verification chart or metrics, read straight from disk.
    Does NOT regenerate them — the chart/metrics are (re)generated only after training
    completes (see run_check_model_script), since re-running checkModel on every page
    view/tab switch would be wasteful and pointless when nothing has changed.
    """
    show_plot = request.args.get("show_plot", "0") == "1"

    if show_plot:
        if os.path.exists(plot_path):
            return send_file(plot_path, mimetype="image/png")
        else:
            return jsonify({
                "status": "error",
                "message": f"Chart not found: {plot_path}"
            }), 404
    else:
        if os.path.exists(metrics_file):
            with open(metrics_file, "r") as f:
                metrics_data = json.load(f)
            return jsonify({"status": "success", "metrics": metrics_data})
        else:
            return jsonify({
                "status": "error",
                "message": f"Metrics file does not exist: {metrics_file}"
            }), 404

@app.route('/showpicresults', endpoint='show_rf')
def show_results_rf_view():
    return get_check_model_results(
        plot_path="/config/pump/plots/predicted_vs_actual_rf.png",
        metrics_file="/config/pump/dane.json"
    )

@app.route('/showpicresultslinear', endpoint='show_linear')
def show_results_linear_view():
    return get_check_model_results(
        plot_path="/config/pump/plots/predicted_vs_actual_linear.png",
        metrics_file="/config/pump/dane_linear.json"
    )

# =============================
# Automatic data logging
# =============================
OPTIONS_FILE = "/data/options.json"
FALLBACK_OPTIONS_FILE = "/config/heatpumptrain/options.json"
CSV_FILE = "/config/pump/daily_temps.csv"
MIN_TRAINING_ROWS = 2


def get_csv_row_count():
    """Counts data rows in CSV_FILE (excluding the header). Returns 0 if the file doesn't exist."""
    if not os.path.exists(CSV_FILE):
        return 0
    with open(CSV_FILE, mode='r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        return sum(1 for _ in reader)

logging_status = {
    "last_run": None,
    "last_status": "Idle",
    "last_error": None,
    "next_run": None
}
logging_lock = threading.Lock()

LOGGING_STATE_FILE = "/config/pump/logging_state.json"


def get_logging_enabled():
    """Whether automatic daily CSV logging is turned on. Defaults to True (enabled) if never toggled."""
    if os.path.exists(LOGGING_STATE_FILE):
        try:
            with open(LOGGING_STATE_FILE, "r") as f:
                return bool(json.load(f).get("enabled", True))
        except Exception as e:
            print(f"Error reading {LOGGING_STATE_FILE}: {e}")
    return True


def set_logging_enabled(enabled):
    os.makedirs(os.path.dirname(LOGGING_STATE_FILE), exist_ok=True)
    with open(LOGGING_STATE_FILE, "w") as f:
        json.dump({"enabled": bool(enabled)}, f)

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
        "avg_temp": COMPUTED_SENSOR_IDS["temp_today"],
        "avg_temp_48h": COMPUTED_SENSOR_IDS["avg_temp_48h"],
        "avg_humidity": COMPUTED_SENSOR_IDS["humidity_today"],
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

def get_full_sensor_state(entity_id):
    token = os.environ.get("SUPERVISOR_TOKEN") or os.environ.get("HA_TOKEN")
    if not token:
        raise ValueError("SUPERVISOR_TOKEN or HA_TOKEN environment variable not set")

    ha_url = os.environ.get("HA_URL", "http://supervisor/core/api")
    url = f"{ha_url}/states/{entity_id}"

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    return response.json()

def push_sensor_state(entity_id, state, attributes=None):
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
    payload = {"state": state}
    if attributes:
        payload["attributes"] = attributes

    response = requests.post(url, headers=headers, json=payload, timeout=10)
    response.raise_for_status()

def publish_mae_sensor(entity_id, metadata_file, friendly_name):
    if not os.path.exists(metadata_file):
        return
    try:
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        mae = metadata.get("mae_test_kwh")
        if mae is None:
            return
        push_sensor_state(entity_id, round(mae, 4), attributes={
            "unit_of_measurement": "kWh",
            "friendly_name": friendly_name,
        })
    except Exception as e:
        print(f"Error publishing sensor {entity_id}: {e}")

def publish_r2_sensor(entity_id, metadata_file, friendly_name):
    if not os.path.exists(metadata_file):
        return
    try:
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        r2 = metadata.get("r2_test")
        if r2 is None:
            return
        push_sensor_state(entity_id, round(r2, 4), attributes={
            "friendly_name": friendly_name,
        })
    except Exception as e:
        print(f"Error publishing sensor {entity_id}: {e}")

def run_check_model_script(script_path):
    """
    Runs a checkModel script (RF or Linear) to regenerate its verification chart
    and metrics JSON (dane.json / dane_linear.json). Returns True on success.
    """
    try:
        result = subprocess.run(
            ["python3", script_path],
            capture_output=True,
            text=True,
            timeout=300
        )
        if result.returncode != 0:
            print(f"Error running {script_path}: {result.stderr}")
            return False
        return True
    except Exception as e:
        print(f"Error running {script_path}: {e}")
        return False

def publish_verification_sensors(metrics_file, mae_entity, r2_entity, mape_entity, friendly_prefix):
    """
    Publishes MAE/R2/MAPE from a checkModel metrics file (dane.json / dane_linear.json)
    as HA sensors. Unlike the MAE/R2 sensors from training metadata (which use a held-out test
    split), these are computed by checkModel over the full CSV history.
    """
    if not os.path.exists(metrics_file):
        return
    try:
        with open(metrics_file, "r") as f:
            metrics = json.load(f)

        mae = metrics.get("mae")
        if mae is not None:
            push_sensor_state(mae_entity, mae, attributes={
                "unit_of_measurement": "kWh",
                "friendly_name": f"{friendly_prefix} Verification MAE",
            })

        r2 = metrics.get("r2")
        if r2 is not None:
            push_sensor_state(r2_entity, r2, attributes={
                "friendly_name": f"{friendly_prefix} Verification R2",
            })

        mape = metrics.get("mape")
        if mape is not None:
            push_sensor_state(mape_entity, mape, attributes={
                "unit_of_measurement": "%",
                "friendly_name": f"{friendly_prefix} Verification Mean Error (%)",
            })
    except Exception as e:
        print(f"Error publishing verification sensors from {metrics_file}: {e}")

# =============================
# Weather / temperature computed sensors
# =============================
COMPUTED_SENSOR_IDS = {
    "temp_today": "sensor.heat_pump_pred_avg_temp_today_actual_and_forcast",
    "temp_tomorrow": "sensor.heat_pump_pred_avg_temp_tomorrow_forecast",
    "humidity_today": "sensor.heat_pump_pred_avg_humidity_today_actual_and_forcast",
    "humidity_tomorrow": "sensor.heat_pump_pred_avg_humidity_tomorrow_forecast",
    "avg_temp_48h": "sensor.heat_pump_pred_avg_temp_48h",
}

def _ha_headers():
    token = os.environ.get("SUPERVISOR_TOKEN") or os.environ.get("HA_TOKEN")
    if not token:
        raise ValueError("SUPERVISOR_TOKEN or HA_TOKEN environment variable not set")
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

def get_history_avg(entity_id, start_dt, end_dt, value_range):
    """Time-weighted average of an entity's numeric states between start_dt and end_dt via HA's History REST API.

    A plain arithmetic mean over raw state-change samples is wrong here: state changes
    land at irregular intervals, so a burst of readings crammed into a few seconds would
    outweigh a value that was genuinely held for hours. Each sample is instead weighted
    by how long it remained in effect within the window.
    """
    ha_url = os.environ.get("HA_URL", "http://supervisor/core/api")
    start_iso = start_dt.astimezone().isoformat()
    end_iso = end_dt.astimezone().isoformat()
    url = f"{ha_url}/history/period/{quote(start_iso, safe='')}"
    params = {
        "filter_entity_id": entity_id,
        "end_time": end_iso,
        "minimal_response": "",
        "no_attributes": "",
        "significant_changes_only": "0",
    }
    response = requests.get(url, headers=_ha_headers(), params=params, timeout=15)
    response.raise_for_status()
    data = response.json()
    if not data or not data[0]:
        return None, 0, []

    low, high = value_range
    samples = []
    for entry in data[0]:
        state_str = entry.get("state")
        if state_str in (None, "unknown", "unavailable", "none", ""):
            continue
        try:
            value = round(float(state_str), 1)
        except (TypeError, ValueError):
            continue
        if not (low <= value <= high):
            continue
        last_changed = entry.get("last_changed")
        if not last_changed:
            continue
        try:
            ts = datetime.fromisoformat(last_changed.replace("Z", "+00:00"))
        except ValueError:
            continue
        samples.append((ts, value))

    if not samples:
        return None, 0, []

    samples.sort(key=lambda pair: pair[0])
    start_utc = start_dt.astimezone(timezone.utc)
    end_utc = end_dt.astimezone(timezone.utc)

    weighted_sum = 0.0
    total_seconds = 0.0
    values = []
    for i, (ts, value) in enumerate(samples):
        segment_start = max(ts, start_utc)
        segment_end = samples[i + 1][0] if i + 1 < len(samples) else end_utc
        segment_end = min(max(segment_end, segment_start), end_utc)
        duration = (segment_end - segment_start).total_seconds()
        values.append(value)
        if duration <= 0:
            continue
        weighted_sum += value * duration
        total_seconds += duration

    if total_seconds <= 0:
        return round(sum(values) / len(values), 2), len(values), values
    return round(weighted_sum / total_seconds, 2), len(values), values


def get_hourly_forecast(weather_entity):
    """Hourly forecast list for a weather entity via the weather.get_forecasts service call."""
    ha_url = os.environ.get("HA_URL", "http://supervisor/core/api")
    url = f"{ha_url}/services/weather/get_forecasts"
    payload = {"entity_id": weather_entity, "type": "hourly"}
    response = requests.post(
        url, headers=_ha_headers(), params={"return_response": ""}, json=payload, timeout=15
    )
    response.raise_for_status()
    data = response.json()
    service_response = data.get("service_response", {}) if isinstance(data, dict) else {}
    entity_data = service_response.get(weather_entity)
    if not entity_data:
        raise RuntimeError(f"No forecast data returned for {weather_entity}")
    return entity_data.get("forecast", [])


def filter_forecast_range(forecast, start_dt, end_dt, field):
    """Average of a numeric field ('temperature'/'humidity') across forecast entries within [start_dt, end_dt)."""
    values = []
    for entry in forecast:
        if field not in entry:
            continue
        try:
            entry_dt = datetime.fromisoformat(entry["datetime"]).replace(tzinfo=None)
        except (TypeError, ValueError):
            continue
        if start_dt <= entry_dt < end_dt:
            try:
                values.append(round(float(entry[field]), 1))
            except (TypeError, ValueError):
                continue

    if not values:
        return None, 0, []
    return round(sum(values) / len(values), 2), len(values), values


def weighted_combine(actual_avg, actual_weight, forecast_avg, forecast_weight):
    """Weighted average of an 'actual' average and a 'forecast' average.

    Both weights must be in the same unit (hours) for this to make sense: the actual
    side is already a time-weighted average over the hours elapsed so far today, and
    the forecast side is an average over N remaining forecast hours. Weighting by raw
    sample counts instead (e.g. number of state changes) would be wrong, since a
    frequently-changing sensor would then dominate a stable one that covers the same
    span of real time.
    """
    if actual_avg is not None and forecast_avg is not None:
        total = actual_weight + forecast_weight
        if total <= 0:
            return round((actual_avg + forecast_avg) / 2, 1)
        return round((actual_avg * actual_weight + forecast_avg * forecast_weight) / total, 1)
    if actual_avg is not None:
        return actual_avg
    return forecast_avg


def update_temp_today(temp_sensor, weather_entity):
    now = datetime.now()
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    today_end = now.replace(hour=23, minute=59, second=59, microsecond=999999)

    actual_avg, actual_n, actual_values = get_history_avg(temp_sensor, midnight, now, (-50, 50))
    actual_hours = (now - midnight).total_seconds() / 3600
    forecast = get_hourly_forecast(weather_entity)
    forecast_avg, forecast_n, forecast_values = filter_forecast_range(forecast, now, today_end, "temperature")

    if actual_avg is None and forecast_avg is None:
        raise RuntimeError("No actual or forecast temperature data available for today")

    combined = weighted_combine(actual_avg, actual_hours, forecast_avg, forecast_n)

    push_sensor_state(COMPUTED_SENSOR_IDS["temp_today"], combined, attributes={
        "unit_of_measurement": "°C",
        "device_class": "temperature",
        "state_class": "measurement",
        "friendly_name": "Average temperature today (actual+forecast)",
        "actual_avg": actual_avg,
        "forecast_avg": forecast_avg,
        "actual_values": actual_values,
        "forecast_values": forecast_values,
        "actual_sample_count": actual_n,
        "actual_hours": round(actual_hours, 2),
        "forecast_hour_count": forecast_n,
        "last_update": now.isoformat(),
        "source_sensor": temp_sensor,
        "source_weather": weather_entity,
    })


def update_temp_tomorrow(weather_entity):
    now = datetime.now()
    tomorrow_start = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    tomorrow_end = tomorrow_start + timedelta(days=1)

    forecast = get_hourly_forecast(weather_entity)
    avg, n, values = filter_forecast_range(forecast, tomorrow_start, tomorrow_end, "temperature")

    if avg is None:
        raise RuntimeError("No forecast temperature data available for tomorrow")

    push_sensor_state(COMPUTED_SENSOR_IDS["temp_tomorrow"], avg, attributes={
        "unit_of_measurement": "°C",
        "device_class": "temperature",
        "state_class": "measurement",
        "friendly_name": "Average temperature tomorrow (forecast)",
        "values": values,
        "hour_count": n,
        "last_update": now.isoformat(),
        "source_weather": weather_entity,
    })


def update_humidity_today(humidity_sensor, weather_entity):
    now = datetime.now()
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    today_end = now.replace(hour=23, minute=59, second=59, microsecond=999999)

    actual_avg, actual_n, actual_values = get_history_avg(humidity_sensor, midnight, now, (0, 100))
    actual_hours = (now - midnight).total_seconds() / 3600
    forecast = get_hourly_forecast(weather_entity)
    forecast_avg, forecast_n, forecast_values = filter_forecast_range(forecast, now, today_end, "humidity")

    if actual_avg is None and forecast_avg is None:
        raise RuntimeError("No actual or forecast humidity data available for today")

    combined = weighted_combine(actual_avg, actual_hours, forecast_avg, forecast_n)

    push_sensor_state(COMPUTED_SENSOR_IDS["humidity_today"], combined, attributes={
        "unit_of_measurement": "%",
        "device_class": "humidity",
        "state_class": "measurement",
        "friendly_name": "Average humidity today (actual+forecast)",
        "actual_avg": actual_avg,
        "forecast_avg": forecast_avg,
        "actual_values": actual_values,
        "forecast_values": forecast_values,
        "actual_sample_count": actual_n,
        "actual_hours": round(actual_hours, 2),
        "forecast_hour_count": forecast_n,
        "last_update": now.isoformat(),
        "source_sensor": humidity_sensor,
        "source_weather": weather_entity,
    })


def update_humidity_tomorrow(weather_entity):
    now = datetime.now()
    tomorrow_start = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    tomorrow_end = tomorrow_start + timedelta(days=1)

    forecast = get_hourly_forecast(weather_entity)
    avg, n, values = filter_forecast_range(forecast, tomorrow_start, tomorrow_end, "humidity")

    if avg is None:
        raise RuntimeError("No forecast humidity data available for tomorrow")

    push_sensor_state(COMPUTED_SENSOR_IDS["humidity_tomorrow"], avg, attributes={
        "unit_of_measurement": "%",
        "device_class": "humidity",
        "state_class": "measurement",
        "friendly_name": "Average humidity tomorrow (forecast)",
        "values": values,
        "hour_count": n,
        "last_update": now.isoformat(),
        "source_weather": weather_entity,
    })


def update_avg_temp_48h(temp_sensor):
    now = datetime.now()
    start = now - timedelta(hours=48)

    avg, n, values = get_history_avg(temp_sensor, start, now, (-50, 50))
    if avg is None:
        raise RuntimeError("No temperature history available for the last 48h")

    push_sensor_state(COMPUTED_SENSOR_IDS["avg_temp_48h"], avg, attributes={
        "unit_of_measurement": "°C",
        "device_class": "temperature",
        "state_class": "measurement",
        "friendly_name": "Average temperature (last 48h)",
        "sample_count": n,
        "last_update": now.isoformat(),
        "source_sensor": temp_sensor,
    })


def run_weather_sensor_updates():
    opts = get_addon_options()
    temp_sensor = opts.get("temp_sensor", "")
    humidity_sensor = opts.get("humidity_sensor", "")
    weather_entity = opts.get("weather_forecast", "")

    if temp_sensor and weather_entity:
        try:
            update_temp_today(temp_sensor, weather_entity)
        except Exception as e:
            print(f"Error updating today's temperature sensor: {e}")
    else:
        print("Skipping today's temperature sensor: temp_sensor/weather_forecast not configured")

    if weather_entity:
        try:
            update_temp_tomorrow(weather_entity)
        except Exception as e:
            print(f"Error updating tomorrow's temperature sensor: {e}")
    else:
        print("Skipping tomorrow's temperature sensor: weather_forecast not configured")

    if humidity_sensor and weather_entity:
        try:
            update_humidity_today(humidity_sensor, weather_entity)
        except Exception as e:
            print(f"Error updating today's humidity sensor: {e}")
    else:
        print("Skipping today's humidity sensor: humidity_sensor/weather_forecast not configured")

    if weather_entity:
        try:
            update_humidity_tomorrow(weather_entity)
        except Exception as e:
            print(f"Error updating tomorrow's humidity sensor: {e}")
    else:
        print("Skipping tomorrow's humidity sensor: weather_forecast not configured")

    if temp_sensor:
        try:
            update_avg_temp_48h(temp_sensor)
        except Exception as e:
            print(f"Error updating 48h average temperature sensor: {e}")
    else:
        print("Skipping 48h average temperature sensor: temp_sensor not configured")


# =============================
# Real-time energy prediction sensors (RF & Linear, today & tomorrow)
# Recomputed whenever temp_sensor/humidity_sensor changes (via HA WebSocket),
# after the weather-blend sensors above have been refreshed
# =============================
PREDICTION_SENSOR_IDS = {
    "today_rf": "sensor.heat_pump_pred_today_rf",
    "today_linear": "sensor.heat_pump_pred_today_linear",
    "tomorrow_rf": "sensor.heat_pump_pred_tomorrow_rf",
    "tomorrow_linear": "sensor.heat_pump_pred_tomorrow_linear",
}

def _predict_and_push_sensor(sensor_key, model_path, avg_temp, avg_temp_48h, avg_humidity, friendly_name):
    result, status_code = predict_energy(model_path, avg_temp, avg_temp_48h, avg_humidity)
    if status_code != 200 or result.get("status") != "success":
        raise RuntimeError(result.get("message", "Prediction failed"))

    energy_kwh = result["prediction"]["energy_kwh"]
    push_sensor_state(PREDICTION_SENSOR_IDS[sensor_key], energy_kwh, attributes={
        "unit_of_measurement": "kWh",
        "device_class": "energy",
        "state_class": "measurement",
        "friendly_name": friendly_name,
        "avg_temp": avg_temp,
        "avg_temp_48h": avg_temp_48h,
        "avg_humidity": avg_humidity,
        "last_update": datetime.now().isoformat(),
    })
    return energy_kwh

def update_prediction_sensors():
    """Recompute today/tomorrow RF & Linear energy predictions from the computed weather sensors."""
    try:
        avg_temp_48h = get_sensor_state(COMPUTED_SENSOR_IDS["avg_temp_48h"])
    except Exception as e:
        print(f"Skipping prediction sensor update, 48h average temperature unavailable: {e}")
        return

    try:
        avg_temp_today = get_sensor_state(COMPUTED_SENSOR_IDS["temp_today"])
        avg_humidity_today = get_sensor_state(COMPUTED_SENSOR_IDS["humidity_today"])
        _predict_and_push_sensor("today_rf", "/config/pump/heatpump_model_rf.pkl",
                                  avg_temp_today, avg_temp_48h, avg_humidity_today,
                                  "Predicted energy consumption today (Random Forest)")
        _predict_and_push_sensor("today_linear", "/config/pump/heatpump_model_linear.pkl",
                                  avg_temp_today, avg_temp_48h, avg_humidity_today,
                                  "Predicted energy consumption today (Linear Regression)")
    except Exception as e:
        print(f"Error updating today's prediction sensors: {e}")

    try:
        avg_temp_tomorrow = get_sensor_state(COMPUTED_SENSOR_IDS["temp_tomorrow"])
        avg_humidity_tomorrow = get_sensor_state(COMPUTED_SENSOR_IDS["humidity_tomorrow"])
        _predict_and_push_sensor("tomorrow_rf", "/config/pump/heatpump_model_rf.pkl",
                                  avg_temp_tomorrow, avg_temp_48h, avg_humidity_tomorrow,
                                  "Predicted energy consumption tomorrow (Random Forest)")
        _predict_and_push_sensor("tomorrow_linear", "/config/pump/heatpump_model_linear.pkl",
                                  avg_temp_tomorrow, avg_temp_48h, avg_humidity_tomorrow,
                                  "Predicted energy consumption tomorrow (Linear Regression)")
    except Exception as e:
        print(f"Error updating tomorrow's prediction sensors: {e}")


def _ha_ws_url():
    if os.environ.get("SUPERVISOR_TOKEN"):
        return "ws://supervisor/core/websocket"

    ha_url = os.environ.get("HA_URL", "http://supervisor/core/api")
    base = ha_url[:-4] if ha_url.endswith("/api") else ha_url
    base = base.replace("https://", "wss://").replace("http://", "ws://")
    return f"{base}/api/websocket"

_last_watched_sensor_states = {}

def watched_sensor_ws_listener():
    """
    Keeps a persistent WebSocket connection to Home Assistant open, subscribed to
    state_changed events for temp_sensor and humidity_sensor, and recomputes the
    weather-blend sensors (today/tomorrow temp & humidity, 48h average) plus the
    RF/Linear prediction sensors every time either source sensor's state changes.
    """
    global _last_watched_sensor_states

    if websocket is None:
        print("websocket-client is not installed; real-time sensor listener disabled")
        return

    token = os.environ.get("SUPERVISOR_TOKEN") or os.environ.get("HA_TOKEN")
    if not token:
        print("No HA token available; real-time sensor listener disabled")
        return

    ws_url = _ha_ws_url()

    while True:
        opts = get_addon_options()
        watched_entities = {e for e in (opts.get("temp_sensor", ""), opts.get("humidity_sensor", "")) if e}
        if not watched_entities:
            print("temp_sensor/humidity_sensor not configured; retrying WebSocket listener in 30s")
            time.sleep(30)
            continue

        ws = None
        try:
            ws = websocket.create_connection(ws_url, timeout=30)

            auth_required = json.loads(ws.recv())
            if auth_required.get("type") != "auth_required":
                raise RuntimeError(f"Unexpected WebSocket handshake message: {auth_required}")

            ws.send(json.dumps({"type": "auth", "access_token": token}))
            auth_result = json.loads(ws.recv())
            if auth_result.get("type") != "auth_ok":
                raise RuntimeError(f"WebSocket authentication failed: {auth_result}")

            ws.send(json.dumps({"id": 1, "type": "subscribe_events", "event_type": "state_changed"}))
            sub_result = json.loads(ws.recv())
            if not sub_result.get("success", False):
                raise RuntimeError(f"Failed to subscribe to state_changed events: {sub_result}")

            print(f"HA WebSocket connected, watching {', '.join(sorted(watched_entities))} for changes")

            while True:
                message = json.loads(ws.recv())
                if message.get("type") != "event":
                    continue

                event_data = message.get("event", {}).get("data", {})
                entity_id = event_data.get("entity_id")
                if entity_id not in watched_entities:
                    continue

                new_state = event_data.get("new_state") or {}
                state_str = new_state.get("state")
                if state_str in (None, "unavailable", "unknown") or state_str == _last_watched_sensor_states.get(entity_id):
                    continue

                _last_watched_sensor_states[entity_id] = state_str
                print(f"{entity_id} changed to {state_str}; recomputing weather and prediction sensors")
                try:
                    run_weather_sensor_updates()
                    update_prediction_sensors()
                except Exception as e:
                    print(f"Error recomputing sensors after {entity_id} change: {e}")

        except Exception as e:
            print(f"HA WebSocket listener error: {e}; reconnecting in 10s")
            time.sleep(10)
        finally:
            if ws is not None:
                try:
                    ws.close()
                except Exception:
                    pass


def fetch_all_sensors():
    sensors = get_sensor_names()
    missing = [k for k, v in sensors.items() if not v]
    if missing:
        raise ValueError(f"No sensors configured for: {', '.join(missing)}")

    results = {}
    for name, entity_id in sensors.items():
        try:
            results[name] = get_sensor_state(entity_id)
        except Exception as e:
            raise RuntimeError(f"Error fetching sensor '{name}' ({entity_id}): {e}")
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

_last_weather_update_hour = None

def scheduler_thread():
    global _last_weather_update_hour
    print("Starting background thread for daily logging...")
    while True:
        try:
            now = datetime.now()
            today_str = now.strftime("%Y-%m-%d")

            hour_key = now.strftime("%Y-%m-%d %H")
            if now.minute == 1 and _last_weather_update_hour != hour_key:
                print(f"Running hourly weather/temperature sensor updates at {now}...")
                run_weather_sensor_updates()
                update_prediction_sensors()
                _last_weather_update_hour = hour_key

            if now.hour == AUTO_TRAIN_HOUR and now.minute == AUTO_TRAIN_MINUTE:
                with status_lock:
                    already_ran_today = auto_train_status["last_run"] == today_str
                if not already_ran_today:
                    print(f"Starting automatic daily model retraining at {now}...")
                    with status_lock:
                        auto_train_status["last_run"] = today_str
                        rf_idle = not training_status["rf"]["is_training"]
                        linear_idle = not training_status["linear"]["is_training"]
                    if rf_idle:
                        threading.Thread(target=run_training_rf, daemon=True).start()
                    if linear_idle:
                        threading.Thread(target=run_training_linear, daemon=True).start()

            with logging_lock:
                last_run = logging_status["last_run"]

            next_run_str = f"{today_str} 23:59:00"
            next_run_dt = datetime.strptime(next_run_str, "%Y-%m-%d %H:%M:%S")
            if now >= next_run_dt:
                next_run_dt = next_run_dt + timedelta(days=1)

            with logging_lock:
                logging_status["next_run"] = next_run_dt.strftime("%Y-%m-%d %H:%M:%S")

            if now.hour == 23 and now.minute == 59 and last_run != today_str:
                if not get_logging_enabled():
                    with logging_lock:
                        logging_status["last_status"] = "Skipped (data logging is turned off)"
                elif get_addon_options().get("energy_consumption_sensor"):
                    print(f"Starting automatic daily data logging at {now}...")
                    row_data, action = log_daily_data(today_str)

                    with logging_lock:
                        logging_status["last_run"] = today_str
                        logging_status["last_status"] = f"SUCCESS ({action})"
                        logging_status["last_error"] = None
                    print(f"Automatic logging completed successfully: {row_data} ({action})")
                else:
                    with logging_lock:
                        logging_status["last_status"] = "Skipped (no sensors configured)"

        except Exception as e:
            print(f"Error in scheduler thread: {e}")
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

@app.route('/log_toggle', methods=['POST'])
def log_toggle_view():
    try:
        new_state = not get_logging_enabled()
        set_logging_enabled(new_state)
        return jsonify({
            "status": "success",
            "logging_enabled": new_state,
            "message": f"Automatic data logging turned {'ON' if new_state else 'OFF'}"
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

    status_copy["logging_enabled"] = get_logging_enabled()
    status_copy["configured_sensors"] = get_sensor_names()

    row_count = get_csv_row_count()
    status_copy["csv_row_count"] = row_count
    status_copy["min_training_rows"] = MIN_TRAINING_ROWS
    if row_count < MIN_TRAINING_ROWS:
        status_copy["training_data_warning"] = (
            f"Only {row_count} day(s) of data logged. "
            f"At least {MIN_TRAINING_ROWS} days of data are needed before the model can be trained."
        )
    else:
        status_copy["training_data_warning"] = None

    return jsonify(status_copy), 200


@app.route('/computed_sensors', methods=['GET'])
def computed_sensors_view():
    sensors = []
    for entity_id in list(COMPUTED_SENSOR_IDS.values()) + list(PREDICTION_SENSOR_IDS.values()) + VERIFICATION_SENSOR_IDS:
        entry = {"entity_id": entity_id}
        try:
            data = get_full_sensor_state(entity_id)
            attributes = data.get("attributes", {})
            entry["state"] = data.get("state")
            entry["friendly_name"] = attributes.get("friendly_name", entity_id)
            entry["unit_of_measurement"] = attributes.get("unit_of_measurement")
            entry["last_update"] = attributes.get("last_update")
            entry["available"] = True
        except Exception as e:
            entry["state"] = None
            entry["friendly_name"] = entity_id
            entry["unit_of_measurement"] = None
            entry["last_update"] = None
            entry["available"] = False
            entry["error"] = str(e)
        sensors.append(entry)

    return jsonify({"status": "success", "sensors": sensors}), 200


CSV_COLUMNS = ["date", "avg_temp", "avg_temp_48h", "avg_humidity", "energy_consumption"]

@app.route('/csv_data', methods=['GET'])
def csv_data_view():
    if not os.path.exists(CSV_FILE):
        return jsonify({"status": "error", "message": "CSV data file not found", "rows": []}), 404

    try:
        with open(CSV_FILE, mode='r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = [{col: row.get(col, "") for col in CSV_COLUMNS} for row in reader]

        # jsonify sorts keys alphabetically, so we build the JSON manually
        # to preserve the requested column order
        payload = json.dumps({"status": "success", "columns": CSV_COLUMNS, "rows": rows}, ensure_ascii=False)
        return Response(payload, mimetype='application/json'), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# =============================
# Endpoint - list of available endpoints
# =============================
ENDPOINT_INFO = {
    'home': {
        "description": "Control panel (dashboard HTML)",
    },
    'train_rf_view': {
        "description": "Starts Random Forest model training in the background",
    },
    'train_linear_view': {
        "description": "Starts Linear Regression model training in the background",
    },
    'status_rf_view': {
        "description": "RF model training status and metrics",
    },
    'status_linear_view': {
        "description": "Linear model training status and metrics",
    },
    'predict_rf': {
        "description": "Energy consumption prediction using the RF model (parameters: avg_temp, avg_temp_48h, avg_humidity)",
        "usage": "/prediction?avg_temp=16.5&avg_temp_48h=15.12&avg_humidity=12",
    },
    'predict_linear': {
        "description": "Energy consumption prediction using the Linear model (parameters: avg_temp, avg_temp_48h, avg_humidity)",
        "usage": "/predictionlinear?avg_temp=16.5&avg_temp_48h=15.12&avg_humidity=12",
    },
    'show_rf': {
        "description": "RF model verification chart or metrics, generated after training (parameter: show_plot=1 for the PNG chart)",
        "usage": "/showpicresults?show_plot=1",
    },
    'show_linear': {
        "description": "Linear model verification chart or metrics, generated after training (parameter: show_plot=1 for the PNG chart)",
        "usage": "/showpicresultslinear?show_plot=1",
    },
    'log_now_view': {
        "description": "Manually logs the current sensor readings to CSV",
    },
    'log_status_view': {
        "description": "Status of automatic daily data logging",
    },
    'csv_data_view': {
        "description": "Returns data from daily_temps.csv as JSON (columns: date, avg_temp, avg_temp_48h, avg_humidity, energy_consumption)",
    },
    'computed_sensors_view': {
        "description": "Current state of the weather-averaging and RF/Linear energy prediction sensors this add-on computes and pushes to Home Assistant",
    },
    'list_endpoints_view': {
        "description": "List of all available endpoints with descriptions",
    },
}

@app.route('/list', methods=['GET'])
def list_endpoints_view():
    endpoints = []
    for rule in app.url_map.iter_rules():
        if rule.endpoint == 'static':
            continue
        methods = sorted(m for m in rule.methods if m not in ('HEAD', 'OPTIONS'))
        info = ENDPOINT_INFO.get(rule.endpoint, {})
        endpoints.append({
            "path": str(rule),
            "methods": methods,
            "description": info.get("description", ""),
            "usage": info.get("usage", "")
        })
    endpoints.sort(key=lambda e: e["path"])
    return jsonify({"status": "success", "endpoints": endpoints}), 200

# =============================
# Server startup
# =============================
if __name__ == '__main__':
    print("="*70)
    print("FLASK SERVER - HEAT PUMP MODEL TRAINING")
    print("="*70)

    # Publish HA sensors from any existing metadata so they're available right after a restart
    publish_mae_sensor(RF_MAE_ENTITY_ID, RF_METADATA_FILE, "RF Model Test MAE")
    publish_mae_sensor(LR_MAE_ENTITY_ID, LINEAR_METADATA_FILE, "Linear Model Test MAE")
    publish_r2_sensor(RF_R2_ENTITY_ID, RF_METADATA_FILE, "RF Model Test R2")
    publish_r2_sensor(LR_R2_ENTITY_ID, LINEAR_METADATA_FILE, "Linear Model Test R2")
    publish_verification_sensors(
        RF_CHECK_METRICS_FILE,
        RF_VERIFICATION_MAE_ENTITY_ID,
        RF_VERIFICATION_R2_ENTITY_ID,
        RF_VERIFICATION_MAPE_ENTITY_ID,
        "RF Model"
    )
    publish_verification_sensors(
        LINEAR_CHECK_METRICS_FILE,
        LR_VERIFICATION_MAE_ENTITY_ID,
        LR_VERIFICATION_R2_ENTITY_ID,
        LR_VERIFICATION_MAPE_ENTITY_ID,
        "Linear Model"
    )

    # Seed the weather/temperature computed sensors so they aren't empty until the next hourly update
    try:
        run_weather_sensor_updates()
    except Exception as e:
        print(f"Error running initial weather sensor updates: {e}")

    # Seed the RF/Linear prediction sensors so they aren't empty until the next temp_sensor change
    try:
        update_prediction_sensors()
    except Exception as e:
        print(f"Error running initial prediction sensor updates: {e}")

    # Start the background thread for automatic logging
    t = threading.Thread(target=scheduler_thread)
    t.daemon = True
    t.start()

    # Start the background thread listening for temp_sensor/humidity_sensor changes over HA's WebSocket API
    t_ws = threading.Thread(target=watched_sensor_ws_listener)
    t_ws.daemon = True
    t_ws.start()

    app.run(host='0.0.0.0', port=8000, debug=False)
