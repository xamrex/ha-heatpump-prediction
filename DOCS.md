# Heat Pump Prediction — Documentation

Home Assistant add-on that trains and runs Random Forest and Linear Regression
models to predict your heat pump's daily energy consumption from outdoor
temperature, humidity, and weather forecast history. It also computes a set of
supporting sensors (blended actual + forecast temperature/humidity, live
energy predictions, and model accuracy metrics) and publishes them back to
Home Assistant so they can be used on dashboards or in automations.

## 1. How to use

### 1.1 Configure the required sensors

Before starting the add-on, open **Settings → Add-ons → Heat Pump prediction →
Configuration** and fill in four entity IDs:

| Option | What it should point to | Example |
|---|---|---|
| `temp_sensor` | Current outdoor temperature sensor | `sensor.outdoor_temperature` |
| `humidity_sensor` | Current outdoor humidity sensor | `sensor.outdoor_humidity` |
| `weather_forecast` | A weather entity that provides an **hourly forecast** (temperature + humidity) | `weather.home` |
| `energy_consumption_sensor` | Heat pump's **daily** energy consumption sensor, in kWh | `sensor.heat_pump_daily_energy` |

The add-on validates these on every dashboard load. If an entity is missing,
unavailable, or doesn't look like the right kind of sensor (wrong unit or
device class), it blocks the dashboard and shows exactly what's wrong instead
of starting up with broken sensors.

**Tips:**
- If you don't have a forecast sensor, install the **Met.no (Meteorologisk
  Institutt)** integration — its `weather.*` entity provides hourly
  temperature and humidity forecasts out of the box.
- If your weather entity doesn't expose current humidity as a plain sensor,
  create a template helper:
  `{{ state_attr('weather.<your_weather_entity>', 'humidity') }}`

### 1.2 Let it collect data

Once configured, the add-on automatically logs one row per day (`date`,
`avg_temp`, `avg_temp_48h`, `avg_humidity`, `energy_consumption`) to
`/config/pump/daily_temps.csv`. This history is what the models are trained
on — the more days logged, the better the predictions.

- **At least 10 days of history are required for the R² metric to be a valid
  number.** With fewer than 10 rows, R² is mathematically undefined
  ("Fit (R²)" will show a tooltip explaining this in the dashboard).
- Training itself only needs 2+ rows to run, but accuracy with very little
  data will be poor.

### 1.3 Train the models

Models retrain automatically every day at **00:10**. You can also trigger
training manually from the dashboard (or by calling `/train` /
`/trainlinear`). Training runs in the background; the dashboard polls
`/status` and `/statuslinear` for progress.

### 1.4 Read the results

The dashboard (served on the add-on's ingress panel) shows, per model:

- Last/next training time
- Test-set MAE and R² (with a tooltip if R² isn't available yet)
- Today's and tomorrow's predicted energy consumption
- A verification chart comparing actual vs. predicted consumption over the
  full history, plus MAE / R² / mean error % computed over that history
- A calculator to manually try out temperature/humidity combinations

## 2. How it works

### 2.1 Computed weather sensors

Rather than requiring you to build your own template sensors, the add-on
computes the blended "actual so far today + forecast for the rest of the
day" averages itself, refreshed **every hour at minute :01**, and
additionally whenever `temp_sensor` or `humidity_sensor` changes state (via a
persistent WebSocket subscription to Home Assistant's `state_changed`
events).

- **Actual, historical part**: pulled from Home Assistant's History REST API
  and averaged **time-weighted** (each reading is weighted by how long it
  stayed in effect), not a plain arithmetic mean — this avoids a burst of
  frequent state changes skewing the result.
- **Forecast part**: pulled from the configured `weather_forecast` entity via
  the `weather.get_forecasts` service call (hourly forecast type).
- **Combination**: the two averages are combined with a weighted mean, where
  the weights are the number of hours each side actually covers (hours
  elapsed today for the actual side, number of forecast hours remaining for
  the forecast side). If only one side has data, that side is used as-is.

### 2.2 Model training and prediction

Two independent models are trained on the same feature set
(`avg_temp`, `avg_temp_48h`, `heating_degree`, `avg_humidity`, where
`heating_degree = max(0, 18 − avg_temp)`):

- **Random Forest** — better with larger datasets (30+ days), captures
  non-linear patterns, but doesn't extrapolate well outside the temperature
  range it was trained on.
- **Linear Regression** — better with small datasets, extrapolates linearly,
  but is generally less accurate once more data is available.

Each model is evaluated two different ways, which is why you'll see two sets
of MAE/R² numbers per model:

- **Test-split metrics** (`sensor.rf_mae` / `sensor.rf_r2`, and the Linear
  equivalents) — computed on a held-out test split during training
  (`train_model.py` / `train_model_linear.py`), reflecting how well the model
  generalizes to unseen days.
- **Verification metrics** (`sensor.rf_verification_*`, `sensor.lr_verification_*`)
  — computed by `checkModel.py` / `checkModel_linear.py` over the **entire**
  CSV history after each training run, and used to draw the "actual vs.
  predicted" verification chart.

### 2.3 Scheduling summary

| What | When |
|---|---|
| Weather/temperature computed sensors + live predictions | Every hour at minute :01, and instantly on any `temp_sensor`/`humidity_sensor` state change |
| Model training (RF + Linear) | Daily at 00:10 |
| Daily CSV data logging | Daily at 23:59 (skippable via the dashboard's logging toggle) |

## 3. Sensors created by this add-on

All sensors below are pushed to Home Assistant via the REST API and can be
used in dashboards, automations, or history graphs like any other sensor.

### 3.1 Weather / temperature (computed)

| Entity ID | Description | Unit |
|---|---|---|
| `sensor.heat_pump_pred_avg_temp_today_actual_and_forcast` | Blended average outdoor temperature for today (actual so far + forecast for the rest of the day) | °C |
| `sensor.heat_pump_pred_avg_temp_tomorrow_forecast` | Average forecast outdoor temperature for tomorrow | °C |
| `sensor.heat_pump_pred_avg_humidity_today_actual_and_forcast` | Blended average outdoor humidity for today (actual so far + forecast for the rest of the day) | % |
| `sensor.heat_pump_pred_avg_humidity_tomorrow_forecast` | Average forecast outdoor humidity for tomorrow | % |
| `sensor.heat_pump_pred_avg_temp_48h` | Average outdoor temperature over the last 48 hours (used as a model input feature) | °C |

### 3.2 Energy predictions (live)

| Entity ID | Description | Unit |
|---|---|---|
| `sensor.heat_pump_pred_today_rf` | Predicted energy consumption for today — Random Forest | kWh |
| `sensor.heat_pump_pred_today_linear` | Predicted energy consumption for today — Linear Regression | kWh |
| `sensor.heat_pump_pred_tomorrow_rf` | Predicted energy consumption for tomorrow — Random Forest | kWh |
| `sensor.heat_pump_pred_tomorrow_linear` | Predicted energy consumption for tomorrow — Linear Regression | kWh |

### 3.3 Model accuracy — test split (updated after each training run)

| Entity ID | Description | Unit |
|---|---|---|
| `sensor.rf_mae` | Random Forest mean absolute error on the held-out test split | kWh |
| `sensor.rf_r2` | Random Forest R² score on the held-out test split | — |
| `sensor.lr_mae` | Linear Regression mean absolute error on the held-out test split | kWh |
| `sensor.lr_r2` | Linear Regression R² score on the held-out test split | — |

### 3.4 Model accuracy — full-history verification (updated after each training run)

| Entity ID | Description | Unit |
|---|---|---|
| `sensor.rf_verification_mae` | Random Forest MAE over the full CSV history | kWh |
| `sensor.rf_verification_r2` | Random Forest R² over the full CSV history | — |
| `sensor.rf_verification_mean_error` | Random Forest mean absolute percentage error over the full CSV history | % |
| `sensor.lr_verification_mae` | Linear Regression MAE over the full CSV history | kWh |
| `sensor.lr_verification_r2` | Linear Regression R² over the full CSV history | — |
| `sensor.lr_verification_mean_error` | Linear Regression mean absolute percentage error over the full CSV history | % |

## 4. API endpoints

The full, always-current list is available at `/list` on the running
add-on. Key endpoints:

| Endpoint | Description |
|---|---|
| `GET /status`, `GET /statuslinear` | Training status and metrics (RF / Linear) |
| `POST /train`, `POST /trainlinear` | Start training in the background |
| `GET /prediction`, `GET /predictionlinear` | One-off prediction (`avg_temp`, `avg_temp_48h`, `avg_humidity` query params) |
| `GET /showpicresults`, `GET /showpicresultslinear` | Verification chart (`?show_plot=1`) or metrics JSON |
| `GET /log_now` | Manually log today's sensor readings to CSV |
| `GET /log_status` | Automatic daily logging status, CSV row count, minimum-rows warning |
| `GET /computed_sensors` | Current state of every computed/prediction/verification sensor |
| `GET /csv_data` | Raw `daily_temps.csv` history as JSON |
