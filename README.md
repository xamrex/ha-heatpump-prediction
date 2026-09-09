# Heat Pump Prediction

A Home Assistant add-on that trains and serves Random Forest and Linear
Regression models predicting your heat pump's daily energy consumption from
outdoor temperature, humidity, and weather forecast history.

The add-on also computes a set of supporting sensors (blended actual +
forecast temperature/humidity, live energy predictions, and model accuracy
metrics) and publishes them back to Home Assistant so they can be used on
dashboards or in automations.

See **[heatpumptrain/DOCS.md](heatpumptrain/DOCS.md)** for full documentation:
required sensor configuration, every sensor the add-on creates, how the
predictions and metrics are computed, and the API endpoints it exposes.

## Features

- Two independent, continuously retrained models — Random Forest and Linear
  Regression — with separate accuracy metrics for each.
- Self-computed weather/temperature sensors (no manual template sensors or
  extra automations required): time-weighted blend of actual history and
  forecast, refreshed hourly and on every sensor state change.
- A dashboard (served on the add-on's ingress panel) showing training
  status, predictions, accuracy metrics, and verification charts.
- Automatic daily data logging and daily model retraining.

## Installation

1. Add this repository to Home Assistant: **Settings → Add-ons → Add-on
   Store → ⋮ → Repositories**, then add this repo's URL
   (`https://github.com/xamrex/ha-heatpump-prediction`).
2. Install **Heat Pump prediction** from the add-on store and start it.
3. Open the add-on's **Configuration** tab and set the four required sensor
   entities (see [DOCS.md](heatpumptrain/DOCS.md#11-configure-the-required-sensors)).
4. Open the add-on's web UI (ingress panel) to monitor training and
   predictions.

## Development

```bash
cd heatpumptrain

# Build the add-on image
docker build -t heatpumptrain .

# Run the server directly (only meaningful where /config resolves,
# e.g. inside the add-on container or on a live HA host)
python3 server.py

# Trigger training/evaluation without going through HTTP
python3 train_model.py          # Random Forest
python3 train_model_linear.py   # Linear Regression
python3 checkModel.py           # RF evaluation + plot
python3 checkModel_linear.py    # Linear evaluation + plot
```

## Repository layout

This is a Home Assistant [add-on repository](https://developers.home-assistant.io/docs/add-ons/repository);
the add-on itself lives in its own subfolder as required by Supervisor.

| Path | Purpose |
|---|---|
| `repository.yaml` | Add-on repository manifest (required for Supervisor to recognize this repo) |
| `heatpumptrain/config.yaml` | Home Assistant add-on manifest |
| `heatpumptrain/Dockerfile` | Add-on container image |
| `heatpumptrain/server.py` | Flask/waitress server — dashboard, training orchestration, prediction/status endpoints, sensor scheduling |
| `heatpumptrain/train_model.py` / `train_model_linear.py` | Standalone model trainers |
| `heatpumptrain/checkModel.py` / `checkModel_linear.py` | Standalone model evaluators (metrics + plots) |
| `heatpumptrain/index.html` | Dashboard UI |
