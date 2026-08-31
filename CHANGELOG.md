# heatpumptrain — Session Notes / Changelog

Running log of what's been built in this add-on beyond the original baseline (Random Forest + Linear Regression training/prediction). Kept up to date so a new session can pick up context without re-reading the full diff.

## Current version: `4.38` (see `config.yaml`)

## 1. Add-on options reworked

`config.yaml` `options`/`schema` now only ask for four raw HA entities — the add-on computes everything else itself:

```yaml
options:
  energy_consumption_sensor: ""
  weather_forecast: ""
  temp_sensor: ""
  humidity_sensor: ""
```

The old `avg_temp_sensor` / `avg_temp_48h_sensor` / `avg_humidity_sensor` options are gone; their equivalents are now computed sensors pushed back to HA by the add-on (see below).

## 2. Computed weather-blend sensors

Reimplemented the logic from `mojskrypt.txt` (a PyScript automation that can't run inside the add-on container) directly in `server.py` using HA's REST API (`get_sensor_state`/`push_sensor_state` helpers, `SUPERVISOR_TOKEN`/`HA_TOKEN` + `HA_URL`):

- `get_history_avg(...)` — average of an entity's history over a time range via `/history/period/...`.
- `get_hourly_forecast(weather_entity)` — hourly forecast via `POST /services/weather/get_forecasts?return_response`. **Unverified against a live instance** — if sensors don't populate, check this function's response-shape assumption first.
- `filter_forecast_range(...)` — slices the forecast list to a datetime window and pulls a field (temperature/humidity).
- `weighted_combine(actual_avg, actual_n, forecast_avg, forecast_n)` — count-weighted blend, matching `mojskrypt.txt` exactly (not a plain 50/50 mean).

Sensors produced (English names, `heat_pump_pred_` prefix — renamed from the original Polish names):

| Sensor | Meaning |
|---|---|
| `sensor.heat_pump_pred_avg_temp_today_actual_and_forcast` | Today's temp: actual-so-far history blended with remaining-day forecast |
| `sensor.heat_pump_pred_avg_temp_tomorrow_forecast` | Tomorrow's temp, pure forecast |
| `sensor.heat_pump_pred_avg_humidity_today_actual_and_forcast` | Same blend, humidity |
| `sensor.heat_pump_pred_avg_humidity_tomorrow_forecast` | Tomorrow's humidity, pure forecast |
| `sensor.heat_pump_pred_avg_temp_48h` | Plain 48h backward-looking history average (no forecast component — needed for the training/prediction feature set) |

`COMPUTED_SENSOR_IDS` in `server.py` maps short keys → these entity IDs. `run_weather_sensor_updates()` recomputes all five (each wrapped in its own try/except so one failure doesn't block the others).

`get_sensor_names()` (feeds the daily CSV logger) now sources `avg_temp`/`avg_humidity`/`avg_temp_48h` from these computed sensors instead of removed config options.

## 3. Real-time energy prediction sensors

Four new sensors, recomputed from the weather-blend sensors above using the existing `predict_energy()` function (same model `.pkl` files used by `/prediction` and `/predictionlinear`):

- `sensor.heat_pump_pred_today_rf`
- `sensor.heat_pump_pred_today_linear`
- `sensor.heat_pump_pred_tomorrow_rf`
- `sensor.heat_pump_pred_tomorrow_linear`

`PREDICTION_SENSOR_IDS` maps keys → entity IDs. `update_prediction_sensors()` pulls `avg_temp_48h` first (bails out if missing), then computes today's pair from `temp_today`/`humidity_today` and tomorrow's pair from `temp_tomorrow`/`humidity_tomorrow`. Each push includes `unit_of_measurement: kWh`, `device_class: energy`, `state_class: measurement`, plus the three inputs as attributes.

**Design note**: `avg_temp_48h` has no forecast equivalent (it's inherently backward-looking), so the same current 48h value is reused for both today's and tomorrow's tomorrow-side predictions. This was a judgment call, flagged to the user, not something they specified.

## 4. Real-time recompute via HA WebSocket

Originally these sensors only refreshed hourly (`:01` scheduler tick, still kept as a fallback since forecast data changes independently of the source sensors). The user asked for true real-time recompute whenever `temp_sensor` or `humidity_sensor` changes, so:

- Added `websocket-client` to `Dockerfile`'s pip install line.
- `_ha_ws_url()` — resolves the WS endpoint: `ws://supervisor/core/websocket` when `SUPERVISOR_TOKEN` is present (running under Supervisor), otherwise derived from `HA_URL` (strip `/api`, swap `http(s)`→`ws(s)`, append `/api/websocket`). **Unverified against a live Supervisor instance.**
- `watched_sensor_ws_listener()` (renamed from an earlier `temp_sensor_ws_listener` that only watched `temp_sensor`) — runs on a daemon thread, connects, authenticates, subscribes to `state_changed`, and on any change to `temp_sensor` or `humidity_sensor` (dedup'd via `_last_watched_sensor_states`, skips `unknown`/`unavailable`) calls `run_weather_sensor_updates()` then `update_prediction_sensors()`. Reconnects with a 10s backoff on any error.
- Both weather-blend and prediction sensors are also seeded once at startup (`if __name__ == '__main__':` block) so they aren't empty before the first sensor change or hourly tick.

**Not implemented (flagged, not requested)**: no debounce. If the source temp/humidity sensor changes very frequently, every change currently triggers a History-API call plus two model predictions. Revisit if this becomes a problem in practice.

## 5. Dashboard (`index.html`) changes

- New "Computed Prediction Sensors" card (`#computed_sensors_grid`) rendering the weather-blend sensors as clickable tiles. Backed by a new `/computed_sensors` Flask endpoint that returns state/friendly_name/unit/last_update/available for every entity in `COMPUTED_SENSOR_IDS` **and** `PREDICTION_SENSOR_IDS` (single endpoint, no duplication).
- Clicking a tile calls `openSensorHistory(entityId)`, which does `window.open('${origin}/history?entity_id=...')` — relies on ingress serving the add-on same-origin with HA's frontend, so a relative link reaches HA's own native History panel. **This satisfies the user's explicit requirement to use HA's own history view, not a custom-drawn chart** — but is unverified without a live instance.
- The 4 RF/Linear prediction sensors are **not** shown in that bottom grid — client-side JS (`PREDICTION_SENSOR_ENTITY_IDS` Set) filters them out of the grid render and instead updates dedicated `.prediction-sensor-link` spans:
  - Random Forest card: two new rows under "Fit (R²)" — `#rf_pred_today` / `#rf_pred_tomorrow`.
  - Linear Regression card: same pattern — `#linear_pred_today` / `#linear_pred_tomorrow`.
  - Each is clickable (`openSensorHistory`) and shows `state + unit` or `N/A`.
- `fetchComputedSensors()` runs on `DOMContentLoaded` and inside `refreshStatus()`'s polling loop.
- All UI text is in English per standing preference, regardless of the (Polish) language used in chat.

## 6. Last-updated timestamps + Temperature/Humidity split (v4.27)

- "Predicted today"/"Predicted tomorrow" rows in the RF and Linear cards now show a small `Updated: ...` line under the value (`.prediction-sensor-updated`, matched by `data-entity-id`), sourced from the same `last_update` attribute already pushed with every prediction sensor.
- The bottom "Computed Prediction Sensors" card was renamed to **"Temperature / Humidity Sensors"** and split into two columns: `#computed_sensors_grid_temp` (left — `avg_temp_today`, `avg_temp_tomorrow`, `avg_temp_48h`) and `#computed_sensors_grid_humidity` (right — `avg_humidity_today`, `avg_humidity_tomorrow`). Routing is done client-side via `TEMPERATURE_SENSOR_ENTITY_IDS`/`HUMIDITY_SENSOR_ENTITY_IDS` Sets in `fetchComputedSensors()`; card markup itself is unchanged (`buildComputedSensorCard()`, extracted from the old inline logic).

## 7. Fixed unstable/wrong 48h (and today-actual) averages (v4.28)

- Bug: `sensor.heat_pump_pred_avg_temp_48h` (and the "actual" half of the today temp/humidity blends) was returning wildly unstable values (e.g. jumping between 23.0°C and -15.97°C within an hour), with `sample_count`/`liczba_pomiarow` stuck at `1`.
- Root cause: `get_history_avg()` calls HA's `/api/history/period/...` REST endpoint but never disabled `significant_changes_only`, which **defaults to `True`**. For a plain numeric sensor, that filter collapses the returned state list down to only "significant" changes — often just 1 entry for the whole window — so the "average" was really just whichever single raw reading survived the filter, re-picked on every websocket-triggered recompute.
- Fix: added `"significant_changes_only": "0"` to the request params in `get_history_avg()` (`server.py`), so the full state history in the window is returned and averaged properly. This is the shared helper behind all history-based averages (48h, and the actual-history side of today's temp/humidity blends), so the fix applies to all of them.
- **Important**: the live sensor attributes reported (`liczba_pomiarow`) don't match this repo's current attribute key (`sample_count`), meaning the deployed add-on container was running an older build. This fix requires an actual rebuild/redeploy of the add-on to take effect — saving the source alone isn't enough.

## 8. Switched history averages from arithmetic mean to time-weighted mean (v4.29)

- Bug: even after fix #7, `get_history_avg()` still computed a **plain arithmetic mean** of the returned state samples — every sample counted equally regardless of how long it was actually in effect. For irregularly-sampled sensors this is wrong: a burst of readings crammed into a few seconds gets the same weight as a value that was genuinely held for hours, so a handful of noisy samples can dominate the result.
- Verified with a real example: a source sensor held `23.0°C` for ~5.4 hours, then received a burst of ~20 garbage readings (`-50`..`50`) within about 2 seconds. Plain mean of the 29 samples: `-15.97°C`. Time-weighted mean (each sample weighted by seconds until the next state change, clamped to the query window): `21.96°C` — the value that was actually true almost the whole time.
- Fix: `get_history_avg()` now sorts samples by `last_changed`, computes each sample's duration-in-window (up to the next sample's timestamp, or `end_dt` for the last one, clamped to `[start_dt, end_dt]`), and returns `sum(value * duration) / sum(duration)` instead of a plain mean. Falls back to plain mean only if all durations come out to zero (e.g. every sample landed at the exact same instant). Applies to every caller of `get_history_avg()` (48h average, and the actual-history side of the today temp/humidity blends).
- Requires `timezone` from `datetime` (added to the top-of-file import).

## 9. Fixed actual/forecast blend weighting for today's temp+humidity (v4.30)

- Bug: `update_temp_today()`/`update_humidity_today()` already got a time-weighted "actual so far" average from `get_history_avg()` (fix #8), but `weighted_combine()` then blended it with the forecast average using **raw sample count** (`actual_n`, the number of state changes) against **forecast hour count** (`forecast_n`) — two different units. A sensor that changes often would out-weigh the forecast even if it only covered an hour of the day; a sensor that barely changes would be under-weighted even if it covered most of the day.
- Fix: `weighted_combine()` now takes `actual_weight`/`forecast_weight` in the same unit (hours). `actual_weight` is the real elapsed time since midnight (`(now - midnight).total_seconds() / 3600`), `forecast_weight` stays as the forecast hour count. So "today's expected average" is now a genuine hours-elapsed vs hours-remaining blend: `(actual_avg * hours_so_far + forecast_avg * hours_remaining) / total_hours`. Raw sample count is still exposed as `actual_sample_count` for diagnostics; the new `actual_hours` attribute shows the weight actually used.
- Applies to both `sensor.heat_pump_pred_avg_temp_today_actual_and_forcast` and `sensor.heat_pump_pred_avg_humidity_today_actual_and_forcast`.

## 10. Removed manual Train buttons in favor of daily auto-retraining, added model tooltips, and added config validation (v4.31)

- **Automatic daily retraining**: `server.py` now retrains both models on its own every day at `00:10` (`AUTO_TRAIN_HOUR`/`AUTO_TRAIN_MINUTE`), shortly after the `23:59` daily CSV write completes, so each day's fresh row is included. Implemented as a new check inside the existing `scheduler_thread()` 30s loop, guarded by an `auto_train_status["last_run"]` date so it fires once per day; each model is only (re)launched if it isn't already training (skips a slot if a manual `/train`/`/trainlinear` call is in flight). The manual `/train` and `/trainlinear` endpoints are unchanged and still work — only the dashboard buttons calling them were removed.
- **"Next training" in the UI**: `/status` and `/statuslinear` now include a `next_training` field (`get_next_auto_training_time()`, the next occurrence of `00:10`). The RF and Linear cards in `index.html` show it in a new "Next training" row under "Last training". The "Train RF model" / "Train Linear model" buttons and the `trainModel()` JS function were removed from `index.html` since training is no longer manually triggered from the dashboard.
- **Model description tooltips**: hovering "Random Forest (RF)" or "Linear Regression" in their card headers now shows a balloon tooltip (new `.tooltip-trigger` CSS, dark glass style matching the rest of the dashboard) explaining when each model is a good fit — Random Forest for larger datasets (30+ days) but unable to extrapolate outside the temperature range it was trained on; Linear Regression for small datasets (under 30 days), simpler and can extrapolate, but generally less accurate once more data exists.
- **Sensor configuration validation**: added `get_missing_required_sensors()`, checking that all four add-on options (`temp_sensor`, `humidity_sensor`, `weather_forecast`, `energy_consumption_sensor`) are present and non-empty in the add-on's configuration. `home_view()` (the `/` route) now calls this first and, if anything is missing, serves a standalone dark-themed error page (`render_config_error_page()`, HTTP 503) listing exactly which sensor(s) need to be set, instead of the normal dashboard. This is a config-presence check only (no live HA lookup of the entity), matching the existing lightweight validation style already used elsewhere in the add-on (e.g. `get_sensor_state`'s empty-string check).

## 11. Added descriptions for the 4 configuration options (v4.32)

- New `translations/en.yaml`, using Supervisor's add-on translation format (`configuration.<option_key>.name` / `.description`). Supervisor reads this directly from the add-on's own directory (like `CHANGELOG.md`/`DOCS.md`) — it does **not** need to be added to `Dockerfile`'s `COPY` list, since it's store/config metadata, not a runtime file.
- Each of the four options (`temp_sensor`, `humidity_sensor`, `weather_forecast`, `energy_consumption_sensor`) now shows a friendly display name and an explanatory description in the add-on's Configuration tab in Home Assistant, so users know which entity to pick and what it's used for.

## 12. Validate sensor entity type/unit, not just presence (v4.33)

- Bug: a wrongly-configured sensor (e.g. typing a `sensor.battery` entity into `weather_forecast`) previously passed the config check from #10 (the option string isn't empty) and the add-on started normally. Nothing errored — the computed sensors that depend on the forecast (`avg_temp_today`, `avg_temp_tomorrow`, `avg_humidity_today`, `avg_humidity_tomorrow`) just silently stopped updating, because `get_hourly_forecast()`/`filter_forecast_range()` got no usable forecast data from a non-weather entity and every downstream call was wrapped in try/except.
- Fix: `get_missing_required_sensors()` was replaced with `get_sensor_config_problems()`, which for each of the 4 required options now: (1) checks it's non-empty, (2) queries HA (`get_full_sensor_state`) to confirm the entity actually exists and has a known state, (3) runs a per-key validator (`SENSOR_VALIDATORS`) checking it's the *right kind* of entity:
  - `weather_forecast` must be a `weather.*` entity.
  - `temp_sensor` must have `device_class: temperature` or a `°C`/`°F` unit.
  - `humidity_sensor` must have `device_class: humidity` or a `%` unit.
  - `energy_consumption_sensor` must report its `unit_of_measurement` as `kWh` (case-insensitive).
- `home_view()` (the `/` route) shows the same dark-themed configuration-error page from #10 for any of these failures, now listing the specific problem per sensor (missing / unreachable / wrong unit or domain) instead of just "not configured".

## 13. Documented the #12 validation requirements in the config UI (v4.34)

- `translations/en.yaml` descriptions for all 4 options now spell out the exact requirement enforced by `SENSOR_VALIDATORS` in `server.py` (§12), so a user filling in the Configuration tab sees the constraint up front instead of only discovering it via the error page after saving: `temp_sensor` needs `device_class: temperature` or unit `°C`/`°F`; `humidity_sensor` needs `device_class: humidity` or unit `%`; `weather_forecast` must be a `weather.*` entity; `energy_consumption_sensor` must have `unit_of_measurement: kWh`.

## 14. Removed "Save now" button; added minimum-training-data warning (v4.35)

- Removed the "Save now" button (`btn_log_now`) and its `logNow()` JS handler from the Data Logging card in `index.html`. The `/log_now` endpoint itself is unchanged and still callable directly — same pattern as the earlier removal of the manual Train buttons (#10), where the UI trigger goes but the underlying endpoint stays.
- Added a minimum-training-data warning: `server.py` now has `MIN_TRAINING_ROWS = 2` and `get_csv_row_count()` (counts data rows in `daily_temps.csv`, excluding the header, returns 0 if the file doesn't exist). `/log_status` includes `csv_row_count`, `min_training_rows`, and `training_data_warning` (a message, or `null` once enough data exists) in its JSON response.
- `index.html`'s Data Logging card shows this as an amber warning banner (`#training_data_warning`, using the existing `--color-warning` token) whenever fewer than 2 days of data have been logged, e.g. "Only 1 day(s) of data logged. At least 2 days of data are needed before the model can be trained."

## 15. Translated standalone training/evaluation scripts to English (v4.36)

- `train_model.py`, `train_model_linear.py`, `checkModel.py`, `checkModel_linear.py` had Polish docstrings, print statements, chart labels/titles, and error messages left over from their original authoring. Translated all of it to English (docstrings, console output, `matplotlib` axis labels/titles/legends) to match the rest of the add-on's English-only UI/output convention. Also replaced Polish-diacritic special characters (e.g. `❌`, `✓`, `°C` arrows) with plain ASCII equivalents in console output for safer terminal/log encoding. No behavioral changes — output file names, JSON keys, and CSV columns are unchanged.

## 16. Validate that the weather forecast includes humidity (v4.37)

- Bug/gap: `_validate_weather_forecast()` (§12) only checked that `weather_forecast` was a `weather.*` entity — it never checked whether that entity's hourly forecast actually contains humidity data. Some weather integrations only forecast temperature, which meant `sensor.heat_pump_pred_avg_humidity_today_actual_and_forcast`/`..._tomorrow_forecast` would silently have no forecast-side data despite the sensor itself passing validation.
- Fix: `_validate_weather_forecast()` now calls `get_hourly_forecast(entity_id)` at startup validation time and fails with a configuration-error page if the call errors, the forecast is empty, or none of the returned hourly entries contain a `humidity` key. The error message and the `weather_forecast` option's description in `translations/en.yaml` both point users at the Met.no (Meteorologisk Institutt) integration as a weather source that does provide humidity forecasts.

## 17. Added a manual ON/OFF toggle for automatic CSV data logging (v4.38)

- `server.py`: new `LOGGING_STATE_FILE` (`/config/pump/logging_state.json`) with `get_logging_enabled()`/`set_logging_enabled()` helpers, persisting the flag across add-on restarts (defaults to `True`/enabled if never toggled, so existing installs keep logging as before). `scheduler_thread()`'s `23:59` auto-logging check now skips (`"Skipped (data logging is turned off)"`) when the flag is off. New `POST /log_toggle` endpoint flips the flag and returns the new state; `/log_status` now also reports `logging_enabled`.
- `index.html`: Data Logging card gets a new "Auto logging" status row and a "Logging: ON/OFF" toggle button (`btn_toggle_logging`, next to "Browse") styled green when on (`.btn-success`) and red when off (new `.btn-off` CSS). `toggleLogging()` calls `/log_toggle`; `updateLoggingToggleUI()` keeps the button/status row in sync on load and on every status poll.
- This only affects the automatic `23:59` daily write — the manual `/log_now` endpoint (no longer exposed in the UI, see #14) is unaffected and always logs immediately regardless of the toggle.

## Open / unverified items for next session

1. `get_hourly_forecast()`'s assumed REST shape for `weather.get_forecasts` — confirm against real HA response once deployed.
2. `_ha_ws_url()` — confirm `ws://supervisor/core/websocket` actually works from inside the Supervisor-managed container.
3. `openSensorHistory()` — confirm the ingress-served page can `window.open()` a same-origin `/history?entity_id=...` link and that it renders correctly.
4. Consider a debounce in `watched_sensor_ws_listener()` if `temp_sensor`/`humidity_sensor` turn out to update very frequently in practice.
