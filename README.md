# Rossmann Store Sales Forecasting

Forecast daily **sales and customers** for 1,115 Rossmann stores up to six weeks ahead, with 90% ranges, and serve the
forecasts to finance analysts and store managers in a web app.

**Live app:** https://rossmann-forecast-nexthikes.streamlit.app/  
**Report (PDF):** `outputs/blog_report.pdf`  
**Start here:** `Attempt 2.ipynb`

## Results (final holdout, 20 Jun to 31 Jul 2015, final run)
| | RMSPE |
|---|---|
| Gradient boosting (production model) | 0.1068 |
| Best baseline (store-by-weekday-and-promotion average) | 0.1468 |
| Customer model | 0.0877 |
| Coverage of the 90% sales ranges | 88.4% (nominal 90%) |

Current production bundle: `models/15-11-2026-21-07-42.pkl`. The holdout was locked during development and scored by the frozen
configuration in the final run (see the evaluation log in `outputs/`).

## Repository
| Path | Contents |
|---|---|
| `Attempt 2.ipynb` | Sections 1 to 12: overview, setup, cleaning, validation design, exploration, features, machine learning, time-series diagnostics, LSTM, customers, explainability, intervals, final evaluation, app, report |
| `src/` | All reusable code (config, logging, style, data, validation, features, baselines, models, timeseries, sequences, lstm, evaluation, customers, explain, intervals, bundle, predict) |
| `app.py`, `.streamlit/`, `app_data/` | Streamlit app, theme and the small data files it needs |
| `data/` (DVC) | raw, cleaned and feature tables; versions `data-v1-raw` to `data-v4-final` |
| `models/` (DVC) | timestamped model bundles `dd-mm-yyyy-HH-MM-SS.pkl` with metadata; version `models-v1` |
| `outputs/` | registers, tables, submission file, interval calibration, report, sample log |
| `figures/` | every numbered figure |
| `mlflow.db`, `mlruns/` | MLflow tracking store and model registry (local; not in Git) |

## Setup
Tested with: Python 3.14.3, TensorFlow 2.22.0rc0, Keras 3.15.1, scikit-learn 1.9.0, Streamlit 1.65.0.
```bash
pip install -r requirements-dev.txt          # the app alone needs only requirements.txt
# No shared DVC remote is configured. Rebuild from the source data instead:
# download the Rossmann Store Sales files (train.csv, test.csv, store.csv) into data/raw/ and run the notebook
```
If TensorFlow 2.22.0rc0 cannot be installed, use Python 3.12 with `pip install tensorflow`, or run the LSTM section on Google Colab.
Run `Attempt 2.ipynb` top to bottom. On a 2-core CPU the full run takes about
1.1 hours (most of it the store-level LSTM and the full-history refits).
Every run creates new MLflow model versions and moves the DVC tags to the latest data and models, so version numbers grow with each run.

## The app
```bash
streamlit run app.py
```
Choose a store (or upload a CSV for many stores), edit promotions or opening days, and download the forecast.
Supported dates: 2015-08-01 to 2015-09-17 (the 48 days after the last day with known sales).

## Tracking
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db     # experiments, runs and the model registry
git tag -l; git log --oneline --decorate --simplify-by-decoration -- data.dvc models.dvc   # data and model versions
```
