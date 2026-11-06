# Rossmann Store Sales Forecasting

Forecast daily **sales and customers** for 1,115 Rossmann stores up to six weeks ahead, with 90% ranges, and serve the
forecasts to finance analysts and store managers in a web app.

## Results (final holdout, 20 Jun to 31 Jul 2015, scored once)
| | RMSPE |
|---|---|
| Gradient boosting (production model) | 0.1068 |
| Best baseline (store-by-weekday-and-promotion average) | 0.1468 |
| Customer model | 0.0877 |
| Coverage of the 90% sales ranges | 88.4% |

## Repository
| Path | Contents |
|---|---|
| `Attempt 1.ipynb` | Sections 1 to 8.1: overview, setup, cleaning, validation design, exploration, features, machine learning, time-series diagnostics |
| `Rossmann_Part2.ipynb` | Interim package and Sections 8.2 to 12: LSTM, customers, explainability, intervals, final evaluation, app, report |
| `src/` | All reusable code (config, logging, style, data, validation, features, baselines, models, timeseries, sequences, lstm, evaluation, customers, explain, intervals, bundle, predict) |
| `app.py`, `.streamlit/`, `app_data/` | Streamlit app, theme and the small data files it needs |
| `data/` (DVC) | raw, cleaned (interim) and feature tables; versions `data-v1-raw` to `data-v4-final` |
| `models/` (DVC) | timestamped model bundles `dd-mm-yyyy-HH-MM-SS.pkl` with metadata; version `models-v1` |
| `outputs/` | registers, tables, submission file, interval calibration, blog draft |
| `figures/` | every numbered figure |
| `mlflow.db`, `mlruns/` | MLflow tracking store and model registry (local; not in Git) |

## Setup
```bash
pip install -r requirements-dev.txt          # the app alone needs only requirements.txt
dvc pull                                     # data and models from the DVC remote (or copy the raw CSVs to data/raw/)
```
Run `Attempt 1.ipynb`, then `Rossmann_Part2.ipynb`, top to bottom. On a 2-core CPU the second notebook takes about
2.0 hours (most of it the store-level LSTM and the full-history refits); a Colab GPU is much faster.

## The app
```bash
streamlit run app.py
```
Choose a store, download the example plan, edit promotions or opening days, upload it, and download the forecast.
Supported dates: 2015-08-01 to 2015-09-17 (the 48 days after the last day with known sales).

## Tracking
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db     # experiments, runs and the model registry
git log --oneline -- data.dvc; git tag -l              # data versions
```
