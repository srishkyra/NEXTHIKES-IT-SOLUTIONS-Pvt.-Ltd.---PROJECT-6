"""Central project configuration. Import values from here instead of hard-coding them."""
from pathlib import Path

# ---------------------------------------------------------------- reproducibility
SEED = 42

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]          # project root = parent of src/
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
FIGURES_DIR = ROOT / "figures"
LOGS_DIR = ROOT / "logs"

RAW_FILES = {
    "train": DATA_RAW / "train.csv",
    "test": DATA_RAW / "test.csv",
    "store": DATA_RAW / "store.csv",
    "sample_submission": DATA_RAW / "sample_submission.csv",
}

# ---------------------------------------------------------------- forecasting design
TARGET_SALES = "Sales"
TARGET_CUSTOMERS = "Customers"
HORIZON_DAYS = 42            # main forecast horizon (six weeks)
TEST_DAYS = 48               # length of the supplied test window
N_ROLLING_WINDOWS = 3        # earlier 42-day validation windows before the final holdout
INTERVAL_LEVEL = 0.90        # nominal coverage of prediction intervals

# ---------------------------------------------------------------- tracking and model files
MLFLOW_TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
MLFLOW_EXPERIMENT = "rossmann-sales-forecasting"
MLFLOW_SETUP_EXPERIMENT = "setup-check"
MODEL_TIMESTAMP_FORMAT = "%d-%m-%Y-%H-%M-%S"   # e.g. 10-08-2026-16-32-31.pkl
