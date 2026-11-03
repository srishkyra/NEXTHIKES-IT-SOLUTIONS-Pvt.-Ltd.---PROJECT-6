"""Models in sklearn pipelines, scoring helpers and the training and scoring loop used for every candidate."""
import os
import pickle
import tempfile
import time

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from . import features as F
from .config import SEED
from .logger import get_logger

logger = get_logger("rossmann.models")

MODEL_LABELS = {"ridge": "Ridge regression", "random_forest": "Random Forest", "hist_gradient_boosting": "Gradient boosting"}


def make_estimator(kind, **params):
    """The scikit-learn estimator behind each model name."""
    if kind == "ridge":
        return Ridge(**params)
    if kind == "random_forest":
        return RandomForestRegressor(random_state=SEED, n_jobs=-1, **params)
    if kind == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(random_state=SEED, **params)
    raise ValueError(f"Unknown model kind: {kind}")


def make_model(kind, features, log_target=True, **params):
    """Preprocessing and estimator in one pipeline. With log_target the model learns log(1 + Sales) and predicts Sales."""
    pipe = Pipeline([("prep", F.make_preprocessor(features)), ("model", make_estimator(kind, **params))])
    if not log_target:
        return pipe
    return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)


def window_data(dataset, splits, name, train_sample=None, seed=SEED, train_filter=None):
    """Training rows (all history before the window) and validation rows (one origin) for a split.

    train_filter: optional function that takes the training rows and returns a True/False mask of rows to keep.
    """
    s = splits.set_index("split").loc[name]
    train = dataset[(dataset["Date"] <= s["train_end"]) & (dataset["use_for_training"] == 1)]
    if train_filter is not None:
        train = train[train_filter(train)]
    if train_sample is not None and len(train) > train_sample:
        train = train.sample(train_sample, random_state=seed)
    valid = dataset[(dataset["origin_date"] == s["train_end"]) & dataset["Date"].between(s["valid_start"], s["valid_end"])
                    & (dataset["use_for_training"] == 1)]
    if name == "seasonal_check":
        valid = valid[valid["has_history_gap"] == 0]                        # only stores with a complete history
    return train, valid


def score(y_true, y_pred, stores=None, test_stores=None):
    """RMSPE (primary), MAE, RMSE and the average percentage error (bias); optionally RMSPE on the test-file stores."""
    y, p = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    out = {"rmspe": F.rmspe(y, p), "mae": float(np.mean(np.abs(y - p))), "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
           "bias_pct": float(100 * np.mean((p - y) / y))}
    if stores is not None and test_stores is not None:
        m = np.asarray(pd.Series(stores).isin(test_stores))
        out["rmspe_test_stores"] = F.rmspe(y[m], p[m])
    return out


def fit_and_score(kind, params, features, train_rows, valid_rows, test_stores=None, log_target=True, sample_weight=None, keep_model=False):
    """Fit one candidate on the training rows, score it on the validation rows and time it."""
    model = make_model(kind, features, log_target=log_target, **params)
    t0 = time.time()
    if sample_weight is not None:
        step = "regressor__model__sample_weight" if log_target else "model__sample_weight"
        model.fit(train_rows[features], train_rows["Sales"], **{step: sample_weight})
    else:
        model.fit(train_rows[features], train_rows["Sales"])
    fit_seconds = time.time() - t0
    t0 = time.time()
    pred = model.predict(valid_rows[features])
    predict_seconds = time.time() - t0
    result = score(valid_rows["Sales"], pred, valid_rows["Store"], test_stores)
    result.update({"fit_seconds": fit_seconds, "predict_seconds": predict_seconds, "train_rows": len(train_rows), "valid_rows": len(valid_rows)})
    return (result, pred, model) if keep_model else (result, pred)


def model_size_mb(model):
    """Size of the pickled model in megabytes (written to a temporary file so it does not need extra memory)."""
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        pickle.dump(model, f, protocol=4)
        path = f.name
    size = os.path.getsize(path) / 1e6
    os.remove(path)
    return size
