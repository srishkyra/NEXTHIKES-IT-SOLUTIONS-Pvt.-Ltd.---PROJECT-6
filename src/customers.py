"""Customer forecasts: the same pipeline and the same forecast-time features as the sales model, with Customers
as the target. Only past customers (history features up to the forecast origin) are inputs; the customers of the
forecast day itself never are."""
import time

import numpy as np
import pandas as pd

from . import models as M
from .evaluation import metrics

TARGET = "Customers"


def customer_baseline(valid_rows, hist, origin):
    """Store-by-weekday-and-promotion average of customers up to the origin (falls back to store-by-weekday, then store)."""
    h = hist[(hist["Date"] <= pd.Timestamp(origin)) & (hist["Open"] == 1) & (hist["Sales"] > 0)]
    by_promo = h.groupby(["Store", "DayOfWeek", "Promo"])["Customers"].mean().rename("by_promo")
    by_dow = h.groupby(["Store", "DayOfWeek"])["Customers"].mean().rename("by_dow")
    by_store = h.groupby("Store")["Customers"].mean().rename("by_store")
    r = valid_rows[["Store", "DayOfWeek", "promo"]].rename(columns={"promo": "Promo"})
    r = r.join(by_promo, on=["Store", "DayOfWeek", "Promo"]).join(by_dow, on=["Store", "DayOfWeek"]).join(by_store, on="Store")
    out = r["by_promo"].fillna(r["by_dow"]).fillna(r["by_store"]).fillna(float(h["Customers"].median()))
    return out.values, float(r["by_promo"].isna().mean())


def fit_and_score(kind, params, features, train_rows, valid_rows, test_stores=None, keep_model=False):
    """Fit the customers model (learns log1p(Customers)) and score it on the validation rows."""
    model = M.make_model(kind, features, log_target=True, **params)
    t0 = time.time()
    model.fit(train_rows[features], train_rows[TARGET])
    fit_seconds = time.time() - t0
    pred = np.maximum(model.predict(valid_rows[features]), 0.0)
    res = metrics(valid_rows[TARGET], pred, valid_rows["Store"], test_stores)
    res.update({"fit_seconds": fit_seconds, "train_rows": len(train_rows), "valid_rows": len(valid_rows)})
    return (res, pred, model) if keep_model else (res, pred)
