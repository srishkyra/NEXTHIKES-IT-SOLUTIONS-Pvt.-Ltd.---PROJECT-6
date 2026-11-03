"""Simple forecasts that need no learning. Every model must beat them."""
import numpy as np
import pandas as pd

BASELINE_LABELS = {
    "last_week_same_day": "Last observed week, same weekday",
    "store_weekday_average": "Store-by-weekday average (all history)",
    "store_weekday_recent": "Store-by-weekday average (last 13 weeks)",
    "promo_aware_average": "Store-by-weekday-and-promotion average",
    "same_day_last_year": "Same weekday one year earlier",
}


def baseline_forecasts(valid_rows, hist, origin):
    """Five baseline forecasts for the rows of one origin, using only history up to that origin.

    A baseline is repeated unchanged over the whole forecast window (it is never updated with later sales).
    Where a baseline has no value (for example no Sunday history), it falls back to the next simpler baseline.
    Returns (forecasts, share of rows that needed a fallback).
    """
    h = hist[(hist["Date"] <= pd.Timestamp(origin)) & (hist["Open"] == 1) & (hist["Sales"] > 0)]
    last_week = h.sort_values("Date").groupby(["Store", "DayOfWeek"])["Sales"].last().rename("last_week")
    promo_avg = h.groupby(["Store", "DayOfWeek", "Promo"])["Sales"].mean().rename("promo_avg")

    r = valid_rows[["Store", "DayOfWeek", "promo", "hist_mean_all", "hist_dow_mean_all", "hist_dow_mean_91", "ly_sales_same_dow"]].copy()
    r["Promo"] = r["promo"]
    r = r.join(last_week, on=["Store", "DayOfWeek"]).join(promo_avg, on=["Store", "DayOfWeek", "Promo"])

    raw = pd.DataFrame({
        "last_week_same_day": r["last_week"],
        "store_weekday_average": r["hist_dow_mean_all"],
        "store_weekday_recent": r["hist_dow_mean_91"],
        "promo_aware_average": r["promo_avg"],
        "same_day_last_year": r["ly_sales_same_dow"],
    })
    store_mean = r["hist_mean_all"]
    fallback = {
        "last_week_same_day": store_mean,
        "store_weekday_average": store_mean,
        "store_weekday_recent": raw["store_weekday_average"].fillna(store_mean),
        "promo_aware_average": raw["store_weekday_average"].fillna(store_mean),
        "same_day_last_year": raw["promo_aware_average"].fillna(raw["store_weekday_average"]).fillna(store_mean),
    }
    out = pd.DataFrame({c: raw[c].fillna(fallback[c]) for c in raw.columns})
    out = out.fillna(float(store_mean.median()))
    return out, raw.isna().mean()
