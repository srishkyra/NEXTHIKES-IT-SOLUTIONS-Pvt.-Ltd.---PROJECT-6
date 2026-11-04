"""Prediction intervals by split conformal calibration on out-of-sample forecast errors.

Errors are measured on the log scale, r = log(1 + actual) - log(1 + forecast), so an interval is a percentage band
around the forecast. For each group (by default the forecast week), the lower and upper bounds are empirical
quantiles of r with the finite-sample correction of split conformal prediction. Calibration errors come from the
rolling validation windows only (never the final holdout).
"""
import numpy as np
import pandas as pd


def week_ahead(horizon_day):
    return ((np.asarray(horizon_day) - 1) // 7 + 1).clip(max=7)


class ConformalIntervals:
    def __init__(self, level=0.90, group_cols=("week_ahead",), min_group_size=200):
        self.level, self.group_cols, self.min_group_size = level, list(group_cols), min_group_size

    @staticmethod
    def _quantile(r, q):
        n = len(r)
        return float(np.quantile(r, min(1.0, max(0.0, np.ceil((n + 1) * q) / n if q > 0.5 else np.floor((n + 1) * q) / n)),
                                 method="higher" if q > 0.5 else "lower"))

    def fit(self, actual, forecast, groups):
        r = np.log1p(np.asarray(actual, float)) - np.log1p(np.asarray(forecast, float))
        a = 1 - self.level
        g = pd.DataFrame(groups)[self.group_cols].reset_index(drop=True).assign(r=r)
        self.overall_ = (self._quantile(r, a / 2), self._quantile(r, 1 - a / 2), len(r))
        rows = []
        for key, d in g.groupby(self.group_cols):
            key = key if isinstance(key, tuple) else (key,)
            if len(d) >= self.min_group_size:
                rows.append((*key, self._quantile(d["r"].values, a / 2), self._quantile(d["r"].values, 1 - a / 2), len(d)))
        self.table_ = pd.DataFrame(rows, columns=self.group_cols + ["q_low", "q_high", "n"])
        return self

    def bounds(self, groups):
        g = pd.DataFrame(groups)[self.group_cols].reset_index(drop=True).copy()
        if "week_ahead" in self.group_cols:
            # days beyond the calibrated horizon (days 43 to 48 when calibrated on 42-day windows) use the last calibrated week
            g["week_ahead"] = g["week_ahead"].clip(upper=int(self.table_["week_ahead"].max()))
        m = g.merge(self.table_, on=self.group_cols, how="left")
        return m["q_low"].fillna(self.overall_[0]).values, m["q_high"].fillna(self.overall_[1]).values

    def to_frame(self):
        """The calibration table (saved for the app)."""
        return self.table_.assign(level=self.level)

    def predict(self, forecast, groups):
        lo, hi = self.bounds(groups)
        f = np.log1p(np.asarray(forecast, float))
        return np.maximum(np.expm1(f + lo), 0.0), np.expm1(f + hi)


def coverage(actual, lower, upper):
    y = np.asarray(actual, float)
    inside = (y >= lower) & (y <= upper)
    return {"coverage_pct": float(100 * inside.mean()), "mean_width_pct": float(100 * np.mean((upper - lower) / np.maximum(y, 1))),
            "below_pct": float(100 * np.mean(y < lower)), "above_pct": float(100 * np.mean(y > upper))}
