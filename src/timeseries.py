"""Time-series helpers: daily series, stationarity tests, differencing and its exact reversal."""
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, adfuller, kpss


def chain_series(hist):
    """Average sales per open store for every day (robust to closures and to stores missing from the data)."""
    open_days = hist[(hist["Open"] == 1) & (hist["Sales"] > 0)]
    return open_days.groupby("Date")["Sales"].mean().asfreq("D").rename("chain_average_sales")


def store_series(hist, store):
    """One store's daily sales; closed days are 0 (that is what a forecast has to say for them)."""
    return hist.loc[hist["Store"] == store].set_index("Date")["Sales"].asfreq("D").rename(f"store_{store}")


def stationarity(x):
    """ADF (null: unit root) and KPSS (null: stationary) around a level and around a trend."""
    x = pd.Series(x).dropna()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adf = adfuller(x, autolag="AIC")
        k_level = kpss(x, regression="c", nlags="auto")
        k_trend = kpss(x, regression="ct", nlags="auto")
    return {"n": len(x), "adf_stat": adf[0], "adf_p": adf[1], "kpss_level_stat": k_level[0], "kpss_level_p": k_level[1],
            "kpss_trend_stat": k_trend[0], "kpss_trend_p": k_trend[1]}


def verdict(row, alpha=0.05):
    """Plain-language reading of the three tests."""
    if row["adf_p"] >= alpha:
        return "unit root not rejected"
    if row["kpss_level_p"] > alpha:
        return "stationary"
    if row["kpss_trend_p"] > alpha:
        return "trend-stationary"
    return "ADF says stationary, KPSS disagrees"


def autocorrelation(x, nlags):
    return acf(pd.Series(x).dropna(), nlags=nlags)


def difference(x, lag=1):
    """x[t] - x[t-lag]."""
    return pd.Series(x).diff(lag).dropna()


def difference_chain(x, lags):
    """Apply several differences one after the other, for example (14, 1)."""
    out = pd.Series(x)
    for lag in lags:
        out = out.diff(lag)
    return out.dropna()


def invert_chain(future_diffs, history, lags):
    """Turn future differenced values back into sales levels, using the levels observed up to the forecast origin."""
    stages = [np.asarray(history, dtype=float)]
    for lag in lags:
        s = stages[-1]
        stages.append(s[lag:] - s[:-lag])
    out = []
    for d in np.asarray(future_diffs, dtype=float):
        stages[-1] = np.append(stages[-1], d)
        for k in range(len(lags), 0, -1):
            level = stages[k][-1] + stages[k - 1][-lags[k - 1]]
            stages[k - 1] = np.append(stages[k - 1], level)
        out.append(stages[0][-1])
    return np.array(out)
