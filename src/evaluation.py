"""Scoring helpers shared by Sections 8 to 10: metrics, breakdowns, chain-level aggregation and paired bootstrap."""
import numpy as np
import pandas as pd

from . import features as F

DAYS = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}


def metrics(y, p, stores=None, test_stores=None):
    """RMSPE (primary), MAE, RMSE, bias (mean percentage error) and share of rows within 10%."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    out = {"rmspe": F.rmspe(y, p), "mae": float(np.mean(np.abs(y - p))), "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
           "bias_pct": float(100 * np.mean((p - y) / y)), "within_10pct": float(100 * np.mean(np.abs(p - y) / y <= 0.10))}
    if stores is not None and test_stores is not None:
        m = np.asarray(pd.Series(stores).isin(test_stores))
        out["rmspe_test_stores"] = F.rmspe(y[m], p[m])
    return out


def add_groups(rows):
    """Columns used for error breakdowns (weekday, holiday, promotion, store type, week ahead)."""
    r = rows.copy()
    r["weekday"] = r["dow"].map(DAYS)
    r["holiday"] = np.where(r["state_holiday"].astype(str) == "0", "Ordinary day", "State holiday")
    r["promotion"] = np.where(r["promo"] == 1, "Promotion", "No promotion")
    r["week_ahead"] = ((r["horizon_day"] - 1) // 7 + 1).clip(upper=7)
    return r


def breakdown(rows, pred_cols, by, actual="Sales"):
    """RMSPE of several forecast columns for each group of `by`."""
    out = {}
    for c in pred_cols:
        sq = ((rows[actual] - rows[c]) / rows[actual]) ** 2
        out[c] = sq.groupby(rows[by]).mean().pow(0.5)
    t = pd.DataFrame(out)
    t.insert(0, "rows", rows.groupby(by).size())
    return t


def chain_level(rows, pred_cols, actual="Sales"):
    """Average per open store and day (the chain series of 8.1) for actuals and forecasts."""
    return rows.groupby("Date")[[actual] + list(pred_cols)].mean()


def paired_bootstrap(rows, col_a, col_b, actual="Sales", cluster="Store", n_boot=1000, seed=42):
    """RMSPE(col_a) - RMSPE(col_b), with a 95% interval from resampling whole stores (clusters).

    Resampling stores keeps each store's days together, so the interval respects the dependence between days of the
    same store. A negative difference means col_a is more accurate.
    """
    g = rows.assign(sa=((rows[actual] - rows[col_a]) / rows[actual]) ** 2,
                    sb=((rows[actual] - rows[col_b]) / rows[actual]) ** 2).groupby(cluster)
    sums = g[["sa", "sb"]].sum()
    n = g.size().values
    sa, sb = sums["sa"].values, sums["sb"].values
    rng = np.random.default_rng(seed)
    k = len(n)
    idx = rng.integers(0, k, size=(n_boot, k))
    na = n[idx].sum(axis=1)
    diff = np.sqrt(sa[idx].sum(axis=1) / na) - np.sqrt(sb[idx].sum(axis=1) / na)
    point = np.sqrt(sa.sum() / n.sum()) - np.sqrt(sb.sum() / n.sum())
    return {"difference": float(point), "ci_low": float(np.quantile(diff, 0.025)), "ci_high": float(np.quantile(diff, 0.975)),
            "share_of_resamples_a_better": float(np.mean(diff < 0))}
