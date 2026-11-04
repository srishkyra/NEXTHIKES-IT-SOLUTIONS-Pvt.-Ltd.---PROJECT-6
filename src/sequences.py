"""Sequence data for the LSTM: store-by-day panels, (-1, 1) scaling and lazy sliding windows.

Design (Section 8.2):
  * direct multi-horizon framing: one forecast origin produces all `horizon` days at once;
  * a sample = (series, origin). Its history is the `window` days ending at the origin, its future inputs are the
    planned calendar of the next `horizon` days, its target is the sales of those days;
  * nothing is materialised: windows are cut from the panel on demand (get_batch), so the panel stays small;
  * missing days (the 2014 gap of 180 stores) are zero-filled and flagged with `has_record`, never interpolated.

This module does not import any deep learning library, so it can be tested and reused without one.
"""
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from . import features as F
from .logger import get_logger

logger = get_logger("rossmann.sequences")

WINDOW_DAYS = 56            # history window (8.1: the two-week cycle needs at least 28 days)
HORIZON_DAYS = 48           # direct horizon (the test window is 48 days; the main horizon is 42)
ORIGIN_STEP_DAYS = 7        # a training origin every 7 days
MIN_HISTORY_SHARE = 0.5     # a history window needs records on at least half of its days

CALENDAR_CHANNELS = ["dow_sin", "dow_cos", "doy_sin", "doy_cos", "days_to_easter", "days_since_easter",
                     "days_to_christmas", "days_since_christmas", "christmas_eve", "new_years_eve"]
STORE_CHANNELS = ["has_record", "open", "promo", "school_holiday", "state_holiday_a", "state_holiday_b",
                  "state_holiday_c"] + CALENDAR_CHANNELS
CHAIN_CHANNELS = ["open_share", "promo_share", "school_holiday_share", "state_holiday_share"] + CALENDAR_CHANNELS


# ----------------------------------------------------------------------------- the panel
@dataclass
class Panel:
    """Series x days arrays. `sales` is NaN wherever the value is unknown (no record, after the last known day,
    or an open day with zero sales). `cov` holds the planned inputs, all inside [-1, 1]."""
    dates: pd.DatetimeIndex
    stores: np.ndarray                       # series ids (store numbers, or [0] for the chain)
    sales: np.ndarray                        # (S, T) float32: log1p(Sales) at store level, raw at chain level
    cov: np.ndarray                          # (S, T, C) float32
    cov_names: list
    level: str = "store"
    last_sales_date: pd.Timestamp = None
    log_sales: bool = True
    static: dict = field(default_factory=dict)      # per-series static inputs (store type, assortment)
    scaled: np.ndarray = None                # (S, T) scaled sales, set by scale_panel
    scaling: object = None

    @property
    def n_days(self):
        return len(self.dates)

    def date_index(self, dates):
        idx = self.dates.get_indexer(pd.DatetimeIndex(pd.to_datetime(dates)))
        if (idx < 0).any():
            raise KeyError("Some dates are outside the panel")
        return idx

    def memory_mb(self):
        arrays = [self.sales, self.cov] + ([self.scaled] if self.scaled is not None else [])
        return sum(a.nbytes for a in arrays) / 1e6


def _calendar_block(plan, dates):
    """Calendar channels for every date (the same holiday calendar as the tree-model features)."""
    cal = F.holiday_calendar(plan)
    d = pd.Series(dates)
    out = pd.DataFrame(index=dates)
    dow = d.dt.dayofweek.values + 1
    doy = d.dt.dayofyear.values
    out["dow_sin"], out["dow_cos"] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)
    out["doy_sin"], out["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    for key, name in [("easter", "easter"), ("christmas", "christmas")]:
        to, since = F._days_to_and_since(d, cal[key], F.HOLIDAY_CAP)
        out[f"days_to_{name}"] = to / F.HOLIDAY_CAP
        out[f"days_since_{name}"] = since / F.HOLIDAY_CAP
    out["christmas_eve"] = ((d.dt.month == 12) & (d.dt.day == 24)).astype(float).values
    out["new_years_eve"] = ((d.dt.month == 12) & (d.dt.day == 31)).astype(float).values
    return out[CALENDAR_CHANNELS].astype("float32")


def store_panel(plan, last_sales_date, log_sales=True):
    """One row per store, one column per day (first train day to last test day).

    Sales after `last_sales_date` are hidden (NaN), so a panel built for a split cannot leak its own future.
    """
    last_sales_date = pd.Timestamp(last_sales_date)
    dates = pd.date_range(plan["Date"].min(), plan["Date"].max(), freq="D")
    stores = np.sort(plan["Store"].unique())
    S, T = len(stores), len(dates)
    si = np.searchsorted(stores, plan["Store"].values)
    ti = dates.get_indexer(plan["Date"])

    sales = np.full((S, T), np.nan, dtype="float32")
    known = (plan["Date"] <= last_sales_date) & plan["Sales"].notna()
    usable = known & ((plan["Open"] == 0) | (plan["Sales"] > 0))           # open zero-sales days stay unknown
    values = plan["Sales"].astype(float).values
    values = np.log1p(values) if log_sales else values
    sales[si[usable.values], ti[usable.values]] = values[usable.values]

    cov = np.zeros((S, T, len(STORE_CHANNELS)), dtype="float32")
    ch = {c: i for i, c in enumerate(STORE_CHANNELS)}
    cov[si, ti, ch["has_record"]] = 1.0
    cov[si, ti, ch["open"]] = plan["Open"].values
    cov[si, ti, ch["promo"]] = plan["Promo"].values
    cov[si, ti, ch["school_holiday"]] = plan["SchoolHoliday"].values
    for code in "abc":
        cov[si, ti, ch[f"state_holiday_{code}"]] = (plan["StateHoliday"] == code).values
    cal = _calendar_block(plan, dates).values
    cov[:, :, ch["dow_sin"]:] = cal[None, :, :]

    meta = plan.drop_duplicates("Store").set_index("Store").reindex(stores)
    static = {
        "store_type": pd.Categorical(meta["StoreType"], categories=list("abcd")).codes.astype("int32"),
        "assortment": pd.Categorical(meta["Assortment"], categories=list("abc")).codes.astype("int32"),
    }
    panel = Panel(dates, stores, sales, cov, list(STORE_CHANNELS), "store", last_sales_date, log_sales, static)
    logger.info(f"Store panel: {S} stores x {T} days x {len(STORE_CHANNELS)} channels ({panel.memory_mb():.0f} MB), sales known to {last_sales_date.date()}")
    return panel


def chain_panel(plan, last_sales_date):
    """The chain-level series of 8.1 (average sales per open store per day) with chain-wide planned shares."""
    last_sales_date = pd.Timestamp(last_sales_date)
    dates = pd.date_range(plan["Date"].min(), plan["Date"].max(), freq="D")
    used = plan[(plan["Date"] <= last_sales_date) & (plan["Open"] == 1) & (plan["Sales"] > 0)]
    avg = used.groupby("Date")["Sales"].mean().reindex(dates)
    by_day = plan.groupby("Date").agg(open_share=("Open", "mean"), promo_share=("Promo", "mean"),
                                      school_holiday_share=("SchoolHoliday", "mean"),
                                      state_holiday_share=("StateHoliday", lambda s: float((s != "0").mean()))).reindex(dates)
    cov = np.concatenate([by_day[CHAIN_CHANNELS[:4]].values.astype("float32"), _calendar_block(plan, dates).values], axis=1)
    panel = Panel(dates, np.array([0]), avg.values.astype("float32")[None, :], cov[None, :, :], list(CHAIN_CHANNELS),
                  "chain", last_sales_date, False, {})
    logger.info(f"Chain panel: {len(dates)} days x {len(CHAIN_CHANNELS)} channels, sales known to {last_sales_date.date()}")
    return panel


def differenced_panel(panel, lag):
    """Replace the series by its `lag`-day difference x[t] - x[t - lag] (NaN for the first `lag` days).

    Used for the chain-level differencing test of 8.3; the scaling must be fitted again on the differenced values and
    forecasts are turned back into levels with timeseries.invert_chain.
    """
    d = np.full_like(panel.sales, np.nan)
    d[:, lag:] = panel.sales[:, lag:] - panel.sales[:, :-lag]
    return replace(panel, sales=d, scaled=None, scaling=None)


# ----------------------------------------------------------------------------- scaling to (-1, 1)
class RangeScaler:
    """Linear map of [lo, hi] onto [-1, 1]. lo and hi are quantiles of the TRAINING values (default: min and max).

    With clip=True, values outside the training range are clipped to the boundary (a robust range ignores a handful
    of extreme days instead of letting them squeeze every other day into a narrow band).
    """

    def __init__(self, lower_q=0.0, upper_q=1.0, clip=True):
        self.lower_q, self.upper_q, self.clip = lower_q, upper_q, clip

    def fit(self, values):
        v = np.asarray(values, dtype=float)
        v = v[np.isfinite(v)]
        self.lo_, self.hi_ = float(np.quantile(v, self.lower_q)), float(np.quantile(v, self.upper_q))
        if self.hi_ <= self.lo_:
            raise ValueError("RangeScaler needs at least two distinct training values")
        self.n_fit_ = int(v.size)
        self.clipped_share_ = float(np.mean((v < self.lo_) | (v > self.hi_)))
        return self

    def transform(self, x):
        z = 2.0 * (np.asarray(x, dtype=float) - self.lo_) / (self.hi_ - self.lo_) - 1.0
        return np.clip(z, -1.0, 1.0) if self.clip else z

    def inverse_transform(self, z):
        return (np.asarray(z, dtype=float) + 1.0) / 2.0 * (self.hi_ - self.lo_) + self.lo_


@dataclass
class Scaling:
    """Everything needed to turn sales into model values and back."""
    train_end: pd.Timestamp
    center: np.ndarray            # per-series centre (store level: mean log sales on open days up to train_end)
    scaler: RangeScaler
    centered: bool

    def to_model(self, sales, series_idx):
        c = self.center[series_idx] if self.centered else 0.0
        return self.scaler.transform(sales - c)

    def to_sales(self, z, series_idx):
        """Model output -> sales (expm1 of log sales at store level)."""
        c = self.center[np.asarray(series_idx)][:, None] if self.centered else 0.0
        level = self.scaler.inverse_transform(z) + c
        return level


def scale_panel(panel, train_end, lower_q=None, upper_q=None):
    """Fit the scaling on days up to `train_end` ONLY and return a copy of the panel with `scaled` values.

    Store level: log1p sales minus the store's own mean (a shared model then sees every store on one scale), then
    a robust (0.05% / 99.95%) range. Closed days carry no sales signal: their value is set to 0 (the centre) and the
    `open` channel tells the model the store is closed. Chain level: raw average sales, plain min / max range.
    """
    train_end = pd.Timestamp(train_end)
    train_cols = panel.dates <= train_end
    opened = panel.cov[:, :, panel.cov_names.index("open")] > 0.5 if panel.level == "store" else np.ones(panel.sales.shape, bool)
    signal = np.where(opened, panel.sales, np.nan)
    if panel.level == "store":
        with np.errstate(invalid="ignore"):
            center = np.nanmean(np.where(train_cols[None, :], signal, np.nan), axis=1)
        center = np.where(np.isfinite(center), center, np.nanmean(center))
        lq, uq = (0.0005 if lower_q is None else lower_q), (0.9995 if upper_q is None else upper_q)
        centered = True
    else:
        center = np.zeros(len(panel.stores))
        lq, uq = (0.0 if lower_q is None else lower_q), (1.0 if upper_q is None else upper_q)
        centered = False
    rel = signal - (center[:, None] if centered else 0.0)
    scaler = RangeScaler(lq, uq).fit(rel[:, train_cols])
    scaled = scaler.transform(rel).astype("float32")
    if panel.level == "store":
        closed_known = (~opened) & np.isfinite(panel.sales)
        scaled[closed_known] = scaler.transform(0.0)
    scaling = Scaling(train_end, center.astype("float32"), scaler, centered)
    return replace(panel, scaled=scaled, scaling=scaling)


# ----------------------------------------------------------------------------- samples and batches
def training_origins(panel, train_end, window=WINDOW_DAYS, horizon=HORIZON_DAYS, step=ORIGIN_STEP_DAYS):
    """Origins every `step` days, anchored at the latest origin whose `horizon` targets all fall on or before train_end."""
    last = pd.Timestamp(train_end) - pd.Timedelta(days=horizon)
    first = panel.dates[0] + pd.Timedelta(days=window - 1)
    out, o = [], last
    while o >= first:
        out.append(o)
        o -= pd.Timedelta(days=step)
    return pd.DatetimeIndex(sorted(out))


def sample_index(panel, origins, window, horizon, last_target_day, complete_target, min_history_share=MIN_HISTORY_SHARE):
    """Valid (series, origin) pairs as a DataFrame with columns series_idx, origin_idx, origin.

    A sample is kept when its history window lies inside the panel and has records on at least `min_history_share`
    of its days, and it has at least one scoreable target day (known sales on an open day, not after last_target_day).
    With complete_target=True every target day must also exist in the panel, be on or before last_target_day and
    have a record (used for training samples).
    """
    origins = pd.DatetimeIndex(origins)
    oi = panel.date_index(origins)
    last_t = int(np.searchsorted(panel.dates, pd.Timestamp(last_target_day), side="right")) - 1
    if "has_record" in panel.cov_names:
        rec = panel.cov[:, :, panel.cov_names.index("has_record")]
    else:                                                  # chain level: a day counts if its value is known
        rec = np.isfinite(panel.sales).astype("float32")
    opened = panel.cov[:, :, panel.cov_names.index("open")] if "open" in panel.cov_names else np.ones(panel.sales.shape, "float32")
    scoreable = np.isfinite(panel.sales) & (opened > 0.5)
    if panel.log_sales:
        scoreable &= panel.sales > 0
    csum_rec = np.concatenate([np.zeros((rec.shape[0], 1)), np.cumsum(rec, axis=1)], axis=1)
    csum_sc = np.concatenate([np.zeros((rec.shape[0], 1)), np.cumsum(scoreable, axis=1)], axis=1)
    rows = []
    for o_pos, o_date in zip(oi, origins):
        h0 = o_pos - window + 1
        if h0 < 0:
            continue
        t1 = o_pos + horizon
        if complete_target and (t1 > last_t or t1 >= panel.n_days):
            continue
        hist_share = (csum_rec[:, o_pos + 1] - csum_rec[:, h0]) / window
        t_end = min(t1, last_t)
        n_score = csum_sc[:, t_end + 1] - csum_sc[:, o_pos + 1] if t_end > o_pos else np.zeros(len(panel.stores))
        ok = (hist_share >= min_history_share) & (n_score >= 1)
        if complete_target:
            ok &= (csum_rec[:, t1 + 1] - csum_rec[:, o_pos + 1]) == horizon
        for s in np.flatnonzero(ok):
            rows.append((s, o_pos, o_date))
    return pd.DataFrame(rows, columns=["series_idx", "origin_idx", "origin"])


def get_batch(panel, series_idx, origin_idx, window, horizon, last_target_day, add_horizon_channel=True):
    """Cut a batch from the panel (nothing is stored). Returns a dict:

      history : (n, window, 1 + C)  scaled sales (store centre where unknown) followed by the planned inputs of each past day
      future  : (n, horizon, C [+1]) planned inputs of each future day (+ horizon position h / horizon)
      target  : (n, horizon)        scaled sales on scoreable days up to last_target_day, NaN elsewhere
      series  : (n,)                series index (for the store embedding)
    """
    if panel.scaled is None:
        raise ValueError("Call scale_panel first")
    s = np.asarray(series_idx, dtype=int)
    o = np.asarray(origin_idx, dtype=int)
    hist_t = o[:, None] + np.arange(-window + 1, 1)[None, :]
    fut_t = o[:, None] + np.arange(1, horizon + 1)[None, :]
    inside = fut_t < panel.n_days
    fut_c = np.minimum(fut_t, panel.n_days - 1)

    fill = float(panel.scaling.scaler.transform(0.0)) if panel.scaling.centered else 0.0      # "typical day" value
    hist_sales = np.nan_to_num(panel.scaled[s[:, None], hist_t], nan=fill)
    history = np.concatenate([hist_sales[:, :, None], panel.cov[s[:, None], hist_t]], axis=2)
    future = panel.cov[s[:, None], fut_c] * inside[:, :, None]
    if add_horizon_channel:
        pos = np.broadcast_to((np.arange(1, horizon + 1) / horizon)[None, :, None], (len(s), horizon, 1))
        future = np.concatenate([future, pos], axis=2)

    target = panel.scaled[s[:, None], fut_c].astype("float32").copy()
    last_t = int(np.searchsorted(panel.dates, pd.Timestamp(last_target_day), side="right")) - 1
    opened = panel.cov[s[:, None], fut_c, panel.cov_names.index("open")] if "open" in panel.cov_names else np.ones_like(target)
    raw = panel.sales[s[:, None], fut_c]
    scoreable = inside & (fut_t <= last_t) & np.isfinite(raw) & (opened > 0.5)
    if panel.log_sales:
        scoreable &= raw > 0
    target[~scoreable] = np.nan
    return {"history": history.astype("float32"), "future": future.astype("float32"), "target": target, "series": s}


def static_inputs(panel, series_idx):
    """One-hot store type and assortment plus the (scaled) store centre: (n, 8)."""
    s = np.asarray(series_idx, dtype=int)
    st = np.eye(4, dtype="float32")[panel.static["store_type"][s]]
    asr = np.eye(3, dtype="float32")[panel.static["assortment"][s]]
    c = panel.scaling.center[s]
    c = ((c - panel.scaling.center.mean()) / (panel.scaling.center.std() + 1e-9)).astype("float32")[:, None]
    return np.concatenate([st, asr, c], axis=1)


def forecasts_to_rows(panel, series_idx, origin_idx, z_pred, horizon):
    """Model outputs (n, horizon) -> long table Store, Date, horizon_day, pred_sales (closed days forced to 0)."""
    s = np.asarray(series_idx, dtype=int)
    o = np.asarray(origin_idx, dtype=int)
    level = panel.scaling.to_sales(z_pred, s)
    sales = np.expm1(level) if panel.log_sales else level
    fut_t = o[:, None] + np.arange(1, horizon + 1)[None, :]
    inside = fut_t < panel.n_days
    fut_c = np.minimum(fut_t, panel.n_days - 1)
    if "open" in panel.cov_names:
        opened = panel.cov[s[:, None], fut_c, panel.cov_names.index("open")] > 0.5
        sales = np.where(opened, sales, 0.0)
    sales = np.maximum(sales, 0.0)
    out = pd.DataFrame({
        "Store": np.repeat(panel.stores[s], horizon),
        "Date": panel.dates[fut_c.ravel()],
        "horizon_day": np.tile(np.arange(1, horizon + 1), len(s)),
        "pred": sales.ravel(),
        "inside": inside.ravel(),
    })
    return out[out["inside"]].drop(columns="inside").reset_index(drop=True)
