"""Feature engineering: known-future features, origin-based history features and the preprocessing pipeline.

Two kinds of information feed the model, and they are kept strictly apart:
  * known-future features: built from the date, the planned inputs (promotion, holidays, opening) and store details;
  * origin-history features: summaries of sales that only use days up to the forecast origin.
Nothing in this module ever reads the sales of a day after the origin to build a feature of that day.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import validation
from .logger import get_logger

logger = get_logger("rossmann.features")

FEATURE_HORIZON_DAYS = 48        # training rows look up to 48 days ahead (the test window is 48 days)
ORIGIN_SPACING_DAYS = 28         # a new forecast origin every four weeks (plus the validation origins)
MIN_HISTORY_DAYS = 91            # an origin needs at least 13 weeks of history before it
HOLIDAY_CAP = 30                 # distances to holidays beyond this are treated as "far away"
LAST_YEAR_LAG_DAYS = 364         # same weekday, one year earlier

ID_COLUMNS = ["Store", "Date", "DayOfWeek", "Open", "Sales", "Customers", "use_for_training", "is_test"]

# ----------------------------------------------------------------------------- feature catalogue
# (name, availability class, kind, description)
FEATURE_SPECS = [
    # calendar
    ("dow", "known_future", "numeric", "Weekday number (1 = Monday)"),
    ("is_weekend", "known_future", "binary", "Saturday or Sunday"),
    ("day_of_month", "known_future", "numeric", "Day of the month"),
    ("month", "known_future", "numeric", "Month number"),
    ("week_of_year", "known_future", "numeric", "ISO week of the year"),
    ("day_of_year", "known_future", "numeric", "Day of the year"),
    ("days_to_month_end", "known_future", "numeric", "Days left in the month"),
    ("month_part", "known_future", "categorical", "Beginning (1-10), middle (11-20) or end (21+) of the month"),
    ("dow_sin", "known_future", "numeric", "Weekday as a cycle (sine)"),
    ("dow_cos", "known_future", "numeric", "Weekday as a cycle (cosine)"),
    ("month_sin", "known_future", "numeric", "Month as a cycle (sine)"),
    ("month_cos", "known_future", "numeric", "Month as a cycle (cosine)"),
    ("doy_sin", "known_future", "numeric", "Day of year as a cycle (sine)"),
    ("doy_cos", "known_future", "numeric", "Day of year as a cycle (cosine)"),
    ("is_christmas_eve", "known_future", "binary", "24 December"),
    ("is_new_years_eve", "known_future", "binary", "31 December"),
    # holidays
    ("state_holiday", "known_future", "categorical", "State holiday type for the store (0, a, b, c)"),
    ("school_holiday", "known_future", "binary", "School holiday affects the store"),
    ("days_to_holiday", "known_future", "numeric", "Days to the next national holiday (capped at 30)"),
    ("days_since_holiday", "known_future", "numeric", "Days since the last national holiday (capped at 30)"),
    ("days_to_easter", "known_future", "numeric", "Days to the next Easter holiday (capped at 30)"),
    ("days_since_easter", "known_future", "numeric", "Days since the last Easter holiday (capped at 30)"),
    ("days_to_christmas", "known_future", "numeric", "Days to the next Christmas holiday (capped at 30)"),
    ("days_since_christmas", "known_future", "numeric", "Days since the last Christmas holiday (capped at 30)"),
    # promotions
    ("promo", "known_future", "binary", "Promotion on this day"),
    ("promo_dow", "known_future", "categorical", "Promotion and weekday together (e.g. P1_D1)"),
    ("promo_week", "known_future", "binary", "Any promotion this calendar week"),
    ("promo_prev_week", "known_future", "binary", "Any promotion in the previous week"),
    # opening
    ("days_since_closure", "known_future", "numeric", "Days since the last closure on an ordinary day (capped at 14)"),
    ("days_since_long_closure", "known_future", "numeric", "Days since a closure of 7+ ordinary days ended (capped at 60)"),
    ("days_to_long_closure", "known_future", "numeric", "Days until a closure of 7+ ordinary days starts (capped at 30)"),
    ("days_since_gap_end", "known_future", "numeric", "Days since the store's long gap in its history ended (capped at 365)"),
    # competition and Promo2 (store details combined with the date)
    ("comp_open_months", "known_future", "numeric", "Months since the competitor opened (0 to 24; unknown date counted as 24)"),
    ("comp_not_yet_open", "known_future", "binary", "Competitor opens after this date"),
    ("comp_opened_last_6m", "known_future", "binary", "Competitor opened within the last 6 months"),
    ("comp_recent_near", "known_future", "binary", "Competitor under 1 km opened within the last 6 months"),
    ("promo2_active", "known_future", "binary", "Store is in Promo2 and its start date has passed"),
    ("promo2_weeks", "known_future", "numeric", "Weeks since Promo2 started (0 if not active, capped at 260)"),
    ("promo2_start_month", "known_future", "binary", "Promo2 round starts again this month"),
    ("horizon_day", "known_future", "numeric", "Days between the forecast origin and this day"),
    # store details
    ("store_type", "store_static", "categorical", "Store type (a, b, c, d)"),
    ("assortment", "store_static", "categorical", "Assortment level (a, b, c)"),
    ("competition_distance_log", "store_static", "numeric", "log(1 + distance to nearest competitor)"),
    ("competition_distance_missing", "store_static", "binary", "Distance was unknown (median used)"),
    ("promo2", "store_static", "binary", "Store takes part in Promo2"),
    ("promo_interval", "store_static", "categorical", "Months in which Promo2 rounds start"),
    ("has_history_gap", "store_static", "binary", "Store is missing a long block of history"),
    # history up to the forecast origin
    ("hist_mean_28", "origin_history", "numeric", "Store's average sales per open day, last 28 days before the origin"),
    ("hist_mean_91", "origin_history", "numeric", "Same, last 91 days"),
    ("hist_mean_all", "origin_history", "numeric", "Same, all history up to the origin"),
    ("hist_customers_91", "origin_history", "numeric", "Store's average customers per open day, last 91 days"),
    ("hist_spc_91", "origin_history", "numeric", "Sales per customer, last 91 days"),
    ("hist_days_91", "origin_history", "numeric", "Number of open days with sales in the last 91 days"),
    ("hist_promo_mean_91", "origin_history", "numeric", "Average sales on Monday-Friday promotion days, last 91 days"),
    ("hist_nopromo_mean_91", "origin_history", "numeric", "Average sales on Monday-Friday non-promotion days, last 91 days"),
    ("hist_promo_lift_all", "origin_history", "numeric", "Promotion vs non-promotion sales ratio, all history to the origin"),
    ("hist_momentum", "origin_history", "numeric", "Last 28 days divided by last 91 days"),
    ("hist_dow_mean_91", "origin_history", "numeric", "Store's average sales on this weekday, last 91 days"),
    ("hist_dow_mean_all", "origin_history", "numeric", "Store's average sales on this weekday, all history to the origin"),
    ("ly_sales_same_dow", "origin_history", "numeric", "Store's sales on the same weekday one year earlier"),
    ("ly_week_mean", "origin_history", "numeric", "Store's average sales in the 7 days around that date one year earlier"),
]
FEATURE_TABLE = pd.DataFrame(FEATURE_SPECS, columns=["feature", "class", "kind", "description"])

KNOWN_AND_STATIC = FEATURE_TABLE.loc[FEATURE_TABLE["class"].isin(["known_future", "store_static"]), "feature"].tolist()
ALL_FEATURES = FEATURE_TABLE["feature"].tolist()


def register_all():
    """Register every engineered feature with the availability policy (src/validation.py)."""
    for klass, names in FEATURE_TABLE.groupby("class")["feature"]:
        validation.register_features(list(names), klass)


def split_by_kind(features):
    """Group a list of feature names into numeric, binary and categorical lists."""
    kinds = FEATURE_TABLE.set_index("feature")["kind"]
    return {k: [f for f in features if kinds[f] == k] for k in ("numeric", "binary", "categorical")}


# ----------------------------------------------------------------------------- planning table
PLAN_COLUMNS = ["Store", "Date", "DayOfWeek", "Open", "Promo", "StateHoliday", "SchoolHoliday", "Sales", "Customers", "use_for_training",
                "StoreType", "Assortment", "CompetitionDistance", "CompetitionDistance_missing", "CompetitionOpenSince",
                "CompetitionOpenDate_known", "Promo2", "Promo2SinceWeek", "Promo2SinceYear", "PromoInterval", "has_history_gap"]


def make_plan(train, test):
    """One table with every store-day (train and test) and its planned inputs. Test rows have no sales."""
    tr = train.assign(is_test=0)
    te = test.assign(is_test=1)
    plan = pd.concat([tr[PLAN_COLUMNS + ["is_test"]], te[[c for c in PLAN_COLUMNS if c in te.columns] + ["is_test"]]], ignore_index=True)
    return plan.sort_values(["Store", "Date"]).reset_index(drop=True)


# ----------------------------------------------------------------------------- helpers
def holiday_calendar(plan, min_share=0.9):
    """Dates of holidays that apply to (almost) every store, from the StateHoliday flags in the data."""
    flagged = plan["StateHoliday"] != "0"
    share = flagged.groupby(plan["Date"]).mean()
    types = plan[flagged].groupby("Date")["StateHoliday"].agg(lambda s: s.mode().iloc[0])
    national = types[share.reindex(types.index) >= min_share]
    return {
        "any": national.index.sort_values(),
        "easter": national[national == "b"].index.sort_values(),
        "christmas": national[national == "c"].index.sort_values(),
    }


def _days_to_and_since(dates, events, cap):
    """For each date: days to the next event (0 on the day) and days since the previous event, both capped."""
    d = dates.values.astype("datetime64[D]").astype(np.int64)
    ev = events.values.astype("datetime64[D]").astype(np.int64)
    if len(ev) == 0:
        return np.full(len(d), cap), np.full(len(d), cap)
    nxt = np.searchsorted(ev, d, side="left")
    prv = nxt - 1
    to = np.where(nxt < len(ev), ev[np.minimum(nxt, len(ev) - 1)] - d, cap)
    since = np.where(prv >= 0, d - ev[np.maximum(prv, 0)], cap)
    return np.minimum(to, cap), np.minimum(since, cap)


def _asof_days(rows, events, event_col, direction, cap, exact):
    """Days from each row's date to the nearest event of the same store (backward = since, forward = until)."""
    left = rows[["Store", "Date"]].reset_index().sort_values("Date")
    right = events[["Store", event_col]].sort_values(event_col)
    merged = pd.merge_asof(left, right, left_on="Date", right_on=event_col, by="Store", direction=direction,
                           allow_exact_matches=exact)
    days = (merged["Date"] - merged[event_col]).dt.days.abs().fillna(cap).clip(upper=cap)
    return pd.Series(days.values, index=merged["index"].values).reindex(rows.index)


def find_gap_end(train):
    """Last missing day of each store's long gap (the day before its history resumes), or NaT."""
    d = train[["Store", "Date"]].sort_values(["Store", "Date"])
    step = d.groupby("Store")["Date"].diff().dt.days
    resumed = d[step >= 30]
    return (resumed["Date"] - pd.Timedelta(days=1)).groupby(resumed["Store"]).max()


_INTERVAL_MONTHS = {"Jan,Apr,Jul,Oct": [1, 4, 7, 10], "Feb,May,Aug,Nov": [2, 5, 8, 11], "Mar,Jun,Sept,Dec": [3, 6, 9, 12]}


# ----------------------------------------------------------------------------- known-future features
def add_known_future_features(plan, gap_end):
    """Features built from the date, planned inputs and store details only (never from sales)."""
    f = pd.DataFrame(index=plan.index)
    d = plan["Date"]

    # calendar
    f["dow"] = (d.dt.dayofweek + 1).astype("int8")
    f["is_weekend"] = (f["dow"] >= 6).astype("int8")
    f["day_of_month"] = d.dt.day.astype("int8")
    f["month"] = d.dt.month.astype("int8")
    f["week_of_year"] = d.dt.isocalendar().week.astype("int16").values
    f["day_of_year"] = d.dt.dayofyear.astype("int16")
    f["days_to_month_end"] = (d.dt.days_in_month - d.dt.day).astype("int8")
    f["month_part"] = np.select([f["day_of_month"] <= 10, f["day_of_month"] <= 20], ["start", "mid"], "end")
    for name, value, period in [("dow", f["dow"], 7), ("month", f["month"], 12), ("doy", f["day_of_year"], 365.25)]:
        f[f"{name}_sin"] = np.sin(2 * np.pi * value / period)
        f[f"{name}_cos"] = np.cos(2 * np.pi * value / period)
    f["is_christmas_eve"] = ((f["month"] == 12) & (f["day_of_month"] == 24)).astype("int8")
    f["is_new_years_eve"] = ((f["month"] == 12) & (f["day_of_month"] == 31)).astype("int8")

    # holidays (distances use only the holiday flags that appear in the data)
    f["state_holiday"] = plan["StateHoliday"].astype(str)
    f["school_holiday"] = plan["SchoolHoliday"].astype("int8")
    cal = holiday_calendar(plan)
    unique_dates = pd.Series(d.unique()).sort_values()
    for key, name in [("any", "holiday"), ("easter", "easter"), ("christmas", "christmas")]:
        to, since = _days_to_and_since(unique_dates, cal[key], HOLIDAY_CAP)
        f[f"days_to_{name}"] = d.map(pd.Series(to, index=unique_dates.values)).astype("int16")
        f[f"days_since_{name}"] = d.map(pd.Series(since, index=unique_dates.values)).astype("int16")

    # promotions
    f["promo"] = plan["Promo"].astype("int8")
    f["promo_dow"] = "P" + f["promo"].astype(str) + "_D" + f["dow"].astype(str)
    week = d - pd.to_timedelta(d.dt.dayofweek, unit="D")
    wk = pd.DataFrame({"Store": plan["Store"], "week": week, "promo": plan["Promo"]})
    wk_flag = wk.groupby(["Store", "week"])["promo"].max().rename("promo_week").reset_index()
    f["promo_week"] = wk.merge(wk_flag, on=["Store", "week"], how="left")["promo_week"].values.astype("int8")
    prev = wk_flag.assign(week=wk_flag["week"] + pd.Timedelta(days=7)).rename(columns={"promo_week": "promo_prev_week"})
    f["promo_prev_week"] = wk.merge(prev, on=["Store", "week"], how="left")["promo_prev_week"].fillna(0).values.astype("int8")

    # closures on ordinary days (Sundays and state holidays do not count as closures)
    keep = ~((plan["Open"] == 0) & ((plan["DayOfWeek"] == 7) | (plan["StateHoliday"] != "0")))
    e = plan.loc[keep, ["Store", "Date", "Open"]].reset_index(drop=True)
    e["closed"] = (e["Open"] == 0)
    run = ((e["closed"] != e["closed"].shift()) | (e["Store"] != e["Store"].shift())).cumsum()
    spells = (e[e["closed"]].groupby(run[e["closed"]])
              .agg(Store=("Store", "first"), start=("Date", "min"), end=("Date", "max"), days=("Date", "size")))
    long_spells = spells[spells["days"] >= 7]
    f["days_since_closure"] = _asof_days(plan, spells, "end", "backward", 14, False).astype("int16")
    f["days_since_long_closure"] = _asof_days(plan, long_spells, "end", "backward", 60, False).astype("int16")
    f["days_to_long_closure"] = _asof_days(plan, long_spells, "start", "forward", 30, True).astype("int16")
    gap = plan["Store"].map(gap_end)
    f["days_since_gap_end"] = np.where(gap.notna() & (d > gap), (d - gap).dt.days.clip(upper=365), 0).astype("int16")

    # competition
    months = (d - plan["CompetitionOpenSince"]).dt.days / 30.44
    known = plan["CompetitionOpenDate_known"] == 1
    f["comp_not_yet_open"] = (known & (months < 0)).astype("int8")
    f["comp_open_months"] = np.where(known, months.clip(0, 24), 24).astype("float32")
    f["comp_opened_last_6m"] = (known & (months >= 0) & (months < 6)).astype("int8")
    f["comp_recent_near"] = ((f["comp_opened_last_6m"] == 1) & (plan["CompetitionDistance"] < 1000)).astype("int8")

    # Promo2
    in_promo2 = plan["Promo2"] == 1
    starts = {}
    for s, y, w in plan.loc[in_promo2, ["Store", "Promo2SinceYear", "Promo2SinceWeek"]].drop_duplicates("Store").itertuples(index=False):
        starts[s] = pd.Timestamp.fromisocalendar(int(y), int(w), 1)
    start = plan["Store"].map(starts)
    active = in_promo2 & start.notna() & (d >= start)
    f["promo2_active"] = active.astype("int8")
    f["promo2_weeks"] = np.where(active, ((d - start).dt.days / 7).clip(0, 260), 0).astype("float32")
    starts_now = np.zeros(len(plan), dtype=bool)
    for interval, months_list in _INTERVAL_MONTHS.items():
        starts_now |= ((plan["PromoInterval"] == interval) & f["month"].isin(months_list)).values
    f["promo2_start_month"] = (active.values & starts_now).astype("int8")

    # store details
    f["store_type"] = plan["StoreType"].astype(str)
    f["assortment"] = plan["Assortment"].astype(str)
    f["competition_distance_log"] = np.log1p(plan["CompetitionDistance"]).astype("float32")
    f["competition_distance_missing"] = plan["CompetitionDistance_missing"].astype("int8")
    f["promo2"] = plan["Promo2"].astype("int8")
    f["promo_interval"] = plan["PromoInterval"].fillna("None").astype(str)      # "None" is read back as missing from the saved file
    f["has_history_gap"] = plan["has_history_gap"].astype("int8")

    out = pd.concat([plan[[c for c in ID_COLUMNS if c in plan.columns]], f], axis=1)
    logger.info(f"Known-future features built for {len(out):,} store-days.")
    return out


# ----------------------------------------------------------------------------- history features
def add_last_year_features(feat, hist, lag_days=LAST_YEAR_LAG_DAYS):
    """Sales of the same store one year (52 weeks) earlier. Needs only days far before the forecast origin."""
    pos = hist[(hist["Open"] == 1) & (hist["Sales"] > 0)]
    dense = pos.pivot(index="Date", columns="Store", values="Sales")
    dense = dense.reindex(pd.date_range(dense.index.min(), dense.index.max()))
    roll = dense.rolling(7, center=True, min_periods=3).mean()
    date_pos = pd.Series(np.arange(len(dense)), index=dense.index)
    store_pos = pd.Series(np.arange(dense.shape[1]), index=dense.columns)
    di = date_pos.reindex(feat["Date"] - pd.Timedelta(days=lag_days)).values
    si = store_pos.reindex(feat["Store"]).values
    ok = ~np.isnan(di) & ~np.isnan(si)
    a, b = np.full(len(feat), np.nan, dtype="float32"), np.full(len(feat), np.nan, dtype="float32")
    a[ok] = dense.values[di[ok].astype(int), si[ok].astype(int)]
    b[ok] = roll.values[di[ok].astype(int), si[ok].astype(int)]
    out = feat.copy()
    out["ly_sales_same_dow"], out["ly_week_mean"] = a, b
    return out


def history_profile(hist, origin):
    """Store summaries using only days up to the forecast origin. Returns (store table, store-by-weekday table)."""
    origin = pd.Timestamp(origin)
    h = hist[(hist["Date"] <= origin) & (hist["Open"] == 1) & (hist["Sales"] > 0)]
    w91 = h[h["Date"] > origin - pd.Timedelta(days=91)]
    w28 = h[h["Date"] > origin - pd.Timedelta(days=28)]
    stores = pd.Index(sorted(hist["Store"].unique()), name="Store")
    prof = pd.DataFrame(index=stores)
    prof["hist_mean_28"] = w28.groupby("Store")["Sales"].mean()
    prof["hist_mean_91"] = w91.groupby("Store")["Sales"].mean()
    prof["hist_mean_all"] = h.groupby("Store")["Sales"].mean()
    prof["hist_customers_91"] = w91.groupby("Store")["Customers"].mean()
    prof["hist_spc_91"] = w91.groupby("Store")["Sales"].sum() / w91.groupby("Store")["Customers"].sum()
    prof["hist_days_91"] = w91.groupby("Store")["Sales"].size()
    ordinary91 = w91[(w91["DayOfWeek"] <= 5) & (w91["StateHoliday"] == "0")]
    by_promo = ordinary91.groupby(["Store", "Promo"])["Sales"].mean().unstack()
    prof["hist_promo_mean_91"] = by_promo.get(1)
    prof["hist_nopromo_mean_91"] = by_promo.get(0)
    ordinary_all = h[(h["DayOfWeek"] <= 5) & (h["StateHoliday"] == "0")]
    by_promo_all = ordinary_all.groupby(["Store", "Promo"])["Sales"].mean().unstack()
    prof["hist_promo_lift_all"] = by_promo_all.get(1) / by_promo_all.get(0)
    prof["hist_momentum"] = prof["hist_mean_28"] / prof["hist_mean_91"]
    dow = pd.DataFrame({
        "hist_dow_mean_91": w91.groupby(["Store", "DayOfWeek"])["Sales"].mean(),
        "hist_dow_mean_all": h.groupby(["Store", "DayOfWeek"])["Sales"].mean(),
    }).reset_index()
    return prof.astype("float32"), dow


def build_origin_rows(feat, hist, origin, horizon=FEATURE_HORIZON_DAYS, max_date=None, only_open=True):
    """All rows for one forecast origin: planned features of each future day plus history up to the origin."""
    origin = pd.Timestamp(origin)
    end = origin + pd.Timedelta(days=horizon)
    if max_date is not None:
        end = min(end, pd.Timestamp(max_date))
    rows = feat[(feat["Date"] > origin) & (feat["Date"] <= end)]
    if only_open:
        rows = rows[rows["Open"] == 1]
    prof, dow = history_profile(hist, origin)
    rows = rows.join(prof, on="Store").merge(dow, left_on=["Store", "DayOfWeek"], right_on=["Store", "DayOfWeek"], how="left")
    rows["origin_date"] = origin
    rows["horizon_day"] = (rows["Date"] - origin).dt.days.astype("int16")
    return rows


def origin_grid(first_date, last_origin, spacing=ORIGIN_SPACING_DAYS, extra=(), min_history=MIN_HISTORY_DAYS):
    """Forecast origins every `spacing` days counted back from the last origin, plus any extra origins."""
    first_ok = pd.Timestamp(first_date) + pd.Timedelta(days=min_history)
    grid, o = [], pd.Timestamp(last_origin)
    while o >= first_ok:
        grid.append(o)
        o -= pd.Timedelta(days=spacing)
    return sorted(set(grid) | {pd.Timestamp(x) for x in extra if pd.Timestamp(x) >= first_ok})


def build_dataset(feat, hist, origins, horizon=FEATURE_HORIZON_DAYS, max_date=None):
    """Stack the rows of every origin into one training table."""
    parts = []
    for i, o in enumerate(origins, 1):
        parts.append(build_origin_rows(feat, hist, o, horizon=horizon, max_date=max_date))
        if i % 10 == 0:
            logger.info(f"  built {i} of {len(origins)} origins")
    return downcast(pd.concat(parts, ignore_index=True))


def downcast(df):
    """Store decimals as float32 (except the targets) so the development and test tables have identical types."""
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == "float64" and c not in ("Sales", "Customers"):
            out[c] = out[c].astype("float32")
    return out


# ----------------------------------------------------------------------------- preprocessing pipeline
def make_preprocessor(features):
    """Impute, scale and encode: one ColumnTransformer used identically in training and in the app."""
    kinds = split_by_kind(features)
    numeric = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    parts = []
    if kinds["numeric"]:
        parts.append(("numeric", numeric, kinds["numeric"]))
    if kinds["binary"]:
        parts.append(("binary", "passthrough", kinds["binary"]))
    if kinds["categorical"]:
        parts.append(("categorical", categorical, kinds["categorical"]))
    return ColumnTransformer(parts, verbose_feature_names_out=False)


# ----------------------------------------------------------------------------- metrics
def rmspe(y_true, y_pred):
    """Root mean squared percentage error on positive sales only."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    m = y_true > 0
    return float(np.sqrt(np.mean(((y_true[m] - y_pred[m]) / y_true[m]) ** 2)))
