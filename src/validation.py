"""Forecasting design: fixed validation splits, the feature-availability policy and leakage checks."""
import pandas as pd

from .config import HORIZON_DAYS, N_ROLLING_WINDOWS
from .logger import get_logger

logger = get_logger("rossmann.validation")

ONE_DAY = pd.Timedelta(days=1)


# ----------------------------------------------------------------------------- splits
def make_splits(first_date, last_date, horizon=HORIZON_DAYS, n_windows=N_ROLLING_WINDOWS):
    """Final holdout plus earlier rolling windows, counted back from the last labelled day in blocks of `horizon` days.

    Every split trains on all history before its validation window (an expanding window) and forecasts the whole
    window from that single cut-off, which is the same for every store.
    """
    first, last = pd.Timestamp(first_date), pd.Timestamp(last_date)
    rows, valid_end = [], last
    for i in range(n_windows + 1):
        valid_start = valid_end - pd.Timedelta(days=horizon - 1)
        rows.append({
            "split": "holdout" if i == 0 else f"rolling_{n_windows - i + 1}",
            "role": "Final holdout (scored once)" if i == 0 else "Rolling validation (tuning and model choice)",
            "train_start": first,
            "train_end": valid_start - ONE_DAY,
            "valid_start": valid_start,
            "valid_end": valid_end,
        })
        valid_end = valid_start - ONE_DAY
    return pd.DataFrame(rows).sort_values("valid_start").reset_index(drop=True)


def seasonal_check_split(first_date, test_first_date, test_days):
    """The same calendar dates one year before the test window (a stress test for the summer-holiday shift)."""
    valid_start = pd.Timestamp(test_first_date) - pd.DateOffset(years=1)
    return pd.DataFrame([{
        "split": "seasonal_check",
        "role": "Seasonal stress test (diagnostic only, not used for model choice)",
        "train_start": pd.Timestamp(first_date),
        "train_end": valid_start - ONE_DAY,
        "valid_start": valid_start,
        "valid_end": valid_start + pd.Timedelta(days=test_days - 1),
    }])


def get_masks(df, splits, name, allow_holdout=False):
    """Boolean row masks (train, valid) for one split. The holdout is locked unless explicitly allowed."""
    if name == "holdout" and not allow_holdout:
        raise PermissionError("The final holdout is locked. Only the final evaluation (Section 10) may unlock it, "
                              "and every use must be written to the evaluation-access log.")
    row = splits.loc[splits["split"] == name].iloc[0]
    train = (df["Date"] >= row["train_start"]) & (df["Date"] <= row["train_end"])
    valid = (df["Date"] >= row["valid_start"]) & (df["Date"] <= row["valid_end"])
    return train, valid


def check_splits(splits, first_date, last_date, horizon=HORIZON_DAYS):
    """Automatic checks that the split design is valid. Returns a table of check results."""
    first, last = pd.Timestamp(first_date), pd.Timestamp(last_date)
    rolling = splits[splits["split"].str.startswith("rolling")].sort_values("valid_start")
    holdout = splits[splits["split"] == "holdout"].iloc[0]
    sequential = pd.concat([rolling, splits[splits["split"] == "holdout"]]).sort_values("valid_start")
    seasonal = splits[splits["split"] == "seasonal_check"]

    checks = [
        ("Every training period ends before its validation window starts", bool((splits["train_end"] < splits["valid_start"]).all())),
        ("Rolling windows and the holdout do not overlap each other",
         bool((sequential["valid_start"].iloc[1:].values > sequential["valid_end"].iloc[:-1].values).all())),
        ("The holdout is the last window and ends on the last labelled day", bool(holdout["valid_end"] == last)),
        (f"Holdout and rolling windows all span exactly {horizon} days",
         bool(((sequential["valid_end"] - sequential["valid_start"]).dt.days + 1 == horizon).all())),
        ("All windows lie inside the labelled data range", bool((splits["valid_start"] >= first).all() and (splits["valid_end"] <= last).all())),
        ("The earliest rolling window still has training history before it", bool((rolling["train_end"] >= first).all())),
        ("The seasonal check does not overlap the final holdout",
         bool(seasonal.empty or (seasonal["valid_end"] < holdout["valid_start"]).all())),
    ]
    return pd.DataFrame(checks, columns=["check", "passed"])


# ----------------------------------------------------------------------------- feature-availability policy
FEATURE_CLASSES = {
    "known_future": "Calendar, schedule and holiday information known before the forecast date (allowed)",
    "store_static": "Store attributes that do not change during the forecast window (allowed)",
    "origin_history": "History summarised using only days up to the forecast origin (allowed)",
    "target": "What we predict (never an input)",
    "forbidden": "Actual values from the forecast window, for example Customers on the forecast day (never an input)",
    "row_control": "Columns that select or flag rows in the cleaning step; derived from sales, so never inputs",
    "identifier": "Row identifiers (never inputs)",
}
ALLOWED_INPUT_CLASSES = {"known_future", "store_static", "origin_history"}

FEATURE_POLICY = {
    # known in advance
    "Date": "known_future", "DayOfWeek": "known_future", "Open": "known_future", "Promo": "known_future",
    "StateHoliday": "known_future", "SchoolHoliday": "known_future",
    # store attributes
    "Store": "store_static", "StoreType": "store_static", "Assortment": "store_static",
    "CompetitionDistance": "store_static", "CompetitionDistance_missing": "store_static",
    "CompetitionOpenSince": "store_static", "CompetitionOpenDate_known": "store_static",
    "CompetitionOpenYear_suspect": "store_static", "Promo2": "store_static", "Promo2SinceWeek": "store_static",
    "Promo2SinceYear": "store_static", "PromoInterval": "store_static",
    "missing_history_days": "store_static", "has_history_gap": "store_static",
    # targets and forbidden inputs
    "Sales": "target", "Customers": "forbidden",
    # cleaning controls and identifiers
    "suspect_zero_sales": "row_control", "use_for_training": "row_control", "sales_outlier_flag": "row_control",
    "sales_robust_z": "row_control", "Open_imputed": "row_control", "Id": "identifier",
}


def register_features(names, feature_class):
    """Add engineered features to the policy (Section 6) with their availability class."""
    if feature_class not in FEATURE_CLASSES:
        raise ValueError(f"Unknown class {feature_class!r}; choose from {list(FEATURE_CLASSES)}")
    for name in names:
        FEATURE_POLICY[name] = feature_class


def check_features(features):
    """Raise an error if any model input is unclassified or not allowed as an input."""
    problems = []
    for f in features:
        klass = FEATURE_POLICY.get(f)
        if klass is None:
            problems.append(f"{f}: not in the feature-availability policy")
        elif klass not in ALLOWED_INPUT_CLASSES:
            problems.append(f"{f}: class '{klass}' is not allowed as a model input")
    if problems:
        raise ValueError("Feature policy violation:\n  " + "\n  ".join(problems))
    logger.info(f"Feature policy check passed for {len(list(features))} features.")
    return True
