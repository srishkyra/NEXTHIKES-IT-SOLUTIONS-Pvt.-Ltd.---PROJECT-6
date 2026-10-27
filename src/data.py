"""Loading, checking and cleaning functions shared by the notebook and the app."""
import numpy as np
import pandas as pd

from .config import DATA_INTERIM, RAW_FILES
from .logger import get_logger

logger = get_logger("rossmann.data")

VALID_STATE_HOLIDAYS = {"0", "a", "b", "c"}
MIN_PLAUSIBLE_COMPETITION_YEAR = 1990   # earlier competitor opening years are treated as unreliable
OUTLIER_Z_THRESHOLD = 5.0               # robust z-score cut-off for FLAGGING (never for deleting) unusual sales
LONG_GAP_DAYS = 30                      # a store missing at least this many days has a "history gap"


# ----------------------------------------------------------------------------- loading
def load_raw():
    """Read the four raw files. StateHoliday is read as text because it mixes '0' with letters."""
    train = pd.read_csv(RAW_FILES["train"], dtype={"StateHoliday": str}, parse_dates=["Date"])
    test = pd.read_csv(RAW_FILES["test"], dtype={"StateHoliday": str}, parse_dates=["Date"])
    store = pd.read_csv(RAW_FILES["store"])
    sample = pd.read_csv(RAW_FILES["sample_submission"])
    logger.info(f"Loaded raw data: train {train.shape}, test {test.shape}, store {store.shape}, sample {sample.shape}")
    return train, test, store, sample


def coverage_by_store(df):
    """Rows, first and last date and number of missing calendar days for each store."""
    n_days = (df["Date"].max() - df["Date"].min()).days + 1
    cov = df.groupby("Store")["Date"].agg(rows="size", first_date="min", last_date="max").reset_index()
    cov["expected_days"] = n_days
    cov["missing_days"] = n_days - cov["rows"]
    return cov


# ----------------------------------------------------------------------------- store.csv
def clean_store(store):
    """Clean store.csv, keeping 'not applicable' (Promo2) separate from 'unknown' (competition)."""
    s = store.copy()

    # Competition distance: unknown for a few stores -> median plus a flag (distance is a store attribute, not the target)
    s["CompetitionDistance_missing"] = s["CompetitionDistance"].isna().astype("int8")
    s["CompetitionDistance"] = s["CompetitionDistance"].fillna(s["CompetitionDistance"].median())

    # Competitor opening date: build one date column. Unknown or implausible years are left unknown (not invented).
    year, month = s["CompetitionOpenSinceYear"], s["CompetitionOpenSinceMonth"]
    s["CompetitionOpenYear_suspect"] = (year < MIN_PLAUSIBLE_COMPETITION_YEAR).astype("int8")
    valid = year.notna() & month.notna() & (year >= MIN_PLAUSIBLE_COMPETITION_YEAR)
    text_date = (year.where(valid).astype("Int64").astype("string") + "-"
                 + month.where(valid).astype("Int64").astype("string") + "-01")
    s["CompetitionOpenSince"] = pd.to_datetime(text_date, errors="coerce")
    s["CompetitionOpenDate_known"] = s["CompetitionOpenSince"].notna().astype("int8")
    s = s.drop(columns=["CompetitionOpenSinceYear", "CompetitionOpenSinceMonth"])

    # Promo2: missing details mean "not participating" (not applicable), so code them explicitly
    not_participating = s["Promo2"] == 0
    participating_missing = bool(s.loc[~not_participating, ["Promo2SinceWeek", "Promo2SinceYear", "PromoInterval"]].isna().any(axis=None))
    assert not participating_missing, "Participating Promo2 stores have missing details; review before coding as not applicable"
    for col in ["Promo2SinceWeek", "Promo2SinceYear"]:
        s[col] = s[col].where(~not_participating, 0).astype("int16")
    s["PromoInterval"] = s["PromoInterval"].where(~not_participating, "None")
    s["Promo2"] = s["Promo2"].astype("int8")

    logger.info("Cleaned store.csv: competition distance imputed with median and flagged; competitor dates rebuilt; Promo2 N/A coded.")
    return s


def add_history_gap(store, train):
    """Add the number of missing calendar days per store and a flag for stores with a long gap."""
    gaps = coverage_by_store(train)[["Store", "missing_days"]].rename(columns={"missing_days": "missing_history_days"})
    s = store.merge(gaps, on="Store", how="left", validate="1:1")
    s["missing_history_days"] = s["missing_history_days"].fillna(0).astype("int16")
    s["has_history_gap"] = (s["missing_history_days"] >= LONG_GAP_DAYS).astype("int8")
    return s


# ----------------------------------------------------------------------------- sales data
def flag_sales_outliers(df, threshold=OUTLIER_Z_THRESHOLD):
    """Flag unusually high (+1) or low (-1) sales relative to the same store on the same weekday.

    Uses a robust z-score on log sales (median and MAD) over open, positive-sales days. This only FLAGS rows:
    nothing is removed or capped, because high sales are often genuine promotion or holiday peaks.
    """
    used = (df["Open"] == 1) & (df["Sales"] > 0)
    log_sales = np.log1p(df.loc[used, "Sales"])
    keys = [df.loc[used, "Store"], df.loc[used, "DayOfWeek"]]
    median = log_sales.groupby(keys).transform("median")
    mad = (log_sales - median).abs().groupby(keys).transform("median") * 1.4826
    z = (log_sales - median) / mad.replace(0, np.nan)

    flag = pd.Series(0, index=df.index, dtype="int8")
    flag.loc[used] = np.where(z > threshold, 1, np.where(z < -threshold, -1, 0)).astype("int8")
    z_full = pd.Series(np.nan, index=df.index, dtype="float32")
    z_full.loc[used] = z.astype("float32")
    return flag, z_full


def clean_train(train):
    """Standardise types, check dates, mark suspect rows and flag unusual sales (no rows are deleted)."""
    df = train.copy()
    df["StateHoliday"] = df["StateHoliday"].astype(str)
    assert set(df["StateHoliday"].unique()) <= VALID_STATE_HOLIDAYS, "Unexpected StateHoliday value in train"
    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)

    expected_dow = df["Date"].dt.dayofweek + 1
    n_bad = int((df["DayOfWeek"] != expected_dow).sum())
    if n_bad:
        logger.warning(f"{n_bad} train rows had a DayOfWeek that disagrees with the date; recomputed from the date.")
        df["DayOfWeek"] = expected_dow

    df["suspect_zero_sales"] = ((df["Open"] == 1) & (df["Sales"] == 0)).astype("int8")
    df["use_for_training"] = ((df["Open"] == 1) & (df["Sales"] > 0)).astype("int8")
    df["sales_outlier_flag"], df["sales_robust_z"] = flag_sales_outliers(df)
    logger.info(f"Cleaned train: {int(df['suspect_zero_sales'].sum())} suspect zero-sales rows flagged, "
                f"{int((df['sales_outlier_flag'] != 0).sum())} unusual-sales rows flagged, none removed.")
    return df


def clean_test(test):
    """Fill missing Open values with the rule 'open unless Sunday or a state holiday' and flag them."""
    df = test.copy()
    df["StateHoliday"] = df["StateHoliday"].astype(str)
    assert set(df["StateHoliday"].unique()) <= VALID_STATE_HOLIDAYS, "Unexpected StateHoliday value in test"
    df["Open_imputed"] = df["Open"].isna().astype("int8")
    open_by_rule = ((df["DayOfWeek"] != 7) & (df["StateHoliday"] == "0")).astype("int8")
    df["Open"] = df["Open"].fillna(open_by_rule).astype("int8")
    logger.info(f"Cleaned test: {int(df['Open_imputed'].sum())} missing Open values filled by rule.")
    return df.sort_values("Id").reset_index(drop=True)


def merge_store(df, store, name):
    """Join store details; checks the join is many-to-one, keeps the row count and matches every store."""
    n_before = len(df)
    out = df.merge(store, on="Store", how="left", validate="m:1")
    assert len(out) == n_before, f"{name}: row count changed after the join"
    assert out["StoreType"].notna().all(), f"{name}: some stores have no store details"
    logger.info(f"Joined store details to {name}: {n_before:,} rows kept, all stores matched.")
    return out


def build_clean_data(train, test, store):
    """Full cleaning pipeline: returns (train_clean, test_clean, store_clean)."""
    store_clean = add_history_gap(clean_store(store), train)
    train_clean = merge_store(clean_train(train), store_clean, "train")
    test_clean = merge_store(clean_test(test), store_clean, "test")
    return train_clean, test_clean, store_clean


# ----------------------------------------------------------------------------- saving and reloading
def save_interim(df, name):
    """Save a cleaned table to data/interim/<name>.csv.gz and return the path."""
    DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    path = DATA_INTERIM / f"{name}.csv.gz"
    df.to_csv(path, index=False)
    logger.info(f"Saved {name}: {df.shape[0]:,} rows x {df.shape[1]} columns -> {path.name}")
    return path


def load_interim(name):
    """Reload a cleaned table with the right types (dates and holiday codes)."""
    path = DATA_INTERIM / f"{name}.csv.gz"
    columns = pd.read_csv(path, nrows=0).columns
    dates = [c for c in ("Date", "CompetitionOpenSince") if c in columns]
    dtypes = {"StateHoliday": str} if "StateHoliday" in columns else None
    return pd.read_csv(path, parse_dates=dates, dtype=dtypes)
