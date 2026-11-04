"""Inference for the app: one store + a table of future dates and planned inputs -> sales, customers and 90% intervals.

The features are built with exactly the training code (src/features.py). Two small precomputed pieces replace the
full history, so the app stays light:
  * the store profile at the forecast origin (features.history_profile, the same call the training rows use);
  * the last 200 days of planned inputs per store (for promotion-week, closure and holiday-distance features).
The holiday calendar of the training data (holidays flagged for at least 90% of stores) is reproduced by adding
placeholder "calendar carrier" rows, so features.holiday_calendar finds exactly the same national holidays from a
single store's rows. Section 11.3 checks that the app's features equal the training features for every test store.
"""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import features as F
from .bundle import load_bundle
from .config import ROOT
from .intervals import week_ahead

APP_DIR = ROOT / "app_data"
SNAPSHOT_DAYS = 200
N_CARRIERS = 19                      # with the store's own row, 19 carriers give a holiday share of at least 0.95
ALIASES = {"IsPromo": "Promo", "IsOpen": "Open", "IsSchoolHoliday": "SchoolHoliday"}
STORE_COLUMNS = ["Store", "StoreType", "Assortment", "CompetitionDistance", "CompetitionDistance_missing", "CompetitionOpenSince",
                 "CompetitionOpenDate_known", "Promo2", "Promo2SinceWeek", "Promo2SinceYear", "PromoInterval", "has_history_gap"]
PLAN_INPUTS = ["Store", "Date", "DayOfWeek", "Open", "Promo", "StateHoliday", "SchoolHoliday"]


# ----------------------------------------------------------------------------- packaging (run once, in the notebook)
def build_app_data(train_clean, test_clean, store_clean, bundle_path, out_dir=APP_DIR):
    """Write the small files the app needs (about 5 MB) and copy the model bundle next to them."""
    import shutil
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    origin = train_clean["Date"].max()
    plan = F.make_plan(train_clean, test_clean)
    recent = train_clean[train_clean["Date"] > origin - pd.Timedelta(days=SNAPSHOT_DAYS)][PLAN_INPUTS]
    recent.to_csv(out / "recent_plan.csv.gz", index=False)
    test_clean[PLAN_INPUTS].to_csv(out / "test_plan.csv.gz", index=False)
    store_clean[STORE_COLUMNS].to_csv(out / "stores.csv", index=False)
    prof, dow = F.history_profile(train_clean, origin)
    prof.to_csv(out / "profile.csv")
    dow.to_csv(out / "profile_weekday.csv", index=False)
    F.find_gap_end(train_clean).rename("gap_end").to_csv(out / "gap_end.csv")
    # sales one year (364 days) before the supported dates, only needed if the model uses the same-year-earlier features
    lag = pd.Timedelta(days=F.LAST_YEAR_LAG_DAYS)
    ly = train_clean[train_clean["Date"].between(origin + pd.Timedelta(days=1) - lag - pd.Timedelta(days=10),
                                                 test_clean["Date"].max() - lag + pd.Timedelta(days=10))]
    ly[["Store", "Date", "Open", "Sales"]].to_csv(out / "last_year_sales.csv.gz", index=False)
    cal = F.holiday_calendar(plan)
    flagged = plan[plan["StateHoliday"] != "0"]
    types = flagged.groupby("Date")["StateHoliday"].agg(lambda s: s.mode().iloc[0])
    pd.DataFrame({"Date": cal["any"], "StateHoliday": types.reindex(cal["any"]).values}).to_csv(out / "national_holidays.csv", index=False)
    # one promotion calendar for the whole chain (Section 5): used for stores that are not in test.csv
    chain_promo = test_clean.groupby("Date")["Promo"].agg(lambda s: int(s.mode().iloc[0]))
    chain_promo.rename("Promo").to_csv(out / "chain_promo_calendar.csv")
    shutil.copy(bundle_path, out / Path(bundle_path).name)
    meta = {"origin": str(origin.date()), "first_date": str((origin + pd.Timedelta(days=1)).date()), "last_date": str(test_clean["Date"].max().date()),
            "bundle": Path(bundle_path).name, "stores": int(store_clean["Store"].nunique())}
    (out / "app_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


# ----------------------------------------------------------------------------- the forecaster used by the app
class Forecaster:
    def __init__(self, app_dir=APP_DIR):
        d = Path(app_dir)
        self.meta = json.loads((d / "app_meta.json").read_text(encoding="utf-8"))
        self.origin = pd.Timestamp(self.meta["origin"])
        self.first_date, self.last_date = pd.Timestamp(self.meta["first_date"]), pd.Timestamp(self.meta["last_date"])
        self.bundle = load_bundle(d / self.meta["bundle"])
        self.features = self.bundle["features"]
        self.recent = pd.read_csv(d / "recent_plan.csv.gz", parse_dates=["Date"], dtype={"StateHoliday": str})
        self.test_plan = pd.read_csv(d / "test_plan.csv.gz", parse_dates=["Date"], dtype={"StateHoliday": str})
        self.stores_tab = pd.read_csv(d / "stores.csv", parse_dates=["CompetitionOpenSince"], keep_default_na=True)
        self.stores_tab["PromoInterval"] = self.stores_tab["PromoInterval"].fillna("None")
        self.profile = pd.read_csv(d / "profile.csv", index_col="Store").astype("float32")
        self.profile_dow = pd.read_csv(d / "profile_weekday.csv")
        self.gap_end = pd.read_csv(d / "gap_end.csv", index_col=0, parse_dates=["gap_end"])["gap_end"]
        nat = pd.read_csv(d / "national_holidays.csv", parse_dates=["Date"], dtype={"StateHoliday": str})
        self.national = nat.set_index("Date")["StateHoliday"]
        self.chain_promo = pd.read_csv(d / "chain_promo_calendar.csv", parse_dates=["Date"]).set_index("Date")["Promo"]
        self.last_year = pd.read_csv(d / "last_year_sales.csv.gz", parse_dates=["Date"])
        cal = self.bundle["interval_calibration"]
        self.size_edges = json.loads(cal["size_edges"].iloc[0])
        self.cal = {t: g.drop(columns=["target", "size_edges", "level"]) for t, g in cal.groupby("target")}

    @property
    def stores(self):
        return sorted(self.stores_tab["Store"].astype(int).tolist())

    # ---------------------------------------------------------------- default plan and example file
    def default_plan(self, store):
        """The planned inputs of every supported date: from test.csv when the store is in it, otherwise from the chain-wide
        promotion calendar, the national holidays and the store's usual Sunday opening."""
        days = pd.date_range(self.first_date, self.last_date)
        tp = self.test_plan[self.test_plan["Store"] == store]
        if len(tp) == len(days):
            out = tp.set_index("Date").reindex(days)
        else:
            hist = self.recent[self.recent["Store"] == store]
            sunday_open = bool(hist.loc[hist["DayOfWeek"] == 7, "Open"].mean() > 0.5) if (hist["DayOfWeek"] == 7).any() else False
            out = pd.DataFrame(index=days)
            out["StateHoliday"] = self.national.reindex(days).fillna("0").values
            out["Promo"] = self.chain_promo.reindex(days).fillna(0).astype(int).values
            out["SchoolHoliday"] = 0
            is_sun = days.dayofweek == 6
            out["Open"] = np.where((out["StateHoliday"] != "0") | (is_sun & ~sunday_open), 0, 1)
        out = out.reset_index().rename(columns={"index": "Date"})
        out["Store"] = store
        return out[["Date", "Open", "Promo", "StateHoliday", "SchoolHoliday"]]

    def example_input(self, store):
        ex = self.default_plan(store)
        ex["IsHoliday"] = (ex["StateHoliday"] != "0").astype(int)
        ex["IsWeekend"] = (ex["Date"].dt.dayofweek >= 5).astype(int)
        ex["Date"] = ex["Date"].dt.strftime("%Y-%m-%d")
        return ex[["Date", "Open", "Promo", "IsHoliday", "StateHoliday", "SchoolHoliday", "IsWeekend"]]

    # ---------------------------------------------------------------- input checks
    def validate(self, store, table):
        """Return (clean table with one row per uploaded date, list of notes). Raises ValueError with a readable message."""
        if store not in set(self.stores):
            raise ValueError(f"Store {store} is not in the data. Choose a store number between {min(self.stores)} and {max(self.stores)}.")
        df = table.rename(columns={c: ALIASES.get(c, c) for c in table.columns}).copy()
        notes = []
        missing = [c for c in ["Date", "Promo"] if c not in df.columns]
        if missing:
            raise ValueError(f"The file is missing the column(s): {', '.join(missing)}. Required: Date and Promo (or IsPromo). "
                             "Optional: Open, StateHoliday or IsHoliday, SchoolHoliday, IsWeekend.")
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        if df["Date"].isna().any():
            raise ValueError(f"{int(df['Date'].isna().sum())} row(s) have a date that cannot be read. Use the format YYYY-MM-DD.")
        if df["Date"].duplicated().any():
            raise ValueError("Each date may appear only once.")
        outside = df[(df["Date"] < self.first_date) | (df["Date"] > self.last_date)]
        if len(outside):
            raise ValueError(f"{len(outside)} date(s) are outside the supported forecast window "
                             f"({self.first_date.date()} to {self.last_date.date()}, the 48 days after the last day with known sales).")
        default = self.default_plan(store).set_index("Date")
        if "StateHoliday" not in df.columns and "IsHoliday" in df.columns:
            df["StateHoliday"] = np.where(pd.to_numeric(df["IsHoliday"], errors="coerce") == 1, "a", "0")
            notes.append("IsHoliday = 1 was read as a public holiday (StateHoliday 'a').")
        for col in ["Open", "StateHoliday", "SchoolHoliday"]:
            if col not in df.columns:
                df[col] = df["Date"].map(default[col])
                notes.append(f"{col} was not supplied; the default plan was used.")
        for col in ["Promo", "Open", "SchoolHoliday"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            bad = ~df[col].isin([0, 1])
            if bad.any():
                raise ValueError(f"Column {col} must contain only 0 or 1 ({int(bad.sum())} other value(s) found).")
            df[col] = df[col].astype(int)
        df["StateHoliday"] = df["StateHoliday"].astype(str).str.strip().replace({"0.0": "0", "nan": "0"})
        if not df["StateHoliday"].isin(["0", "a", "b", "c"]).all():
            raise ValueError("StateHoliday must be 0, a (public holiday), b (Easter) or c (Christmas).")
        if "IsWeekend" in df.columns:
            wrong = pd.to_numeric(df["IsWeekend"], errors="coerce") != (df["Date"].dt.dayofweek >= 5).astype(int)
            if wrong.any():
                raise ValueError(f"IsWeekend disagrees with the date on {int(wrong.sum())} row(s) (Saturday and Sunday are weekend days).")
        return df.sort_values("Date").reset_index(drop=True)[["Date", "Open", "Promo", "StateHoliday", "SchoolHoliday"]], notes

    # ---------------------------------------------------------------- features (exactly the training code)
    def feature_rows(self, store, clean):
        """Training features for the uploaded dates. Dates between the origin and the last uploaded date that were not
        uploaded are filled from the default plan, so week-based features see a complete calendar."""
        default = self.default_plan(store)
        future = default[default["Date"] <= self.last_date].set_index("Date")
        future.update(clean.set_index("Date"))
        future = future.reset_index()
        future["Store"] = store
        hist = self.recent[self.recent["Store"] == store].drop(columns="DayOfWeek")
        rows = pd.concat([hist, future], ignore_index=True)
        rows["DayOfWeek"] = rows["Date"].dt.dayofweek + 1
        days = pd.date_range(rows["Date"].min(), self.last_date)
        carriers = pd.DataFrame({"Date": np.tile(days, N_CARRIERS), "Store": np.repeat(-np.arange(1, N_CARRIERS + 1), len(days))})
        carriers["DayOfWeek"] = carriers["Date"].dt.dayofweek + 1
        carriers["Open"], carriers["Promo"], carriers["SchoolHoliday"] = 1, 0, 0
        carriers["StateHoliday"] = carriers["Date"].map(self.national).fillna("0")
        st = self.stores_tab.set_index("Store")
        neutral = {"StoreType": "a", "Assortment": "a", "CompetitionDistance": 1000.0, "CompetitionDistance_missing": 0,
                   "CompetitionOpenSince": pd.NaT, "CompetitionOpenDate_known": 0, "Promo2": 0, "Promo2SinceWeek": 0,
                   "Promo2SinceYear": 0, "PromoInterval": "None", "has_history_gap": 0}
        rows = rows.join(st, on="Store")
        for c, v in neutral.items():
            carriers[c] = v
        # one carrier takes part in Promo2, so the Promo2 start dates always form a date column (features.py compares
        # them with dates; with a single non-Promo2 store they would otherwise be an all-missing number column)
        first = carriers["Store"] == -1
        carriers.loc[first, ["Promo2", "Promo2SinceYear", "Promo2SinceWeek"]] = [1, 2009, 1]
        carriers.loc[first, "PromoInterval"] = "Jan,Apr,Jul,Oct"
        plan = pd.concat([rows, carriers], ignore_index=True)
        plan["Sales"], plan["Customers"], plan["use_for_training"], plan["is_test"] = np.nan, np.nan, 0, 0
        plan["StateHoliday"] = plan["StateHoliday"].astype(str)
        plan = plan.sort_values(["Store", "Date"]).reset_index(drop=True)
        quiet = logging.getLogger("rossmann.features")
        level = quiet.level
        quiet.setLevel(logging.WARNING)                      # one store at a time: no progress line per call
        try:
            known = F.add_known_future_features(plan[F.PLAN_COLUMNS + ["is_test"]], self.gap_end)
        finally:
            quiet.setLevel(level)
        out = known[(known["Store"] == store) & (known["Date"] > self.origin) & known["Date"].isin(clean["Date"])]
        out = out.join(self.profile, on="Store").merge(self.profile_dow, on=["Store", "DayOfWeek"], how="left")
        out["origin_date"] = self.origin
        out["horizon_day"] = (out["Date"] - self.origin).dt.days.astype("int16")
        if any(f.startswith("ly_") for f in self.features):
            out = F.add_last_year_features(out, self.last_year[self.last_year["Store"] == store])
        return F.downcast(out.reset_index(drop=True))

    # ---------------------------------------------------------------- forecast
    def _interval(self, target, forecast, rows):
        cal = self.cal[target]
        g = pd.DataFrame({"week_ahead": week_ahead(rows["horizon_day"])})
        if "store_size" in cal.columns:
            labels = ["smallest 25%", "lower middle", "upper middle", "largest 25%"]
            g["store_size"] = np.array(labels)[np.searchsorted(self.size_edges, rows["hist_mean_all"].values)]
        g["week_ahead"] = g["week_ahead"].clip(upper=int(cal["week_ahead"].max()))
        m = g.merge(cal, on=[c for c in g.columns], how="left")
        f = np.log1p(forecast)
        return np.maximum(np.expm1(f + m["q_low"].values), 0.0), np.expm1(f + m["q_high"].values)

    def predict(self, store, table):
        store = int(store)
        clean, notes = self.validate(store, table)
        rows = self.feature_rows(store, clean)
        X = rows[self.features]
        sales = np.maximum(self.bundle["sales_model"].predict(X), 0.0)
        cust = np.maximum(self.bundle["customers_model"].predict(X), 0.0)
        s_lo, s_hi = self._interval("Sales", sales, rows)
        c_lo, c_hi = self._interval("Customers", cust, rows)
        closed = rows["Open"].values == 0
        res = pd.DataFrame({
            "Date": rows["Date"].dt.date, "Store": store, "Open": rows["Open"].values, "Promo": rows["promo"].values,
            "StateHoliday": rows["state_holiday"].values, "SchoolHoliday": rows["school_holiday"].values, "days_ahead": rows["horizon_day"].values,
            "sales_forecast": sales, "sales_low_90": s_lo, "sales_high_90": s_hi,
            "customers_forecast": cust, "customers_low_90": c_lo, "customers_high_90": c_hi,
        })
        num = ["sales_forecast", "sales_low_90", "sales_high_90", "customers_forecast", "customers_low_90", "customers_high_90"]
        res.loc[closed, num] = 0.0
        res[num] = res[num].round(0).astype(int)
        res["model_version"] = self.meta["bundle"]
        return res, notes
