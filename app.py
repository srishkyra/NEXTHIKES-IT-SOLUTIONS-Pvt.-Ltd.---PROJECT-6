"""Rossmann store forecast app: choose a store, upload the planned days, get sales and customer forecasts with 90% ranges."""
import io

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.predict import Forecaster
from src.style import FORECAST_COLORS, PALETTE, apply_style

st.set_page_config(page_title="Rossmann Sales Forecast", page_icon="📈", layout="wide")
apply_style()


@st.cache_resource
def get_forecaster():
    return Forecaster()


fc = get_forecaster()

st.title("Rossmann Store Sales Forecast")
st.caption(f"Forecasts for {fc.first_date:%d %b %Y} to {fc.last_date:%d %b %Y}, made from sales known up to {fc.origin:%d %b %Y}. "
           f"Model version: {fc.meta['bundle']}")

with st.sidebar:
    st.header("1. Choose a store")
    store = st.selectbox("Store number", fc.stores, index=0)
    st.header("2. Planned days")
    example = fc.example_input(store)
    st.download_button("Download an example file for this store", example.to_csv(index=False).encode("utf-8"),
                       file_name=f"store_{store}_plan.csv", mime="text/csv")
    uploaded = st.file_uploader("Upload your plan (CSV)", type=["csv"])
    use_example = st.checkbox("Use the example plan instead", value=uploaded is None)
    st.markdown("**Columns:** `Date` (YYYY-MM-DD) and `Promo` (or `IsPromo`) are required. Optional: `Open`, "
                "`StateHoliday` (0, a, b, c) or `IsHoliday` (0/1), `SchoolHoliday`, `IsWeekend`. Missing optional columns use the "
                "store's default plan.")

table = None
if uploaded is not None and not use_example:
    try:
        table = pd.read_csv(uploaded)
    except Exception:
        st.error("The file could not be read as a CSV table.")
elif use_example:
    table = example

if table is None:
    st.info("Upload a plan or tick 'Use the example plan' in the sidebar to see a forecast.")
    st.stop()

try:
    result, notes = fc.predict(store, table)
except ValueError as err:
    st.error(str(err))
    st.stop()

for n in notes:
    st.caption(f"Note: {n}")

open_days = result[result["Open"] == 1]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Forecast sales, whole period", f"{result['sales_forecast'].sum():,.0f}")
c2.metric("Average sales per open day", f"{open_days['sales_forecast'].mean():,.0f}" if len(open_days) else "closed")
c3.metric("Forecast customers, whole period", f"{result['customers_forecast'].sum():,.0f}")
c4.metric("Open days / promotion days", f"{len(open_days)} / {int(result['Promo'].sum())}")

fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
dates = pd.to_datetime(result["Date"])
for ax, (col, label) in zip(axes, [("sales", "Sales"), ("customers", "Customers")]):
    ax.fill_between(dates, result[f"{col}_low_90"], result[f"{col}_high_90"], color=FORECAST_COLORS["interval"], alpha=0.85, label="90% range")
    ax.plot(dates, result[f"{col}_forecast"], color=FORECAST_COLORS["forecast"], linewidth=2.2, marker="o", markersize=3.5, label="Forecast")
    promo = result["Promo"] == 1
    ax.scatter(dates[promo], result.loc[promo, f"{col}_forecast"], marker="^", s=40, color=PALETTE["terracotta"], zorder=3, label="Promotion day")
    closed = result["Open"] == 0
    ax.scatter(dates[closed], [0] * int(closed.sum()), marker="x", s=40, color=FORECAST_COLORS["error"], zorder=3, label="Closed")
    ax.set_ylabel(label)
    ax.legend(fontsize=8, ncol=4, loc="upper left")
axes[0].set_title(f"Store {store}: daily forecasts with 90% ranges")
axes[1].tick_params(axis="x", rotation=25)
st.pyplot(fig)

st.subheader("Forecast table")
st.dataframe(result, width="stretch", hide_index=True)
buf = io.StringIO()
result.to_csv(buf, index=False)
st.download_button("Download the forecast (CSV)", buf.getvalue().encode("utf-8"), file_name=f"store_{store}_forecast.csv", mime="text/csv")

with st.expander("How to read this"):
    st.markdown(
        "- **Forecast** is the expected sales (or customers) for the day, given the plan you entered.\n"
        "- **90% range**: on past data the actual value fell inside this range about 9 times in 10. Ranges widen for later weeks.\n"
        "- **Closed days** are forecast as zero.\n"
        "- Promotions, holidays and opening days are taken from your file; change them to see how the forecast responds.\n"
        "- The model has seen only one promotion calendar for all stores, so promotion effects are learned from that pattern; "
        "forecasts for stores whose competitor opens during this window (stores 688, 1020, 1072) deserve extra care."
    )
