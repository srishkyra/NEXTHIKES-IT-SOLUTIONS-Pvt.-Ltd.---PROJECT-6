"""Minimal Streamlit page that checks the deployment environment (replaced by the full app in Section 11)."""
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.style import FORECAST_COLORS, apply_style

st.set_page_config(page_title="Rossmann Sales Forecast", page_icon="📈", layout="wide")
apply_style()

st.title("Rossmann Store Sales Forecast")
st.caption("Setup check only. The full forecasting app is built in Section 11.")

st.write("If you can read this page and see the chart below, the hosting environment can run the app, "
         "import the project code and draw figures in the project palette.")

# Demo data only (not real sales): a weekly pattern with noise, drawn in the project colours.
rng = np.random.default_rng(42)
days = np.arange(42)
demo = 6000 + 900 * np.sin(2 * np.pi * days / 7) + rng.normal(0, 150, size=days.size)

fig, ax = plt.subplots(figsize=(9, 4))
ax.fill_between(days, demo * 0.92, demo * 1.08, color=FORECAST_COLORS["interval"], label="Interval (demo)")
ax.plot(days, demo, color=FORECAST_COLORS["forecast"], linewidth=2.2, label="Forecast (demo)")
ax.set_xlabel("Days ahead")
ax.set_ylabel("Sales")
ax.set_title("Demo chart in the project palette")
ax.legend()
st.pyplot(fig)

st.success("Environment check passed.")
