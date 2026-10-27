"""Shared look for every figure, table, the Streamlit app and the slides (palette taken from the reference image)."""
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from .config import FIGURES_DIR, OUTPUTS_DIR

# ---------------------------------------------------------------- palette
PALETTE = {
    "espresso":   "#37170C",   # titles, axis text, actual values
    "umber":      "#772F12",   # dark primary, table headers
    "charcoal":   "#584F43",   # secondary text, gridlines
    "clay":       "#845C36",   # secondary series
    "terracotta": "#C44F10",   # primary accent (promo, key bars)
    "rose":       "#DA4A53",   # alerts, errors, closed or negative
    "bronze":     "#BB7534",   # third series
    "stone":      "#8A897A",   # baselines, neutral, "other"
    "marigold":   "#EF920D",   # main sales / forecast line
    "saffron":    "#F0BA25",   # highlights, annotations
    "sand":       "#EEC288",   # fills, prediction-interval bands
    "cream":      "#F7F2DC",   # backgrounds
}

# at most 5 clearly different colours for comparing groups; keep the group-to-colour mapping fixed
CATEGORICAL = [PALETTE[k] for k in ("terracotta", "marigold", "clay", "stone", "rose")]

# fixed meaning for forecasting plots
FORECAST_COLORS = {
    "actual": PALETTE["espresso"],
    "forecast": PALETTE["marigold"],
    "interval": PALETTE["sand"],
    "baseline": PALETTE["stone"],
    "error": PALETTE["rose"],
}

# sequential scale (low to high intensity) and a diverging scale (negative to positive)
SEQ_CMAP = LinearSegmentedColormap.from_list(
    "rossmann_seq",
    [PALETTE[k] for k in ("cream", "sand", "marigold", "terracotta", "umber", "espresso")],
)
DIV_CMAP = LinearSegmentedColormap.from_list(
    "rossmann_div", [PALETTE["charcoal"], PALETTE["cream"], PALETTE["terracotta"]]
)


def apply_style() -> None:
    """Apply the shared matplotlib style (cream background, espresso text, light gridlines)."""
    mpl.rcParams.update({
        "figure.facecolor": PALETTE["cream"], "axes.facecolor": PALETTE["cream"],
        "savefig.facecolor": PALETTE["cream"],
        "axes.edgecolor": PALETTE["charcoal"], "axes.labelcolor": PALETTE["espresso"],
        "axes.titlecolor": PALETTE["espresso"], "text.color": PALETTE["espresso"],
        "xtick.color": PALETTE["charcoal"], "ytick.color": PALETTE["charcoal"],
        "axes.grid": True, "grid.color": PALETTE["charcoal"], "grid.alpha": 0.15,
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.labelsize": 11,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.frameon": False, "legend.fontsize": 10,
        "figure.figsize": (10, 5.5), "figure.dpi": 110,
        "savefig.dpi": 200, "savefig.bbox": "tight",
        "axes.prop_cycle": mpl.cycler(color=CATEGORICAL),
    })


# ---------------------------------------------------------------- figure and table numbering
# Numbers are stable: re-running a cell reuses the same number for the same (section, title).
_numbers = {}                  # (kind, section, title) -> n
_per_section = {}              # (kind, section) -> highest n used
figure_index, table_index = [], []


def _number(kind, section, title):
    key = (kind, str(section), title)
    if key not in _numbers:
        _per_section[(kind, str(section))] = _per_section.get((kind, str(section)), 0) + 1
        _numbers[key] = _per_section[(kind, str(section))]
    return _numbers[key]


def _slug(text, max_len=60):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:max_len]


def save_figure(fig, section, title):
    """Save a figure as figures/fig_<section>_<n>_<title>.png and return its label, e.g. 'Figure 5.2'."""
    n = _number("Figure", section, title)
    label = f"Figure {section}.{n}"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"fig_{section}_{n:02d}_{_slug(title)}.png"
    fig.savefig(path)
    entry = {"label": label, "title": title, "file": path.name}
    if entry not in figure_index:
        figure_index.append(entry)
    print(f"{label}: {title}")
    return label


def show_table(df, section, title, save=True):
    """Display a numbered table caption above a DataFrame; optionally save it to outputs/tables/."""
    from IPython.display import Markdown, display
    n = _number("Table", section, title)
    label = f"Table {section}.{n}"
    display(Markdown(f"**{label}: {title}**"))
    display(df)
    file = ""
    if save:
        folder = OUTPUTS_DIR / "tables"
        folder.mkdir(parents=True, exist_ok=True)
        file = f"table_{section}_{n:02d}_{_slug(title)}.csv"
        df.to_csv(folder / file)
    entry = {"label": label, "title": title, "file": file}
    if entry not in table_index:
        table_index.append(entry)
    return label


def save_index():
    """Save the figure and table indexes to outputs/ so the report can cite them."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(figure_index).to_csv(OUTPUTS_DIR / "figure_index.csv", index=False)
    pd.DataFrame(table_index).to_csv(OUTPUTS_DIR / "table_index.csv", index=False)


def palette_preview():
    """Draw the 12 colours as labelled swatches and return the figure."""
    fig, ax = plt.subplots(figsize=(12, 3.4))
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 2)
    ax.axis("off")
    for i, (name, hex_code) in enumerate(PALETTE.items()):
        col, row = i % 6, 1 - i // 6
        ax.add_patch(plt.Rectangle((col + 0.05, row + 0.28), 0.9, 0.62, color=hex_code,
                                   ec=PALETTE["charcoal"], lw=0.8))
        ax.text(col + 0.5, row + 0.17, f"{name}\n{hex_code}", ha="center", va="top", fontsize=9)
    ax.set_title("Project colour palette (12 colours from the reference image)")
    return fig
