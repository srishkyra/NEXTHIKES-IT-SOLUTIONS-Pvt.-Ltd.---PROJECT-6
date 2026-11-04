"""Timestamped model bundles: everything needed to serve a forecast, saved as dd-mm-yyyy-HH-MM-SS.pkl."""
import json
from datetime import datetime
from pathlib import Path

import joblib

from .config import MODEL_TIMESTAMP_FORMAT, MODELS_DIR


def timestamp(now=None):
    return (now or datetime.now()).strftime(MODEL_TIMESTAMP_FORMAT)


def save_bundle(bundle, metadata, folder=MODELS_DIR, now=None):
    """Save the bundle as <folder>/<timestamp>.pkl and its metadata as <timestamp>.json; return the .pkl path."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    ts = timestamp(now)
    path = folder / f"{ts}.pkl"
    joblib.dump({**bundle, "metadata": {**metadata, "bundle": path.name}}, path, compress=3)
    (folder / f"{ts}.json").write_text(json.dumps({**metadata, "bundle": path.name}, indent=2, default=str), encoding="utf-8")
    return path


def list_bundles(folder=MODELS_DIR):
    """All bundles in a folder, oldest first (ordered by the timestamp in the file name, not by file dates)."""
    out = []
    for p in Path(folder).glob("*.pkl"):
        try:
            out.append((datetime.strptime(p.stem, MODEL_TIMESTAMP_FORMAT), p))
        except ValueError:
            continue
    return [p for _, p in sorted(out)]


def load_bundle(path=None, folder=MODELS_DIR):
    """Load a bundle; without a path, the latest one in the folder."""
    if path is None:
        bundles = list_bundles(folder)
        if not bundles:
            raise FileNotFoundError(f"No timestamped model bundle found in {folder}")
        path = bundles[-1]
    return joblib.load(path)
