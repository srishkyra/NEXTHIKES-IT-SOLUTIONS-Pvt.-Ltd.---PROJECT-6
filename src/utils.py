"""Small helpers: reproducibility seeding and requirements file writing."""
import os
import random
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and (if installed) TensorFlow / PyTorch."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def write_requirements(path, packages) -> pd.DataFrame:
    """Write pinned requirements (pkg==version) to `path`.

    Returns a DataFrame with columns ['package', 'version'].
    Packages that are not installed are skipped in the file and shown as 'not installed' in the table.
    """
    rows, lines = [], []
    for pkg in packages:
        try:
            ver = metadata.version(pkg)
            lines.append(f"{pkg}=={ver}")
        except metadata.PackageNotFoundError:
            ver = "not installed"
        rows.append({"package": pkg, "version": ver})

    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return pd.DataFrame(rows)
