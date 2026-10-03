"""Time and numeric helpers shared across the API."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd


def utc(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def py_dt(ts) -> Optional[datetime]:
    if ts is None or pd.isna(ts):
        return None
    return utc(ts).to_pydatetime()


def epoch_s(index) -> np.ndarray:
    idx = pd.DatetimeIndex(index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    return ((idx - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(seconds=1)).to_numpy()


def align(ts, index: pd.DatetimeIndex) -> pd.Timestamp:
    """Express ts in the tz convention of `index` so slicing and comparison work."""
    ts = utc(ts)
    return ts.tz_convert(index.tz) if index.tz is not None else ts.tz_localize(None)


def fnum(x) -> Optional[float]:
    return None if x is None or pd.isna(x) else float(x)


def clean(arr) -> list:
    """NaN -> None so the values are JSON-serialisable."""
    return [None if (v is None or v != v) else float(v) for v in arr]
