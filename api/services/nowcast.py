"""Nowcast status for the latest sample in a snapshot."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from api.config import GOES_FACTOR, THRESHOLD_RATIO
from api.schemas import NowcastResponse
from api.snapshot import Snapshot
from api.timeutil import fnum, py_dt


def goes_class_from_flux(w: Optional[float]) -> Optional[str]:
    if w is None or w <= 0:
        return None
    for letter, lo in (("X", 1e-4), ("M", 1e-5), ("C", 1e-6), ("B", 1e-7)):
        if w >= lo:
            return f"{letter}{w / lo:.1f}"
    return f"A{w / 1e-8:.1f}"


def trailing_crossing_seconds(flux: np.ndarray, baseline: np.ndarray, valid: np.ndarray, index) -> Optional[float]:
    """Length of the threshold crossing that is still in progress at the last sample."""
    cross = valid & (flux > THRESHOLD_RATIO * baseline)  # NaN compares False
    if cross.size == 0 or not cross[-1]:
        return None
    gaps = np.flatnonzero(~cross)
    start = int(gaps[-1]) + 1 if gaps.size else 0
    return float((index[-1] - index[start]).total_seconds())


def compute_nowcast(snap: Snapshot) -> NowcastResponse:
    if snap.nowcast is None:
        return NowcastResponse(note="No data at or before the requested time.")

    df, res = snap.df, snap.nowcast
    alert = res["alert_level"]
    base = res["baseline"]
    if len(alert) == 0:
        return NowcastResponse(note="No data available to compute nowcast status.")

    ts = alert.index[-1]
    level = str(alert.iloc[-1])
    baseline = fnum(base.iloc[-1])

    flux_all = df["solexs_counts"].to_numpy(dtype=float)
    valid_all = (df["solexs_is_valid"].to_numpy(dtype=bool)
                 if "solexs_is_valid" in df.columns else np.ones(len(df), dtype=bool))
    flux_valid = bool(valid_all[-1])
    flux = float(flux_all[-1]) if flux_valid and not np.isnan(flux_all[-1]) else None

    ratio = flux / baseline if (flux is not None and baseline) else None
    pending = trailing_crossing_seconds(flux_all, base.reindex(df.index).to_numpy(dtype=float),
                                        valid_all, df.index)

    goes_class = None
    catalog = res.get("flare_catalog")
    if level == "ALERT" and catalog is not None and len(catalog) > 0:
        gc = catalog.sort_values("detection_time").iloc[-1]["goes_class"]
        goes_class = None if pd.isna(gc) else str(gc)

    return NowcastResponse(
        as_of=py_dt(ts), alert_level=level, current_flux=flux, flux_valid=flux_valid,
        baseline=baseline, ratio=ratio, pending_s=pending,
        goes_equiv_class=goes_class_from_flux(flux * GOES_FACTOR) if flux is not None else None,
        goes_class=goes_class,
        note=None if flux_valid else "Latest sample is outside a good-time interval; flux unavailable.",
    )
