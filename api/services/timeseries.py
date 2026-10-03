"""Windowed, downsampled series for the dashboard chart."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from api.config import LEVEL_CODE, SUSTAIN_MINUTES
from api.schemas import TimeseriesEvent, TimeseriesResponse
from api.snapshot import Snapshot
from api.timeutil import align, clean, epoch_s


def _empty() -> TimeseriesResponse:
    return TimeseriesResponse(t=[], flux=[], baseline=[], level=[], hel1os=[], events=[])


def compute_timeseries(snap: Snapshot, start: Optional[str], end: Optional[str], max_points: int) -> TimeseriesResponse:
    df = snap.df
    if df.empty or snap.nowcast is None:
        return _empty()

    idx = df.index
    i0 = idx.searchsorted(align(start, idx)) if start else 0
    i1 = idx.searchsorted(align(end, idx), side="right") if end else len(idx)
    if i1 <= i0:
        return _empty()

    res = snap.nowcast
    sl = slice(i0, i1)
    t = epoch_s(idx[sl])

    flux = df["solexs_counts"].to_numpy(dtype=float)[sl].copy()
    if "solexs_is_valid" in df.columns:
        flux[~df["solexs_is_valid"].to_numpy(dtype=bool)[sl]] = np.nan
    base = res["baseline"].reindex(idx).to_numpy(dtype=float)[sl]
    lvl = (res["alert_level"].reindex(idx).astype(str).map(LEVEL_CODE).fillna(0).to_numpy(dtype=int))[sl]
    hel = np.full(len(t), np.nan)
    if "hel1os_ctr" in df.columns:
        hel = df["hel1os_ctr"].to_numpy(dtype=float)[sl].copy()
        if "hel1os_is_valid" in df.columns:
            hel[~df["hel1os_is_valid"].to_numpy(dtype=bool)[sl]] = np.nan

    # Max-per-bucket keeps flare peaks visible when zoomed out (stride decimation would drop them).
    n = len(t)
    bucket = (np.arange(n) * max_points // n) if n > max_points else np.arange(n)
    g = pd.DataFrame({"t": t, "flux": flux, "base": base, "lvl": lvl, "hel": hel, "b": bucket}).groupby("b", sort=True)
    out = pd.DataFrame({"t": g["t"].first(), "flux": g["flux"].max(), "baseline": g["base"].mean(),
                        "level": g["lvl"].max(), "hel1os": g["hel"].mean()})

    events: list[TimeseriesEvent] = []
    cat = res.get("flare_catalog")
    if cat is not None and len(cat) > 0:
        lo, hi = t[0], t[-1]
        for _, r in cat.iterrows():
            det = float(epoch_s(pd.DatetimeIndex([r["detection_time"]]))[0])
            if lo <= det <= hi:
                pk = None if pd.isna(r.get("peak_time")) else float(epoch_s(pd.DatetimeIndex([r["peak_time"]]))[0])
                events.append(TimeseriesEvent(
                    onset=det - SUSTAIN_MINUTES * 60, detection=det, peak=pk,
                    goes_class=None if pd.isna(r.get("goes_class")) else str(r["goes_class"])))

    return TimeseriesResponse(
        t=out["t"].tolist(), flux=clean(out["flux"]), baseline=clean(out["baseline"]),
        level=out["level"].astype(int).tolist(), hel1os=clean(out["hel1os"]), events=events)
