"""Flare catalog -> API rows."""
from __future__ import annotations

import pandas as pd

from api.config import SUSTAIN_MINUTES
from api.schemas import FlareHistoryEntry, FlareHistoryResponse
from api.snapshot import Snapshot
from api.timeutil import fnum, py_dt, utc


def compute_history(snap: Snapshot, max_events: int) -> FlareHistoryResponse:
    catalog = snap.nowcast.get("flare_catalog") if snap.nowcast else None
    if catalog is None or len(catalog) == 0:
        return FlareHistoryResponse(events=[], note="No confirmed flare events in the current data window.")
    recent = catalog.sort_values("detection_time", ascending=False).head(max_events)
    events = []
    for _, r in recent.iterrows():
        det = utc(r["detection_time"])
        events.append(FlareHistoryEntry(
            onset_time=(det - pd.Timedelta(minutes=SUSTAIN_MINUTES)).to_pydatetime(),
            detection_time=det.to_pydatetime(),
            peak_time=py_dt(r.get("peak_time")),
            peak_flux=fnum(r.get("peak_counts")),
            goes_class=None if pd.isna(r.get("goes_class")) else str(r["goes_class"]),
            hel1os_corroboration=None if pd.isna(r.get("hel1os_corroboration"))
            else str(r["hel1os_corroboration"]),
        ))
    return FlareHistoryResponse(events=events)
