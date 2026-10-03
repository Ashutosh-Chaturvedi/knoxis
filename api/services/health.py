"""Data-health summary and live/replay mode resolution."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from api.config import FEED_ACTIVE_S, MODE_OVERRIDE, STALE_AFTER_S
from api.schemas import DataHealth
from api.snapshot import Snapshot
from api.timeutil import py_dt


def compute_health(snap: Snapshot) -> DataHealth:
    df = snap.df
    if df.empty:
        return DataHealth(file_updated=snap.file_updated)
    sx = float(df["solexs_is_valid"].mean()) if "solexs_is_valid" in df.columns else None
    zero = None
    if "hel1os_ctr" in df.columns:
        h = df["hel1os_ctr"]
        if "hel1os_is_valid" in df.columns:
            h = h[df["hel1os_is_valid"].astype(bool)]
        zero = float((h == 0).mean()) if len(h) else None
    return DataHealth(rows=int(len(df)), solexs_valid_fraction=sx, hel1os_zero_fraction=zero,
                      file_updated=snap.file_updated)


def resolve_mode(snap: Snapshot, as_of: Optional[str]) -> str:
    if as_of:
        return "replay"
    if MODE_OVERRIDE in ("live", "simulated", "replay"):
        return MODE_OVERRIDE
    if snap.df.empty:
        return "replay"
    now = datetime.now(timezone.utc)
    if (now - py_dt(snap.df.index[-1])).total_seconds() <= STALE_AFTER_S:
        return "live"                       # samples are recent: a real feed
    if (now - snap.file_updated).total_seconds() <= FEED_ACTIVE_S:
        return "simulated"                  # file is being rewritten but samples are old
    return "replay"                         # static archive
