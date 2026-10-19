"""Multi-day archive without a database.

A directory of parquet files, indexed by timestamp only. Files may be one per day or one big
multi-day file, may overlap (for example a day file that carries the previous day as lead-in)
and need no naming convention: days are UTC dates discovered from the timestamps.

Parquet is the right store for 1 Hz samples (a day is a few MB, read in well under a second).
A database would add a server to run without making these queries faster.
"""
from __future__ import annotations

import logging
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import HTTPException

from api.config import DAYS_DIR, LEADIN_HOURS, MIN_DAY_ROWS
from api.schemas import DayInfo
from api.timeutil import align, epoch_s

log = logging.getLogger("knoxis.api")

DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
DAY_S = 86400.0


def _read_index_seconds(path: Path) -> np.ndarray:
    """Epoch seconds of every row, reading only the index (no data columns)."""
    idx = None
    try:
        idx = pd.read_parquet(path, columns=[]).index
    except Exception:
        pass
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.DatetimeIndex(pd.read_parquet(path, columns=["timestamp"])["timestamp"])
    return epoch_s(idx)


def _load_frame(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if df.index.name != "timestamp" and "timestamp" in df.columns:
        df = df.set_index("timestamp")
    return df.sort_index()


def _utc(sec: float) -> datetime:
    return datetime.fromtimestamp(sec, tz=timezone.utc)


class DayStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._files: dict[Path, tuple[tuple, np.ndarray]] = {}
        self._key: Optional[tuple] = None
        self._days: list[DayInfo] = []

    def _paths(self) -> list[Path]:
        return sorted(DAYS_DIR.rglob("*.parquet")) if DAYS_DIR.exists() else []

    def scan(self) -> list[DayInfo]:
        """Cheap when nothing changed (one stat per file); re-reads only changed files."""
        with self._lock:
            paths, sigs = [], []
            for p in self._paths():
                try:
                    st = p.stat()
                except FileNotFoundError:
                    continue
                paths.append(p)
                sigs.append((str(p), st.st_mtime_ns, st.st_size))
            key = tuple(sigs)
            if key == self._key:
                return self._days

            arrays = []
            for p, sig in zip(paths, sigs):
                cached = self._files.get(p)
                if cached is None or cached[0] != sig:
                    try:
                        self._files[p] = (sig, _read_index_seconds(p))
                    except Exception:
                        log.exception("could not index %s; skipping it", p)
                        self._files.pop(p, None)
                        continue
                arrays.append(self._files[p][1])
            for stale in set(self._files) - set(paths):
                del self._files[stale]

            self._days = self._build(np.unique(np.concatenate(arrays)) if arrays else np.array([]))
            self._key = key
            return self._days

    @staticmethod
    def _build(sec: np.ndarray) -> list[DayInfo]:
        if sec.size == 0:
            return []
        lead = LEADIN_HOURS * 3600
        bins = np.floor(sec / DAY_S).astype(np.int64)
        uniq, first, counts = np.unique(bins, return_index=True, return_counts=True)
        out = []
        for d, i0, n in zip(uniq, first, counts):
            if n < MIN_DAY_ROWS:
                continue
            day_start = float(d) * DAY_S
            lo, hi = np.searchsorted(sec, day_start - lead), np.searchsorted(sec, day_start)
            out.append(DayInfo(
                day=_utc(day_start).strftime("%Y-%m-%d"), start=_utc(sec[i0]), end=_utc(sec[i0 + n - 1]),
                rows=int(n), hours=round(float(sec[i0 + n - 1] - sec[i0]) / 3600, 2),
                leadin_ok=bool((hi - lo) >= 0.5 * lead)))
        return out

    def signature(self) -> tuple:
        self.scan()
        return self._key or ()

    def info(self, day: str) -> DayInfo:
        if not DAY_RE.match(day):
            raise HTTPException(422, f"day must look like 2026-06-21, got {day!r}")
        for d in self.scan():
            if d.day == day:
                return d
        have = ", ".join(d.day for d in self._days[:10]) or "none"
        raise HTTPException(404, f"No data for {day} in {DAYS_DIR}. Days found: {have}.")

    def load_day(self, day: str) -> tuple[pd.DataFrame, pd.Timestamp]:
        """The selected UTC day plus LEADIN_HOURS before it, de-duplicated across overlapping files.
        Returns the frame and the day's start expressed in the frame's own timezone convention."""
        self.info(day)                                    # validates and 404s
        d0 = pd.Timestamp(day, tz="UTC")
        d1, lo = d0 + pd.Timedelta(days=1), d0 - pd.Timedelta(hours=LEADIN_HOURS)
        frames = []
        with self._lock:
            hits = [p for p, (_, arr) in self._files.items()
                    if arr.size and arr.max() >= lo.timestamp() and arr.min() < d1.timestamp()]
        for p in sorted(hits):
            frames.append(_load_frame(p))
        if not frames:
            raise HTTPException(404, f"No files cover {day}.")
        df = pd.concat(frames)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df = df.loc[align(lo, df.index):]
        df = df[df.index < align(d1, df.index)]
        return df, align(d0, df.index)


day_store = DayStore()
