"""Data loading and the snapshot cache.

A Snapshot is everything derived from the data file at one version and one `as_of`:
the (possibly truncated) frame, the nowcast result and the forecast features.
Every endpoint reads from a snapshot, so a dashboard refresh costs one engine run.
"""
from __future__ import annotations

import logging
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional

import pandas as pd
from fastapi import HTTPException

from api.config import DATA_FILE_PATH
from api.state import runtime
from api.timeutil import align, py_dt
from forecast.pipeline.compute_features import compute_day_features

log = logging.getLogger("knoxis.api")


@dataclass
class Snapshot:
    df: pd.DataFrame
    nowcast: Optional[dict]
    features: Optional[pd.DataFrame]
    feature_error: Optional[str]
    span_start: Optional[datetime]
    span_end: Optional[datetime]
    file_updated: datetime


def stat_data_file() -> os.stat_result:
    try:
        return DATA_FILE_PATH.stat()
    except FileNotFoundError:
        raise HTTPException(503, f"Data file not found at {DATA_FILE_PATH}. No live data source configured.")


def load_current_data() -> pd.DataFrame:
    try:
        df = pd.read_parquet(DATA_FILE_PATH)
    except FileNotFoundError:
        raise HTTPException(503, f"Data file not found at {DATA_FILE_PATH}.")
    except Exception as exc:
        raise HTTPException(503, f"Data file unreadable: {exc}") from exc
    if df.index.name != "timestamp" and "timestamp" in df.columns:
        df = df.set_index("timestamp")
    return df.sort_index()


class LRUCache:
    """Builds under the lock: concurrent identical requests compute once, and the shared
    NowcastEngine is never run from two threads at the same time."""

    def __init__(self, size: int = 24):
        self._lock = threading.Lock()
        self._items: OrderedDict = OrderedDict()
        self._size = size

    def get(self, key, build: Callable[[], Snapshot]) -> Snapshot:
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                return self._items[key]
            value = build()
            self._items[key] = value
            while len(self._items) > self._size:
                self._items.popitem(last=False)
            return value


_cache = LRUCache()


def build_snapshot(as_of: Optional[pd.Timestamp], st: os.stat_result) -> Snapshot:
    df = load_current_data()
    span_start = py_dt(df.index[0]) if len(df) else None
    span_end = py_dt(df.index[-1]) if len(df) else None
    updated = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc)

    if as_of is not None:
        # Causality: the engine and the feature code never see rows after as_of.
        df = df.loc[: align(as_of, df.index)]
    if df.empty:
        return Snapshot(df, None, None, None, span_start, span_end, updated)

    nowcast = runtime.engine.run(df)

    features, ferr = None, None
    if runtime.model is not None:
        try:
            features = compute_day_features(df)
        except Exception as exc:  # keep the nowcast alive if feature code fails
            log.exception("compute_day_features failed")
            ferr = f"{type(exc).__name__}: {exc}"
    return Snapshot(df, nowcast, features, ferr, span_start, span_end, updated)


def get_snapshot(as_of: Optional[str]) -> Snapshot:
    st = stat_data_file()
    parsed = None
    if as_of:
        try:
            parsed = pd.Timestamp(as_of)
        except (ValueError, TypeError):
            raise HTTPException(422, f"as_of is not a valid timestamp: {as_of!r}")
    key = (st.st_mtime_ns, st.st_size, parsed.value if parsed is not None else None)
    return _cache.get(key, lambda: build_snapshot(parsed, st))
