"""Data loading, snapshot cache, and the background keeper for the latest snapshot.

A Snapshot is everything derived from the data file at one version and one `as_of`:
the (possibly truncated) frame, the nowcast result and the forecast features.

Three paths, because engine.run + compute_day_features can take seconds on a 24 h file:
- as_of is None ("latest"): a background thread rebuilds the snapshot whenever the file changes.
  Requests read the last completed snapshot and never wait on an engine run. If the file is
  rewritten faster than a build finishes (the simulator ticks every few seconds), the served
  snapshot is at most one build behind instead of every request triggering its own rebuild.
- as_of set, slice mode: the latest full snapshot is cut at as_of (instant). Only used after a
  startup self-check (verify_causality) has shown that truncating the input gives the same answer
  as cutting the full run. In-progress event fields are hidden rather than leaked.
- as_of set, exact mode: the engine is re-run on truncated data, cached in a small LRU. Always
  correct, slow on large files.

All engine/feature work is serialised by _build_lock, so the shared NowcastEngine is never run
from two threads at once.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional

import numpy as np
import pandas as pd
from fastapi import HTTPException

from api.config import DATA_FILE_PATH, FEATURE_COLUMNS, PENDING_AFTER_PEAK_MIN, REPLAY_MODE
from api.state import runtime
from api.timeutil import align, py_dt
from forecast.pipeline.compute_features import compute_day_features

log = logging.getLogger("knoxis.api")

_build_lock = threading.Lock()


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


def build_snapshot(as_of: Optional[pd.Timestamp], st: os.stat_result) -> Snapshot:
    """Caller must hold _build_lock."""
    df = load_current_data()
    span_start = py_dt(df.index[0]) if len(df) else None
    span_end = py_dt(df.index[-1]) if len(df) else None
    updated = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc)

    if as_of is not None:
        # Causality: the engine and the feature code never see rows after as_of.
        df = df.loc[: align(as_of, df.index)]
    if df.empty:
        return Snapshot(df, None, None, None, span_start, span_end, updated)

    t0 = time.perf_counter()
    nowcast = runtime.engine.run(df)
    t1 = time.perf_counter()

    features, ferr = None, None
    if runtime.model is not None:
        try:
            features = compute_day_features(df)
        except Exception as exc:  # keep the nowcast alive if feature code fails
            log.exception("compute_day_features failed")
            ferr = f"{type(exc).__name__}: {exc}"
    t2 = time.perf_counter()
    log.info("snapshot rows=%d as_of=%s engine.run=%.2fs features=%.2fs",
             len(df), as_of, t1 - t0, t2 - t1)
    return Snapshot(df, nowcast, features, ferr, span_start, span_end, updated)


def _mask_catalog(cat: pd.DataFrame, cut: pd.Timestamp) -> pd.DataFrame:
    """Catalog as it could have looked at `cut`, derived from a full run. Events detected after the
    cut are dropped. Fields that depend on later data are hidden: peak until it has happened,
    HEL1OS corroboration until PENDING_AFTER_PEAK_MIN after the peak."""
    if cat is None or len(cat) == 0:
        return cat
    cat = cat[cat["detection_time"] <= cut].copy()
    if len(cat) == 0:
        return cat
    peak_pending = cat["peak_time"] > cut
    corr_pending = cut < cat["peak_time"] + pd.Timedelta(minutes=PENDING_AFTER_PEAK_MIN)
    cat["peak_time"] = cat["peak_time"].where(~peak_pending, pd.NaT)
    cat["peak_counts"] = cat["peak_counts"].where(~peak_pending, np.nan)
    cat["goes_class"] = cat["goes_class"].astype(object).where(~peak_pending, None)
    cat["hel1os_corroboration"] = cat["hel1os_corroboration"].astype(object).where(~corr_pending, None)
    return cat


def slice_snapshot(full: Snapshot, as_of: pd.Timestamp) -> Snapshot:
    if full.df.empty or full.nowcast is None:
        return full
    df = full.df.loc[: align(as_of, full.df.index)]
    if df.empty:
        return Snapshot(df, None, None, None, full.span_start, full.span_end, full.file_updated)
    res = full.nowcast
    nowcast = {**res,
               "baseline": res["baseline"].loc[: df.index[-1]],
               "alert_level": res["alert_level"].loc[: df.index[-1]],
               "flare_catalog": _mask_catalog(res.get("flare_catalog"), df.index[-1])}
    feats = full.features.loc[: df.index[-1]] if full.features is not None else None
    return Snapshot(df, nowcast, feats, full.feature_error, full.span_start, full.span_end, full.file_updated)


def _settled_events(full: Snapshot) -> int:
    cat = full.nowcast.get("flare_catalog") if full.nowcast else None
    if cat is None or len(cat) == 0 or full.df.empty:
        return 0
    horizon = full.df.index[-1] - pd.Timedelta(minutes=PENDING_AFTER_PEAK_MIN)
    return int((cat["peak_time"] <= horizon).sum())


def verify_causality(full: Snapshot) -> tuple[bool, str]:
    """Sample-based check that output at t does not depend on data after t. Not a proof: the
    pytest suite in tests/test_causality.py is the thorough version. Caller holds _build_lock."""
    df, res = full.df, full.nowcast
    n = len(df)
    positions = {n // 2}
    cat = res.get("flare_catalog")
    event = None
    if cat is not None and len(cat) > 0:
        event = cat.sort_values("detection_time").iloc[0]
        for ts, off in ((event["detection_time"], pd.Timedelta(seconds=60)),
                        (event["peak_time"], pd.Timedelta(minutes=PENDING_AFTER_PEAK_MIN))):
            i = int(df.index.searchsorted(ts + off))
            if 0 < i < n:
                positions.add(i)

    problems: list[str] = []
    for i in sorted(positions):
        part_df = df.iloc[: i + 1]
        t = part_df.index[-1]
        part = runtime.engine.run(part_df)
        if part["alert_level"].iloc[-1] != res["alert_level"].iloc[i]:
            problems.append(f"alert level differs at {t}")
        if not np.isclose(part["baseline"].iloc[-1], res["baseline"].iloc[i], equal_nan=True):
            problems.append(f"baseline differs at {t}")
        if runtime.model is not None and full.features is not None and t in full.features.index:
            pf = compute_day_features(part_df)
            a = pf.iloc[-1][FEATURE_COLUMNS].astype(float).to_numpy()
            b = full.features.loc[t, FEATURE_COLUMNS].astype(float).to_numpy()
            if not np.allclose(a, b, equal_nan=True, rtol=1e-9):
                problems.append(f"forecast features differ at {t} (day-wide statistics?)")
        if event is not None and t >= event["peak_time"] + pd.Timedelta(minutes=PENDING_AFTER_PEAK_MIN):
            pc = part["flare_catalog"]
            m = pc[pc["detection_time"] == event["detection_time"]]
            if len(m) != 1 or m.iloc[0]["hel1os_corroboration"] != event["hel1os_corroboration"]:
                problems.append("HEL1OS corroboration changes when later data is removed")
    if problems:
        return False, "; ".join(dict.fromkeys(problems))
    return True, f"verified at {len(positions)} sample point(s)"


class LRUCache:
    """Small cache for as_of snapshots. The lock guards the dict only; builds happen outside it
    (under _build_lock), so lookups are never blocked by a build in progress."""

    def __init__(self, size: int = 24):
        self._lock = threading.Lock()
        self._items: OrderedDict = OrderedDict()
        self._size = size

    def _lookup(self, key) -> Optional[Snapshot]:
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                return self._items[key]
        return None

    def get(self, key, build: Callable[[], Snapshot]) -> Snapshot:
        hit = self._lookup(key)
        if hit is not None:
            return hit
        with _build_lock:
            hit = self._lookup(key)          # built by someone else while we waited
            if hit is not None:
                return hit
            value = build()
        with self._lock:
            self._items[key] = value
            while len(self._items) > self._size:
                self._items.popitem(last=False)
        return value


_cache = LRUCache()


class LatestKeeper:
    """Keeps the newest as_of=None snapshot current in a background thread."""

    def __init__(self, poll_s: float = 1.0):
        self._poll_s = poll_s
        self._lock = threading.Lock()
        self._snap: Optional[Snapshot] = None
        self._key = None
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # observability (read by /diagnostics)
        self.builds = 0
        self.last_build_s: Optional[float] = None
        self.last_error: Optional[str] = None
        self.causal: Optional[bool] = None          # None = not checked yet
        self.causal_detail = "not checked yet"
        self._verified_settled = -1

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="knoxis-latest", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def peek(self) -> Optional[Snapshot]:
        with self._lock:
            return self._snap

    def replay_strategy(self) -> str:
        if REPLAY_MODE == "slice":
            return "slice"
        if REPLAY_MODE == "auto" and self.causal is True:
            return "slice"
        return "exact"

    def _maybe_verify(self, snap: Snapshot) -> None:
        if snap.nowcast is None:
            return
        settled = _settled_events(snap)
        if settled <= self._verified_settled:
            return
        self._verified_settled = settled
        try:
            with _build_lock:
                ok, detail = verify_causality(snap)
        except Exception as exc:
            ok, detail = False, f"self-check failed: {type(exc).__name__}: {exc}"
        self.causal = ok if self.causal is None else (self.causal and ok)
        self.causal_detail = detail
        (log.info if ok else log.warning)("causality self-check: %s", detail)

    def latest(self, wait_s: float = 60.0) -> Optional[Snapshot]:
        if self._snap is None and self._thread is not None:
            self._ready.wait(wait_s)         # first build; set even if it failed
        with self._lock:
            return self._snap

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                st = DATA_FILE_PATH.stat()
            except FileNotFoundError:
                self._stop.wait(self._poll_s)
                continue
            key = (st.st_mtime_ns, st.st_size)
            if key != self._key:
                snap = None
                try:
                    t0 = time.perf_counter()
                    with _build_lock:
                        snap = build_snapshot(None, st)
                    with self._lock:
                        self._snap = snap
                    self.builds += 1
                    self.last_build_s = time.perf_counter() - t0
                    self.last_error = None
                except Exception as exc:
                    log.exception("background snapshot build failed")
                    self.last_error = f"{type(exc).__name__}: {exc}"
                self._key = key              # on failure, wait for the next file change
                self._ready.set()
                if snap is not None:
                    self._maybe_verify(snap)
                continue                     # re-check immediately: the file may have changed during the build
            self._stop.wait(self._poll_s)


latest_keeper = LatestKeeper()


def get_snapshot(as_of: Optional[str]) -> Snapshot:
    st = stat_data_file()                    # 503 if the file is missing
    if not as_of:
        snap = latest_keeper.latest()
        if snap is not None:
            return snap
    parsed = None
    if as_of:
        try:
            parsed = pd.Timestamp(as_of)
        except (ValueError, TypeError):
            raise HTTPException(422, f"as_of is not a valid timestamp: {as_of!r}")
        if latest_keeper.replay_strategy() == "slice":
            full = latest_keeper.peek()
            if full is not None:
                return slice_snapshot(full, parsed)
    key = (st.st_mtime_ns, st.st_size, parsed.value if parsed is not None else None)
    return _cache.get(key, lambda: build_snapshot(parsed, st))
