"""Multi-day archive without a database.

A directory of parquet files. Files may be one per day or one big multi-day file, may overlap (for
example a day file that carries the previous day as lead-in) and need no naming convention: days are
UTC dates discovered from the timestamps.

Indexing is cheap by design (this is what kept the day list slow and memory hungry before):
- A file's time range and row count come from the parquet footer statistics. No data is read.
  Only a file that spans several UTC dates has its timestamp column read, to count rows per date.
- The result is kept per file in a small JSON index on disk, keyed by the file's mtime and size.
  A restart, or a new request, re-reads only files that are new or changed.
- No timestamp arrays are kept in memory; the whole index is a few hundred bytes per file.

Parquet is the right store for 1 Hz samples (a day is a few MB). A database would add a server to run
without making these lookups faster.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from fastapi import HTTPException

from api.config import DAYS_DIR, INDEX_FILE_PATH, LEADIN_HOURS, MIN_DAY_ROWS, SCAN_TTL_S, shown
from api.schemas import DayInfo
from api.timeutil import align, epoch_s

log = logging.getLogger("knoxis.api")

DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
DAY_S = 86400.0
INDEX_VERSION = 2


def _date_str(sec: float) -> str:
    return datetime.fromtimestamp(sec, tz=timezone.utc).strftime("%Y-%m-%d")


def _utc(sec: float) -> datetime:
    return datetime.fromtimestamp(sec, tz=timezone.utc)


def _stat_seconds(v) -> Optional[float]:
    """Parquet statistics value -> epoch seconds (naive values are taken as UTC), or None if unusable."""
    if isinstance(v, (int, np.integer, float)):
        return None                                   # unit unknown: fall back to reading the column
    ts = pd.Timestamp(v)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.timestamp()


def _footer_range(path: Path) -> Optional[tuple[float, float, int]]:
    """(first, last, rows) from the footer alone, or None if the file has no usable statistics."""
    md = pq.ParquetFile(path).metadata
    names = [md.schema.column(i).name for i in range(md.num_columns)]
    if "timestamp" not in names:
        return None
    ci = names.index("timestamp")
    lows, highs = [], []
    for g in range(md.num_row_groups):
        st = md.row_group(g).column(ci).statistics
        if st is None or not st.has_min_max:
            return None
        lows.append(_stat_seconds(st.min))
        highs.append(_stat_seconds(st.max))
    if not lows or None in lows or None in highs:
        return None
    return min(lows), max(highs), int(md.num_rows)


def _read_index_seconds(path: Path) -> np.ndarray:
    """Epoch seconds of every row; reads only the index/timestamp column."""
    idx = None
    try:
        idx = pd.read_parquet(path, columns=[]).index
    except Exception:
        pass
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.DatetimeIndex(pd.read_parquet(path, columns=["timestamp"])["timestamp"])
    return np.sort(epoch_s(idx))


def _index_file(path: Path) -> dict:
    """Per-file index entry: overall range plus, per UTC date, [rows, first, last]."""
    fr = _footer_range(path)
    if fr is not None and math.floor(fr[0] / DAY_S) == math.floor(fr[1] / DAY_S):
        lo, hi, rows = fr
        return {"tmin": lo, "tmax": hi, "rows": rows, "dates": {_date_str(lo): [rows, lo, hi]}}
    sec = _read_index_seconds(path)                   # file spans several dates (or has no statistics)
    if sec.size == 0:
        return {"tmin": 0.0, "tmax": 0.0, "rows": 0, "dates": {}}
    bins = np.floor(sec / DAY_S).astype(np.int64)
    uniq, first, counts = np.unique(bins, return_index=True, return_counts=True)
    dates = {_date_str(float(d) * DAY_S): [int(n), float(sec[i]), float(sec[i + n - 1])]
             for d, i, n in zip(uniq, first, counts)}
    return {"tmin": float(sec[0]), "tmax": float(sec[-1]), "rows": int(sec.size), "dates": dates}


def _load_frame(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if df.index.name != "timestamp" and "timestamp" in df.columns:
        df = df.set_index("timestamp")
    return df.sort_index()


class DayStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._files: dict[str, dict] = {}             # relative path -> {sig, tmin, tmax, rows, dates}
        self._loaded_disk = False
        self._key: Optional[tuple] = None
        self._days: list[DayInfo] = []
        self._last_scan = -1e18

    # ---- persistence -------------------------------------------------------------------------
    def _load_disk_index(self) -> None:
        self._loaded_disk = True
        try:
            data = json.loads(INDEX_FILE_PATH.read_text())
            if data.get("version") == INDEX_VERSION:
                self._files = data["files"]
        except FileNotFoundError:
            pass
        except Exception:
            log.warning("ignoring unreadable day index %s", INDEX_FILE_PATH.name)

    def _save_disk_index(self) -> None:
        try:
            tmp = INDEX_FILE_PATH.with_name(INDEX_FILE_PATH.name + ".tmp")
            tmp.write_text(json.dumps({"version": INDEX_VERSION, "files": self._files}))
            os.replace(tmp, INDEX_FILE_PATH)
        except OSError as exc:                        # read-only folder: keep working, index in memory only
            log.warning("could not save the day index (%s); it will be rebuilt on restart", exc.__class__.__name__)

    # ---- scanning ----------------------------------------------------------------------------
    def _paths(self) -> list[Path]:
        return sorted(DAYS_DIR.rglob("*.parquet")) if DAYS_DIR.exists() else []

    def scan(self, force: bool = False) -> list[DayInfo]:
        """One stat per file when nothing changed, and at most once per SCAN_TTL_S."""
        with self._lock:
            if not force and self._key is not None and time.monotonic() - self._last_scan < SCAN_TTL_S:
                return self._days
            if not self._loaded_disk:
                self._load_disk_index()

            found: dict[str, tuple[Path, list]] = {}
            for p in self._paths():
                try:
                    st = p.stat()
                except FileNotFoundError:
                    continue
                found[p.relative_to(DAYS_DIR).as_posix()] = (p, [st.st_mtime_ns, st.st_size])
            key = tuple(sorted((rel, *sig) for rel, (_, sig) in found.items()))

            self._last_scan = time.monotonic()
            if key == self._key:
                return self._days

            todo = [rel for rel, (_, sig) in found.items()
                    if rel not in self._files or self._files[rel].get("sig") != sig]
            changed = bool(todo) or set(self._files) != set(found)
            if todo:
                t0 = time.perf_counter()

                def work(rel: str):
                    try:
                        return rel, _index_file(found[rel][0])
                    except Exception:
                        log.exception("could not index %s; skipping it", rel)
                        return rel, None
                with ThreadPoolExecutor(max_workers=min(8, len(todo))) as pool:
                    for rel, entry in pool.map(work, todo):
                        if entry is None:
                            self._files.pop(rel, None)
                        else:
                            entry["sig"] = found[rel][1]
                            self._files[rel] = entry
                log.info("indexed %d file(s) in %.2fs", len(todo), time.perf_counter() - t0)
            for gone in set(self._files) - set(found):
                del self._files[gone]
            if changed:
                self._save_disk_index()

            self._days = self._build(self._files)
            self._key = key
            return self._days

    @staticmethod
    def _build(files: dict[str, dict]) -> list[DayInfo]:
        agg: dict[str, list] = {}                     # date -> [rows, first, last]
        for e in files.values():
            for d, (n, a, b) in e["dates"].items():
                cur = agg.get(d)
                if cur is None:
                    agg[d] = [n, a, b]
                else:                                 # overlapping files: rows are not added twice
                    cur[0], cur[1], cur[2] = max(cur[0], n), min(cur[1], a), max(cur[2], b)
        lead = LEADIN_HOURS * 3600
        out = []
        for d in sorted(agg):
            n, first, last = agg[d]
            if n < MIN_DAY_ROWS:
                continue
            day_start = datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()
            coverage = 0.0                            # rows in [day_start - lead, day_start), by date overlap
            for k in range(1, int(math.ceil(lead / DAY_S)) + 2):
                pd_start = day_start - k * DAY_S
                prev = agg.get(_date_str(pd_start))
                if prev is None:
                    continue
                overlap = max(0.0, min(day_start, pd_start + DAY_S) - max(day_start - lead, pd_start))
                coverage += prev[0] * overlap / DAY_S
            out.append(DayInfo(day=d, start=_utc(first), end=_utc(last), rows=int(n),
                               hours=round((last - first) / 3600, 2), leadin_ok=bool(coverage >= 0.5 * lead)))
        return out

    # ---- queries -----------------------------------------------------------------------------
    def info(self, day: str) -> DayInfo:
        if not DAY_RE.match(day):
            raise HTTPException(422, f"day must look like 2026-06-21, got {day!r}")
        days = self.scan()
        for d in days:
            if d.day == day:
                return d
        have = ", ".join(d.day for d in days[-10:]) or "none"
        raise HTTPException(404, f"No data for {day} in {shown(DAYS_DIR)}. Most recent days found: {have}.")

    def _files_for(self, day: str) -> list[str]:
        with self._lock:
            return self._files_for_locked(day)

    def signature_for(self, day: str) -> tuple:
        """Signature of only the files this day depends on, so unrelated changes don't invalidate its cache."""
        self.info(day)                                # validates and 404s
        with self._lock:
            return tuple((rel, *self._files[rel]["sig"]) for rel in self._files_for_locked(day))

    def _files_for_locked(self, day: str) -> list[str]:
        d0 = pd.Timestamp(day, tz="UTC")
        lo, hi = (d0 - pd.Timedelta(hours=LEADIN_HOURS)).timestamp(), (d0 + pd.Timedelta(days=1)).timestamp()
        return sorted(rel for rel, e in self._files.items() if e["tmax"] >= lo and e["tmin"] < hi)

    def load_day(self, day: str) -> tuple[pd.DataFrame, pd.Timestamp]:
        """The selected UTC day plus LEADIN_HOURS before it, de-duplicated across overlapping files.
        Returns the frame and the day's start expressed in the frame's own timezone convention."""
        self.info(day)
        d0 = pd.Timestamp(day, tz="UTC")
        d1, lo = d0 + pd.Timedelta(days=1), d0 - pd.Timedelta(hours=LEADIN_HOURS)
        rels = self._files_for(day)
        if not rels:
            raise HTTPException(404, f"No files cover {day}.")
        df = pd.concat([_load_frame(DAYS_DIR / rel) for rel in rels])
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df = df.loc[align(lo, df.index):]
        df = df[df.index < align(d1, df.index)]
        return df, align(d0, df.index)


day_store = DayStore()
