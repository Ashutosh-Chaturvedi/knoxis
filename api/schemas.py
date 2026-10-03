"""Pydantic response models."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from api.config import GOES_FACTOR, LEVELS, SUSTAIN_MINUTES, THRESHOLD_RATIO


class NowcastResponse(BaseModel):
    as_of: Optional[datetime] = None
    alert_level: Optional[str] = None
    current_flux: Optional[float] = None       # counts/s; None if latest sample is outside a GTI
    flux_valid: Optional[bool] = None
    baseline: Optional[float] = None
    ratio: Optional[float] = None              # flux / baseline
    threshold_ratio: float = THRESHOLD_RATIO
    pending_s: Optional[float] = None          # seconds the current threshold crossing has lasted
    sustain_s: float = SUSTAIN_MINUTES * 60
    goes_equiv_class: Optional[str] = None     # from the two-flare calibration; indicative only
    goes_class: Optional[str] = None           # catalog class, only while level is ALERT
    note: Optional[str] = None


class ForecastResponse(BaseModel):
    as_of: Optional[datetime] = None
    alert_level: Optional[str] = None
    flare_probability: Optional[float] = None
    base_rate: Optional[float] = None
    thresholds: dict[str, float] = {}
    model_version: Optional[str] = None
    note: Optional[str] = None


class DataHealth(BaseModel):
    rows: int = 0
    solexs_valid_fraction: Optional[float] = None
    hel1os_zero_fraction: Optional[float] = None
    file_updated: Optional[datetime] = None


class StatusResponse(BaseModel):
    mode: str                                   # "live" | "simulated" | "replay"
    server_time: datetime
    data_time: Optional[datetime] = None        # last sample included in this response
    data_age_s: Optional[float] = None          # server time minus newest sample time
    file_age_s: Optional[float] = None          # server time minus last write of the data file
    span_start: Optional[datetime] = None       # full extent of the data file
    span_end: Optional[datetime] = None
    nowcast: NowcastResponse
    forecast: ForecastResponse
    data_health: DataHealth


class FlareHistoryEntry(BaseModel):
    onset_time: Optional[datetime] = None       # detection_time - sustain window (estimate)
    detection_time: datetime
    peak_time: Optional[datetime] = None
    peak_flux: Optional[float] = None
    goes_class: Optional[str] = None
    hel1os_corroboration: Optional[str] = None


class FlareHistoryResponse(BaseModel):
    events: list[FlareHistoryEntry]
    note: Optional[str] = None


class TimeseriesEvent(BaseModel):
    onset: float
    detection: float
    peak: Optional[float] = None
    goes_class: Optional[str] = None


class TimeseriesResponse(BaseModel):
    t: list[float]                              # epoch seconds, UTC
    flux: list[Optional[float]]                 # max per bucket, None outside GTI
    baseline: list[Optional[float]]
    level: list[int]                            # index into level_names, max per bucket
    hel1os: list[Optional[float]]
    events: list[TimeseriesEvent]
    level_names: list[str] = LEVELS
    threshold_ratio: float = THRESHOLD_RATIO
    sustain_s: float = SUSTAIN_MINUTES * 60
    goes_factor: float = GOES_FACTOR


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    data_file_accessible: bool
    checked_at: datetime
