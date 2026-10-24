"""HTTP endpoints. Thin: parse params, get a snapshot, call a service."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

import hmac

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse

from api.config import (ADMIN_TOKEN, DASHBOARD_FILE, DATA_FILE_PATH, DAYS_DIR, LEADIN_HOURS, METRICS_FILE_PATH,
                        PRODUCTION, shown)
from api.daystore import day_store
from api.schemas import (DaysResponse, FlareHistoryResponse, ForecastResponse, HealthResponse,
                         NowcastResponse, StatusResponse, TimeseriesResponse)
from api.services.diagnostics import run_diagnostics
from api.services.evaluation import compute_evaluation
from api.services.forecast import compute_forecast
from api.services.health import compute_health, resolve_mode
from api.services.history import compute_history
from api.services.nowcast import compute_nowcast
from api.services.timeseries import compute_timeseries
from api.snapshot import get_snapshot
from api.state import runtime
from api.timeutil import py_dt

router = APIRouter()


def require_admin(request: Request) -> None:
    """In production, /diagnostics (which lists server paths and configuration) needs the admin token."""
    if not PRODUCTION:
        return
    sent = request.headers.get("x-admin-token", "")
    if not ADMIN_TOKEN or not hmac.compare_digest(sent, ADMIN_TOKEN):
        raise HTTPException(404, "Diagnostics are disabled on this server.")

AsOf = Query(None, description="ISO timestamp. Replays the system using only data up to this time.")
Day = Query(None, description="UTC date, YYYY-MM-DD. Selects a day from the archive instead of the live file.")


@router.get("/", include_in_schema=False)
def dashboard():
    """Serves the dashboard from the API itself, so it is same-origin (no CORS, no second server)."""
    if not DASHBOARD_FILE.exists():
        return JSONResponse(status_code=404, content={
            "detail": f"Dashboard file not found ({shown(DASHBOARD_FILE)}). Start uvicorn from the repo root "
                      "or set KNOXIS_DASHBOARD_FILE."})
    return FileResponse(DASHBOARD_FILE, media_type="text/html")


@router.get("/days", response_model=DaysResponse)
def days():
    """Days available in the archive folder (UTC dates), and whether a live file exists."""
    return DaysResponse(days=day_store.scan(), live_available=DATA_FILE_PATH.exists(),
                        days_dir=shown(DAYS_DIR), leadin_hours=LEADIN_HOURS)


@router.get("/diagnostics", dependencies=[Depends(require_admin)])
def diagnostics():
    """Checklist of what is configured and what is missing, with a hint for each failure."""
    return run_diagnostics()


@router.get("/evaluation")
def evaluation(day: Optional[str] = Day):
    """Nowcast detection statistics, hit rate / false alarms / latency vs a truth catalog (if provided),
    and the offline batch metrics (if provided)."""
    return compute_evaluation(get_snapshot(None, day))


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", production=PRODUCTION, model_loaded=runtime.model is not None,
                          data_file_accessible=DATA_FILE_PATH.exists(), checked_at=datetime.now(timezone.utc))


@router.get("/status", response_model=StatusResponse)
def status(as_of: Optional[str] = AsOf, day: Optional[str] = Day):
    snap = get_snapshot(as_of, day)
    last = py_dt(snap.df.index[-1]) if len(snap.df) else None
    now = datetime.now(timezone.utc)
    return StatusResponse(
        mode=resolve_mode(snap, as_of, day), server_time=now,
        day_info=day_store.info(day) if day else None, data_time=last,
        data_age_s=(now - last).total_seconds() if last else None,
        file_age_s=(now - snap.file_updated).total_seconds(),
        span_start=snap.span_start, span_end=snap.span_end,
        nowcast=compute_nowcast(snap), forecast=compute_forecast(snap), data_health=compute_health(snap))


@router.get("/nowcast", response_model=NowcastResponse)
def nowcast(as_of: Optional[str] = AsOf, day: Optional[str] = Day):
    return compute_nowcast(get_snapshot(as_of, day))


@router.get("/forecast", response_model=ForecastResponse)
def forecast(as_of: Optional[str] = AsOf, day: Optional[str] = Day):
    return compute_forecast(get_snapshot(as_of, day))


@router.get("/flare-history", response_model=FlareHistoryResponse)
def flare_history(as_of: Optional[str] = AsOf, day: Optional[str] = Day, max_events: int = Query(10, ge=1, le=100)):
    return compute_history(get_snapshot(as_of, day), max_events)


@router.get("/timeseries", response_model=TimeseriesResponse)
def timeseries(as_of: Optional[str] = AsOf, day: Optional[str] = Day, start: Optional[str] = None,
               end: Optional[str] = None, max_points: int = Query(1500, ge=50, le=5000)):
    return compute_timeseries(get_snapshot(as_of, day), start, end, max_points)


@router.get("/metrics")
def metrics():
    """Evaluation results written by an offline batch job (persistence baseline, bootstrap CIs,
    hit/false-alarm rates, latency, calibration residuals). Schema is whatever that job writes."""
    if not METRICS_FILE_PATH.exists():
        return {"available": False, "detail": f"No metrics file ({shown(METRICS_FILE_PATH)})."}
    try:
        return {"available": True, "metrics": json.loads(METRICS_FILE_PATH.read_text())}
    except Exception as exc:
        raise HTTPException(503, "Metrics file unreadable.") from exc
