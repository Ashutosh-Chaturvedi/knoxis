from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from nowcast.nowcast import NowcastEngine
from forecast.pipeline.compute_features import compute_day_features

FEATURE_COLUMNS = [
    "solexs_mean", "solexs_slope", "solexs_std",
    "hel1os_mean", "hel1os_slope",
    "hardness_ratio", "hr_rate_of_change",
    "peak_to_mean_ratio", "hel1os_saturation_fraction",
]


DATA_FILE_PATH = Path(os.environ.get("KNOXIS_DATA_FILE", "data/training/output/latest_data.parquet"))
MODEL_FILE_PATH = Path(os.environ.get("KNOXIS_MODEL_FILE", "data/training/output/knoxis_binary_flare_model.joblib"))
WATCH_THRESHOLD = float(os.environ.get("KNOXIS_WATCH_THRESHOLD", "0.3"))
WARNING_THRESHOLD = float(os.environ.get("KNOXIS_WARNING_THRESHOLD", "0.5"))
ALERT_THRESHOLD = float(os.environ.get("KNOXIS_ALERT_THRESHOLD", "0.7"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    if MODEL_FILE_PATH.exists():
        app.state.forecast_model = joblib.load(MODEL_FILE_PATH)
    else:
        app.state.forecast_model = None
        print(f"WARNING: forecast model not found at {MODEL_FILE_PATH} -- "
              f"/forecast and /status will report this as unavailable.")

    app.state.nowcast_engine = NowcastEngine()
    yield


app = FastAPI(title="Knoxis API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)



class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    data_file_accessible: bool
    checked_at: datetime


class NowcastResponse(BaseModel):
    as_of: Optional[datetime] = None
    alert_level: Optional[str] = None
    current_flux: Optional[float] = None
    baseline: Optional[float] = None
    goes_class: Optional[str] = None
    note: Optional[str] = None


class ForecastResponse(BaseModel):
    as_of: Optional[datetime] = None
    alert_level: Optional[str] = None
    flare_probability: Optional[float] = None
    note: Optional[str] = None


class StatusResponse(BaseModel):
    nowcast: NowcastResponse
    forecast: ForecastResponse



def load_current_data() -> pd.DataFrame:
    """Reads the data file fresh. Raises HTTPException (503) if it's
    missing or unreadable -- a real, honest failure state, not silently
    returning an empty/default response."""
    if not DATA_FILE_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Data file not found at {DATA_FILE_PATH}. No live data source configured.",
        )
    try:
        df = pd.read_parquet(DATA_FILE_PATH)
    except Exception as exc: 
        raise HTTPException(status_code=503, detail=f"Data file unreadable: {exc}") from exc

    if df.index.name != "timestamp" and "timestamp" in df.columns:
        df = df.set_index("timestamp")
    return df.sort_index()


def compute_nowcast_status(df: pd.DataFrame, engine: NowcastEngine) -> NowcastResponse:
    result = engine.run(df)
    alert_series = result["alert_level"]
    baseline_series = result["baseline"]

    if len(alert_series) == 0:
        return NowcastResponse(note="No data available to compute nowcast status.")

    latest_ts = alert_series.index[-1]
    latest_alert = alert_series.iloc[-1]
    latest_baseline = baseline_series.iloc[-1]
    latest_flux = df["solexs_counts"].asof(latest_ts) if "solexs_counts" in df.columns else None

    goes_class = None
    catalog = result.get("flare_catalog")
    if catalog is not None and len(catalog) > 0:
        last_event = catalog.iloc[-1]
        if last_event["detection_time"] <= latest_ts <= last_event.get("end_time", last_event["detection_time"]):
            goes_class = last_event.get("goes_class")

    return NowcastResponse(
        as_of=latest_ts.to_pydatetime() if hasattr(latest_ts, "to_pydatetime") else latest_ts,
        alert_level=str(latest_alert),
        current_flux=float(latest_flux) if latest_flux is not None and not pd.isna(latest_flux) else None,
        baseline=float(latest_baseline) if not pd.isna(latest_baseline) else None,
        goes_class=goes_class,
    )


def compute_forecast_status(df: pd.DataFrame, model) -> ForecastResponse:
    if model is None:
        return ForecastResponse(note="Forecast model not loaded on this server.")

    features = compute_day_features(df)
    if len(features) == 0:
        return ForecastResponse(note="No data available to compute forecast features.")

    latest_ts = features.index[-1]
    latest_features = features.iloc[[-1]][FEATURE_COLUMNS]

    if latest_features.isna().any(axis=1).iloc[0]:
        return ForecastResponse(
            as_of=latest_ts.to_pydatetime() if hasattr(latest_ts, "to_pydatetime") else latest_ts,
            note="Insufficient or untrustworthy data for the most recent window "
                 "(e.g. instrument saturation or missing history) -- no forecast produced.",
        )

    proba = model.predict_proba(latest_features)[:, list(model.classes_).index(True)]
    flare_prob = float(proba[0])

    level = "QUIET"
    if flare_prob >= ALERT_THRESHOLD:
        level = "ALERT"
    elif flare_prob >= WARNING_THRESHOLD:
        level = "WARNING"
    elif flare_prob >= WATCH_THRESHOLD:
        level = "WATCH"

    return ForecastResponse(
        as_of=latest_ts.to_pydatetime() if hasattr(latest_ts, "to_pydatetime") else latest_ts,
        alert_level=level,
        flare_probability=flare_prob,
    )


# --------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------- #
@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_loaded=app.state.forecast_model is not None,
        data_file_accessible=DATA_FILE_PATH.exists(),
        checked_at=datetime.now(timezone.utc),
    )


@app.get("/nowcast", response_model=NowcastResponse)
def nowcast():
    df = load_current_data()
    return compute_nowcast_status(df, app.state.nowcast_engine)


@app.get("/forecast", response_model=ForecastResponse)
def forecast():
    df = load_current_data()
    return compute_forecast_status(df, app.state.forecast_model)


@app.get("/status", response_model=StatusResponse)
def status():
    df = load_current_data()
    return StatusResponse(
        nowcast=compute_nowcast_status(df, app.state.nowcast_engine),
        forecast=compute_forecast_status(df, app.state.forecast_model),
    )
