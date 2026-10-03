"""Forecast status: model probability -> graduated level."""
from __future__ import annotations

from api.config import FEATURE_COLUMNS
from api.schemas import ForecastResponse
from api.snapshot import Snapshot
from api.state import runtime
from api.timeutil import py_dt


def compute_forecast(snap: Snapshot) -> ForecastResponse:
    model, meta, thr = runtime.model, runtime.meta, runtime.thresholds
    base = dict(base_rate=meta.get("base_rate"), thresholds=thr, model_version=meta.get("model_version"))

    if model is None:
        return ForecastResponse(note="Forecast model not loaded on this server.", **base)
    if snap.features is None:
        msg = (f"Feature computation failed ({snap.feature_error})." if snap.feature_error
               else "No data available to compute forecast features.")
        return ForecastResponse(note=msg, **base)
    if len(snap.features) == 0:
        return ForecastResponse(note="No data available to compute forecast features.", **base)

    ts = snap.features.index[-1]
    row = snap.features.iloc[[-1]][FEATURE_COLUMNS]
    if row.isna().any(axis=1).iloc[0]:
        return ForecastResponse(
            as_of=py_dt(ts),
            note="Insufficient or untrustworthy data for the most recent window "
                 "(for example instrument saturation or missing history). No forecast produced.",
            **base)

    p = float(model.predict_proba(row)[:, list(model.classes_).index(True)][0])
    level = "QUIET"
    if p >= thr["alert"]:
        level = "ALERT"
    elif p >= thr["warning"]:
        level = "WARNING"
    elif p >= thr["watch"]:
        level = "WATCH"
    return ForecastResponse(as_of=py_dt(ts), alert_level=level, flare_probability=p, **base)
