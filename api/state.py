"""Process-wide runtime objects (engine, model, metadata), loaded once at startup.

Kept in a plain module so services and the snapshot builder can read it without
importing the FastAPI app (which would be a circular import).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import joblib

from api.config import DEFAULT_THRESHOLDS, META_FILE_PATH, MODEL_FILE_PATH, resolve_thresholds
from nowcast.nowcast import NowcastEngine

log = logging.getLogger("knoxis.api")


@dataclass
class Runtime:
    engine: Optional[Any] = None
    model: Optional[Any] = None
    meta: dict = field(default_factory=dict)
    thresholds: dict = field(default_factory=lambda: dict(DEFAULT_THRESHOLDS))


runtime = Runtime()


def load_runtime() -> Runtime:
    if MODEL_FILE_PATH.exists():
        runtime.model = joblib.load(MODEL_FILE_PATH)
    else:
        runtime.model = None
        log.warning("forecast model not found at %s; forecast will report unavailable", MODEL_FILE_PATH)

    meta: dict = {}
    if META_FILE_PATH.exists():
        try:
            meta = json.loads(META_FILE_PATH.read_text())
        except Exception:
            log.exception("could not read %s", META_FILE_PATH)
    else:
        log.warning("no model meta at %s; using default thresholds and no base rate", META_FILE_PATH)
    runtime.meta = meta
    runtime.thresholds = resolve_thresholds(meta)
    runtime.engine = NowcastEngine()
    return runtime
