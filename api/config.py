"""Environment-driven configuration and constants."""
from __future__ import annotations

import os
from pathlib import Path

FEATURE_COLUMNS = [
    "solexs_mean", "solexs_slope", "solexs_std",
    "hel1os_mean", "hel1os_slope",
    "hardness_ratio", "hr_rate_of_change",
    "peak_to_mean_ratio", "hel1os_saturation_fraction",
]

DATA_FILE_PATH = Path(os.environ.get("KNOXIS_DATA_FILE", "data/training/output/latest_data.parquet"))
MODEL_FILE_PATH = Path(os.environ.get("KNOXIS_MODEL_FILE", "data/training/output/knoxis_binary_flare_model.joblib"))
META_FILE_PATH = Path(os.environ.get("KNOXIS_META_FILE", str(MODEL_FILE_PATH.with_name("model_meta.json"))))
# Multi-day archive: a folder of parquet files (one per day, or multi-day, overlap allowed).
# Days are UTC dates found from the timestamps; no naming convention is needed.
DAYS_DIR = Path(os.environ.get("KNOXIS_DAYS_DIR", "data/days"))
# History loaded before the selected day so the trailing baseline is warm at 00:00.
LEADIN_HOURS = float(os.environ.get("KNOXIS_LEADIN_HOURS", "24"))
# A date with fewer samples than this (for example a sliver of lead-in) is not listed as a day.
MIN_DAY_ROWS = int(os.environ.get("KNOXIS_MIN_DAY_ROWS", "600"))

METRICS_FILE_PATH = Path(os.environ.get("KNOXIS_METRICS_FILE", "data/training/output/metrics.json"))

# Display-only; must match NowcastEngine. The engine remains the source of truth for levels.
THRESHOLD_RATIO = float(os.environ.get("KNOXIS_THRESHOLD_RATIO", "1.4"))
SUSTAIN_MINUTES = float(os.environ.get("KNOXIS_SUSTAIN_MINUTES", "3"))

# GOES-equivalent W/m^2 per count/s, reconstructed from the two matched flares in
# nowcast/README.md (M2.6 @ 1886, M6.8 @ 4931 counts/s). Replace with the engine's own value.
GOES_FACTOR = float(os.environ.get("KNOXIS_GOES_FACTOR", "1.379e-8"))

STALE_AFTER_S = float(os.environ.get("KNOXIS_STALE_AFTER_S", "600"))
MODE_OVERRIDE = os.environ.get("KNOXIS_MODE")  # "live" | "simulated" | "replay" | unset
# A data file rewritten within this many seconds counts as an active feed even if its
# timestamps are old (live_feed_simulator.py replays historical days into the file).
FEED_ACTIVE_S = float(os.environ.get("KNOXIS_FEED_ACTIVE_S", "30"))

# Replay (as_of) strategy: "exact" re-runs the engine on truncated data for every position (always
# correct, slow); "slice" cuts one full run (fast; valid only if the code is causal); "auto" uses
# slice once a startup self-check has passed and falls back to exact otherwise.
REPLAY_MODE = os.environ.get("KNOXIS_REPLAY", "auto")
# In slice mode, an event's HEL1OS corroboration is reported only this long after its peak, because
# the corroboration window may still be growing before that.
PENDING_AFTER_PEAK_MIN = 35.0

# Optional ground truth for /evaluation: CSV with start_time, peak_time (optional), end_time (optional), goes_class.
TRUTH_FILE_PATH = Path(os.environ.get("KNOXIS_TRUTH_FILE", "data/truth/noaa_events.csv"))

DASHBOARD_FILE = Path(os.environ.get(
    "KNOXIS_DASHBOARD_FILE", str(Path(__file__).resolve().parent.parent / "dashboard" / "index.html")))

# ---- production ---------------------------------------------------------------------------------
# KNOXIS_PRODUCTION=1 turns off /docs, hides server paths from responses, restricts CORS and hosts, and
# puts /diagnostics behind KNOXIS_ADMIN_TOKEN (sent as the X-Admin-Token header).
PRODUCTION = os.environ.get("KNOXIS_PRODUCTION", "0") == "1"
ADMIN_TOKEN = os.environ.get("KNOXIS_ADMIN_TOKEN", "")
_DOMAIN_ORIGINS = "https://knoxis.in,https://www.knoxis.in"
CORS_ORIGINS = [o.strip() for o in os.environ.get("KNOXIS_CORS_ORIGINS", _DOMAIN_ORIGINS if PRODUCTION else "*").split(",") if o.strip()]
ALLOWED_HOSTS = [h.strip() for h in os.environ.get(
    "KNOXIS_ALLOWED_HOSTS", "knoxis.in,www.knoxis.in,127.0.0.1,localhost" if PRODUCTION else "*").split(",") if h.strip()]
# Requests per minute per client IP (0 = off). Per-IP counting needs uvicorn --proxy-headers behind nginx.
RATE_LIMIT_PER_MIN = int(os.environ.get("KNOXIS_RATE_LIMIT_PER_MIN", "240" if PRODUCTION else "0"))
# How many engine builds may be running or waiting at once; further requests get 503 instead of queueing forever.
MAX_BUILD_QUEUE = int(os.environ.get("KNOXIS_MAX_BUILD_QUEUE", "6"))
# The day index is re-checked against the folder at most this often (seconds).
SCAN_TTL_S = float(os.environ.get("KNOXIS_SCAN_TTL_S", "5"))
INDEX_FILE_PATH = Path(os.environ.get("KNOXIS_INDEX_FILE", str(DAYS_DIR / ".knoxis_index.json")))


def shown(path) -> str:
    """A path as it may appear in a response: full in development, file name only in production."""
    return Path(path).name if PRODUCTION else str(path)


DEFAULT_THRESHOLDS = {"watch": 0.3, "warning": 0.5, "alert": 0.7}
LEVELS = ["QUIET", "WATCH", "WARNING", "ALERT"]
LEVEL_CODE = {name: i for i, name in enumerate(LEVELS)}


def resolve_thresholds(meta: dict) -> dict[str, float]:
    """Defaults, then model_meta.json, then explicit KNOXIS_<LEVEL>_THRESHOLD env vars."""
    thr = dict(DEFAULT_THRESHOLDS)
    for k, v in (meta.get("thresholds") or {}).items():
        if k in thr:
            thr[k] = float(v)
    for k in thr:
        env = os.environ.get(f"KNOXIS_{k.upper()}_THRESHOLD")
        if env is not None:
            thr[k] = float(env)
    return thr
