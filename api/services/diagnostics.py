"""Self-diagnosis for the common "why does nothing work" problems."""
from __future__ import annotations

from datetime import datetime, timezone

from api.config import (DASHBOARD_FILE, DATA_FILE_PATH, GOES_FACTOR, META_FILE_PATH, METRICS_FILE_PATH,
                        MODEL_FILE_PATH, REPLAY_MODE, SUSTAIN_MINUTES, THRESHOLD_RATIO, TRUTH_FILE_PATH)
from api.snapshot import latest_keeper
from api.state import runtime

REQUIRED_COLUMNS = ["solexs_counts", "solexs_is_valid", "hel1os_ctr", "hel1os_is_valid"]


def run_diagnostics() -> dict:
    checks: list[dict] = []

    def add(name: str, ok: bool, detail: str, hint: str = "", required: bool = True) -> None:
        checks.append({"name": name, "ok": ok, "required": required, "detail": detail, "hint": "" if ok else hint})

    now = datetime.now(timezone.utc)

    add("Dashboard file", DASHBOARD_FILE.exists(), str(DASHBOARD_FILE),
        "Start the API from the repo root, or set KNOXIS_DASHBOARD_FILE.", required=False)

    exists = DATA_FILE_PATH.exists()
    add("Data file", exists, str(DATA_FILE_PATH),
        "Set KNOXIS_DATA_FILE to the parquet the ingestion job or live_feed_simulator.py writes, "
        "and start the writer first.")

    snap = latest_keeper.peek()
    if exists:
        if latest_keeper.last_error:
            add("Latest snapshot", False, latest_keeper.last_error,
                "The data file could not be processed. Check the uvicorn log for the traceback.")
        elif snap is None:
            add("Latest snapshot", False, "not built yet",
                "The first build is still running (see the engine.run time in the uvicorn log), or the file is empty.")
        else:
            add("Latest snapshot", True,
                f"{len(snap.df):,} rows, {snap.span_start:%Y-%m-%d %H:%M} to {snap.span_end:%Y-%m-%d %H:%M} UTC, "
                f"last build {latest_keeper.last_build_s:.1f}s, {latest_keeper.builds} build(s)")
            missing = [c for c in REQUIRED_COLUMNS if c not in snap.df.columns]
            add("Required columns", not missing,
                "all present" if not missing else "missing: " + ", ".join(missing),
                "The file must be the output of DataIngestionPipeline.run(), indexed by timestamp.")
            age = (now - snap.file_updated).total_seconds()
            add("Data file freshness", True, f"last written {age:.0f}s ago", required=False)

    add("Forecast model", runtime.model is not None, str(MODEL_FILE_PATH),
        "Train it or set KNOXIS_MODEL_FILE. Without it the forecast panel reports 'not loaded'.", required=False)
    add("Model metadata", bool(runtime.meta), str(META_FILE_PATH),
        "Write model_meta.json (model_version, base_rate, thresholds). Without it the base rate is blank "
        "and default thresholds apply.", required=False)
    add("Evaluation metrics", METRICS_FILE_PATH.exists(), str(METRICS_FILE_PATH),
        "Optional: write metrics.json from your offline evaluation job.", required=False)
    add("Truth catalog", TRUTH_FILE_PATH.exists(), str(TRUTH_FILE_PATH),
        "Optional: a CSV of NOAA events (start_time, peak_time, end_time, goes_class) enables hit rate, "
        "false alarms and latency in the Evaluation panel.", required=False)

    return {
        "ok": all(c["ok"] for c in checks if c["required"]),
        "server_time": now.isoformat(),
        "checks": checks,
        "config": {
            "threshold_ratio": THRESHOLD_RATIO, "sustain_minutes": SUSTAIN_MINUTES, "goes_factor": GOES_FACTOR,
            "forecast_thresholds": runtime.thresholds, "replay_setting": REPLAY_MODE,
            "replay_strategy": latest_keeper.replay_strategy(),
            "causality_self_check": latest_keeper.causal_detail,
            "causality_self_check_passed": latest_keeper.causal,
        },
    }
