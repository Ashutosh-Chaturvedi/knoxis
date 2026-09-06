"""
Knoxis Backend - Step 2: The Live State Engine
==============================================
This file is the "Brain" of the backend. 
It remembers the recent stream of satellite readings, computes the normal baseline,
and checks every second whether flux has jumped high enough to trigger an ALERT.
"""

from __future__ import annotations

import sys
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# Ensure root directory is in path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Try importing from nowcast.nowcast; provide pure-Python fallbacks so the server NEVER crashes
try:
    from nowcast.nowcast import NowcastConfig, classify_flare, counts_to_flux
except ImportError:
    COUNTS_TO_FLUX_FACTOR = 1.379e-8
    GOES_BOUNDARIES = {"A": 1e-8, "B": 1e-7, "C": 1e-6, "M": 1e-5, "X": 1e-4}

    def counts_to_flux(counts: float) -> float:
        return counts * COUNTS_TO_FLUX_FACTOR

    def classify_flare(peak_counts: float) -> str:
        flux = counts_to_flux(peak_counts)
        if flux < GOES_BOUNDARIES["A"]:
            return "Quiet"
        for letter in ["X", "M", "C", "B", "A"]:
            lower = GOES_BOUNDARIES[letter]
            if flux >= lower:
                return f"{letter}{flux / lower:.1f}"
        return "Quiet"

    @dataclass
    class NowcastConfig:
        threshold_ratio: float = 1.4
        sustain_minutes: float = 3.0
        watch_fraction: float = 0.6
        warning_fraction: float = 0.85


class KnoxisEngine:
    """
    Manages live satellite state in memory and applies Knoxis nowcast rules.
    """

    def __init__(self, default_baseline: float = 45.0):
        self.config = NowcastConfig()
        
        # Current normal "quiet Sun" baseline count rate
        self.baseline: float = default_baseline
        
        # Latest live satellite readings
        self.latest_timestamp: datetime = datetime.now(timezone.utc)
        self.latest_counts: float = default_baseline
        self.latest_is_valid: bool = True
        
        # Alert tracking
        self.current_alert_level: str = "QUIET"   # QUIET, WATCH, WARNING, ALERT
        self.crossing_start_time: datetime | None = None
        self.sustained_seconds: float = 0.0
        
        # Memory buffer: keep the last 3,600 readings (1 hour at 1Hz)
        self.history = deque(maxlen=3600)
        
        # Recorded flare events (our live flare catalog)
        self.flare_catalog: list[dict] = []
        self._current_event: dict | None = None

    def process_tick(self, counts: float, timestamp: datetime | None = None, is_valid: bool = True) -> dict:
        """
        Receives ONE second of satellite flux and evaluates the alert level.
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        self.latest_timestamp = timestamp
        self.latest_counts = counts
        self.latest_is_valid = is_valid

        # Save to short-term history buffer
        self.history.append({
            "timestamp": timestamp.isoformat(),
            "counts": counts,
            "is_valid": is_valid,
        })

        # Calculate current ratio against the baseline
        ratio = (counts / self.baseline) if self.baseline > 0 else 1.0
        
        # Scientific boundaries
        watch_boundary = 1.0 + self.config.watch_fraction * (self.config.threshold_ratio - 1.0)      # ~1.24
        warning_boundary = 1.0 + self.config.warning_fraction * (self.config.threshold_ratio - 1.0)  # ~1.34
        alert_boundary = self.config.threshold_ratio                                                  # 1.40
        sustain_required = self.config.sustain_minutes * 60.0                                          # 180 seconds

        # Check threshold crossing & sustained duration
        is_above_threshold = (ratio >= alert_boundary) and is_valid

        if is_above_threshold:
            if self.crossing_start_time is None:
                self.crossing_start_time = timestamp
            self.sustained_seconds = (timestamp - self.crossing_start_time).total_seconds()
        else:
            # Dropped back down below 1.4x
            if self._current_event is not None:
                self._current_event["end_time"] = timestamp.isoformat()
                self.flare_catalog.append(self._current_event)
                self._current_event = None

            self.crossing_start_time = None
            self.sustained_seconds = 0.0

        # Determine graduated alert level
        is_confirmed = is_above_threshold and (self.sustained_seconds >= sustain_required)

        if is_confirmed:
            self.current_alert_level = "ALERT"
            if self._current_event is None:
                self._current_event = {
                    "detection_time": timestamp.isoformat(),
                    "start_time": self.crossing_start_time.isoformat() if self.crossing_start_time else timestamp.isoformat(),
                    "peak_counts": counts,
                    "goes_class": classify_flare(counts),
                    "status": "IN_PROGRESS",
                }
            else:
                if counts > self._current_event["peak_counts"]:
                    self._current_event["peak_counts"] = counts
                    self._current_event["goes_class"] = classify_flare(counts)
        elif ratio >= warning_boundary:
            self.current_alert_level = "WARNING"
        elif ratio >= watch_boundary:
            self.current_alert_level = "WATCH"
        else:
            self.current_alert_level = "QUIET"

        return self.get_status()

    def get_status(self) -> dict:
        """Returns the complete real-time status of the engine."""
        flux_w_m2 = counts_to_flux(self.latest_counts)
        goes_class = classify_flare(self.latest_counts)
        ratio = (self.latest_counts / self.baseline) if self.baseline > 0 else 1.0

        return {
            "alert_level": self.current_alert_level,
            "active_flare": self.current_alert_level == "ALERT",
            "current_reading": {
                "timestamp": self.latest_timestamp.isoformat(),
                "solexs_counts": round(float(self.latest_counts), 2),
                "baseline_counts": round(float(self.baseline), 2),
                "ratio_to_baseline": round(float(ratio), 3),
                "flux_w_m2": f"{flux_w_m2:.3e}",
                "goes_class": goes_class,
                "is_valid": bool(self.latest_is_valid),
            },
            "sustained_seconds": round(float(self.sustained_seconds), 1),
            "recent_flares_count": len(self.flare_catalog),
            "active_event": self._current_event,
        }


    def reset(self, default_baseline: float = 45.0) -> None:
        """Resets engine state and clears historical flare memories back to quiet sun."""
        self.baseline = default_baseline
        self.latest_counts = default_baseline
        self.latest_is_valid = True
        self.current_alert_level = "QUIET"
        self.crossing_start_time = None
        self.sustained_seconds = 0.0
        self.history.clear()
        
        # Populate history with quiet baseline readings so the 60-min forecast resets immediately
        now = datetime.now(timezone.utc)
        for _ in range(60):
            self.history.append({
                "timestamp": now.isoformat(),
                "counts": default_baseline,
                "is_valid": True,
            })
        self._current_event = None


# Singleton instance shared across the backend
engine = KnoxisEngine()
