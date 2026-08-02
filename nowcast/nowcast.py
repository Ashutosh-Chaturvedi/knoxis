from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

COUNTS_TO_FLUX_FACTOR = 1.379e-8  # W/m^2 per count/sec

GOES_BOUNDARIES = {
    "A": 1e-8,
    "B": 1e-7,
    "C": 1e-6,
    "M": 1e-5,
    "X": 1e-4,
}


def counts_to_flux(counts) -> float:
    """Converts raw SoLEXS counts/sec to GOES-equivalent flux (W/m^2)."""
    return counts * COUNTS_TO_FLUX_FACTOR


def classify_flare(peak_counts: float) -> str:
    """Converts peak SoLEXS counts/sec into a GOES-style class string, e.g. 'M6.8'."""
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
    """Tunable parameters for the nowcast engine."""

    baseline_window: str = "24h"       # trailing window for the quiet baseline
    baseline_min_periods: int = 3600   # require at least 1hr of valid samples
    threshold_ratio: float = 1.4       # confirmed-flare trigger, per project spec
    sustain_minutes: float = 3.0       # required sustained duration before confirming
    watch_fraction: float = 0.6        # WATCH begins at this fraction of the way to threshold
    warning_fraction: float = 0.85     # WARNING begins at this fraction of the way to threshold
    hel1os_corroboration_ratio: float = 1.4  # HEL1OS "also elevated" check threshold


def compute_rolling_baseline(df: pd.DataFrame, config: NowcastConfig) -> pd.Series:
    valid_counts = df["solexs_counts"].where(df["solexs_is_valid"])
    baseline = valid_counts.rolling(
        config.baseline_window, min_periods=config.baseline_min_periods
    ).median()
    return baseline


def detect_threshold_crossings(
    df: pd.DataFrame, baseline: pd.Series, config: NowcastConfig
) -> pd.Series:
    
    ratio = df["solexs_counts"] / baseline
    is_crossing = (ratio >= config.threshold_ratio) & df["solexs_is_valid"]
    return is_crossing.fillna(False)


def enforce_sustained_duration(crossing: pd.Series, config: NowcastConfig) -> pd.Series:
 
    # Each time `crossing` flips (True->False or False->True), start a new
    # group. Cumulative sum of the flip-indicator gives a unique group id
    # for every contiguous run of identical values.
    group_id = (crossing != crossing.shift(fill_value=False)).cumsum()

    confirmed = pd.Series(False, index=crossing.index)
    sustain_delta = pd.Timedelta(minutes=config.sustain_minutes)

    for _, group_index in crossing.groupby(group_id).groups.items():
        if not crossing.loc[group_index[0]]:
            continue  # this run is a False-run, nothing to confirm
        run_start = group_index[0]
        elapsed = group_index.to_series().index - run_start
        confirmed.loc[group_index] = elapsed >= sustain_delta

    return confirmed


def compute_alert_level(
    df: pd.DataFrame, baseline: pd.Series, confirmed: pd.Series, config: NowcastConfig
) -> pd.Series:
    ratio = (df["solexs_counts"] / baseline).fillna(0.0)

    watch_boundary = 1.0 + config.watch_fraction * (config.threshold_ratio - 1.0)
    warning_boundary = 1.0 + config.warning_fraction * (config.threshold_ratio - 1.0)

    level = pd.Series("QUIET", index=df.index)
    level[ratio >= watch_boundary] = "WATCH"
    level[ratio >= warning_boundary] = "WARNING"
    level[confirmed] = "ALERT"
    return level


def check_hel1os_corroboration(
    df: pd.DataFrame, event_start, event_end, config: NowcastConfig
) -> str:
  
    window = df.loc[event_start:event_end]
    valid_window = window[window["hel1os_is_valid"]]

    if len(valid_window) == 0:
        return "inconclusive"

    # Simple quiet-reference: median HEL1OS count rate over the whole
    # dataset's valid rows, as a rough "is this elevated" yardstick.
    hel1os_quiet = df.loc[df["hel1os_is_valid"], "hel1os_ctr"].median()
    if hel1os_quiet == 0 or np.isnan(hel1os_quiet):
        return "inconclusive"

    peak_hel1os = valid_window["hel1os_ctr"].max()
    if peak_hel1os / hel1os_quiet >= config.hel1os_corroboration_ratio:
        return "yes"
    return "no"


def build_flare_catalog(
    df: pd.DataFrame, baseline: pd.Series, confirmed: pd.Series, config: NowcastConfig
) -> pd.DataFrame:

    if not confirmed.any():
        return pd.DataFrame(
            columns=["detection_time", "peak_time", "peak_counts",
                     "goes_class", "hel1os_corroboration"]
        )

    group_id = (confirmed != confirmed.shift(fill_value=False)).cumsum()
    events = []

    for _, group_index in confirmed.groupby(group_id).groups.items():
        if not confirmed.loc[group_index[0]]:
            continue

        event_window = df.loc[group_index]
        detection_time = group_index[0]  # first CONFIRMED timestamp
        peak_time = event_window["solexs_counts"].idxmax()
        peak_counts = event_window.loc[peak_time, "solexs_counts"]
        goes_class = classify_flare(peak_counts)

        # Extend the event window slightly for HEL1OS corroboration check,
        # since HEL1OS's own peak may lag/lead SoLEXS's slightly.
        corroboration = check_hel1os_corroboration(
            df, group_index[0] - pd.Timedelta(minutes=2),
            group_index[-1] + pd.Timedelta(minutes=2), config,
        )

        events.append({
            "detection_time": detection_time,
            "peak_time": peak_time,
            "peak_counts": peak_counts,
            "goes_class": goes_class,
            "hel1os_corroboration": corroboration,
        })

    return pd.DataFrame(events)


class NowcastEngine:
    

    def __init__(self, config: NowcastConfig | None = None):
        self.config = config or NowcastConfig()

    def run(self, df: pd.DataFrame) -> dict:
        baseline = compute_rolling_baseline(df, self.config)
        crossing = detect_threshold_crossings(df, baseline, self.config)
        confirmed = enforce_sustained_duration(crossing, self.config)
        alert_level = compute_alert_level(df, baseline, confirmed, self.config)
        catalog = build_flare_catalog(df, baseline, confirmed, self.config)

        return {
            "baseline": baseline,
            "alert_level": alert_level,
            "flare_catalog": catalog,
        }