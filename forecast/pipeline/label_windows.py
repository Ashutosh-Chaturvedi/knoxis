"""
label_windows.py

Step 4 of the forecast training pipeline: labels every feature window
(output of compute_features.py) with the strongest GOES class of any
flare beginning in the next 60 minutes, using the NOAA flare catalog
(output of parse_noaa_events.py).

Design: instead of scanning the catalog per window (slow at ~226,000
windows across 787 days), converts catalog flares to real physical flux
values on a 1-minute timeline, then uses a REVERSED rolling-max trick:
reverse the timeline, take a trailing 60-min rolling max (fast,
vectorized), reverse back. Every minute now holds the max flux of any
flare beginning in the next hour, computed once for the whole date
range. Flares crossing midnight are handled correctly for free, since
the timeline is global, not restricted per calendar day like the
feature files are.

Horizon definition (confirmed): ANY flare starting anytime in the next
60 minutes (0-60 min ahead), not restricted to a 30-60 min band.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

GOES_BOUNDARIES = {
    "A": 1e-8,
    "B": 1e-7,
    "C": 1e-6,
    "M": 1e-5,
    "X": 1e-4,
}


def goes_class_to_flux(goes_class: str) -> float:
    """Converts a real GOES class string (e.g. 'M2.6') to its physical
    flux value in W/m^2. NOAA's catalog already gives true GOES units --
    no calibration factor needed here (unlike SoLEXS counts, which DID
    need the derived 1.379e-8 factor)."""
    letter, number = goes_class[0], float(goes_class[1:])
    return GOES_BOUNDARIES[letter] * number


def flux_to_goes_class(flux: float) -> str:
    """Inverse of goes_class_to_flux -- converts a flux value back into
    a class string, e.g. for reporting the strongest flare in a window
    after a rolling-max operation on raw flux values."""
    if flux <= 0 or pd.isna(flux):
        return "QUIET"
    for letter in ["X", "M", "C", "B", "A"]:
        lower = GOES_BOUNDARIES[letter]
        if flux >= lower:
            return f"{letter}{flux / lower:.1f}"
    return "QUIET"


def build_forward_max_flux_timeline(
    catalog: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    horizon_minutes: int = 60,
) -> pd.Series:
    """
    Builds a 1-minute-resolution Series spanning [start, end] where each
    minute T holds the max real flux of any flare in `catalog` whose
    begin_time falls in (T, T + horizon_minutes].

    Vectorized via the reverse-rolling-max trick described in the module
    docstring -- NOT a per-minute catalog scan.
    """
    timeline_index = pd.date_range(start, end, freq="1min", tz="UTC")
    flux_at_minute = pd.Series(0.0, index=timeline_index)

    catalog = catalog.copy()
    catalog["flux"] = catalog["goes_class"].apply(goes_class_to_flux)
    catalog["begin_minute"] = catalog["begin_time"].dt.floor("min")

    # Multiple flares can begin in the same minute -- keep the max flux
    # per minute before placing onto the timeline.
    per_minute_max = catalog.groupby("begin_minute")["flux"].max()
    per_minute_max = per_minute_max[
        (per_minute_max.index >= start) & (per_minute_max.index <= end)
    ]
    flux_at_minute.loc[per_minute_max.index] = per_minute_max.values

    # Reverse -> trailing rolling max (fast, vectorized) -> reverse back.
    # A trailing window on the REVERSED series is equivalent to a
    # forward-looking window on the original series.
    reversed_series = flux_at_minute.iloc[::-1]
    reversed_rolling_max = reversed_series.rolling(
        f"{horizon_minutes}min", min_periods=1
    ).max()
    forward_max_flux = reversed_rolling_max.iloc[::-1]

    # The rolling window as constructed includes T itself (a flare
    # beginning AT T), but per the confirmed horizon definition we only
    # want flares beginning AFTER T (strictly forward-looking, excluding
    # the current instant) up to T+horizon. Shift by one minute to
    # exclude T itself from its own window.
    forward_max_flux = forward_max_flux.shift(-1)

    return forward_max_flux


def label_feature_file(
    features_path: Path,
    forward_max_flux: pd.Series,
    horizon_minutes: int = 60,
) -> pd.DataFrame:
    """
    Labels one day's feature file using a precomputed forward_max_flux
    timeline (built once for the whole date range, not per day).
    """
    features = pd.read_parquet(features_path)
    if features.index.name != "timestamp":
        features = features.set_index("timestamp") if "timestamp" in features.columns else features

    # Feature timestamps are on a 5-min grid; forward_max_flux is on a
    # 1-min grid -- direct index lookup (not asof) since 5-min marks are
    # always present on a 1-min-resolution index.
    matched_flux = forward_max_flux.reindex(features.index)

    features = features.copy()
    features["label_flux"] = matched_flux
    features["label_class"] = matched_flux.apply(flux_to_goes_class)

    return features


def label_all_features(
    features_dir: Path,
    flare_catalog_path: Path,
    output_dir: Path,
    horizon_minutes: int = 60,
) -> None:
    """
    Batch entry point: labels every feature file in features_dir.
    Builds the forward_max_flux timeline ONCE for the full span (not
    per file), since that's the expensive part and is fully reusable.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    catalog = pd.read_parquet(flare_catalog_path)

    feature_files = sorted(features_dir.glob("*_features.parquet"))
    if not feature_files:
        print("No feature files found.")
        return

    # Determine the full span needed: from the first feature file's
    # start to the last feature file's end PLUS the horizon (so flares
    # just after the last day's data are still captured).
    first_day = pd.read_parquet(feature_files[0])
    last_day = pd.read_parquet(feature_files[-1])
    span_start = (first_day.index if first_day.index.name == "timestamp" else first_day["timestamp"]).min()
    span_end = (last_day.index if last_day.index.name == "timestamp" else last_day["timestamp"]).max() \
        + pd.Timedelta(minutes=horizon_minutes + 5)

    print(f"Building forward-max-flux timeline for {span_start} -> {span_end} ...")
    forward_max_flux = build_forward_max_flux_timeline(catalog, span_start, span_end, horizon_minutes)
    print("Timeline built. Labeling individual days...")

    failed_days = []
    n_done = 0
    for feat_path in feature_files:
        day_str = feat_path.stem.replace("_features", "")
        out_path = output_dir / f"{day_str}_labeled.parquet"
        if out_path.exists():
            continue
        try:
            labeled = label_feature_file(feat_path, forward_max_flux, horizon_minutes)
            labeled.to_parquet(out_path)
            n_done += 1
        except Exception as exc:  # noqa: BLE001
            failed_days.append((day_str, str(exc)))
            print(f"FAILED: {day_str} -- {exc}")

    print(f"\nDone. {n_done} days labeled, {len(failed_days)} failed.")
    if failed_days:
        (output_dir / "label_failures.txt").write_text(
            "\n".join(f"{d}: {r}" for d, r in failed_days)
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Label feature windows using the flare catalog")
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--flare-catalog", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--horizon-minutes", type=int, default=60)
    args = parser.parse_args()

    label_all_features(
        Path(args.features_dir), Path(args.flare_catalog), Path(args.output_dir),
        args.horizon_minutes,
    )
