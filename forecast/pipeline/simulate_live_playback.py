"""
simulate_live_playback.py

Tests whether the nowcast engine + forecast model are actually safe to
run in a LIVE setting, not just batch historical evaluation, without
rebuilding anything -- by proving a specific, checkable claim:

    Since every rolling computation in this pipeline is trailing (not
    centered) by design, running the full pipeline once on historical
    data and reading off values chronologically should be
    mathematically IDENTICAL to what a true incremental, live
    computation would produce at each point in time.

This is directly tested, not assumed: at several checkpoints through a
real day, the data is TRUNCATED (simulating "this is all we know right
now"), everything is recomputed independently on just the truncated
data, and the result at that exact timestamp is compared against the
full-day batch computation.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from nowcast.nowcast import NowcastEngine
from forecast.pipeline.compute_features import compute_day_features

FEATURE_COLUMNS = [
    "solexs_mean", "solexs_slope", "solexs_std",
    "hel1os_mean", "hel1os_slope",
    "hardness_ratio", "hr_rate_of_change",
    "peak_to_mean_ratio", "hel1os_saturation_fraction",
]


def compute_combined_output(combined_df: pd.DataFrame, forecast_model,
                             window_minutes: int = 30, step_minutes: int = 5) -> pd.DataFrame:
    """Runs both the nowcast engine and the forecast model on the same
    combined SoLEXS+HEL1OS data, returning one aligned time series."""
    nowcast_engine = NowcastEngine()
    nowcast_result = nowcast_engine.run(combined_df)

    features = compute_day_features(combined_df, window_minutes, step_minutes)
    features_clean = features.dropna(subset=FEATURE_COLUMNS)

    flare_prob = pd.Series(np.nan, index=features.index, name="flare_probability")
    if len(features_clean) > 0:
        proba = forecast_model.predict_proba(features_clean[FEATURE_COLUMNS])[:, 1]
        flare_prob.loc[features_clean.index] = proba

    combined_output = pd.DataFrame({
        "nowcast_alert_level": nowcast_result["alert_level"].reindex(features.index, method="nearest"),
        "nowcast_baseline": nowcast_result["baseline"].reindex(features.index, method="nearest"),
        "flare_probability": flare_prob,
    })
    return combined_output


def check_live_safety(combined_df: pd.DataFrame, forecast_model,
                       check_times: list, window_minutes: int = 30, step_minutes: int = 5) -> bool:
    """THE key test: for each check_time, truncates combined_df to only
    data up to that time, recomputes the full pipeline independently on
    JUST the truncated data, and confirms the result matches the
    full-batch computation exactly."""
    full_output = compute_combined_output(combined_df, forecast_model, window_minutes, step_minutes)

    all_match = True
    print(f"{'Check time':<28} {'Full-batch alert':<18} {'Truncated alert':<18} "
          f"{'Full prob':<12} {'Truncated prob':<14} Match?")
    print("-" * 100)

    for check_time in check_times:
        truncated_df = combined_df.loc[:check_time]
        truncated_output = compute_combined_output(truncated_df, forecast_model, window_minutes, step_minutes)

        full_slice = full_output.loc[:check_time]
        full_row = full_slice.iloc[-1] if len(full_slice) else None
        trunc_row = truncated_output.iloc[-1] if len(truncated_output) else None

        if full_row is None or trunc_row is None:
            print(f"{str(check_time):<28} -- insufficient data, skipping --")
            continue

        alert_match = full_row["nowcast_alert_level"] == trunc_row["nowcast_alert_level"]
        baseline_match = np.isclose(full_row["nowcast_baseline"], trunc_row["nowcast_baseline"],
                                     equal_nan=True, rtol=1e-4)
        prob_match = (
            (pd.isna(full_row["flare_probability"]) and pd.isna(trunc_row["flare_probability"]))
            or np.isclose(full_row["flare_probability"], trunc_row["flare_probability"], equal_nan=True)
        )
        row_match = alert_match and baseline_match and prob_match
        all_match = all_match and row_match

        print(f"{str(check_time):<28} {str(full_row['nowcast_alert_level']):<18} "
              f"{str(trunc_row['nowcast_alert_level']):<18} "
              f"{full_row['flare_probability']:<12.4f} {trunc_row['flare_probability']:<14.4f} "
              f"{'YES' if row_match else 'NO -- MISMATCH!'}")
        if not baseline_match:
            print(f"    -> baseline mismatch: full={full_row['nowcast_baseline']:.4f} "
                  f"vs truncated={trunc_row['nowcast_baseline']:.4f}")

    print()
    if all_match:
        print("ALL CHECKPOINTS MATCH -- pipeline confirmed safe for live/streaming use.")
    else:
        print("MISMATCH FOUND -- pipeline is NOT currently safe for live use as-is.")

    return all_match


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test live/streaming safety of the combined pipeline")
    parser.add_argument("--combined-day-with-leadin", required=True)
    parser.add_argument("--forecast-model", required=True)
    parser.add_argument("--check-times", nargs="+", required=True)
    args = parser.parse_args()

    combined_df = pd.read_parquet(args.combined_day_with_leadin)
    if "timestamp" in combined_df.columns:
        combined_df = combined_df.set_index("timestamp")

    model = joblib.load(args.forecast_model)
    check_times = [pd.Timestamp(t, tz="UTC") for t in args.check_times]

    check_live_safety(combined_df, model, check_times)
