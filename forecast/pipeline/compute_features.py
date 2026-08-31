"""
compute_features.py

Step 3 of the forecast training pipeline: computes 9 engineered features
over rolling windows for one combined day, and batches this across all
combined days.

Now includes HEL1OS saturation handling: `hel1os_ctr == 0` AND
`hel1os_err == 0` during elevated SoLEXS flux is a real, confirmed
instrument behavior (CZT1 detector saturation/dead-time at extreme
photon flux), verified independently on two real flares -- Feb 12, 2024
(M6) and June 21, 2026 (M6.8). `hel1os_is_valid` stays True throughout
these windows (the instrument reports them as good acquisitions), so
this can't be caught by the existing validity flag -- it needs its own
detection and masking, or it silently teaches a model "near-zero HEL1OS
= low activity" during exactly the highest-stakes events, where it
actually means the opposite.

Features (9 total):
    1. solexs_mean
    2. solexs_slope       -- endpoint slope, not full regression (see note)
    3. solexs_std
    4. hel1os_mean         -- windowed SUM / count of VALID (non-saturated,
                              non-GTI-excluded) seconds in the window --
                              NOT a fixed window-duration denominator,
                              which would silently dilute the mean
                              whenever saturated seconds are masked out
    5. hel1os_slope
    6. hardness_ratio      -- windowed-sum HEL1OS / windowed-sum SoLEXS,
                              computed AFTER saturation masking
    7. hr_rate_of_change
    8. peak_to_mean_ratio
    9. hel1os_saturation_fraction -- NEW: fraction of the window flagged
                              as suspected saturation. A fraction, not a
                              single boolean, so it's directly usable as
                              a numeric feature and tells the model HOW
                              MUCH of the window was affected.

Note on 'slope': endpoint slope (current value minus value one window
ago, divided by window duration), not a full least-squares regression --
chosen for vectorized speed at 912-day scale. A tree-based model like
Random Forest doesn't need a precise linear fit, just directional signal.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def flag_suspected_saturation(
    df: pd.DataFrame,
    saturation_multiplier: float = 20.0,
) -> pd.Series:
    """
    Flags CONTIGUOUS RUNS of hel1os_ctr==0 AND hel1os_err==0 (the
    confirmed saturation signature -- distinct from HEL1OS's normal
    ~75-80% baseline zero-rate, where err is still a real nonzero
    statistical uncertainty) as suspected saturation, if ANY second
    within that run has elevated SoLEXS flux.

    Design history -- why block-based, not per-second: an earlier
    per-second version (require elevated flux AT THE SAME SECOND as the
    zero readout) was checked against two real flares and badly
    under-detected one of them (only 4.4% of a confirmed 40-minute
    zero-region cleared the threshold). Critically, the SAME real
    zero-region on the OTHER flare spanned SoLEXS values from near-quiet
    (5 counts) to peak (3902 counts) -- meaning a single contiguous
    null-region can include both genuinely elevated-flux seconds and
    seconds where flux has already dropped back near baseline. This is
    consistent with a real detector recovery-lag mechanism (saturates at
    peak, stays in a null state through part of the decline) rather than
    an instantaneous per-second effect, and also connects to an earlier
    finding in this project: CZT1 and CZT2 go to zero in near-perfect
    synchrony (~81% both-zero, ~0% either-alone), suggesting a shared,
    systemic onboard cause layered on top of simple flux-driven
    saturation, not something a single-second flux check can capture.

    Treating each contiguous zero-run as ONE atomic event -- flagging
    the entire run if any part of it clears the flux threshold --
    matches this mechanism far better.

    "Elevated" is this day's own MEDIAN valid SoLEXS count times
    saturation_multiplier (not a percentile -- see prior design note:
    percentiles get dragged around by how much of a day a given flare
    happens to occupy, medians don't).

    Vectorized via groupby().transform('max') rather than a per-run
    Python loop, for speed at 912-day scale.
    """
    valid_solexs = df["solexs_counts"].where(df["solexs_is_valid"])
    elevated_threshold = valid_solexs.median() * saturation_multiplier

    zero_ctr_and_err = (df["hel1os_ctr"] == 0) & (df["hel1os_err"] == 0)
    elevated_flux = (df["solexs_counts"] > elevated_threshold).astype(int)

    # Assign a unique group id to each contiguous run of identical
    # zero_ctr_and_err values (same trick used in nowcast.py's
    # enforce_sustained_duration).
    group_id = (zero_ctr_and_err != zero_ctr_and_err.shift(fill_value=False)).cumsum()

    # For every row, look up whether ANY row in its own contiguous group
    # has elevated flux -- vectorized broadcast-back-to-every-row-in-group,
    # not a Python loop over each group.
    group_has_elevated_flux = elevated_flux.groupby(group_id).transform("max").astype(bool)

    return zero_ctr_and_err & group_has_elevated_flux


def compute_day_features(
    df: pd.DataFrame,
    window_minutes: int = 30,
    step_minutes: int = 5,
    min_valid_fraction: float = 0.5,
    saturation_multiplier: float = 20.0,
    saturation_nan_threshold: float = 0.10,
) -> pd.DataFrame:
    """
    Computes all 9 features over rolling windows for one combined day.

    Parameters
    ----------
    df : pd.DataFrame
        Output of combine_day.py -- indexed by timestamp (1Hz), columns
        solexs_counts, solexs_is_valid, hel1os_ctr, hel1os_err, hel1os_is_valid.
    """
    window_str = f"{window_minutes}min"
    window_seconds = window_minutes * 60
    min_periods = int(window_seconds * min_valid_fraction)

    solexs = df["solexs_counts"].where(df["solexs_is_valid"])

    # --- Saturation detection, then mask affected seconds BEFORE any
    # windowed computation touches them -- masking after the fact would
    # be too late, the damage (zeros pulling sums/ratios down) is done
    # the moment a rolling window includes them. ---
    suspected_saturation = flag_suspected_saturation(df, saturation_multiplier)
    hel1os = df["hel1os_ctr"].where(df["hel1os_is_valid"] & ~suspected_saturation)

    # --- SoLEXS features (unaffected by any of this -- rolling on raw
    # per-second values, SoLEXS isn't sparse like HEL1OS) ---
    solexs_mean = solexs.rolling(window_str, min_periods=min_periods).mean()
    solexs_std = solexs.rolling(window_str, min_periods=min_periods).std()
    solexs_max = solexs.rolling(window_str, min_periods=min_periods).max()
    solexs_slope = (solexs - solexs.shift(window_seconds)) / window_seconds
    peak_to_mean_ratio = solexs_max / solexs_mean

    # --- HEL1OS features: windowed SUM (not raw per-second mean, since
    # HEL1OS is ~75-80% zero at 1s resolution even normally -- summing
    # over the window is what makes the signal usable at all). ---
    hel1os_sum = hel1os.rolling(window_str, min_periods=1).sum()

    # Corrected denominator: count of ACTUALLY-VALID seconds in the
    # window (excludes both GTI-invalid AND saturated seconds), not a
    # fixed window_seconds -- otherwise masking silently dilutes the
    # mean whenever a window contains masked seconds.
    valid_seconds_in_window = hel1os.rolling(window_str, min_periods=1).count()
    hel1os_mean = (hel1os_sum / valid_seconds_in_window).replace([np.inf, -np.inf], np.nan)
    hel1os_mean = hel1os_mean.where(valid_seconds_in_window > 0)  # explicit 0-valid-seconds -> NaN

    hel1os_slope = (hel1os_sum - hel1os_sum.shift(window_seconds)) / window_seconds

    # --- Hardness Ratio ---
    solexs_sum = solexs.rolling(window_str, min_periods=min_periods).sum()
    hardness_ratio = hel1os_sum / solexs_sum
    hr_rate_of_change = (hardness_ratio - hardness_ratio.shift(window_seconds)) / window_seconds

    # --- Fraction of each window flagged as suspected saturation.
    # Rolling mean of a boolean series = fraction of True values in the
    # window -- a clean, direct way to get this. ---
    hel1os_saturation_fraction = suspected_saturation.astype(float).rolling(window_str, min_periods=1).mean()

    # --- THE ACTUAL FIX for the collapsed-HR problem (testing revealed
    # masking a rolling SUM does nothing -- a zero contributes nothing
    # to a sum whether it's excluded or included, so hel1os_sum and
    # hardness_ratio were UNCHANGED by masking above; only hel1os_mean's
    # count-based denominator was genuinely fixed by it). ---
    #
    # The real fix, consistent with how this project has always handled
    # untrustworthy data (never fabricate a value -- label it honestly
    # as missing instead, same as SoLEXS's GTI-invalid rows): when a
    # window's saturation fraction is too high, we don't actually KNOW
    # what HEL1OS was doing for enough of that window to trust anything
    # derived from it. NaN it out rather than report a number computed
    # from data we know is corrupted.
    heavily_saturated = hel1os_saturation_fraction > saturation_nan_threshold
    hel1os_mean = hel1os_mean.where(~heavily_saturated)
    hel1os_slope = hel1os_slope.where(~heavily_saturated)
    hardness_ratio = hardness_ratio.where(~heavily_saturated)
    hr_rate_of_change = hr_rate_of_change.where(~heavily_saturated)

    features = pd.DataFrame({
        "solexs_mean": solexs_mean,
        "solexs_slope": solexs_slope,
        "solexs_std": solexs_std,
        "hel1os_mean": hel1os_mean,
        "hel1os_slope": hel1os_slope,
        "hardness_ratio": hardness_ratio,
        "hr_rate_of_change": hr_rate_of_change,
        "peak_to_mean_ratio": peak_to_mean_ratio,
        "hel1os_saturation_fraction": hel1os_saturation_fraction,
    })

    # Downsample to the requested stride using direct positional slicing
    # (exact, since data is a clean regular 1Hz series) rather than
    # .resample(), which introduced a real off-by-bucket bug earlier --
    # positional slicing lands exactly on 00:00, 00:05, 00:10, ... with
    # zero rounding error.
    step_seconds = step_minutes * 60
    features = features.iloc[::step_seconds]

    return features


def compute_all_features(
    combined_days_dir: Path,
    output_dir: Path,
    window_minutes: int = 30,
    step_minutes: int = 5,
) -> None:
    """
    Batch entry point: computes features for every combined-day parquet
    in combined_days_dir, CONTINUOUSLY -- carrying a trailing buffer
    (the last window_minutes of raw data) from each day forward into the
    next, so rolling windows have real trailing history from 00:00:00
    onward instead of starting cold every single day.

    Without this, EVERY day's first ~window_minutes shows NaN features
    purely from lacking history -- not a real data problem, but a
    systematic ~4%/day loss across the whole dataset (confirmed via the
    sanity check: NaN rate was ~equal between QUIET and flare-labeled
    windows, consistent with a uniform per-day artifact rather than
    anything tied to real data quality).

    The buffer is only used across a GENUINELY CONTIGUOUS day boundary
    (previous day's last timestamp is exactly one calendar day before
    this day's first timestamp, no gap). After a real data gap (e.g. the
    126 missing SoLEXS days), the next available day legitimately starts
    cold again -- there's no real continuous history to borrow from.

    Still resumable: skips any day whose output file already exists.
    NOTE: because feature values change with this fix, existing labeled
    output (Step 4) becomes STALE and must be regenerated after
    re-running this -- delete old features/ AND labeled/ output before
    re-running both steps, or the resumable "skip if exists" logic will
    silently keep the old, uncorrected results.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    day_files = sorted(combined_days_dir.glob("*_combined.parquet"))
    failed_days = []
    n_done = 0
    n_skipped = 0
    n_continuous = 0
    n_cold_start = 0

    carry_buffer = None       # trailing raw data from the previously PROCESSED day
    carry_buffer_end = None   # that previous day's last real timestamp

    for day_path in day_files:
        day_str = day_path.stem.replace("_combined", "")
        out_path = output_dir / f"{day_str}_features.parquet"

        try:
            df = pd.read_parquet(day_path)
            if "timestamp" not in df.columns:
                df = df.reset_index()
            df = df.set_index("timestamp")
        except Exception as exc:  # noqa: BLE001
            failed_days.append((day_str, str(exc)))
            print(f"FAILED to load (skipped, run continues): {day_str} -- {exc}")
            continue

        current_day_start = df.index.min().normalize()

        # Is the carried buffer genuinely contiguous with THIS day?
        is_contiguous = (
            carry_buffer is not None
            and carry_buffer_end is not None
            and carry_buffer_end.normalize() == current_day_start - pd.Timedelta(days=1)
        )

        if out_path.exists():
            n_skipped += 1
            # Still need to update the carry buffer even on a skipped
            # day, so the NEXT day's computation stays correct.
            carry_buffer = df.tail(int((window_minutes + 5) * 60))
            carry_buffer_end = df.index.max()
            continue

        try:
            if is_contiguous:
                lead_in = carry_buffer
                combined_with_lead_in = pd.concat([lead_in, df])
                n_continuous += 1
            else:
                combined_with_lead_in = df
                n_cold_start += 1

            features = compute_day_features(combined_with_lead_in, window_minutes, step_minutes)
            # Trim off any borrowed lead-in rows -- only save THIS day's
            # own windows, so each day still produces exactly one file
            # with no duplication across day boundaries.
            features_this_day = features[features.index >= current_day_start]
            features_this_day.to_parquet(out_path)
            n_done += 1
        except Exception as exc:  # noqa: BLE001
            failed_days.append((day_str, str(exc)))
            print(f"FAILED (skipped, run continues): {day_str} -- {exc}")

        # Carry forward the tail of THIS day's raw data for the next iteration.
        carry_buffer = df.tail(int((window_minutes + 5) * 60))
        carry_buffer_end = df.index.max()

    print(f"\nDone. {n_done} days processed ({n_continuous} with real trailing "
          f"history, {n_cold_start} cold-started after a gap or as the first day), "
          f"{n_skipped} already existed, {len(failed_days)} failed.")
    if failed_days:
        log_path = output_dir / "feature_failures.txt"
        log_path.write_text("\n".join(f"{d}: {reason}" for d, reason in failed_days))
        print(f"Failure log: {log_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute rolling-window forecast features")
    parser.add_argument("--combined-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--window-minutes", type=int, default=30)
    parser.add_argument("--step-minutes", type=int, default=5)
    args = parser.parse_args()

    compute_all_features(
        Path(args.combined_dir), Path(args.output_dir),
        args.window_minutes, args.step_minutes,
    )
