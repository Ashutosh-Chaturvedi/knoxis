"""
ingest_new_day.py

Turns "I just manually downloaded a new real day's SoLEXS + HEL1OS
files from PRADAN" into a single command that updates the live data
file the API reads from.

This is NOT live telemetry automation -- PRADAN requires authenticated
portal access with no documented public API for automated polling (a
real, published research project using the same portal for a different
Aditya-L1 instrument documents the same manual-download workflow). This
script instead makes the MANUAL step -> LIVE SYSTEM step a single
command instead of several separately-run pipeline stages.

Reuses the already-tested pipeline logic directly:
    - manage_helios_dumps.py's find_all_dump_files/resolve_day/merge_day
      for reconciling the day's HEL1OS dump zips
    - combine_day.py's combine_solexs_helios_day for aligning with SoLEXS

No labeling here -- labeling only applies to building the TRAINING set,
using the NOAA catalog to look into a KNOWN future. For a new, genuinely
current day, there is no known future yet -- this just produces fresh
input features for the ALREADY-TRAINED model to predict from.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from manage_helios_dumps import find_all_dump_files, resolve_day, merge_day
from combine_day import combine_solexs_helios_day


def ingest_new_day(
    helios_zip_dir: Path,
    solexs_lc_path: Path,
    solexs_gti_path: Path,
    live_data_file: Path,
    day_str: str,
    rolling_window_days: int = 3,
) -> None:
    """
    Reconciles and merges one new real day's raw files, then appends it
    onto the live data file the API reads from -- atomically, so the
    API never reads a partially-written file (the same real race
    condition found and fixed in live_feed_simulator.py applies here
    too, since this file is read by the same API).
    """
    print(f"=== Ingesting new day: {day_str} ===")

    # --- Step 1: reconcile the day's HEL1OS dump zips (reusing the
    # exact same tested logic as the full-dataset pipeline) ---
    print("Resolving HEL1OS dump files...")
    dumps = find_all_dump_files(helios_zip_dir)
    if not dumps:
        raise ValueError(f"No HEL1OS dump files found in {helios_zip_dir}")

    winners, resolution_log = resolve_day(dumps)
    print(f"  {len(dumps)} dump file(s) found, {len(winners)} winner(s) after resolution.")
    for entry in resolution_log:
        print(f"    {entry['reason']}: kept {Path(entry['chosen']).name}")

    bad_files_log: list[str] = []
    helios_output_dir = live_data_file.parent / "_helios_merge_tmp"
    merged_helios_path = merge_day(winners, day_str, helios_output_dir, bad_files_log)
    if merged_helios_path is None:
        raise ValueError(f"HEL1OS merge produced no usable data for {day_str}. "
                          f"Bad files: {bad_files_log}")
    if bad_files_log:
        print(f"  WARNING: {len(bad_files_log)} HEL1OS file(s) failed to parse: {bad_files_log}")

    # --- Step 2: align with SoLEXS (reusing the exact same tested logic) ---
    print("Combining with SoLEXS...")
    new_day_df = combine_solexs_helios_day(solexs_lc_path, solexs_gti_path, merged_helios_path)
    print(f"  New day combined: {len(new_day_df)} rows, "
          f"{new_day_df.index.min()} -> {new_day_df.index.max()}")

    # --- Step 3: append onto the existing live data file ---
    if live_data_file.exists():
        existing_df = pd.read_parquet(live_data_file)
        if existing_df.index.name != "timestamp" and "timestamp" in existing_df.columns:
            existing_df = existing_df.set_index("timestamp")
        combined = pd.concat([existing_df, new_day_df])
        combined = combined[~combined.index.duplicated(keep="last")]  # new data wins on overlap
        combined = combined.sort_index()
        print(f"  Appended onto existing file ({len(existing_df)} -> {len(combined)} rows).")
    else:
        combined = new_day_df
        print(f"  No existing live data file -- starting fresh with this day.")

    # --- Step 4: trim to a rolling window, so the file doesn't grow
    # unbounded and every request doesn't get slower forever (a real,
    # documented limitation of always reprocessing the whole file) ---
    cutoff = combined.index.max() - pd.Timedelta(days=rolling_window_days)
    before_trim = len(combined)
    combined = combined[combined.index >= cutoff]
    if before_trim != len(combined):
        print(f"  Trimmed to last {rolling_window_days} days ({before_trim} -> {len(combined)} rows).")

    # --- Step 5: ATOMIC write -- same fix as live_feed_simulator.py.
    # This file is read by the live API; a direct write risks the API
    # reading a partially-written file (confirmed: 225/500 real read
    # failures under concurrent access without this fix). ---
    tmp_path = live_data_file.with_suffix(live_data_file.suffix + ".tmp")
    combined.to_parquet(tmp_path)
    tmp_path.replace(live_data_file)
    print(f"Live data file updated: {live_data_file} ({len(combined)} rows)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest one new real day into the live data file")
    parser.add_argument("--helios-zip-dir", required=True,
                         help="Folder containing this day's raw HEL1OS dump zips")
    parser.add_argument("--solexs-lc", required=True)
    parser.add_argument("--solexs-gti", required=True)
    parser.add_argument("--live-data-file", required=True,
                         help="The file your API's KNOXIS_DATA_FILE points to")
    parser.add_argument("--day", required=True, help="Date being ingested, YYYYMMDD")
    parser.add_argument("--rolling-window-days", type=int, default=3)
    args = parser.parse_args()

    ingest_new_day(
        Path(args.helios_zip_dir), Path(args.solexs_lc), Path(args.solexs_gti),
        Path(args.live_data_file), args.day, args.rolling_window_days,
    )
