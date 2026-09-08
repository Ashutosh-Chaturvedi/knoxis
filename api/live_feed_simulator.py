"""
live_feed_simulator.py

Simulates a continuously-updating live data feed by replaying a real
historical day at a configurable speed -- without this, latest_data.parquet
is a static file and the API's "live" status never actually changes.

How it works: starts with a small amount of "lead-in" history already
in the file, then periodically appends the next chunk of real historical
data, advancing the "current" timestamp forward -- exactly like a real
feed would grow over time, just compressed into minutes instead of
playing out over real hours.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd


def run_simulation(
    source_file: Path,
    output_file: Path,
    initial_leadin_hours: float = 24.0,
    step_minutes: float = 5.0,
    interval_seconds: float = 3.0,
    speed_label: str = "demo",
) -> None:
    """
    Replays source_file (a real combined day, or multiple concatenated
    days) into output_file, advancing forward in fixed steps on a timer.

    Parameters
    ----------
    source_file : the full historical data to replay from (e.g. a
        real combine_day.py output, or a multi-day concatenation)
    output_file : the file the API's KNOXIS_DATA_FILE should point to
    initial_leadin_hours : how much history to seed the file with before
        the simulation starts advancing (needs to be >=24h for the
        nowcast engine's baseline to be meaningful from the first step)
    step_minutes : how far the "current" timestamp advances per tick
    interval_seconds : real wall-clock seconds between ticks
    """
    full_data = pd.read_parquet(source_file)
    if full_data.index.name != "timestamp" and "timestamp" in full_data.columns:
        full_data = full_data.set_index("timestamp")
    full_data = full_data.sort_index()

    start_time = full_data.index.min()
    leadin_end = start_time + pd.Timedelta(hours=initial_leadin_hours)
    end_time = full_data.index.max()

    if leadin_end >= end_time:
        raise ValueError(
            f"source_file only spans {(end_time - start_time)}, not enough for a "
            f"{initial_leadin_hours}h lead-in plus room to advance. Use a longer source file."
        )

    print(f"[{speed_label}] Simulating a live feed from {source_file.name}")
    print(f"  Full range available: {start_time} -> {end_time}")
    print(f"  Starting with {initial_leadin_hours}h lead-in, advancing "
          f"{step_minutes} min every {interval_seconds}s of real time")
    print(f"  Writing to: {output_file}")
    print("  Press Ctrl+C to stop.\n")

    current_time = leadin_end
    while current_time <= end_time:
        window = full_data.loc[:current_time]
        # Atomic write: save to a temp file first, then rename onto the
        # real target. A rename is atomic at the OS level (both Windows
        # and Linux), so a reader (the API) can only ever see the
        # complete OLD file or the complete NEW file -- never a
        # partially-written one. Writing directly to output_file was
        # tested and found to cause real read failures ~45% of the time
        # under concurrent access -- this fixes that.
        tmp_path = output_file.with_suffix(output_file.suffix + ".tmp")
        window.to_parquet(tmp_path)
        tmp_path.replace(output_file)
        print(f"  [{pd.Timestamp.now().strftime('%H:%M:%S')}] "
              f"latest_data.parquet now covers up to {current_time} "
              f"({len(window)} rows)")

        current_time += pd.Timedelta(minutes=step_minutes)
        time.sleep(interval_seconds)

    print("\nReached the end of the source data. Simulation complete.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Simulate a live data feed for the Knoxis API")
    parser.add_argument("--source-file", required=True,
                         help="Real historical data to replay (e.g. june21_with_leadin.parquet)")
    parser.add_argument("--output-file", default="latest_data.parquet",
                         help="Where to write the growing 'live' file (point KNOXIS_DATA_FILE here)")
    parser.add_argument("--leadin-hours", type=float, default=24.0)
    parser.add_argument("--step-minutes", type=float, default=5.0)
    parser.add_argument("--interval-seconds", type=float, default=3.0,
                         help="Real seconds between updates -- lower = faster demo")
    args = parser.parse_args()

    run_simulation(
        Path(args.source_file), Path(args.output_file),
        args.leadin_hours, args.step_minutes, args.interval_seconds,
    )