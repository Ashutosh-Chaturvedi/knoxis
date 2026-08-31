"""
combine_day.py

Step 2 of the forecast training pipeline: combines one day's SoLEXS
(loaded fresh from FITS via load_solexs) with that day's already-merged
HEL1OS parquet (produced by manage_helios_dumps.py), into one aligned
per-day DataFrame -- the same shape DataIngestionPipeline produces, but
skipping the now-redundant HEL1OS FITS-parsing step since that work is
already done and sitting in the merged parquet.

Reuses the SAME merge_asof alignment + dtype-safe is_valid fix logic
already verified in DataIngestionPipeline._align() -- not a new design,
just repointed at a different HEL1OS input source.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import sys
notebook_dir = Path.cwd()
project_root = notebook_dir.parents[1] 
sys.path.append(str(project_root))
from ingestion.data_ingestion  import load_solexs, DataIngestionError


def combine_solexs_helios_day(
    solexs_lc_path: Path,
    solexs_gti_path: Path,
    helios_parquet_path: Path,
    merge_tolerance: str = "2s",
) -> pd.DataFrame:
    """
    Combines one day's SoLEXS + pre-merged HEL1OS data into one aligned
    DataFrame, indexed by timestamp (UTC), with columns:
        solexs_counts, solexs_is_valid, hel1os_ctr, hel1os_err, hel1os_is_valid

    Parameters
    ----------
    solexs_lc_path, solexs_gti_path : Path
        Raw SoLEXS files for this day (loaded via load_solexs, same as
        the normal ingestion pipeline).
    helios_parquet_path : Path
        Output of manage_helios_dumps.py for this day -- already has
        columns timestamp, hel1os_ctr, hel1os_err, is_valid.
    """
    solexs_df = load_solexs(solexs_lc_path, solexs_gti_path)
    solexs_df = solexs_df.rename(columns={"is_valid": "solexs_is_valid"})

    helios_df = pd.read_parquet(helios_parquet_path)
    if "timestamp" not in helios_df.columns:
        helios_df = helios_df.reset_index()
    helios_df = helios_df.rename(columns={"is_valid": "hel1os_is_valid"})

    required_helios_cols = {"timestamp", "hel1os_ctr", "hel1os_err", "hel1os_is_valid"}
    missing = required_helios_cols - set(helios_df.columns)
    if missing:
        raise DataIngestionError(
            f"HEL1OS parquet {helios_parquet_path} is missing expected columns: {missing}"
        )

    merged = pd.merge_asof(
        solexs_df.sort_values("timestamp"),
        helios_df.sort_values("timestamp")[["timestamp", "hel1os_ctr", "hel1os_err", "hel1os_is_valid"]],
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(merge_tolerance),
    )

    # Same fix as DataIngestionPipeline._align(): merge_asof leaves NaN
    # for unmatched rows, which upcasts the boolean hel1os_is_valid
    # column to float64. An unmatched row = no real HEL1OS data = invalid.
    merged["hel1os_is_valid"] = merged["hel1os_is_valid"].fillna(False).astype(bool)

    merged = merged.set_index("timestamp").sort_index()
    return merged


def combine_all_days(
    solexs_dir: Path,
    helios_merged_dir: Path,
    output_dir: Path,
    start_date: str,
    end_date: str,
    helios_filename_pattern: str = "{yyyymmdd}_hel1os_czt1.parquet",
) -> None:
    """
    Batch entry point: combines every day in [start_date, end_date] where
    BOTH a SoLEXS file pair and a merged HEL1OS parquet exist. Resumable
    (skips already-combined days) and resilient (one bad day is logged
    and skipped, never crashes the whole run) -- same discipline as
    every other batch script in this pipeline.

    SoLEXS files are expected in the real, confirmed nested structure:
        {solexs_dir}/lc/{yyyy}-{mm}/{yyyymmdd}.lc
        {solexs_dir}/gti/{yyyy}-{mm}/{yyyymmdd}.gti
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    failed_days = []
    n_combined = 0
    n_skipped_existing = 0
    n_skipped_missing = 0

    for date in pd.date_range(start_date, end_date, freq="D"):
        yyyymmdd = date.strftime("%Y%m%d")
        yyyy_mm = date.strftime("%Y-%m")
        out_path = output_dir / f"{yyyymmdd}_combined.parquet"

        if out_path.exists():
            n_skipped_existing += 1
            continue

        lc_path = solexs_dir / "lc" / yyyy_mm / f"{yyyymmdd}.lc"
        gti_path = solexs_dir / "gti" / yyyy_mm / f"{yyyymmdd}.gti"
        helios_path = helios_merged_dir / helios_filename_pattern.format(yyyymmdd=yyyymmdd)

        if not (lc_path.exists() and gti_path.exists() and helios_path.exists()):
            missing_bits = []
            if not lc_path.exists():
                missing_bits.append("SoLEXS .lc")
            if not gti_path.exists():
                missing_bits.append("SoLEXS .gti")
            if not helios_path.exists():
                missing_bits.append("HEL1OS merged parquet")
            n_skipped_missing += 1
            failed_days.append((yyyymmdd, f"missing: {', '.join(missing_bits)}"))
            continue

        try:
            combined = combine_solexs_helios_day(lc_path, gti_path, helios_path)
            combined.to_parquet(out_path)
            n_combined += 1
        except Exception as exc:  # noqa: BLE001 -- deliberately broad, see module docstring policy
            failed_days.append((yyyymmdd, str(exc)))
            print(f"FAILED (skipped, run continues): {yyyymmdd} -- {exc}")

    print(f"\nDone. {n_combined} days combined, {n_skipped_existing} already existed, "
          f"{n_skipped_missing} missing required input files, "
          f"{len(failed_days) - n_skipped_missing} genuine processing failures.")

    if failed_days:
        log_path = output_dir / "combine_failures.txt"
        log_path.write_text("\n".join(f"{d}: {reason}" for d, reason in failed_days))
        print(f"Full failure/missing log: {log_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Combine SoLEXS + merged HEL1OS into aligned per-day parquets")
    parser.add_argument("--solexs-dir", required=True, help="Folder with SoLEXS .lc/.gti files")
    parser.add_argument("--helios-merged-dir", required=True, help="Folder with {yyyymmdd}_hel1os_czt1.parquet files")
    parser.add_argument("--output-dir", required=True, help="Where to write combined per-day parquets")
    parser.add_argument("--start", required=True, help="Start date, e.g. 2024-02-01")
    parser.add_argument("--end", required=True, help="End date, e.g. 2026-08-01")
    args = parser.parse_args()

    combine_all_days(
        Path(args.solexs_dir), Path(args.helios_merged_dir), Path(args.output_dir),
        args.start, args.end,
    )
