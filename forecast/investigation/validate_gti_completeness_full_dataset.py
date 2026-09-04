"""
validate_gti_completeness_full_dataset.py

Extends the spot-check to the FULL dataset: computes the real GTI-valid
fraction for every one of the ~912 target days, not just the 10 days
that happened to show up in one investigation window. This produces
the true, complete picture of data reliability -- replacing the
original "126 missing days" metric (which only checked file existence)
with a proper completeness measure that also catches days where a file
exists but is substantially GTI-invalid.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits


def fix_byte_order(arr: np.ndarray) -> np.ndarray:
    if arr.dtype.byteorder not in ("=", "|"):
        arr = arr.byteswap().view(arr.dtype.newbyteorder("="))
    return arr.astype(np.float64, copy=False)


def check_one_day(solexs_dir: Path, date: pd.Timestamp) -> dict:
    yyyymmdd = date.strftime("%Y%m%d")
    yyyy_mm = date.strftime("%Y-%m")
    gti_path = solexs_dir / "gti" / yyyy_mm / f"{yyyymmdd}.gti"
    lc_path = solexs_dir / "lc" / yyyy_mm / f"{yyyymmdd}.lc"

    if not lc_path.exists():
        return {"lc_exists": False, "gti_exists": False, "valid_fraction_pct": 0.0, "n_windows": 0}

    if not gti_path.exists():
        # LC file present but no GTI file at all -- worth knowing about
        # separately, since it's a different failure mode than either
        # "fully missing" or "GTI-degraded".
        return {"lc_exists": True, "gti_exists": False, "valid_fraction_pct": np.nan, "n_windows": 0}

    try:
        with fits.open(gti_path) as hdul:
            gti_hdu = hdul["GTI"] if "GTI" in hdul else hdul[1]
            starts = fix_byte_order(gti_hdu.data["START"])
            stops = fix_byte_order(gti_hdu.data["STOP"])
        total_valid_seconds = float(np.sum(stops - starts))
        return {
            "lc_exists": True,
            "gti_exists": True,
            "n_windows": len(starts),
            "valid_fraction_pct": 100 * total_valid_seconds / 86400,
        }
    except Exception as exc:  # noqa: BLE001
        return {"lc_exists": True, "gti_exists": True, "error": str(exc), "valid_fraction_pct": np.nan, "n_windows": 0}


def run(solexs_dir: Path, start_date: str, end_date: str, output_path: Path) -> pd.DataFrame:
    results = []
    dates = pd.date_range(start_date, end_date, freq="D")
    print(f"Checking {len(dates)} days... (this will take a few minutes)")

    for i, date in enumerate(dates):
        r = check_one_day(solexs_dir, date)
        r["date"] = date.date()
        results.append(r)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(dates)} checked...")

    report = pd.DataFrame(results).set_index("date")
    report.to_parquet(output_path)
    print(f"\nFull report saved to: {output_path}")
    return report


def summarize(report: pd.DataFrame) -> None:
    total_days = len(report)
    fully_missing = (~report["lc_exists"]).sum()
    lc_but_no_gti = (report["lc_exists"] & ~report["gti_exists"]).sum()
    present_with_gti = (report["lc_exists"] & report["gti_exists"])

    full_validity = (report.loc[present_with_gti, "valid_fraction_pct"] >= 90).sum()
    moderate_degraded = ((report.loc[present_with_gti, "valid_fraction_pct"] >= 50) &
                          (report.loc[present_with_gti, "valid_fraction_pct"] < 90)).sum()
    severely_degraded = (report.loc[present_with_gti, "valid_fraction_pct"] < 50).sum()

    print("\n" + "=" * 60)
    print("FULL DATASET GTI COMPLETENESS SUMMARY")
    print("=" * 60)
    print(f"Total days in range: {total_days}")
    print(f"Fully missing (no .lc file at all): {fully_missing} ({100*fully_missing/total_days:.1f}%)")
    print(f"LC file present but NO GTI file: {lc_but_no_gti} ({100*lc_but_no_gti/total_days:.1f}%)")
    print(f"\nOf the {present_with_gti.sum()} days with both files present:")
    print(f"  Full validity (>=90%): {full_validity} ({100*full_validity/present_with_gti.sum():.1f}%)")
    print(f"  MODERATELY degraded (50-90%): {moderate_degraded} ({100*moderate_degraded/present_with_gti.sum():.1f}%)")
    print(f"  SEVERELY degraded (<50%): {severely_degraded} ({100*severely_degraded/present_with_gti.sum():.1f}%)")

    print(f"\n=== Comparison to the original gap metric ===")
    print(f"Original '126 missing days' figure only counted: {fully_missing} fully-missing days")
    print(f"This analysis additionally reveals: {moderate_degraded + severely_degraded} "
          f"'present but substantially GTI-invalid' days -- a real, previously "
          f"unquantified reliability gap on top of the known 126.")

    if severely_degraded > 0:
        print(f"\n=== Worst {min(20, severely_degraded)} severely-degraded days ===")
        worst = report[present_with_gti].sort_values("valid_fraction_pct").head(20)
        print(worst[["valid_fraction_pct", "n_windows"]].to_string())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--solexs-dir", required=True)
    parser.add_argument("--start", default="2024-02-01")
    parser.add_argument("--end", default="2026-08-01")
    parser.add_argument("--output", default="gti_completeness_full_report.parquet")
    args = parser.parse_args()

    report = run(Path(args.solexs_dir), args.start, args.end, Path(args.output))
    summarize(report)
