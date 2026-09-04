"""
validate_solexs_completeness.py

Checks the ACTUAL COMPLETENESS of every raw SoLEXS .lc file, not just
whether it exists. The original 126-day gap analysis almost certainly
checked file existence only -- this checks real row counts, which
would catch PARTIAL-coverage days (a file exists, but only contains a
fraction of a full day's 86,400 expected 1-second samples) that a
pure existence check would silently miss.

This directly tests whether the newly-discovered NaN-spike days
(2025-11-18, 2025-12-05, 2025-12-09, etc.) are a genuine, previously
unquantified data gap -- not a bug introduced by our own processing.
"""

from pathlib import Path

import pandas as pd
from astropy.io import fits

EXPECTED_ROWS_PER_DAY = 86400


def check_one_day(lc_path: Path) -> dict:
    """Opens one .lc file and reports its actual row count and time span,
    WITHOUT doing the full byte-order fix / GTI processing -- just a
    fast completeness check."""
    if not lc_path.exists():
        return {"exists": False, "row_count": 0, "completeness_pct": 0.0}

    try:
        with fits.open(lc_path) as hdul:
            # Find the RATE table the same way load_solexs does
            data_hdu = None
            for hdu in hdul:
                if hdu.data is not None and getattr(hdu, "columns", None) is not None:
                    if "TIME" in hdu.columns.names and "COUNTS" in hdu.columns.names:
                        data_hdu = hdu
                        break
            if data_hdu is None and len(hdul) > 1:
                data_hdu = hdul[1]
            row_count = len(data_hdu.data) if data_hdu is not None else 0
    except Exception as exc:  # noqa: BLE001
        return {"exists": True, "row_count": 0, "completeness_pct": 0.0, "error": str(exc)}

    return {
        "exists": True,
        "row_count": row_count,
        "completeness_pct": 100 * row_count / EXPECTED_ROWS_PER_DAY,
    }


def validate_all_days(solexs_dir: Path, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Walks every day in [start_date, end_date], checks the raw .lc file's
    actual completeness, and returns a full report -- a much more
    reliable "which days are actually usable" reference than a simple
    existence check.
    """
    results = []
    for date in pd.date_range(start_date, end_date, freq="D"):
        yyyymmdd = date.strftime("%Y%m%d")
        yyyy_mm = date.strftime("%Y-%m")
        lc_path = solexs_dir / "lc" / yyyy_mm / f"{yyyymmdd}.lc"

        result = check_one_day(lc_path)
        result["date"] = date.date()
        results.append(result)

    report = pd.DataFrame(results).set_index("date")
    return report


def summarize(report: pd.DataFrame) -> None:
    total_days = len(report)
    fully_missing = (~report["exists"]).sum()
    present = report["exists"].sum()

    full_coverage = (report["completeness_pct"] >= 99).sum()
    partial_coverage = ((report["completeness_pct"] >= 50) & (report["completeness_pct"] < 99)).sum()
    severe_partial = ((report["completeness_pct"] > 0) & (report["completeness_pct"] < 50)).sum()

    print(f"Total days checked: {total_days}")
    print(f"Fully missing (no file at all): {fully_missing} ({100*fully_missing/total_days:.1f}%)")
    print(f"Present, full coverage (>=99%): {full_coverage} ({100*full_coverage/total_days:.1f}%)")
    print(f"Present, PARTIAL coverage (50-99%): {partial_coverage} ({100*partial_coverage/total_days:.1f}%)")
    print(f"Present, SEVERE partial coverage (<50%): {severe_partial} ({100*severe_partial/total_days:.1f}%)")

    print(f"\n=== This directly explains the original gap count ===")
    print(f"Original '126 missing days' analysis (file-existence only) would have found: {fully_missing} days")
    print(f"This analysis additionally finds {partial_coverage + severe_partial} days with a file present "
          f"but INCOMPLETE data -- invisible to a pure existence check.")

    if partial_coverage + severe_partial > 0:
        print(f"\n=== Worst partial-coverage days (real gaps, previously unflagged) ===")
        partial_days = report[(report["completeness_pct"] > 0) & (report["completeness_pct"] < 99)]
        print(partial_days.sort_values("completeness_pct").head(20)[["exists", "row_count", "completeness_pct"]])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validate raw SoLEXS file completeness, not just existence")
    parser.add_argument("--solexs-dir", required=True, help="Root of your solexs_sdd2 folder (containing lc/, gti/)")
    parser.add_argument("--start", default="2024-02-01")
    parser.add_argument("--end", default="2026-08-01")
    parser.add_argument("--output", default="solexs_completeness_report.parquet")
    args = parser.parse_args()

    report = validate_all_days(Path(args.solexs_dir), args.start, args.end)
    report.to_parquet(args.output)
    print(f"Full report saved to: {args.output}\n")
    summarize(report)
