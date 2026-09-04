"""
check_gti_validity_for_flagged_days.py

Checks the REAL GTI-valid fraction (not just raw row count) for the
specific dates flagged by the earlier NaN-spike investigation. If these
dates show low GTI-valid fractions, that fully and benignly explains
the NaN spike -- the instrument itself flagged large chunks of these
specific days as unreliable, exactly the same real signal already
established earlier in this project (SoLEXS GTI gaps were verified to
line up exactly with real NaN patterns). If GTI-valid fraction is HIGH
despite the feature-level NaN spike, that points to a genuine bug
somewhere in the processing pipeline instead (combine_day.py's
merge_asof, or the continuity buffer logic) -- worth knowing which
case we're in before assuming either way.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits


def fix_byte_order(arr: np.ndarray) -> np.ndarray:
    if arr.dtype.byteorder not in ("=", "|"):
        arr = arr.byteswap().view(arr.dtype.newbyteorder("="))
    return arr.astype(np.float64, copy=False)


def check_gti_validity(solexs_dir: Path, date_str: str) -> dict:
    """Loads the raw GTI file for one date and computes the real
    fraction of the day that's actually GTI-valid."""
    date = pd.Timestamp(date_str)
    yyyymmdd = date.strftime("%Y%m%d")
    yyyy_mm = date.strftime("%Y-%m")
    gti_path = solexs_dir / "gti" / yyyy_mm / f"{yyyymmdd}.gti"

    if not gti_path.exists():
        return {"gti_exists": False, "valid_seconds": 0, "valid_fraction_pct": 0.0}

    try:
        with fits.open(gti_path) as hdul:
            gti_hdu = hdul["GTI"] if "GTI" in hdul else hdul[1]
            starts = fix_byte_order(gti_hdu.data["START"])
            stops = fix_byte_order(gti_hdu.data["STOP"])
        total_valid_seconds = float(np.sum(stops - starts))
        return {
            "gti_exists": True,
            "n_windows": len(starts),
            "valid_seconds": total_valid_seconds,
            "valid_fraction_pct": 100 * total_valid_seconds / 86400,
        }
    except Exception as exc:  # noqa: BLE001
        return {"gti_exists": True, "error": str(exc), "valid_fraction_pct": None}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--solexs-dir", required=True)
    args = parser.parse_args()

    # The 10 dates flagged by the earlier notebook investigation as
    # having elevated NaN rates, previously unflagged by the raw
    # existence-only gap analysis.
    flagged_dates = [
        "2025-11-18", "2025-12-05", "2025-12-09", "2025-12-17",
        "2025-12-22", "2026-01-11", "2026-01-17", "2026-02-03",
        "2026-02-05", "2026-02-10",
    ]

    print(f"{'Date':<12} {'GTI valid %':>12}  {'Windows':>8}")
    print("-" * 40)
    results = []
    for d in flagged_dates:
        r = check_gti_validity(Path(args.solexs_dir), d)
        results.append({"date": d, **r})
        vf = r.get("valid_fraction_pct")
        vf_str = f"{vf:.1f}%" if vf is not None else "ERROR"
        print(f"{d:<12} {vf_str:>12}  {r.get('n_windows', '-'):>8}")

    report = pd.DataFrame(results)
    avg_valid = report["valid_fraction_pct"].mean()
    print(f"\nAverage GTI-valid fraction across these 10 flagged days: {avg_valid:.1f}%")
    print("(Compare against a normal day, which should be close to 90-100% valid)")

    if avg_valid < 70:
        print("\n-> LOW GTI validity confirms this is real, instrument-flagged bad data,")
        print("   the same legitimate signal already established earlier in this project.")
        print("   Not a pipeline bug -- a genuine, previously under-quantified data gap.")
    else:
        print("\n-> GTI validity is high despite the feature-level NaN spike -- this points")
        print("   to something in the PROCESSING pipeline instead, not the raw source data.")
        print("   Worth investigating combine_day.py and the continuity buffer logic next.")
