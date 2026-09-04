"""
final_validation_suite.py

Comprehensive, final validation of the master training table -- goes
beyond the earlier sanity checks by testing BREADTH: a random sample of
many real flares across the whole dataset (not just the 2 famous ones),
plus basic feature sanity checks that could catch a real, undiscovered
bug no synthetic test would think to check for.
"""

from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "solexs_mean", "solexs_slope", "solexs_std",
    "hel1os_mean", "hel1os_slope",
    "hardness_ratio", "hr_rate_of_change",
    "peak_to_mean_ratio", "hel1os_saturation_fraction",
]


def check_random_flare_sample(master_table: pd.DataFrame, catalog: pd.DataFrame,
                               n_samples: int = 30, random_state: int = 42) -> dict:
    """Picks N random real flares from the catalog and verifies each
    one's label appears correctly in the windows leading up to it."""
    sample = catalog.sample(n=min(n_samples, len(catalog)), random_state=random_state)

    results = []
    for _, flare in sample.iterrows():
        begin_time = flare["begin_time"]
        expected_class = flare["goes_class"]

        check_time = begin_time - pd.Timedelta(minutes=5)
        nearby = master_table.loc[
            (master_table.index >= check_time - pd.Timedelta(minutes=10)) &
            (master_table.index <= check_time)
        ]

        if len(nearby) == 0:
            results.append({"begin_time": begin_time, "expected_class": expected_class,
                             "status": "NO_DATA (likely a known gap day)"})
            continue

        found_match = (nearby["label_class"] == expected_class).any()
        found_any_flare = (nearby["label_class"] != "QUIET").any()

        if found_match:
            status = "PASS"
        elif found_any_flare:
            status = "PARTIAL (flagged as flare, but different/stronger class nearby)"
        else:
            status = "FAIL (no flare label found at all)"

        results.append({"begin_time": begin_time, "expected_class": expected_class, "status": status})

    results_df = pd.DataFrame(results)
    n_pass = (results_df["status"] == "PASS").sum()
    n_partial = results_df["status"].str.startswith("PARTIAL").sum()
    n_fail = results_df["status"].str.startswith("FAIL").sum()
    n_no_data = results_df["status"].str.startswith("NO_DATA").sum()

    print(f"Random flare spot-check ({len(results_df)} flares sampled):")
    print(f"  PASS: {n_pass}  PARTIAL: {n_partial}  FAIL: {n_fail}  NO_DATA (known gap): {n_no_data}")

    if n_fail > 0:
        print(f"\n  Flares with a genuine FAIL (worth investigating individually):")
        print(results_df[results_df["status"].str.startswith("FAIL")].to_string())

    return {"results": results_df, "n_pass": n_pass, "n_fail": n_fail, "n_no_data": n_no_data}


def check_feature_sanity(master_table: pd.DataFrame) -> dict:
    """Basic sanity checks for impossible values."""
    issues = []

    for col in ["solexs_mean", "hel1os_mean"]:
        n_negative = (master_table[col] < 0).sum()
        if n_negative > 0:
            issues.append(f"{col}: {n_negative} rows with impossible NEGATIVE values")

    below_one = (master_table["peak_to_mean_ratio"] < 1.0).sum()
    if below_one > 0:
        issues.append(f"peak_to_mean_ratio: {below_one} rows below 1.0 (impossible)")

    n_negative_hr = (master_table["hardness_ratio"] < 0).sum()
    if n_negative_hr > 0:
        issues.append(f"hardness_ratio: {n_negative_hr} rows with impossible NEGATIVE values")

    out_of_range = ((master_table["hel1os_saturation_fraction"] < 0) |
                     (master_table["hel1os_saturation_fraction"] > 1)).sum()
    if out_of_range > 0:
        issues.append(f"hel1os_saturation_fraction: {out_of_range} rows outside valid [0,1] range")

    for col in FEATURE_COLUMNS:
        n_inf = np.isinf(master_table[col]).sum()
        if n_inf > 0:
            issues.append(f"{col}: {n_inf} INFINITE values (likely an unguarded divide-by-zero)")

    print(f"\nFeature sanity checks ({len(FEATURE_COLUMNS)} columns checked):")
    if issues:
        print(f"  {len(issues)} ISSUE(S) FOUND:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("  No impossible values found.")

    return {"issues": issues}


def check_saturation_fix_is_live(master_table: pd.DataFrame,
                                  known_flare_time: str = "2026-06-21 19:29:00") -> dict:
    """Regression check: confirms the saturation-handling fix is
    reflected in the CURRENT file, not a stale pre-fix version."""
    flare_ts = pd.Timestamp(known_flare_time, tz="UTC")
    window = master_table.loc[flare_ts - pd.Timedelta(minutes=15):flare_ts + pd.Timedelta(minutes=25)]

    if len(window) == 0:
        print(f"\nSaturation-fix regression check: NO DATA at {known_flare_time}.")
        return {"status": "NO_DATA"}

    has_nan_hr = window["hardness_ratio"].isna().any()
    has_sat_fraction = (window["hel1os_saturation_fraction"] > 0).any()

    print(f"\nSaturation-fix regression check (known M6.8 flare, {known_flare_time}):")
    print(f"  Any NaN hardness_ratio in this window: {has_nan_hr} (expect True)")
    print(f"  Any nonzero saturation_fraction: {has_sat_fraction} (expect True)")

    if has_nan_hr and has_sat_fraction:
        print("  PASS -- saturation fix is confirmed live in this file.")
        return {"status": "PASS"}
    else:
        print("  WARNING -- this looks like it might be a STALE file from before the fix.")
        return {"status": "FAIL_POSSIBLY_STALE"}


def run(master_table_path: Path, flare_catalog_path: Path, n_flare_samples: int = 30) -> None:
    print(f"Loading {master_table_path} ...")
    master_table = pd.read_parquet(master_table_path)
    if master_table.index.name != "timestamp" and "timestamp" in master_table.columns:
        master_table = master_table.set_index("timestamp")

    catalog = pd.read_parquet(flare_catalog_path)
    catalog_in_range = catalog[
        (catalog["begin_time"] >= master_table.index.min()) &
        (catalog["begin_time"] <= master_table.index.max())
    ]

    print(f"\n{'='*70}\nFINAL VALIDATION SUITE\n{'='*70}\n")

    flare_check = check_random_flare_sample(master_table, catalog_in_range, n_samples=n_flare_samples)
    feature_check = check_feature_sanity(master_table)
    sat_check = check_saturation_fix_is_live(master_table)

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    flare_ok = flare_check["n_fail"] == 0
    feature_ok = len(feature_check["issues"]) == 0
    sat_ok = sat_check["status"] == "PASS"
    overall_pass = flare_ok and feature_ok and sat_ok

    print(f"Random flare spot-check: {'PASS' if flare_ok else str(flare_check['n_fail']) + ' FAILURES'}")
    print(f"Feature sanity: {'PASS' if feature_ok else str(len(feature_check['issues'])) + ' ISSUES'}")
    print(f"Saturation fix regression: {sat_check['status']}")
    print(f"\nOVERALL: {'ALL CHECKS PASSED' if overall_pass else 'SOME CHECKS NEED ATTENTION -- see details above'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--master-table", required=True)
    parser.add_argument("--flare-catalog", required=True)
    parser.add_argument("--n-flare-samples", type=int, default=30)
    args = parser.parse_args()

    run(Path(args.master_table), Path(args.flare_catalog), args.n_flare_samples)
