"""
sanity_check_labeled.py

Final sanity pass across the FULL labeled dataset, not just the two
known flares -- checks structural integrity, class distribution (cross-
checked against the real flare catalog), NaN patterns, and whether any
real catalog flares got zero representation anywhere in the labels
(which the two-flare spot-check couldn't reveal).
"""

from pathlib import Path

import pandas as pd


def run_sanity_check(labeled_dir: Path, flare_catalog_path: Path) -> pd.DataFrame:
    labeled_files = sorted(labeled_dir.glob("*_labeled.parquet"))
    print(f"Found {len(labeled_files)} labeled day-files.\n")

    # --- 1. Load and concatenate everything ---
    frames = []
    for f in labeled_files:
        df = pd.read_parquet(f)
        if df.index.name != "timestamp" and "timestamp" in df.columns:
            df = df.set_index("timestamp")
        frames.append(df)
    all_labeled = pd.concat(frames)
    all_labeled = all_labeled.sort_index()

    print("=== 1. Basic structure ===")
    print(f"Total windows: {len(all_labeled)}")
    print(f"Date range: {all_labeled.index.min()} -> {all_labeled.index.max()}")
    print(f"Columns: {list(all_labeled.columns)}")

    # --- 2. Duplicate/overlapping timestamps across day boundaries ---
    n_dupes = all_labeled.index.duplicated().sum()
    print(f"\n=== 2. Duplicate timestamps across day boundaries ===")
    print(f"Duplicates found: {n_dupes} {'(OK)' if n_dupes == 0 else '(INVESTIGATE)'}")

    # --- 3. Label class distribution ---
    print(f"\n=== 3. Label distribution (window counts, not distinct flares) ===")
    label_counts = all_labeled["label_class"].apply(
        lambda c: "QUIET" if c == "QUIET" else c[0]
    ).value_counts()
    print(label_counts)
    quiet_fraction = label_counts.get("QUIET", 0) / len(all_labeled)
    print(f"QUIET fraction: {quiet_fraction:.1%} (expect the large majority -- flares are rare)")

    # --- 4. NaN feature patterns, split by label ---
    print(f"\n=== 4. NaN feature rate, split by QUIET vs flare-labeled windows ===")
    feature_cols = [c for c in all_labeled.columns if c not in ("label_flux", "label_class")]
    is_quiet = all_labeled["label_class"] == "QUIET"
    nan_rate_quiet = all_labeled.loc[is_quiet, feature_cols].isna().any(axis=1).mean()
    nan_rate_flare = all_labeled.loc[~is_quiet, feature_cols].isna().any(axis=1).mean()
    print(f"NaN rate in QUIET windows: {nan_rate_quiet:.1%}")
    print(f"NaN rate in flare-labeled windows: {nan_rate_flare:.1%}")
    if nan_rate_flare > nan_rate_quiet * 1.5:
        print("WARNING: flare-labeled windows have a meaningfully higher NaN rate than "
              "quiet windows -- worth investigating why (e.g. does high flux trip the "
              "min_valid_fraction check more often?).")

    # --- 5. Cross-check against the real flare catalog: which real
    # flares got ZERO representation anywhere in the labels? ---
    print(f"\n=== 5. Coverage check: real flares with NO representation in labels ===")
    catalog = pd.read_parquet(flare_catalog_path)
    catalog_in_range = catalog[
        (catalog["begin_time"] >= all_labeled.index.min() - pd.Timedelta(hours=2))
        & (catalog["begin_time"] <= all_labeled.index.max())
    ].copy()

    represented_classes = set(all_labeled.loc[~is_quiet, "label_class"])
    # A flare counts as "represented" if its exact class string appears
    # at least once in the labels (exact string match, since flux->class
    # conversion is deterministic and should round-trip exactly for a
    # real represented flare).
    catalog_in_range["represented"] = catalog_in_range["goes_class"].isin(represented_classes)

    missed = catalog_in_range[~catalog_in_range["represented"]]
    print(f"Catalog flares in labeled date range: {len(catalog_in_range)}")
    print(f"Flares with NO representation in any label: {len(missed)} "
          f"({100*len(missed)/max(len(catalog_in_range),1):.1f}%)")
    if len(missed):
        print("\nBreakdown of missed flares by class:")
        print(missed["goes_class"].str[0].value_counts())
        print("\nStrongest missed flares (worth investigating individually):")
        missed_sorted = missed.copy()
        missed_sorted["flux_val"] = missed_sorted["goes_class"].apply(
            lambda c: {"B": 1e-7, "C": 1e-6, "M": 1e-5, "X": 1e-4}[c[0]] * float(c[1:])
        )
        print(missed_sorted.sort_values("flux_val", ascending=False).head(10)[
            ["begin_time", "goes_class", "region"]
        ].to_string())

    return all_labeled


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Sanity-check the full labeled dataset")
    parser.add_argument("--labeled-dir", required=True)
    parser.add_argument("--flare-catalog", required=True)
    args = parser.parse_args()

    run_sanity_check(Path(args.labeled_dir), Path(args.flare_catalog))
