#!/usr/bin/env python3
"""
SoLEXS Pass 1 (v2): EXTRACTION ONLY (non-destructive) -- SDD2, LC + GTI only

What this does:
- Walks your downloaded SoLEXS zip archive.
- Groups zips by observation date; if duplicate versions exist for the same
  date, only the HIGHEST version is used (lower version logged as skipped).
- From each selected zip, extracts ONLY the SDD2 detector's .lc.gz and
  .gti.gz files (SDD1 and all .pi.gz spectral files are ignored entirely).
- Decompresses them (source is gzip; output is the plain FITS content).
- Writes them into two separate trees, organized by month, with short
  date-based names:

    OUTPUT_ROOT/lc/2024-02/20240212.lc
    OUTPUT_ROOT/gti/2024-02/20240212.gti

- Writes manifest.csv logging every zip's outcome.
- Does NOT delete or modify anything in your source zip archive.
"""

import zipfile
import gzip
import csv
import re
import sys
from pathlib import Path
from collections import defaultdict

# =====================================================
# CONFIGURATION
# =====================================================

SOURCE_ROOT = Path(r"D:\Knoxis\data\training\pradan1.issdc.gov.in")
OUTPUT_ROOT = Path(r"D:\Knoxis\data\training\solexs_sdd2")
MANIFEST_PATH = Path(r"D:\Knoxis\data\training\solexs_extract_manifest.csv")

DETECTOR = "SDD2"  # only this detector's files are extracted

# =====================================================

FILENAME_PATTERN = re.compile(r"AL1_SLX_L1_(\d{8})_v(\d+)\.(\d+)\.zip")


def find_zips(root: Path):
    return sorted(root.rglob("AL1_SLX_L1_*.zip"))


def parse_date_and_version(zip_path: Path):
    m = FILENAME_PATTERN.match(zip_path.name)
    if not m:
        return None, None
    date_str, major, minor = m.groups()
    return date_str, (int(major), int(minor))


def select_zips_per_date(zips):
    by_date = defaultdict(list)
    unparsed = []

    for zp in zips:
        date_str, version = parse_date_and_version(zp)
        if date_str is None:
            unparsed.append(zp)
            continue
        by_date[date_str].append((zp, version))

    selected = {}   # date_str -> zip_path
    superseded = []

    for date_str, entries in by_date.items():
        entries.sort(key=lambda e: e[1], reverse=True)
        best_zip, best_version = entries[0]
        selected[date_str] = best_zip

        for zp, version in entries[1:]:
            superseded.append((zp, f"superseded by v{best_version[0]}.{best_version[1]}"))

    return selected, superseded, unparsed


def extract_sdd2_lc_gti(zip_path: Path, date_str: str, output_root: Path):
    """Extract + decompress SDD2's .lc.gz and .gti.gz into short-named files
    under lc/YYYY-MM/ and gti/YYYY-MM/. Returns (status, found_types, note)."""

    year_month = f"{date_str[0:4]}-{date_str[4:6]}"

    lc_dir = output_root / "lc" / year_month
    gti_dir = output_root / "gti" / year_month
    lc_dir.mkdir(parents=True, exist_ok=True)
    gti_dir.mkdir(parents=True, exist_ok=True)

    found_types = []

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = zf.namelist()

            lc_member = next(
                (m for m in members if f"/{DETECTOR}/" in m and m.endswith(".lc.gz")),
                None
            )
            gti_member = next(
                (m for m in members if f"/{DETECTOR}/" in m and m.endswith(".gti.gz")),
                None
            )

            if lc_member:
                target = lc_dir / f"{date_str}.lc"
                if not (target.exists() and target.stat().st_size > 0):
                    with zf.open(lc_member) as src:
                        raw = gzip.decompress(src.read())
                        target.write_bytes(raw)
                if target.exists() and target.stat().st_size > 0:
                    found_types.append("lc")

            if gti_member:
                target = gti_dir / f"{date_str}.gti"
                if not (target.exists() and target.stat().st_size > 0):
                    with zf.open(gti_member) as src:
                        raw = gzip.decompress(src.read())
                        target.write_bytes(raw)
                if target.exists() and target.stat().st_size > 0:
                    found_types.append("gti")

        if not found_types:
            return "none_found", found_types, f"No {DETECTOR} lc/gti members in zip"

        if len(found_types) < 2:
            missing = {"lc", "gti"} - set(found_types)
            return "partial", found_types, f"Missing: {missing} (detector may have had no valid data that day)"

        return "ok", found_types, ""

    except zipfile.BadZipFile as e:
        return "corrupt_zip", [], str(e)
    except Exception as e:
        return "error", [], str(e)


def main():
    if not SOURCE_ROOT.exists():
        print(f"SOURCE_ROOT does not exist: {SOURCE_ROOT}")
        sys.exit(1)

    all_zips = find_zips(SOURCE_ROOT)
    print(f"Found {len(all_zips)} SoLEXS zip files under {SOURCE_ROOT}")

    if not all_zips:
        sys.exit(1)

    selected, superseded, unparsed = select_zips_per_date(all_zips)

    print(f"Selected for extraction (highest version per date): {len(selected)}")
    print(f"Superseded, skipped (lower version, same date): {len(superseded)}")
    if unparsed:
        print(f"WARNING: {len(unparsed)} file(s) unparsed:")
        for u in unparsed:
            print(f"  {u}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    counts = {"ok": 0, "partial": 0, "none_found": 0, "corrupt_zip": 0, "error": 0}

    with open(MANIFEST_PATH, "w", newline="") as mf:
        writer = csv.writer(mf)
        writer.writerow(["date", "zip_path", "status", "found_types", "note"])

        for i, (date_str, zip_path) in enumerate(sorted(selected.items()), start=1):
            status, found, note = extract_sdd2_lc_gti(zip_path, date_str, OUTPUT_ROOT)
            counts[status] += 1

            writer.writerow([date_str, str(zip_path), status, ";".join(found), note])

            print(f"[{i}/{len(selected)}] {date_str}: {status}"
                  + (f" ({note})" if note else ""))

        for zip_path, reason in superseded:
            writer.writerow(["-", str(zip_path), "skipped_superseded", "", reason])

    print("\n" + "=" * 50)
    print("EXTRACTION SUMMARY")
    print("=" * 50)
    for k, v in counts.items():
        print(f"  {k:12s}: {v}")
    print(f"  superseded (skipped): {len(superseded)}")
    print(f"\nManifest written to: {MANIFEST_PATH}")
    print(f"LC files under:  {OUTPUT_ROOT / 'lc'}")
    print(f"GTI files under: {OUTPUT_ROOT / 'gti'}")
    print("\nNothing in your source zip archive was modified or deleted.")


if __name__ == "__main__":
    main()