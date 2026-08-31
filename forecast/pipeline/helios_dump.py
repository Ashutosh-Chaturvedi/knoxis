"""
manage_helios_dumps.py

Resolves HEL1OS's messy per-day dump structure (multiple zips per day,
overlapping time windows, multiple versions, multiple folders) into a
clean set of "winning" files per day, then parses and MERGES each day's
files into a single clean parquet file — timestamp, hel1os_ctr,
hel1os_err, is_valid — ready for feature engineering. No raw FITS files
are kept afterward, only the merged output, which is far smaller.

Requires data_ingestion.py (load_helios, DataIngestionError) to be
importable -- keep this script in the same folder, or install the
project package.

Robust to corrupted/incomplete zip downloads: a bad zip is logged and
skipped, not allowed to crash the whole batch. Given 2000+ files, some
partial/failed downloads are expected, not exceptional.
"""

from __future__ import annotations

import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import sys
notebook_dir = Path.cwd()

project_root = notebook_dir.parents[1] 
sys.path.append(str(project_root))
from ingestion.data_ingestion import load_helios, DataIngestionError

FILENAME_PATTERN = re.compile(
    r"HLS_(\d{8})_(\d{6})_(\d+)sec_lev(\d+)_V(\d+)\.zip"
)

ZIP_MAGIC = b"PK\x03\x04"


@dataclass
class DumpFile:
    path: Path
    date_str: str
    start: datetime
    duration_s: int
    level: int
    version: int

    @property
    def end(self) -> datetime:
        return self.start + timedelta(seconds=self.duration_s)


def parse_dump_filename(path: Path) -> DumpFile | None:
    """Parses a HEL1OS dump zip filename. Returns None if it doesn't match
    the expected pattern (logged separately so nothing is silently skipped)."""
    m = FILENAME_PATTERN.match(path.name)
    if not m:
        return None
    date_str, time_str, duration_str, level_str, version_str = m.groups()
    start = datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")
    return DumpFile(
        path=path, date_str=date_str, start=start,
        duration_s=int(duration_str), level=int(level_str), version=int(version_str),
    )


def is_valid_zip(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    with open(path, "rb") as f:
        header = f.read(4)
    return header == ZIP_MAGIC


def fractional_overlap(a: DumpFile, b: DumpFile) -> float:
    latest_start = max(a.start, b.start)
    earliest_end = min(a.end, b.end)
    overlap_seconds = max(0.0, (earliest_end - latest_start).total_seconds())
    shorter_duration = min(a.duration_s, b.duration_s)
    if shorter_duration == 0:
        return 0.0
    return overlap_seconds / shorter_duration


def intervals_overlap(a: DumpFile, b: DumpFile, overlap_threshold: float = 0.5) -> bool:
    return fractional_overlap(a, b) >= overlap_threshold


def group_overlapping(dumps: list[DumpFile]) -> list[list[DumpFile]]:
    dumps_sorted = sorted(dumps, key=lambda d: d.start)
    groups: list[list[DumpFile]] = []
    for d in dumps_sorted:
        placed = False
        for group in groups:
            if any(intervals_overlap(d, existing) for existing in group):
                group.append(d)
                placed = True
                break
        if not placed:
            groups.append([d])
    return groups


def resolve_day(dumps: list[DumpFile]) -> tuple[list[DumpFile], list[dict]]:
    groups = group_overlapping(dumps)
    winners: list[DumpFile] = []
    resolution_log: list[dict] = []

    for group in groups:
        if len(group) == 1:
            winners.append(group[0])
            continue

        max_version = max(d.version for d in group)
        top_version_files = sorted(
            (d for d in group if d.version == max_version),
            key=lambda d: str(d.path),
        )
        chosen = top_version_files[0]
        winners.append(chosen)

        reason = "highest_version" if len(top_version_files) == 1 else "tie_broken_arbitrarily"
        resolution_log.append({
            "chosen": str(chosen.path),
            "discarded": [str(d.path) for d in group if d.path != chosen.path],
            "reason": reason,
        })

    return winners, resolution_log


def find_all_dump_files(root: Path) -> list[DumpFile]:
    dumps = []
    unparsed = []
    for zip_path in root.rglob("*.zip"):
        parsed = parse_dump_filename(zip_path)
        if parsed:
            dumps.append(parsed)
        else:
            unparsed.append(zip_path)
    if unparsed:
        print(f"WARNING: {len(unparsed)} zip file(s) didn't match the expected naming pattern:")
        for p in unparsed:
            print(f"  {p}")
    return dumps


def extract_lightcurve(zip_path: Path, temp_dir: Path) -> Path | None:
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.endswith("lightcurve_czt1.fits"):
                dest = temp_dir / f"{zip_path.stem}__lightcurve_czt1.fits"
                with zf.open(name) as src, open(dest, "wb") as dst:
                    dst.write(src.read())
                return dest
    return None


def merge_day(winners: list[DumpFile], day_str: str, output_root: Path,
              bad_files_log: list[str]) -> Path | None:
    frames = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for dump in winners:
            if not is_valid_zip(dump.path):
                bad_files_log.append(f"{dump.path}  (failed pre-check: missing, empty, or not a valid zip)")
                continue
            try:
                lc_path = extract_lightcurve(dump.path, tmp_path)
            except zipfile.BadZipFile:
                bad_files_log.append(f"{dump.path}  (BadZipFile on open -- corrupted or incomplete download)")
                continue

            if lc_path is None:
                bad_files_log.append(f"{dump.path}  (opened fine, but no lightcurve_czt1.fits inside)")
                continue

            try:
                df = load_helios(lc_path, hdu_index=5)
                frames.append(df)
            except DataIngestionError as e:
                bad_files_log.append(f"{dump.path}  (FITS parsing failed: {e})")
                continue

    return _save_merged(frames, day_str, output_root)


def _save_merged(frames: list, day_str: str, output_root: Path) -> Path | None:
    if not frames:
        return None
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    output_root.mkdir(parents=True, exist_ok=True)
    out_path = output_root / f"{day_str}_hel1os_czt1.parquet"
    merged.to_parquet(out_path)
    return out_path


# --------------------------------------------------------------------- #
# Recovery path
#
# NOTE: the leftover extracted data is NOT flat "<dumpstem>__lightcurve_
# czt1.fits" files -- it's the zip's full internal structure preserved
# as-is, e.g.:
#   <day>/HLS_20250925_000011_43178sec_lev1_V111/czt/lightcurve_czt1.fits
# The dump's identity lives in the GRANDPARENT folder name, not the
# filename (which is always just the generic "lightcurve_czt1.fits").
# So we search recursively and parse the grandparent folder's name.
# --------------------------------------------------------------------- #

def find_lightcurve_files_recursive(day_dir: Path) -> tuple[list[DumpFile], list[Path]]:
    """Recursively finds every lightcurve_czt1.fits under day_dir,
    deriving each file's dump identity (start time, duration, version)
    from its grandparent folder name. Returns (dumps, unmatched_paths)
    so unmatched files are visible rather than silently dropped."""
    dumps: list[DumpFile] = []
    unmatched: list[Path] = []

    for f in day_dir.rglob("lightcurve_czt1.fits"):
        dump_folder_name = f.parent.parent.name  # .../<dumpstem>/czt/lightcurve_czt1.fits
        parsed = parse_dump_filename(Path(dump_folder_name + ".zip"))
        if parsed is None:
            unmatched.append(f)
            continue
        parsed.path = f
        dumps.append(parsed)

    return dumps, unmatched


def recover_day_from_extracted(old_extracted_day_dir: Path, day_str: str,
                                output_root: Path, bad_files_log: list[str]) -> Path | None:
    dumps, unmatched = find_lightcurve_files_recursive(old_extracted_day_dir)

    for u in unmatched:
        bad_files_log.append(f"{u}  (found during recovery, but parent folder name "
                              f"didn't match the expected dump-filename pattern)")

    if not dumps:
        return None

    winners, _ = resolve_day(dumps)

    frames = []
    for dump in winners:
        try:
            df = load_helios(dump.path, hdu_index=5)
            frames.append(df)
        except DataIngestionError as e:
            bad_files_log.append(f"{dump.path}  (FITS parsing failed during recovery: {e})")
            continue

    return _save_merged(frames, day_str, output_root)


def recover_all_from_extracted(old_extracted_root: Path, output_root: Path,
                                delete_after: bool = False) -> None:
    day_dirs = sorted(p for p in old_extracted_root.iterdir() if p.is_dir())
    bad_files_log: list[str] = []
    n_recovered = 0
    n_failed = 0

    for day_dir in day_dirs:
        day_str = day_dir.name
        out_path = recover_day_from_extracted(day_dir, day_str, output_root, bad_files_log)
        if out_path is not None:
            n_recovered += 1
            print(f"{day_str}: recovered -> {out_path.name}")
            if delete_after:
                for entry in day_dir.iterdir():
                    if entry.is_dir():
                        shutil.rmtree(entry)
                    else:
                        entry.unlink()
                day_dir.rmdir()
        else:
            n_failed += 1
            print(f"{day_str}: recovery FAILED -- no usable extracted files")

    print(f"\nRecovery done. {n_recovered} days recovered, {n_failed} failed.")
    if bad_files_log:
        log_path = output_root / "recovery_bad_files_log.txt"
        log_path.write_text("\n".join(bad_files_log))
        print(f"{len(bad_files_log)} individual bad files logged to {log_path}")


def process_all_days(download_root: Path, output_root: Path) -> None:
    day_dirs = sorted(p for p in download_root.glob("level1/*/*/*") if p.is_dir())

    output_root.mkdir(parents=True, exist_ok=True)
    resolution_log_path = output_root / "resolution_log.txt"
    bad_files_log_path = output_root / "bad_files_log.txt"
    resolution_lines = []
    bad_files_log: list[str] = []

    n_days_processed = 0
    n_days_failed = 0
    n_tie_breaks = 0

    for day_dir in day_dirs:
        day_dumps = find_all_dump_files(day_dir)
        if not day_dumps:
            continue

        winners, resolution_entries = resolve_day(day_dumps)
        for entry in resolution_entries:
            if entry["reason"] == "tie_broken_arbitrarily":
                n_tie_breaks += 1
            resolution_lines.append(f"{day_dir}: {entry}")

        day_str = winners[0].date_str
        out_path = merge_day(winners, day_str, output_root, bad_files_log)

        if out_path is not None:
            n_days_processed += 1
            print(f"{day_str}: merged {len(winners)} dump(s) -> {out_path.name}")
        else:
            n_days_failed += 1
            print(f"{day_str}: FAILED -- no usable dumps (see bad_files_log.txt)")

    resolution_log_path.write_text("\n".join(resolution_lines))
    bad_files_log_path.write_text("\n".join(bad_files_log))

    print(f"\nDone. {n_days_processed} days merged successfully, {n_days_failed} days failed entirely.")
    print(f"{n_tie_breaks} arbitrary tie-breaks (N00_0000 vs UNP_9999-type ambiguity).")
    print(f"{len(bad_files_log)} individual bad/unusable zip files skipped.")
    print(f"See {resolution_log_path} and {bad_files_log_path} for full detail.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Resolve, parse, and merge HEL1OS dump files into one parquet per day")
    parser.add_argument("--download-root", help="Root of the PRADAN download tree (contains 'level1/') -- normal mode")
    parser.add_argument("--output-root", required=True, help="Where to write merged per-day parquet files")
    parser.add_argument("--recover-from-extracted", help="Root folder of an OLDER run's leftover extracted "
                                                           "FITS files (day-named subfolders) -- use this instead "
                                                           "of --download-root when the raw zips are already gone")
    parser.add_argument("--delete-after-recovery", action="store_true",
                         help="Delete the old leftover extracted FITS files after successful recovery")
    args = parser.parse_args()

    if args.recover_from_extracted:
        recover_all_from_extracted(
            Path(args.recover_from_extracted), Path(args.output_root),
            delete_after=args.delete_after_recovery,
        )
    elif args.download_root:
        process_all_days(Path(args.download_root), Path(args.output_root))
    else:
        parser.error("Provide either --download-root (normal mode) or --recover-from-extracted (recovery mode)")