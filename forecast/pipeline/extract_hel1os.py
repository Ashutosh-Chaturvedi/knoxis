"""
manage_helios_dumps.py (fixed)

Resolves HEL1OS's messy per-day dump structure (multiple zips per day,
overlapping time windows, multiple versions, multiple folders) into a
clean set of "winning" files per day, then extracts only the files our
pipeline actually needs (czt1 lightcurve, optionally hk.fits) from those
winners.

Fixes applied to the original version:
1. Same-version ties across folders (e.g. N00_0000 vs UNP_9999) are now
   genuinely NEVER auto-deleted, matching what the docstring always
   claimed. They're extracted from BOTH files (so you have the data
   either way) and flagged loudly for manual review. Nothing ambiguous
   is discarded until a human decides.
2. --delete-zips now only deletes a zip if extraction from THAT SPECIFIC
   zip actually produced the wanted file(s). Empty/failed extraction
   never triggers deletion.
3. Corrupt or unreadable zips are caught per-file and logged, instead of
   crashing the entire run. A bad zip 400 files in no longer costs you
   the previous 3 hours of work.
4. Output day-folder is derived from the actual level1/YYYY/MM/DD path
   the zip was found under, not from a filename-parsed date -- avoids
   mislabeling dumps that start near midnight.
5. Every zip's outcome (not just resolution decisions) is written to a
   manifest CSV, same pattern used for the SoLEXS extraction scripts.
"""

from __future__ import annotations

import re
import csv
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

FILENAME_PATTERN = re.compile(
    r"HLS_(\d{8})_(\d{6})_(\d+)sec_lev(\d+)_V(\d+)\.zip"
)


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
    m = FILENAME_PATTERN.match(path.name)
    if not m:
        return None
    date_str, time_str, duration_str, level_str, version_str = m.groups()
    start = datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")
    return DumpFile(
        path=path, date_str=date_str, start=start,
        duration_s=int(duration_str), level=int(level_str), version=int(version_str),
    )


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


def resolve_day(dumps: list[DumpFile]) -> tuple[list[DumpFile], list[DumpFile], list[dict]]:
    """
    Resolves one day's dump files.

    Policy:
    - Same window, different version -> auto-resolve, keep highest version.
      This is a standard reprocessing pattern, safe to resolve automatically.
    - Same window, SAME version, different source (e.g. N00_0000 vs
      UNP_9999) -> genuinely unresolved. Both files are kept as
      "ambiguous" and returned separately, NOT silently merged into
      `winners`. Nothing about this case is guessed.

    Returns
    -------
    (winners, ambiguous, resolution_log)
        winners: files safe to treat as the single answer for their window.
        ambiguous: files involved in an unresolved same-version tie --
            extracted from all of them, never auto-deleted.
        resolution_log: human-readable record of every decision made.
    """
    groups = group_overlapping(dumps)
    winners: list[DumpFile] = []
    ambiguous: list[DumpFile] = []
    resolution_log: list[dict] = []

    for group in groups:
        if len(group) == 1:
            winners.append(group[0])
            continue

        max_version = max(d.version for d in group)
        top_version_files = [d for d in group if d.version == max_version]

        if len(top_version_files) == 1:
            chosen = top_version_files[0]
            winners.append(chosen)
            discarded = [d for d in group if d.path != chosen.path]
            resolution_log.append({
                "kind": "auto_resolved_version",
                "kept": str(chosen.path),
                "discarded": [str(d.path) for d in discarded],
            })
        else:
            # Genuine unresolved ambiguity: same window, same version,
            # different source folder. Do NOT pick one. Keep all of them
            # so nothing is silently dropped.
            ambiguous.extend(top_version_files)
            resolution_log.append({
                "kind": "UNRESOLVED_AMBIGUOUS_TIE",
                "files": [str(d.path) for d in top_version_files],
                "note": ("Same time window, same version, different source "
                         "folder. Not auto-resolved -- extracted from all, "
                         "review manually before deleting any of them."),
            })

    return winners, ambiguous, resolution_log


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


def extract_needed_files(zip_path: Path, out_dir: Path, keep_hk: bool = False):
    """Pulls only the CZT1 lightcurve (and optionally hk.fits) out of a
    HEL1OS dump zip into out_dir. Returns (status, extracted_names, error)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted_suffixes = ["lightcurve_czt1.fits"]
    if keep_hk:
        wanted_suffixes.append("hk.fits")

    extracted = []
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for name in zf.namelist():
                if any(name.endswith(suffix) for suffix in wanted_suffixes):
                    dest_name = f"{zip_path.stem}__{Path(name).name}"
                    dest_path = out_dir / dest_name
                    with zf.open(name) as src, open(dest_path, "wb") as dst:
                        dst.write(src.read())
                    if dest_path.exists() and dest_path.stat().st_size > 0:
                        extracted.append(dest_name)

        if not extracted:
            return "none_found", [], "wanted file(s) not present in this zip"
        return "ok", extracted, ""

    except zipfile.BadZipFile as e:
        return "corrupt_zip", [], str(e)
    except Exception as e:
        return "error", [], str(e)


def process_all_days(download_root: Path, output_root: Path, keep_hk: bool = False,
                      delete_zips_after: bool = False) -> None:
    """
    Walks the whole PRADAN download tree (hel1os/level1/YYYY/MM/DD/...),
    resolves each day independently, and extracts winning + ambiguous
    files into output_root/<YYYY><MM><DD>/.
    """
    day_dirs = sorted(p for p in download_root.glob("level1/*/*/*") if p.is_dir())

    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "resolution_log.txt"
    manifest_path = output_root / "extraction_manifest.csv"
    log_lines = []

    n_days = 0
    n_auto_resolved = 0
    n_unresolved_ties = 0
    n_extract_ok = 0
    n_extract_failed = 0
    n_deleted = 0

    with open(manifest_path, "w", newline="") as mf:
        writer = csv.writer(mf)
        writer.writerow(["day_folder", "zip_path", "role", "status", "extracted_files", "note", "deleted"])

        for day_dir in day_dirs:
            # Derive the day's output folder from the REAL portal path,
            # not from any single dump's filename-parsed date.
            try:
                yyyy, mm, dd = day_dir.parts[-3], day_dir.parts[-2], day_dir.parts[-1]
                day_out_dir = output_root / f"{yyyy}{mm}{dd}"
            except Exception:
                day_out_dir = output_root / day_dir.name

            day_dumps = find_all_dump_files(day_dir)
            if not day_dumps:
                continue
            n_days += 1

            winners, ambiguous, resolution_log = resolve_day(day_dumps)

            for entry in resolution_log:
                if entry["kind"] == "auto_resolved_version":
                    n_auto_resolved += 1
                    log_lines.append(
                        f"{day_dir}: [auto] kept {entry['kept']}, discarded {entry['discarded']}"
                    )
                else:
                    n_unresolved_ties += 1
                    log_lines.append(
                        f"{day_dir}: [UNRESOLVED TIE, needs manual review] {entry['files']}"
                    )

            # Winners: safe to auto-delete on success, since these are
            # confirmed single-answer files for their time window.
            for dump in winners:
                status, extracted, note = extract_needed_files(dump.path, day_out_dir, keep_hk=keep_hk)
                deleted = False

                if status == "ok":
                    n_extract_ok += 1
                    if delete_zips_after:
                        dump.path.unlink()
                        deleted = True
                        n_deleted += 1
                else:
                    n_extract_failed += 1

                writer.writerow([day_out_dir.name, str(dump.path), "winner",
                                  status, ";".join(extracted), note, deleted])
                print(f"{dump.path.name}: {status}" + (f" ({note})" if note else ""))

            # Ambiguous ties: extract from ALL of them, but NEVER delete,
            # regardless of --delete-zips. This is the one category the
            # original script's docstring promised safety on but didn't
            # actually enforce.
            for dump in ambiguous:
                status, extracted, note = extract_needed_files(dump.path, day_out_dir, keep_hk=keep_hk)

                if status == "ok":
                    n_extract_ok += 1
                else:
                    n_extract_failed += 1

                writer.writerow([day_out_dir.name, str(dump.path), "AMBIGUOUS_UNRESOLVED",
                                  status, ";".join(extracted), note, False])
                print(f"{dump.path.name}: {status} [AMBIGUOUS -- zip NOT deleted regardless of --delete-zips]")

    log_path.write_text("\n".join(log_lines))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Days processed:              {n_days}")
    print(f"Auto-resolved (version tie): {n_auto_resolved}")
    print(f"UNRESOLVED ambiguous ties:   {n_unresolved_ties}  <-- review these manually")
    print(f"Extractions ok:              {n_extract_ok}")
    print(f"Extractions failed:          {n_extract_failed}")
    print(f"Zips deleted:                {n_deleted}")
    print(f"\nManifest: {manifest_path}")
    print(f"Resolution log: {log_path}")
    if n_unresolved_ties:
        print(f"\n{n_unresolved_ties} day(s) had files that could not be safely auto-resolved.")
        print("Both versions were extracted, but their source zips were NOT deleted.")
        print("Search the resolution log for 'UNRESOLVED TIE' to review them.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Resolve and extract HEL1OS dump files")
    parser.add_argument("--download-root", required=True, help="Root of the PRADAN download tree (contains 'level1/')")
    parser.add_argument("--output-root", required=True, help="Where to write extracted per-day files")
    parser.add_argument("--keep-hk", action="store_true", help="Also extract hk.fits per dump")
    parser.add_argument("--delete-zips", action="store_true",
                         help="Delete zip files after successful extraction. "
                              "Never deletes files involved in an unresolved ambiguous tie.")
    args = parser.parse_args()

    process_all_days(
        Path(args.download_root), Path(args.output_root),
        keep_hk=args.keep_hk, delete_zips_after=args.delete_zips,
    )