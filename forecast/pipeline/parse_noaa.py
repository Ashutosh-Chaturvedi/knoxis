"""
parse_noaa_events.py

Parses NOAA's daily "Edited Events" solar event reports into a clean
flare catalog: (date, begin_time, max_time, end_time, region, goes_class).

Format reference (classic NOAA event report, used ~unchanged for decades):
    #Event  Begin  Max   End   Obs  Q  Type  Loc/Frq   Particulars   Reg#
    1000    0017   0020  0023  G15  5  XRA   1-8A      B2.1          13563

Only rows with Type == 'XRA' (X-ray flare events) are kept -- the
'Particulars' column directly holds the real GOES class for those rows.

IMPORTANT: this parser's exact column assumptions have NOT yet been
verified against a real downloaded file from the target archive (fetch
attempts were rate-limited while building this). Run
`inspect_raw_file()` on one real file FIRST and sanity check the output
before trusting this across your full 912-day range.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


def inspect_raw_file(path: Path, n_lines: int = 40) -> None:
    """Prints the first N lines of a raw events file, unprocessed. Run
    this on a real file before trusting parse_events_file() on it --
    compare what you see against the expected format in this module's
    docstring, and adjust the parser if anything doesn't match."""
    with open(path, "r", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= n_lines:
                break
            print(f"{i:3d}: {line.rstrip()}")


def parse_events_file(path: Path) -> pd.DataFrame:
    """
    Parses one daily events.txt file into a DataFrame of X-ray flare
    events only.

    Returns
    -------
    pd.DataFrame
        Columns: ['begin_time', 'max_time', 'end_time', 'region', 'goes_class']
        Times are pd.Timestamp (UTC), combining the file's date with each
        event's HHMM. Empty DataFrame (not an error) if the file has no
        XRA events that day, or the file is missing/empty.
    """
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=["begin_time", "max_time", "end_time", "region", "goes_class"])

    # Date comes from the filename itself (YYYYMMDDevents.txt), not
    # parsed from file content, since that's the one thing guaranteed
    # to be unambiguous and consistent.
    date_match = re.match(r"(\d{8})", path.stem)
    if not date_match:
        raise ValueError(f"Can't parse date from filename: {path.name}")
    file_date = pd.Timestamp(date_match.group(1), tz="UTC")

    events = []
    with open(path, "r", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith(("#", ":")):
                continue  # header / comment lines

            tokens = line.split()
            if len(tokens) < 7:
                continue  # too short to be a real data row

            # Expected token layout: Event Begin Max End Obs Q Type Loc/Frq Particulars [Reg#]
            # We locate the 'Type' field by finding a token that's a
            # known 3-letter event-type code, rather than assuming a
            # fixed column index -- more robust to minor spacing/format
            # drift across different years' files.
            event_type = None
            type_idx = None
            for i, tok in enumerate(tokens):
                if tok in ("XRA", "FLA", "RSP", "RBR", "RNS", "RBU", "FIL", "FOR", "SPY"):
                    event_type = tok
                    type_idx = i
                    break

            if event_type != "XRA" or type_idx is None:
                continue  # not an X-ray flare event -- skip

            # Begin/Max/End are anchored RELATIVE TO the Type token
            # position (Type is always preceded by Q, then Obs, then
            # End, Max, Begin in that order) -- NOT relative to the start
            # of the line. This matters because some rows have a '+'
            # flag token right after the event number, which shifts
            # every column after it by one position. Anchoring backward
            # from the dynamically-located Type token is immune to that
            # shift, verified against real rows both with and without
            # the '+' flag.
            if type_idx - 5 < 0:
                continue  # not enough tokens before Type to be a real data row
            begin_str = tokens[type_idx - 5]
            max_str = tokens[type_idx - 4]
            end_str = tokens[type_idx - 3]

            # Particulars (the GOES class) is the token right after
            # Loc/Frq, i.e. two tokens after Type -- this part is
            # unaffected by the '+' shift since it's on the other side
            # of Type and was already being computed relative to it.
            if type_idx + 2 >= len(tokens):
                continue
            particulars = tokens[type_idx + 2]

            goes_match = re.match(r"^([ABCMX])(\d+(\.\d+)?)$", particulars)
            if not goes_match:
                continue  # doesn't look like a real GOES class -- skip defensively

            goes_class = particulars
            region = tokens[-1] if tokens[-1].isdigit() and 4 <= len(tokens[-1]) <= 5 else None

            def hhmm_to_timestamp(hhmm: str, base_date: pd.Timestamp) -> pd.Timestamp | None:
                if not hhmm.isdigit() or len(hhmm) != 4:
                    return None
                hour, minute = int(hhmm[:2]), int(hhmm[2:])
                ts = base_date + pd.Timedelta(hours=hour, minutes=minute)
                return ts

            begin_time = hhmm_to_timestamp(begin_str, file_date)
            max_time = hhmm_to_timestamp(max_str, file_date)
            end_time = hhmm_to_timestamp(end_str, file_date)

            if begin_time is None or max_time is None:
                continue

            # Handle events that cross midnight (End < Begin numerically
            # implies it rolled into the next day).
            if end_time is not None and end_time < begin_time:
                end_time += pd.Timedelta(days=1)
            if max_time < begin_time:
                max_time += pd.Timedelta(days=1)

            events.append({
                "begin_time": begin_time,
                "max_time": max_time,
                "end_time": end_time,
                "region": region,
                "goes_class": goes_class,
            })

    return pd.DataFrame(events)


def build_flare_catalog(events_dir: Path, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Parses every daily events file in `events_dir` across the given date
    range into one combined flare catalog.
    """
    all_events = []
    for date in pd.date_range(start_date, end_date, freq="D"):
        fname = f"{date.strftime('%Y%m%d')}events.txt"
        fpath = events_dir / fname
        day_events = parse_events_file(fpath)
        if len(day_events):
            all_events.append(day_events)

    if not all_events:
        return pd.DataFrame(columns=["begin_time", "max_time", "end_time", "region", "goes_class"])

    catalog = pd.concat(all_events, ignore_index=True)
    catalog = catalog.sort_values("begin_time").reset_index(drop=True)
    return catalog


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Parse NOAA daily solar event reports")
    parser.add_argument("--events-dir", help="Folder containing YYYYMMDDevents.txt files")
    parser.add_argument("--start", help="Start date, e.g. 2024-02-01")
    parser.add_argument("--end", help="End date, e.g. 2026-08-01")
    parser.add_argument("--out", default="flare_catalog.parquet", help="Output Parquet path")
    parser.add_argument("--inspect", help="Just print the raw content of this one file and exit")
    args = parser.parse_args()

    if args.inspect:
        inspect_raw_file(Path(args.inspect))
    elif args.events_dir and args.start and args.end:
        catalog = build_flare_catalog(Path(args.events_dir), args.start, args.end)
        print(f"Parsed {len(catalog)} X-ray flare events.")
        print(catalog["goes_class"].str[0].value_counts())  # class letter distribution
        catalog.to_parquet(args.out)
        print(f"Saved to {args.out}")
    else:
        parser.error(
            "Either provide --inspect <file> (to check one file's raw format), "
            "OR provide all of --events-dir, --start, and --end (to run the full parse)."
        )