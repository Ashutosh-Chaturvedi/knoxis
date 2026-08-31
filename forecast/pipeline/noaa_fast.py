"""
download_noaa_events_fast.py

Fast, resumable bulk downloader for NOAA daily solar event reports.

Key improvements over the serial version:
- Concurrent downloads with a bounded ThreadPoolExecutor.
- A separate requests.Session per worker thread (Session is not shared
  between threads).
- Retries with exponential backoff for transient HTTP/network failures.
- 404 is treated as a normal "no report" result.
- Existing files are skipped before submitting network work.
- Atomic file writes prevent partially-written files from looking complete.
- Failed days are logged for an easy re-run.
"""

from __future__ import annotations

import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = (
    "https://www.ngdc.noaa.gov/stp/space-weather/"
    "swpc-products/daily_reports/solar_event_reports"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}

# NOAA is a public service, so don't go crazy with concurrency.
# 8-16 is usually a good starting range.
DEFAULT_WORKERS = 12
DEFAULT_TIMEOUT = 20

_thread_local = threading.local()


def make_session() -> requests.Session:
    """Create a requests session for one worker thread."""
    session = requests.Session()
    session.headers.update(HEADERS)

    retry = Retry(
        total=4,
        connect=4,
        read=4,
        status=4,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
        raise_on_status=False,
    )

    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=1,
        pool_maxsize=1,
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def get_session() -> requests.Session:
    """Return one persistent Session for the current worker thread."""
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = make_session()
        _thread_local.session = session
    return session


def iter_dates(start_date: str, end_date: str):
    """Yield date objects from start_date through end_date, inclusive."""
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)

    if start > end:
        raise ValueError("start date must be <= end date")

    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def download_one(day: date, out_dir: Path, timeout: float):
    """Download one NOAA report. Returns (status, day, detail)."""
    yyyy = day.strftime("%Y")
    mm = day.strftime("%m")
    dd = day.strftime("%Y%m%d")

    url = f"{BASE_URL}/{yyyy}/{mm}/{dd}events.txt"
    out_path = out_dir / f"{dd}events.txt"

    # Resume support: no request is made if the file already exists.
    if out_path.exists():
        return "skipped", dd, None

    try:
        response = get_session().get(url, timeout=timeout)

        if response.status_code == 404:
            return "no_data", dd, None

        response.raise_for_status()

        # Write to a temporary file first. If the process dies while
        # writing, there is no chance of leaving a fake "complete" file.
        temp_path = out_path.with_suffix(out_path.suffix + ".part")
        temp_path.write_bytes(response.content)
        temp_path.replace(out_path)

        return "downloaded", dd, None

    except requests.exceptions.RequestException as exc:
        return "failed", dd, str(exc)
    except OSError as exc:
        return "failed", dd, f"file error: {exc}"


def download_range(
    start_date: str,
    end_date: str,
    out_dir: Path,
    workers: int = DEFAULT_WORKERS,
    timeout: float = DEFAULT_TIMEOUT,
) -> None:
    """
    Download every day's events.txt in [start_date, end_date].

    Uses bounded concurrency, so several days are downloaded at once
    without creating an uncontrolled number of connections.
    """
    if workers < 1:
        raise ValueError("workers must be >= 1")

    out_dir.mkdir(parents=True, exist_ok=True)

    days = list(iter_dates(start_date, end_date))
    pending = [
        day
        for day in days
        if not (out_dir / f"{day:%Y%m%d}events.txt").exists()
    ]

    already_exists = len(days) - len(pending)

    print(f"Total days:      {len(days)}")
    print(f"Already present: {already_exists}")
    print(f"To download:     {len(pending)}")
    print(f"Workers:         {workers}")
    print()

    if not pending:
        print("Nothing to download.")
        return

    downloaded = 0
    no_data = 0
    failed: list[tuple[str, str]] = []
    completed = 0

    # ThreadPoolExecutor is a good fit here because the workload is
    # network I/O, not CPU-heavy computation.
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(download_one, day, out_dir, timeout): day
            for day in pending
        }

        for future in as_completed(futures):
            completed += 1

            try:
                status, dd, detail = future.result()
            except Exception as exc:
                # Last-resort isolation: one unexpected worker exception
                # cannot kill the entire batch.
                day = futures[future]
                dd = day.strftime("%Y%m%d")
                status = "failed"
                detail = repr(exc)

            if status == "downloaded":
                downloaded += 1
            elif status == "no_data":
                no_data += 1
            elif status == "failed":
                failed.append((dd, detail))

            if completed % 25 == 0 or completed == len(pending):
                print(
                    f"Progress: {completed}/{len(pending)} | "
                    f"downloaded={downloaded}, "
                    f"no_data={no_data}, "
                    f"failed={len(failed)}"
                )

    print()
    print(
        f"Done. downloaded={downloaded}, "
        f"already_present={already_exists}, "
        f"no_data={no_data}, failed={len(failed)}"
    )

    if failed:
        failed.sort()
        failed_log = out_dir / "failed_downloads.txt"
        failed_log.write_text(
            "\n".join(f"{day}: {reason}" for day, reason in failed),
            encoding="utf-8",
        )
        print(f"Failed days logged to: {failed_log}")
        print("Re-run the same command to retry failed/missing days.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fast bulk downloader for NOAA daily solar event reports"
    )
    parser.add_argument("--start", required=True, help="Start date, e.g. 2024-02-01")
    parser.add_argument("--end", required=True, help="End date, e.g. 2026-08-01")
    parser.add_argument(
        "--out-dir",
        default="noaa_events",
        help="Output directory (default: noaa_events)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Concurrent downloads (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"Per-request timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )

    args = parser.parse_args()

    download_range(
        args.start,
        args.end,
        Path(args.out_dir),
        workers=args.workers,
        timeout=args.timeout,
    )
