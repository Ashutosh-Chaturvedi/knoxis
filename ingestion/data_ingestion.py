"""
Data Ingestion & Preprocessing Module.

Loads SoLEXS and HEL1OS FITS data, fixes byte-order issues, flags data
quality, and merges both instruments onto a common time index.

See ingestion/README.md for design rationale and known limitations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time

logger = logging.getLogger("knoxis.data_ingestion")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    )
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class DataIngestionError(Exception):
    """Raised when a source file cannot be parsed into a usable DataFrame."""


def fix_byte_order(arr: np.ndarray) -> np.ndarray:
    """Converts a big-endian FITS array to native-order float64."""
    if arr.dtype.byteorder not in ("=", "|"):
        arr = arr.byteswap().view(arr.dtype.newbyteorder("="))
    return arr.astype(np.float64, copy=False)


@dataclass
class IngestionConfig:
    """Paths and parameters controlling a single ingestion run."""

    solexs_lc_path: Path
    solexs_gti_path: Path
    hel1os_paths: list
    hel1os_hdu_index: int = 5
    merge_tolerance: str = "2s"
    output_dir: Path = field(default_factory=lambda: Path("./processed"))

    def __post_init__(self) -> None:
        self.solexs_lc_path = Path(self.solexs_lc_path)
        self.solexs_gti_path = Path(self.solexs_gti_path)
        if isinstance(self.hel1os_paths, (str, Path)):
            self.hel1os_paths = [Path(self.hel1os_paths)]
        else:
            self.hel1os_paths = [Path(p) for p in self.hel1os_paths]
        self.output_dir = Path(self.output_dir)


def load_solexs_gti(path: Path) -> pd.DataFrame:
    """Loads a SoLEXS GTI file. Returns DataFrame with columns ['start', 'stop']."""
    path = Path(path)
    if not path.exists():
        raise DataIngestionError(f"SoLEXS GTI file not found: {path}")

    logger.info("Loading SoLEXS GTI file: %s", path)

    try:
        with fits.open(path) as hdul:
            gti_hdu = hdul["GTI"] if "GTI" in hdul else hdul[1]
            raw_start = gti_hdu.data["START"]
            raw_stop = gti_hdu.data["STOP"]
    except (KeyError, OSError) as exc:
        raise DataIngestionError(f"Failed to parse SoLEXS GTI file {path}: {exc}") from exc

    gti_df = pd.DataFrame(
        {"start": fix_byte_order(raw_start), "stop": fix_byte_order(raw_stop)}
    ).sort_values("start").reset_index(drop=True)

    total_valid_seconds = (gti_df["stop"] - gti_df["start"]).sum()
    logger.info(
        "SoLEXS GTI: %d valid windows, %.0f total valid seconds",
        len(gti_df), total_valid_seconds,
    )
    return gti_df


def _flag_gti_validity(unix_time: np.ndarray, gti_df: pd.DataFrame) -> np.ndarray:
    """Vectorized check: is each timestamp inside any [start, stop] GTI window?"""
    starts = gti_df["start"].to_numpy()[np.newaxis, :]
    stops = gti_df["stop"].to_numpy()[np.newaxis, :]
    t = unix_time[:, np.newaxis]
    return ((t >= starts) & (t <= stops)).any(axis=1)


def load_solexs(lc_path: Path, gti_path: Path) -> pd.DataFrame:
    """
    Loads a SoLEXS light curve and flags each row valid/invalid via GTI.

    Returns DataFrame: ['timestamp', 'solexs_counts', 'is_valid'].
    Never drops or interpolates rows — labeling only.
    """
    lc_path = Path(lc_path)
    if not lc_path.exists():
        raise DataIngestionError(f"SoLEXS LC file not found: {lc_path}")

    logger.info("Loading SoLEXS light curve: %s", lc_path)

    try:
        with fits.open(lc_path) as hdul:
            data_hdu = _find_solexs_data_hdu(hdul)
            raw_time = data_hdu.data["TIME"]
            raw_counts = data_hdu.data["COUNTS"]
    except (KeyError, OSError) as exc:
        raise DataIngestionError(f"Failed to parse SoLEXS LC file {lc_path}: {exc}") from exc

    unix_time = fix_byte_order(raw_time)
    counts = fix_byte_order(raw_counts)

    gti_df = load_solexs_gti(gti_path)
    is_valid = _flag_gti_validity(unix_time, gti_df)

    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(unix_time, unit="s", utc=True).as_unit("ns"),
            "solexs_counts": counts,
            "is_valid": is_valid,
        }
    )

    before = len(df)
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        logger.warning("Dropped %d duplicate timestamp rows.", dropped)

    n_invalid = (~df["is_valid"]).sum()
    n_nan = df["solexs_counts"].isna().sum()
    logger.info(
        "SoLEXS: %d samples (%s -> %s); %d flagged GTI-invalid, %d NaN counts",
        len(df), df["timestamp"].min(), df["timestamp"].max(), n_invalid, n_nan,
    )
    return df


def _find_solexs_data_hdu(hdul: fits.HDUList) -> fits.BinTableHDU:
    """Locates the binary table HDU containing TIME/COUNTS (name 'RATE')."""
    for hdu in hdul:
        has_table = hdu.data is not None and getattr(hdu, "columns", None) is not None
        if has_table and "TIME" in hdu.columns.names and "COUNTS" in hdu.columns.names:
            return hdu
    if len(hdul) > 1 and hdul[1].data is not None:
        return hdul[1]
    raise DataIngestionError("Could not locate TIME/COUNTS binary table in SoLEXS file.")


def load_helios(path: Path, hdu_index: int = 5) -> pd.DataFrame:
    """
    Loads one HEL1OS detector's light curve file at the given energy-band HDU.

    Returns DataFrame: ['timestamp', 'hel1os_ctr', 'hel1os_err', 'is_valid'].
    'is_valid' here is a NaN/negative-value check only — HEL1OS's Level-1
    GTI is a documented no-op, so it can't be used the way SoLEXS's is.
    """
    path = Path(path)
    if not path.exists():
        raise DataIngestionError(f"HEL1OS file not found: {path}")

    logger.info("Loading HEL1OS light curve: %s (HDU[%d])", path, hdu_index)

    try:
        with fits.open(path) as hdul:
            if hdu_index >= len(hdul):
                raise DataIngestionError(
                    f"HDU index {hdu_index} out of range "
                    f"(file has {len(hdul)} HDUs): {path}"
                )
            band_hdu = hdul[hdu_index]
            detector_name = band_hdu.header.get("DETNAM", "UNKNOWN")
            mjd_raw = band_hdu.data["MJD"]
            isot_raw = band_hdu.data["ISOT"]
            ctr_raw = band_hdu.data["CTR"]
            err_raw = band_hdu.data["STAT_ERR"]
    except KeyError as exc:
        raise DataIngestionError(
            f"Expected column missing in HEL1OS HDU[{hdu_index}] of {path}: {exc}"
        ) from exc
    except OSError as exc:
        raise DataIngestionError(f"Failed to open HEL1OS file {path}: {exc}") from exc

    mjd = fix_byte_order(mjd_raw)
    ctr = fix_byte_order(ctr_raw)
    err = fix_byte_order(err_raw)

    timestamp = Time(mjd, format="mjd", scale="utc").datetime64
    timestamp = pd.to_datetime(timestamp, utc=True).as_unit("ns")

    isot_parsed = pd.to_datetime(np.array(isot_raw, dtype=str), utc=True, errors="coerce")
    time_diff = pd.Series(timestamp) - pd.Series(isot_parsed)
    mjd_isot_mismatch = (time_diff.abs() > pd.Timedelta(seconds=1)).to_numpy()
    n_mismatch = int(mjd_isot_mismatch.sum())
    if n_mismatch:
        logger.warning(
            "%d/%d HEL1OS rows have MJD/ISOT mismatch > 1s; keeping MJD-derived "
            "timestamp, flagging rows invalid.", n_mismatch, len(timestamp),
        )

    is_valid = (~np.isnan(ctr)) & (ctr >= 0) & (~mjd_isot_mismatch)

    df = pd.DataFrame(
        {
            "timestamp": timestamp,
            "hel1os_ctr": ctr,
            "hel1os_err": err,
            "is_valid": is_valid,
        }
    )

    before = len(df)
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        logger.warning("Dropped %d duplicate timestamp rows.", dropped)

    logger.info(
        "HEL1OS [%s]: %d samples (%s -> %s); %d flagged invalid",
        detector_name, len(df), df["timestamp"].min(), df["timestamp"].max(),
        (~df["is_valid"]).sum(),
    )
    return df


def load_helios_multi(paths: list[Path], hdu_index: int = 5) -> pd.DataFrame:
    """
    Loads and stitches together multiple HEL1OS chunk files for ONE detector
    (a single day is often split across several dump files — see README).

    Raises DataIngestionError if the files don't all share the same DETNAM.
    Returns the same shape as load_helios(), spanning all input files.
    """
    if not paths:
        raise DataIngestionError("load_helios_multi() received an empty file list.")

    frames = []
    detectors_seen = set()

    for p in paths:
        df = load_helios(p, hdu_index=hdu_index)
        with fits.open(p) as hdul:
            detectors_seen.add(hdul[hdu_index].header.get("DETNAM", "UNKNOWN"))
        frames.append(df)

    if len(detectors_seen) > 1:
        raise DataIngestionError(
            f"load_helios_multi() received files from DIFFERENT detectors "
            f"({detectors_seen}) — pass files from a single detector only."
        )

    combined = pd.concat(frames, ignore_index=True)

    before = len(combined)
    combined = (
        combined.drop_duplicates(subset="timestamp")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    dropped = before - len(combined)
    if dropped:
        logger.warning(
            "load_helios_multi: dropped %d duplicate timestamp rows across "
            "%d input files (likely overlapping chunk boundaries).",
            dropped, len(paths),
        )

    time_diffs = combined["timestamp"].diff().dropna()
    max_gap = time_diffs.max() if len(time_diffs) else pd.Timedelta(0)
    if max_gap > pd.Timedelta(seconds=5):
        logger.warning(
            "load_helios_multi: largest gap between consecutive samples is "
            "%s — you may be missing a chunk file.", max_gap,
        )

    logger.info(
        "load_helios_multi [%s]: combined %d files into %d samples (%s -> %s)",
        detectors_seen, len(paths), len(combined),
        combined["timestamp"].min(), combined["timestamp"].max(),
    )
    return combined


class DataIngestionPipeline:
    """Loads SoLEXS + HEL1OS and merges them onto a common time index."""

    def __init__(self, config: IngestionConfig):
        self.config = config

    def run(self) -> pd.DataFrame:
        """Returns the merged DataFrame, indexed by timestamp (UTC)."""
        solexs_df = load_solexs(self.config.solexs_lc_path, self.config.solexs_gti_path)
        solexs_df = solexs_df.rename(columns={"is_valid": "solexs_is_valid"})

        hel1os_df = load_helios_multi(self.config.hel1os_paths, self.config.hel1os_hdu_index)
        hel1os_df = hel1os_df.rename(columns={"is_valid": "hel1os_is_valid"})

        merged = self._align(solexs_df, hel1os_df)
        merged = merged.set_index("timestamp").sort_index()

        logger.info(
            "Pipeline complete: %d aligned rows spanning %s -> %s",
            len(merged), merged.index.min(), merged.index.max(),
        )
        return merged

    def _align(self, solexs_df: pd.DataFrame, hel1os_df: pd.DataFrame) -> pd.DataFrame:
        """Aligns HEL1OS onto the SoLEXS time grid via nearest-neighbor merge."""
        merged = pd.merge_asof(
            solexs_df.sort_values("timestamp"),
            hel1os_df.sort_values("timestamp")[
                ["timestamp", "hel1os_ctr", "hel1os_err", "hel1os_is_valid"]
            ],
            on="timestamp",
            direction="nearest",
            tolerance=pd.Timedelta(self.config.merge_tolerance),
        )

        unmatched = merged["hel1os_ctr"].isna().sum()
        if unmatched:
            logger.warning(
                "%d/%d SoLEXS samples had no HEL1OS match within tolerance (%s).",
                unmatched, len(merged), self.config.merge_tolerance,
            )

        # merge_asof leaves NaN for unmatched rows, which upcasts this bool
        # column to float64. An unmatched row = no real HEL1OS data = invalid.
        merged["hel1os_is_valid"] = merged["hel1os_is_valid"].fillna(False).astype(bool)

        return merged

    def save(self, df: pd.DataFrame, filename: str = "aligned_flux.parquet") -> Path:
        """Persists the aligned DataFrame to disk as Parquet."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.config.output_dir / filename
        df.to_parquet(out_path)
        logger.info("Saved aligned dataset -> %s", out_path)
        return out_path
