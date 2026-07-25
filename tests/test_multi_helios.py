"""Test load_helios_multi: 2 chunk files, same detector, different time
windows on the same day — mirroring the real czt1.fits / czt1_2.fits case."""

import numpy as np
from astropy.io import fits
from astropy.time import Time
from pathlib import Path
import tempfile
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ingestion"))

from data_ingestion import (
    IngestionConfig, DataIngestionPipeline, load_helios_multi, DataIngestionError,
)

tmp = Path(tempfile.gettempdir()) / "knoxis_test3"
tmp.mkdir(exist_ok=True)


def make_helios_chunk(fname, t_start, n, detector="CZT1"):
    mjd_arr = Time(t_start + np.arange(n), format="unix").mjd.astype(">f8")
    isot_arr = np.array(
        [t.isot for t in Time(t_start + np.arange(n), format="unix")], dtype="S30"
    )
    ctr_arr = (30 + 5 * np.cos(np.arange(n) / 20)).astype(">f8")
    err_arr = np.full(n, 0.5, dtype=">f8")
    cols = [
        fits.Column(name="MJD", format="D", array=mjd_arr),
        fits.Column(name="ISOT", format="30A", array=isot_arr),
        fits.Column(name="CTR", format="D", array=ctr_arr),
        fits.Column(name="STAT_ERR", format="D", array=err_arr),
    ]
    hdul = [fits.PrimaryHDU()]
    for _ in range(5):
        band_hdu = fits.BinTableHDU.from_columns(cols)
        band_hdu.header["DETNAM"] = detector
        hdul.append(band_hdu)
    fits.HDUList(hdul).writeto(tmp / fname, overwrite=True)


t0 = 1782000000.0
# Chunk 1: seconds 0-299 of the day. Chunk 2: seconds 300-599 (no overlap,
# mirroring "czt1" covering the first part of the day, "czt1_2" the rest).
make_helios_chunk("czt1.fits", t0, 300, detector="CZT1")
make_helios_chunk("czt1_2.fits", t0 + 300, 300, detector="CZT1")

# ---- Test 1: load_helios_multi stitches both chunks into one continuous span ----
combined = load_helios_multi([tmp / "czt1.fits", tmp / "czt1_2.fits"], hdu_index=5)
print("--- Combined HEL1OS (2 chunks) ---")
print(combined.head(3))
print(combined.tail(3))
print("Total rows:", len(combined))
assert len(combined) == 600, f"Expected 600 combined rows, got {len(combined)}"
assert combined["timestamp"].is_monotonic_increasing, "Timestamps should be sorted"
print("Chunk stitching PASSED\n")

# ---- Test 2: mismatched detectors should raise an error, not silently mix data ----
make_helios_chunk("czt2_wrong.fits", t0 + 300, 300, detector="CZT2")
try:
    load_helios_multi([tmp / "czt1.fits", tmp / "czt2_wrong.fits"], hdu_index=5)
    print("FAILED: expected DataIngestionError for mismatched detectors, none raised")
except DataIngestionError as e:
    print("Mismatched-detector guard PASSED:", e)

# ---- Test 3: full pipeline with a list of HEL1OS files ----
import shutil
shutil.copy(tmp / "czt1.fits", tmp / "solexs_placeholder_not_used.fits")  # unused, just tidy

# Reuse synthetic SoLEXS + GTI from earlier test setup
n = 600
time_arr = (t0 + np.arange(n)).astype(">f8")
counts_arr = (50 + 10 * np.sin(np.arange(n) / 30)).astype(">f8")
col1 = fits.Column(name="TIME", format="D", array=time_arr)
col2 = fits.Column(name="COUNTS", format="D", array=counts_arr)
rate_hdu = fits.BinTableHDU.from_columns([col1, col2], name="RATE")
fits.HDUList([fits.PrimaryHDU(), rate_hdu]).writeto(tmp / "solexs.lc", overwrite=True)

starts = np.array([t0], dtype=">f8")
stops = np.array([t0 + n - 1], dtype=">f8")
gti_col1 = fits.Column(name="START", format="D", array=starts)
gti_col2 = fits.Column(name="STOP", format="D", array=stops)
gti_hdu = fits.BinTableHDU.from_columns([gti_col1, gti_col2], name="GTI")
fits.HDUList([fits.PrimaryHDU(), gti_hdu]).writeto(tmp / "solexs.gti", overwrite=True)

config = IngestionConfig(
    solexs_lc_path=tmp / "solexs.lc",
    solexs_gti_path=tmp / "solexs.gti",
    hel1os_paths=[tmp / "czt1.fits", tmp / "czt1_2.fits"],  # <-- list of 2 chunks
    hel1os_hdu_index=5,
    output_dir=tmp / "processed",
)
pipeline = DataIngestionPipeline(config)
df = pipeline.run()
print("\n--- Full pipeline with multi-file HEL1OS ---")
print(df.shape)
assert len(df) == 600
assert df["hel1os_ctr"].isna().sum() == 0, "Expected full HEL1OS coverage, found gaps"
print("Full pipeline with chunked HEL1OS PASSED")

print("\nALL TESTS PASSED")