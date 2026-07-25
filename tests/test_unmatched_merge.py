"""Test: deliberately leave a gap in HEL1OS coverage so some SoLEXS rows
have NO HEL1OS match, and confirm hel1os_is_valid stays a clean bool column."""

import numpy as np
from astropy.io import fits
from astropy.time import Time
from pathlib import Path
import tempfile
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ingestion"))

from data_ingestion import IngestionConfig, DataIngestionPipeline

tmp = Path(tempfile.gettempdir()) / "knoxis_test4"
tmp.mkdir(exist_ok=True)

t0 = 1782000000.0
n = 600

# SoLEXS: full 600-second coverage, no gaps.
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

# HEL1OS: DELIBERATELY only covers seconds 0-399 (missing the last 200
# seconds entirely) — simulating a missing chunk file.
n_hel1os = 400
mjd_arr = Time(t0 + np.arange(n_hel1os), format="unix").mjd.astype(">f8")
isot_arr = np.array(
    [t.isot for t in Time(t0 + np.arange(n_hel1os), format="unix")], dtype="S30"
)
ctr_arr = (30 + 5 * np.cos(np.arange(n_hel1os) / 20)).astype(">f8")
err_arr = np.full(n_hel1os, 0.5, dtype=">f8")
cols = [
    fits.Column(name="MJD", format="D", array=mjd_arr),
    fits.Column(name="ISOT", format="30A", array=isot_arr),
    fits.Column(name="CTR", format="D", array=ctr_arr),
    fits.Column(name="STAT_ERR", format="D", array=err_arr),
]
hdul = [fits.PrimaryHDU()]
for _ in range(5):
    band_hdu = fits.BinTableHDU.from_columns(cols)
    band_hdu.header["DETNAM"] = "CZT1"
    hdul.append(band_hdu)
fits.HDUList(hdul).writeto(tmp / "czt1_incomplete.fits", overwrite=True)

config = IngestionConfig(
    solexs_lc_path=tmp / "solexs.lc",
    solexs_gti_path=tmp / "solexs.gti",
    hel1os_paths=[tmp / "czt1_incomplete.fits"],
    hel1os_hdu_index=5,
    output_dir=tmp / "processed",
)
pipeline = DataIngestionPipeline(config)
df = pipeline.run()

print("--- dtypes ---")
print(df.dtypes)

print("\n--- hel1os_is_valid value counts ---")
print(df["hel1os_is_valid"].value_counts())

# The critical check: this line used to crash with
# "TypeError: bad operand type for unary ~: 'float'"
n_invalid = (~df["hel1os_is_valid"]).sum()
print("\nInvalid HEL1OS rows (via ~ operator):", n_invalid)

assert df["hel1os_is_valid"].dtype == bool, "hel1os_is_valid should be a clean bool dtype"
# We expect ~200 rows (the missing tail) to be correctly flagged invalid.
assert 195 <= n_invalid <= 200, f"Expected ~200 unmatched/invalid rows (allowing for merge tolerance), got {n_invalid}"

print("\nFIX VERIFIED")