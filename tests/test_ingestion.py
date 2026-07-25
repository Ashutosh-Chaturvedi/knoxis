"""Smoke test: synthetic SoLEXS .lc + .gti + HEL1OS .fits files (big-endian,
mimicking real PRADAN structure), run through the full pipeline."""

import numpy as np
from astropy.io import fits
from astropy.time import Time
from pathlib import Path
import sys
from pathlib import Path
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ingestion"))
from data_ingestion import IngestionConfig, DataIngestionPipeline, load_solexs, load_solexs_gti

tmp = Path(tempfile.gettempdir()) / "knoxis_test2"
tmp.mkdir(exist_ok=True)

# ---- synthetic SoLEXS .lc file: 600s, with 2 deliberate NaN gaps ----
n = 600
t0 = 1782000000.0
time_arr = (t0 + np.arange(n)).astype(">f8")
counts_arr = (50 + 10 * np.sin(np.arange(n) / 30)).astype(">f8")
# Deliberate gaps at index 100 and 300, matching the real pattern:
# NaN in counts AND a corresponding GTI gap.
counts_arr[100] = np.nan
counts_arr[300] = np.nan

col1 = fits.Column(name="TIME", format="D", array=time_arr)
col2 = fits.Column(name="COUNTS", format="D", array=counts_arr)
rate_hdu = fits.BinTableHDU.from_columns([col1, col2], name="RATE")
fits.HDUList([fits.PrimaryHDU(), rate_hdu]).writeto(tmp / "solexs_test.lc", overwrite=True)

# ---- synthetic GTI file: two windows, with a gap exactly at index 100 and 300 ----
starts = np.array([t0, t0 + 101, t0 + 301], dtype=">f8")
stops = np.array([t0 + 99, t0 + 299, t0 + n - 1], dtype=">f8")
gti_col1 = fits.Column(name="START", format="D", array=starts)
gti_col2 = fits.Column(name="STOP", format="D", array=stops)
gti_hdu = fits.BinTableHDU.from_columns([gti_col1, gti_col2], name="GTI")
fits.HDUList([fits.PrimaryHDU(), gti_hdu]).writeto(tmp / "solexs_test.gti", overwrite=True)

# ---- synthetic HEL1OS file, 6 HDUs (index 0 primary + 5 bands), HDU[5] used ----
mjd_arr = Time(t0 + np.arange(0, n, 2), format="unix").mjd.astype(">f8")
isot_arr = np.array(
    [t.isot for t in Time(t0 + np.arange(0, n, 2), format="unix")], dtype="S30"
)
ctr_arr = (30 + 5 * np.cos(np.arange(len(mjd_arr)) / 20)).astype(">f8")
err_arr = np.full(len(mjd_arr), 0.5, dtype=">f8")

cols = [
    fits.Column(name="MJD", format="D", array=mjd_arr),
    fits.Column(name="ISOT", format="30A", array=isot_arr),
    fits.Column(name="CTR", format="D", array=ctr_arr),
    fits.Column(name="STAT_ERR", format="D", array=err_arr),
]

hdul = [fits.PrimaryHDU()]
for i in range(5):
    band_hdu = fits.BinTableHDU.from_columns(cols)
    band_hdu.header["DETNAM"] = "CZT1"
    hdul.append(band_hdu)
fits.HDUList(hdul).writeto(tmp / "hel1os_test.fits", overwrite=True)

# ---- Test 1: load_solexs_gti in isolation ----
gti_df = load_solexs_gti(tmp / "solexs_test.gti")
print("--- GTI windows ---")
print(gti_df)
assert gti_df["start"].dtype == np.float64

# ---- Test 2: load_solexs with GTI validity flagging ----
solexs_df = load_solexs(tmp / "solexs_test.lc", tmp / "solexs_test.gti")
print("\n--- SoLEXS head ---")
print(solexs_df.head(5))
print("\n--- Rows flagged invalid ---")
print(solexs_df[~solexs_df["is_valid"]])

# The two deliberate NaNs (index 100, 300) should both be flagged invalid,
# since we built the GTI gaps to line up with them exactly.
nan_rows = solexs_df[solexs_df["solexs_counts"].isna()]
print("\n--- NaN rows all flagged invalid?", (~nan_rows["is_valid"]).all(), "---")
assert (~nan_rows["is_valid"]).all(), "Expected all NaN rows to be GTI-invalid"

# ---- Test 3: full pipeline ----
config = IngestionConfig(
    solexs_lc_path=tmp / "solexs_test.lc",
    solexs_gti_path=tmp / "solexs_test.gti",
    hel1os_paths=[tmp / "hel1os_test.fits"],
    hel1os_hdu_index=5,
    output_dir=tmp / "processed",
)
pipeline = DataIngestionPipeline(config)
df = pipeline.run()
out_path = pipeline.save(df)

print("\n--- Merged pipeline output head ---")
print(df.head())
print("\n--- dtypes ---")
print(df.dtypes)

assert df["solexs_counts"].dtype == np.float64
assert df["hel1os_ctr"].dtype == np.float64
assert "solexs_is_valid" in df.columns
assert "hel1os_is_valid" in df.columns

print("\nSMOKE TEST PASSED")