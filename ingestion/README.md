# `data_ingestion.py` — How It Works

This module turns raw Aditya-L1 satellite files (SoLEXS + HEL1OS) into one
clean, merged dataset the rest of the pipeline can trust. This doc holds all
the *why* behind the code's design decisions — the code itself is kept
deliberately lean (short docstrings only), so read this alongside it rather
than expecting the reasoning to live in-line.

## The core problem this module solves

Two instruments, same spacecraft, same sun — but nothing else matches:
different time systems, different file structures, different byte ordering
vs. our machines, and different ways of signaling "this data point isn't
trustworthy." This module's job is to make both instruments speak the same
language before nowcast/forecast have to think about them.

## Function-by-function design notes

### `fix_byte_order(arr)`
FITS files store numbers in **big-endian** byte order (an old astronomy
standard); our machines are little-endian. NumPy 2.0 refuses certain
operations (rolling means, hashing, indexing) on big-endian arrays outright,
raising errors like `"Big-endian buffer not supported on little-endian
compiler"`.

The fix is two independent steps, not one:
1. `byteswap()` — physically flips the bytes in memory.
2. `.view(newdtype)` — relabels the dtype as "native" *without* touching the
   bytes again (touching them twice would corrupt the values).

Every raw numeric FITS column passes through this before touching pandas.

### `load_solexs_gti()` + `_flag_gti_validity()`
SoLEXS ships with a separate **GTI (Good Time Interval)** file — a small
table of `[start, stop]` windows during which the instrument was actually
working properly (Sun in view, detector healthy, etc.). We verified
empirically on real June 21, 2026 data that every NaN in SoLEXS's counts
column falls exactly inside a gap *between* GTI windows — meaning NaN and
"GTI-invalid" are the same signal. That's why **we never interpolate SoLEXS
gaps** — doing so would mean inventing values during a period ISRO's own
pipeline flags as unreliable. We only label rows via `is_valid` and let
downstream rolling-window logic skip them (e.g. via `min_periods`).

`_flag_gti_validity` checks every timestamp against every GTI window in one
vectorized NumPy broadcast (`(N,1)` timestamps against `(1,M)` windows,
producing an `(N,M)` grid, collapsed with `.any(axis=1)`) rather than
looping row-by-row in Python — same logic, much faster, since M (GTI
windows per day) is small but N (samples per day) is 86,400.

### `load_solexs()`
Loads TIME/COUNTS, fixes byte order, labels `is_valid` via GTI. **Never
drops or fills NaNs** — labeling and cleaning are kept as separate concerns
on purpose, so the cleaning strategy can change later without touching the
parsing code.

### `load_helios()`
Structurally mirrors `load_solexs`, but the internals differ because the
instrument differs:
- Converts MJD → UTC datetime via `astropy.time.Time`, then forces
  nanosecond resolution (`.as_unit("ns")`) — this fixes a real bug we hit
  where `datetime64[s]` vs `datetime64[ns]` silently broke `merge_asof`.
- Cross-checks MJD against the redundant ISOT string column, flagging rows
  where they disagree by more than a second (usually a corrupted row).
- **Important asymmetry vs. SoLEXS**: HEL1OS's own GTI file is a documented
  no-op at Level-1 (it just spans the entire data dump), so we *can't* use
  the SoLEXS approach here. Instead, `is_valid` falls back to a weaker
  signal: is the count rate NaN, negative (physically impossible), or
  internally inconsistent (MJD/ISOT mismatch)?

### `load_helios_multi()`
**Why this exists:** unlike SoLEXS (one `.lc` file = one full calendar day),
HEL1OS data is dumped in chunks of arbitrary duration — the manual's own
zip naming convention (`..._43191sec_...`) confirms a dump is *not*
guaranteed to be a fixed 86,400-second day. In practice, a single day for
one detector is often split across two or more files (e.g. `czt1.fits` and
`czt1_2.fits` covering different time windows of the same day). Calling
`load_helios()` on just one of these silently gives you an incomplete day
with no error — a real gap we caught before shipping.

This function loads every file, verifies they all report the same `DETNAM`
(catches an accidental mix-up, e.g. passing a CZT1 file and a CZT2 file
together — this would silently mix two instruments' data if not caught),
concatenates them, and re-sorts/deduplicates on timestamp, since chunks
aren't guaranteed to arrive in time order. It also warns if there's a
suspiciously large gap between consecutive samples — a sign a chunk file is
missing entirely, not just a normal instrument dropout.

### `DataIngestionPipeline._align()`
SoLEXS is the reference time grid (it's the primary nowcast signal); HEL1OS
gets aligned onto it via `pd.merge_asof` — a **nearest-timestamp** join, not
an exact-match join, since the two instruments don't tick at identical
instants. `tolerance` caps how far apart a "match" is allowed to be before
pandas gives up and inserts NaN instead.

**A real bug we hit and fixed here:** `merge_asof` leaves NaN in any row
with no match within tolerance. A boolean column (`hel1os_is_valid`) can't
hold NaN, so pandas silently widens the whole column to `float64` — which
then crashes the moment anyone runs `~df["hel1os_is_valid"]` (`TypeError:
bad operand type for unary ~: 'float'`). The fix: explicitly treat
unmatched rows as `False` (no real HEL1OS data = definitely not valid) and
restore the proper `bool` dtype after the merge.

One subtlety to know about: because of the `2s` tolerance, a SoLEXS sample
up to 2 seconds past HEL1OS's last real sample can still get "matched" by
reusing that last known value — so `hel1os_is_valid=True` means "a real
HEL1OS sample was close enough in time," not "we have a fresh sample at
this exact second."

## Known limitations / things to watch for

- **One HEL1OS file = one detector's worth of data.** There are 4 separate
  physical detectors (CdTe1, CdTe2, CZT1, CZT2), each potentially split
  across multiple chunk files. `load_helios_multi()` combines chunks for
  *one* detector; combining across detectors is not handled and isn't
  currently needed (the project uses CZT full-broadband, HDU[5]).
- **Small numbers of invalid rows at day boundaries are expected, not
  bugs.** On real June 21, 2026 data we saw ~14 HEL1OS rows flagged invalid,
  clustered at the very start and end of the day (`00:00:00`-`00:00:03` and
  `23:59:50`-`23:59:59`). This is because HEL1OS data dumps don't
  necessarily start/stop at exact midnight — the manual's own example
  header shows a dump starting at `23:59:56.024 UT`. Don't chase this as a
  bug; it's a benign consequence of two independently-scheduled
  instruments.
- **Don't assume `.astype(int64)` behaves the same across datetime
  dtypes.** We hit a real bug where `datetime64[s]` vs `datetime64[ns]`
  needed completely different conversion math when checking timestamps
  against GTI windows — always check the dtype before converting.
- **NaNs in SoLEXS are meaningful, not noise.** Don't be tempted to
  `.fillna()` or `.interpolate()` them — we have direct evidence they
  correspond to real, instrument-flagged bad data (see `load_solexs_gti`
  section above).

## Verification status

Tested against synthetic FITS files that mimic the real structure
(`tests/test_ingestion.py`, `tests/test_multi_helios.py`,
`tests/test_unmatched_merge.py`), and cross-checked against real 2026-06-21
SDD2 SoLEXS data + CZT1 HEL1OS data (2 chunk files). Real-data results:
86,400 rows (one full day), 5 SoLEXS rows flagged invalid (matches manual
GTI investigation exactly), 14 HEL1OS rows flagged invalid (explained
above).

## Not yet done
- Running against multiple days / multiple detectors beyond CZT1
- Deciding how `is_valid=False` rows get handled downstream in the
  nowcast/forecast modules (currently: labeled only, not dropped)