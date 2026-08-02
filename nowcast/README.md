## Knoxis — Nowcast Engine.

Detects solar flares in progress from real-time SoLEXS flux, classifies
their strength using a real GOES-equivalent calibration (derived from two
matched flares on 2026-06-21: M2.6 and M6.8), and raises a graduated
QUIET/WATCH/WARNING/ALERT signal. HEL1OS is used as a non-blocking
corroboration signal only — never as a gate on the alert — since HEL1OS
is known to drop to zero for extended periods during extreme flares
(likely detector saturation), which could otherwise suppress an alert at
exactly the moment it matters most.

### Design notes 
    - The rolling baseline is TRAILING (uses only past data), not centered.
      A live nowcast system cannot see the future; a centered window (appropriate for offline analysis) would leak future information and is not valid here.
    - A threshold crossing only becomes a confirmed ALERT once it has been
      continuously sustained for `sustain_minutes` — this naturally delays
      confirmation by that duration in real time, which is the correct
      behavior for a live system (you cannot know something has been
      sustained for 3 minutes until 3 minutes have actually passed).
    - WATCH/WARNING boundaries are a linear graduation toward the real
      1.4x confirmed-flare threshold. They are a design choice, not
      empirically derived — documented as such.

### Calibration: 
Real GOES-equivalent conversion, derived from two matched flares on 2026-06-21 (M2.6 @ 1886 counts/s, M6.8 @ 4931 counts/s — the two independently-derived conversion factors agreed to within 0.02%).

### Step 1 — rolling baseline (trailing only — no future leakage)
---
#### def `compute_rolling_baseline`
- Computes a trailing rolling median baseline of solexs_counts, using only
    GTI-valid samples. Trailing (not centered) because a live system can
    only ever know the past, never the future.

    ### Returns
    -------
    pd.Series
        Baseline flux estimate, indexed the same as df. NaN wherever there
        aren't yet enough valid samples in the trailing window (e.g. the
        very start of a dataset with no prior history).

### Step 2 — threshold crossing + sustained-duration confirmation
---
#### def `detect_threshold_crossing`
---
- Returns a boolean Series: True wherever live flux exceeds
    threshold_ratio * baseline. Not yet duration-filtered — a single noisy
    second can trip this; see enforce_sustained_duration().

#### def `enforce_sustained_duration`
---
- Confirms a crossing only once it has been CONTINUOUSLY True for at least
    `sustain_minutes`. This delays confirmation by that duration relative to
    when the crossing actually began — correct behavior for a live system,
    since sustained-ness cannot be known until the time has actually passed.

- Implementation: identify contiguous True-runs, then within each run mark
    a row confirmed once the elapsed time since the run's start reaches the
    required duration.

### Step 3 — graduated alert level
---
#### def `compute_alert_level`
---
- Returns a categorical Series: 'QUIET', 'WATCH', 'WARNING', or 'ALERT'.

- ALERT requires a CONFIRMED (sustained) threshold crossing. WATCH/WARNING
    are earlier-stage signals as flux approaches the threshold, even before
    it's confirmed — giving graduated lead time rather than a single binary
    flag. Their exact boundaries are a design choice (linear graduation
    toward the real 1.4x threshold), not independently validated.

### Step 4 — HEL1OS corroboration (non-blocking)
---
#### def `check_hel1os_corroboration`
---
- Checks whether HEL1OS was ALSO elevated during a detected event window.
    This NEVER gates the alert — it's purely informational, logged
    alongside the event. Returns 'yes', 'no', or 'inconclusive' (the last
    when HEL1OS data in that window isn't trustworthy per its own
    is_valid flag — e.g. the known saturation dropout at extreme peaks).

- Note: HEL1OS's own quiet baseline is computed the same way as SoLEXS's,
    but independently, since the two instruments have very different count
    rates and are not directly comparable in absolute terms.

### Step 5 — build a flare catalog from confirmed events
---
#### def `build_flare_catalog`
- Groups contiguous CONFIRMED rows into discrete flare events.

    ### Returns
    -------
- pd.DataFrame
        Columns: detection_time (when the ALERT actually fired, i.e.
        sustain_minutes after true onset), peak_time, peak_counts,
        goes_class, hel1os_corroboration.


### Orchestration
---
### class `NowcastEngine`
Runs the full nowcast pipeline over an ingested SoLEXS+HEL1OS DataFrame
### def `run`:
    Parameters
    ----------
    df : pd.DataFrame
        Output of DataIngestionPipeline.run() — indexed by timestamp,
        with solexs_counts, solexs_is_valid, hel1os_ctr, hel1os_is_valid.

    Returns
    -------
    dict with keys:
        'baseline', 'alert_level' (per-timestamp Series),
        'flare_catalog' (DataFrame of discrete detected events