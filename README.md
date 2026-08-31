# Knoxis

Dual-mode solar flare **nowcasting** and **forecasting** system built on real Aditya-L1 satellite data (ISRO's PRADAN portal).

## Overview

Knoxis combines soft X-ray (SoLEXS) and hard X-ray (HEL1OS) data from ISRO's Aditya-L1 mission to:

1. **Nowcast** — detect a solar flare already in progress, in near real time, and classify its strength (B/C/M/X, following the real GOES scale)
2. **Forecast** — predict the likelihood and class of a flare in the next 60 minutes, using a machine-learning model trained on real historical data

## A note on the Hardness Ratio

Early in this project, the ratio between HEL1OS and SoLEXS flux (the "Hardness Ratio") was investigated as a possible **real-time precursor signal** — the idea being that a rise in this ratio might warn of an approaching flare before flux itself crosses any threshold.

**This was tested rigorously and rejected as a nowcast trigger.** Comparing the ratio's behavior before two confirmed real flares against a matched quiet-period control showed the same elevated, oscillating pattern occurs during ordinary quiet conditions — meaning it does not reliably discriminate "a flare is coming" from "nothing is happening." This negative result is documented, not hidden: the Hardness Ratio is retained only as one of several engineered inputs to the forecasting model, where its actual predictive value (if any) is learned and validated statistically through training, rather than assumed from visual inspection.

## System Design

### Nowcast Engine
- **Trigger:** live SoLEXS flux crossing 1.4x a rolling 24-hour baseline, sustained for at least 3 minutes before being confirmed (filters out momentary noise spikes)
- **Classification:** a real, evidence-based counts-to-GOES-flux calibration, derived from two independently confirmed real flares (M2.6 and M6.8 on 2026-06-21), agreeing to within 0.02%
- **Alerts:** four-level graduated scale (QUIET / WATCH / WARNING / ALERT) based on proximity to the confirmed-flare threshold
- **HEL1OS's role:** non-blocking corroboration only — logged alongside each detection, never gating the alert. This is deliberate: HEL1OS is known to read exactly `0.0` (with `0.0` statistical error) during the most extreme flares, likely detector saturation/dead-time at extreme photon flux. Making HEL1OS a hard requirement risked suppressing an alert for exactly the flares that matter most.

### Forecast Engine
- **Data:** ~2.5 years of real Aditya-L1 data (Feb 2024 - Aug 2026), ingested, combined, and feature-engineered day by day
- **Features:** 9 engineered features per 30-minute rolling window (SoLEXS mean/slope/std, HEL1OS mean/slope, Hardness Ratio, HR rate of change, peak-to-mean ratio, and a HEL1OS-saturation-fraction feature — see note below)
- **Labels:** real GOES-class outcomes from NOAA's daily solar event reports, matched to a 60-minute forward-looking horizon per window
- **Model:** Random Forest (training not yet complete — dataset is built and verified, train/test split and training are the next step)

### HEL1OS saturation handling
Confirmed on two independent real flares: HEL1OS's CZT1 detector reads exactly `ctr=0, err=0` (not just low counts) during the most extreme flux — a real instrument behavior (likely saturation/dead-time), not a data-quality artifact. Left unhandled, this would teach a model that extreme flares look like *low* hard-X-ray activity — the opposite of the truth. The feature pipeline detects these episodes (as contiguous zero-runs coinciding with elevated flux) and reports the affected HEL1OS-derived features as missing rather than fabricating a misleading number, while exposing the saturation episode itself as its own feature.

## Repository Structure

```
knoxis/
├── ingestion/      # SoLEXS + HEL1OS data loaders and preprocessing
├── nowcast/        # Threshold-based detection + classification engine
├── forecast/       # Feature engineering, NOAA-based labeling, model training pipeline
├── tests/          # Test scripts
├── data/           # Raw/intermediate/output data (gitignored)
├── requirements.txt
└── README.md
```

## Data Sources

- **SoLEXS** (`.lc`/`.gti` FITS) — soft X-ray flux, 1-second cadence, from Aditya-L1's Solar Low Energy X-ray Spectrometer
- **HEL1OS** (CZT1 lightcurve FITS) — hard X-ray flux, 1-second cadence, from Aditya-L1's High Energy L1 Orbiting X-ray Spectrometer
- **NOAA Solar Event Reports** — real historical flare classifications (begin/max/end time, GOES class), used as ground-truth labels for the forecast model

Aditya-L1 data sourced from [ISRO's PRADAN portal](https://pradan.issdc.gov.in/). NOAA event reports from [NCEI's Space Weather archive](https://www.ngdc.noaa.gov/stp/space-weather/swpc-products/daily_reports/solar_event_reports/).

## Tech Stack

Python, pandas, NumPy, astropy, scikit-learn, Matplotlib, FastAPI.

## Status

- Data ingestion (SoLEXS + HEL1OS) — built, tested against real data
- Nowcast engine — built and tested, using a real evidence-based flux-to-GOES calibration
- Forecast training dataset — complete: 787 real days, 226,656 labeled 30-minute windows, verified against known real flares and cross-checked for coverage gaps
- Forecast model training — dataset is ready; train/test split and Random Forest training are next
- Live dashboard / API — not currently in scope

## Known Limitations

- SoLEXS data has real gaps totaling 126 of 912 days across the training period (~14%), costing roughly 8-15% of labeled flare events depending on class. This loss is approximately uniform across severity classes, not concentrated in the rarest/most valuable events.
- HEL1OS saturation detection is a heuristic (contiguous-zero-run + elevated-flux check), not derived from the instrument's own housekeeping saturation counters — a documented, deferred follow-up if post-training diagnostics suggest it's a meaningful source of error.
