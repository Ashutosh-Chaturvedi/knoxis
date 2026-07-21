# Knoxis

Dual-mode solar flare forecasting and nowcasting system built on real Aditya-L1 satellite data (ISRO PRADAN portal).

## Overview

Knoxis combines soft X-ray (SoLEXS) and hard X-ray (HEL1OS CZT) data from Aditya-L1 to detect and forecast solar flares. Its core novelty is the use of a **Hardness Ratio** (HEL1OS / SoLEXS flux) as a precursor signal — real data from June 21, 2026 shows a ~20-minute hardness-ratio shift preceding two detected flares, ahead of any flux threshold crossing.

## System Design

- **Nowcast Engine** — rule-based threshold detection (flux > 1.4x 24hr baseline, sustained >3 min, confirmed via HEL1OS), classifies flares as B/C/M/X class, outputs a flare catalog.
- **Forecast Engine** — Random Forest classifier over an 8-feature rolling 30-minute window (SoLEXS mean/slope/std, HEL1OS mean/slope, Hardness Ratio, HR rate of change, peak-to-mean ratio), predicting flare probability 30–60 minutes ahead.
- **Alerting** — four-level system: QUIET / WATCH / WARNING / ALERT.
- **Interfaces** — FastAPI backend, Streamlit dashboard.

## Repository Structure

```
Knoxis/
├── ingestion/    # SoLEXS + HEL1OS data loaders and preprocessing
├── nowcast/      # Threshold-based detection engine
├── forecast/     # Random Forest forecasting pipeline
├── api/          # FastAPI backend
├── dashboard/    # Streamlit dashboard
├── tests/        # Test scripts
├── notebooks/    # Exploratory analysis
├── data/         # Raw/processed data (gitignored)
├── requirements.txt
└── README.md
```

## Data Sources

- **SoLEXS** (`.lc` FITS) — soft X-ray flux, `TIME` (Unix seconds) and `COUNTS` columns, 1-second cadence.
- **HEL1OS CZT** (`.fits`) — hard X-ray flux, 5 energy-band HDUs; broadband channel (18–160 keV) at HDU[5], with `MJD`, `ISOT`, `CTR`, `STAT_ERR` columns.

Data sourced from [ISRO's PRADAN portal](https://pradan.issdc.gov.in/).

## Tech Stack

Python, NumPy, Pandas, astropy, scikit-learn, Matplotlib, Streamlit, FastAPI.

## Status

Actively in development. Data ingestion pipeline in progress.
