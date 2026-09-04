# Knoxis

Dual-mode solar flare **nowcasting** and **forecasting** system built on real Aditya-L1 satellite data (ISRO's PRADAN portal).

Combines soft X-ray (SoLEXS) and hard X-ray (HEL1OS) data to:
1. **Nowcast** — detect a solar flare already in progress and classify its strength
2. **Forecast** — estimate the likelihood of a flare in the next 60 minutes

Each module below has its own README with full design details, findings, and current status — this file is just a map.

## Repository Structure

```
knoxis/
├── ingestion/      # SoLEXS + HEL1OS data loaders and preprocessing
├── nowcast/        # Real-time flare detection and classification
├── forecast/       # Feature engineering, labeling, and the forecast model
├── tests/          # Test scripts
├── data/           # Raw/intermediate/output data (gitignored)
├── requirements.txt
└── README.md       # you are here
```

See each folder's own `README.md` for details on what it does, how it works, and what's been tried.

## Data Sources

- **SoLEXS / HEL1OS** — Aditya-L1's soft/hard X-ray instruments, via [ISRO's PRADAN portal](https://pradan.issdc.gov.in/)
- **NOAA Solar Event Reports** — real historical flare classifications, used as ground truth, via [NCEI's Space Weather archive](https://www.ngdc.noaa.gov/stp/space-weather/swpc-products/daily_reports/solar_event_reports/)

## Tech Stack

Python, pandas, NumPy, astropy, scikit-learn, LightGBM, Matplotlib, FastAPI.

## Status

| Module | Status |
|---|---|
| Data ingestion | Done |
| Nowcast engine | Done |
| Forecast module | Done — see `forecast/README.md` for what it can and can't currently do |
| API / serving layer | In progress |
| Dashboard | Not started |
