# Knoxis Forecast Module

Predicts elevated solar flare risk over the next 60 minutes, using engineered features from combined SoLEXS + HEL1OS data.

## Folder structure

- **`pipeline/`** — current, live code: data combination, feature engineering, labeling, model training/tuning, alert thresholding, and validation/testing scripts. This is what actually runs.
- **`investigation/`** — diagnostic scripts and notebooks from the model-development investigation (signal-existence checks, covariate-shift testing, GTI-completeness validation). Not run as part of the normal pipeline, but kept as a real record of how the current design was reached.
- **`archive/`** — superseded model attempts (original Random Forest, two-stage architecture, 5-class LightGBM), kept for reference. **Not the current model** — see below.

## Current status

The forecast target is **binary** (flare / no-flare in the next 60 minutes), not multi-class severity prediction. This was a deliberate scope correction, not the original plan — a 5-class model (QUIET/B/C/M/X) was tried first and found to have real, working skill at detecting *whether* a flare is coming, but near-chance skill at predicting *which exact class*. The binary reframing matches the model's actual demonstrated capability, and is consistent with published literature (X-ray flux alone tends to have limited skill for pre-onset severity forecasting; magnetic field data is typically needed for that specific task).

**Live model:** `pipeline/train_binary_flare_model.py` (or the tuned version via `pipeline/tune_binary_flare_model.py`)
**Result:** PR-AUC 1.29–1.30x above the no-skill baseline on real, held-out test data — a real, modest, statistically confirmed signal.

### How this conclusion was reached

Five hypotheses were tested, in order, before settling on the binary framing above:

1. **Covariate shift** (train/test periods showing genuinely different feature distributions) — confirmed real, but ruled out as the *dominant* cause via a within-single-period control test.
2. **Class imbalance architecture** (a two-stage flare-detection → severity-classification pipeline) — didn't meaningfully improve real-data results; removing QUIET just moved the imbalance problem down a level (the C:X ratio stayed ~80:1 even after excluding QUIET).
3. **Algorithm choice** (Random Forest → LightGBM) — a genuine, real improvement in rare-class recall, but revealed the 5-class probability output was poorly calibrated (a near-flat precision-recall curve on real data).
4. **Shorter forecast horizon** (60 min → 20 min) — tested via a cheap statistical pre-check; showed no improvement, so a full retrain wasn't pursued.
5. **Hyperparameter tuning** — negligible gain, itself informative: if the bottleneck were poor model configuration, tuning should have helped more than it did.

A direct statistical test confirmed real (if modest) signal exists in several features for predicting *whether* a flare occurs, but severity-class separation was consistently weak. Reframing to binary classification produced a working, honestly-characterized result, consistent with published literature — X-ray flux alone tends to have limited skill for pre-onset severity forecasting specifically, since it measures energy already being released rather than the stored magnetic energy that actually determines what's about to happen.

## A note on `forecast_alert.py`

Converts the model's raw probability output into graduated QUIET/WATCH/WARNING/ALERT levels (matching the nowcast engine's alert philosophy), with thresholds chosen from a real precision-recall table rather than guessed. Does **not** provide a severity-class estimate — an earlier version did, but that capability was removed after being shown to perform below random chance (19.6% vs. a 25% baseline for 4 classes).
