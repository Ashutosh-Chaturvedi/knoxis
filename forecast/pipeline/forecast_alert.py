"""
forecast_alert.py

Converts the trained LightGBM model's raw class probabilities into a
graduated alert output -- QUIET/WATCH/WARNING/ALERT, matching the
nowcast engine's philosophy -- instead of forcing a single hard
classification.

Why: the hard-classification LightGBM run showed a real, dramatic
improvement in rare-class recall over Random Forest, but at a serious
cost -- 57.5% of real QUIET windows got misclassified as some flare
class. That's the DEFAULT decision boundary's tradeoff point, not the
only one available. Exposing the model's actual probability (via
predict_proba) and choosing thresholds deliberately lets you pick WHERE
on the precision/recall tradeoff curve to operate.

Does NOT retrain anything -- this is a post-processing layer on top of
the already-trained model.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

FEATURE_COLUMNS = [
    "solexs_mean", "solexs_slope", "solexs_std",
    "hel1os_mean", "hel1os_slope",
    "hardness_ratio", "hr_rate_of_change",
    "peak_to_mean_ratio", "hel1os_saturation_fraction",
]


def compute_flare_probability(model, X: pd.DataFrame) -> pd.Series:
    """
    Returns P(flare) directly for every row -- the model's confidence
    that a flare will occur in the next 60 minutes.

    Works with the BINARY model (classes_ = [False, True]) -- this was
    originally written for the earlier 5-class model (which looked up
    "QUIET" by name and computed 1 - P(QUIET)). That approach broke
    outright once the project moved to binary classification, since
    there's no "QUIET" string in a boolean model's classes_ at all.
    """
    proba = model.predict_proba(X)
    class_order = list(model.classes_)
    # Binary model: classes_ is [False, True] (or occasionally [0, 1]
    # depending on how labels were passed at fit time) -- find whichever
    # entry represents "flare occurred" rather than assuming a fixed index.
    if True in class_order:
        flare_idx = class_order.index(True)
    elif 1 in class_order:
        flare_idx = class_order.index(1)
    else:
        raise ValueError(
            f"Expected a binary True/False (or 1/0) model, got classes_={class_order}. "
            "If this model is the OLDER 5-class version, this function no longer "
            "supports it -- that model has been superseded, see project docs Section 8."
        )
    flare_prob = proba[:, flare_idx]
    return pd.Series(flare_prob, index=X.index, name="flare_probability")


# NOTE: compute_severity_estimate() has been REMOVED, not just left
# broken. It reported a "most likely severity class" (B/C/M/X) alongside
# the alert level -- but this project's own investigation (see
# Knoxis_Complete_Project_Documentation.md, Section 8.6/8.9) found
# severity classification performs at or below random chance (19.6%
# match rate vs a 25% baseline for 4 classes). Patching this function to
# merely "run" against the binary model would still produce a
# confident-looking but empirically disproven output. If severity
# estimation is revisited later with a genuinely working approach, add
# it back deliberately -- don't resurrect this specific function as-is.


def compute_alert_level(flare_prob: pd.Series, watch_threshold: float,
                         warning_threshold: float, alert_threshold: float) -> pd.Series:
    """Graduated QUIET/WATCH/WARNING/ALERT based on flare_probability
    crossing progressively higher, user-chosen thresholds."""
    level = pd.Series("QUIET", index=flare_prob.index)
    level[flare_prob >= watch_threshold] = "WATCH"
    level[flare_prob >= warning_threshold] = "WARNING"
    level[flare_prob >= alert_threshold] = "ALERT"
    return level


def suggest_thresholds(model, X_test: pd.DataFrame, y_test_is_flare: pd.Series) -> None:
    """Prints a precision-recall table for the binary flare-vs-quiet
    decision, at a range of candidate thresholds -- so thresholds are
    CHOSEN based on real held-out evidence, not guessed."""
    flare_prob = compute_flare_probability(model, X_test)
    precision, recall, thresholds = precision_recall_curve(y_test_is_flare, flare_prob)

    print("Threshold  Precision  Recall  (at that flare-probability cutoff)")
    print("-" * 60)
    sample_idx = np.linspace(0, len(thresholds) - 1, min(20, len(thresholds))).astype(int)
    for i in sample_idx:
        print(f"{thresholds[i]:>9.3f}  {precision[i]:>9.3f}  {recall[i]:>7.3f}")

    print("\nPick WATCH/WARNING/ALERT thresholds based on this table:")
    print("  - A LOWER threshold catches more real flares (higher recall) but")
    print("    raises more false alarms on real quiet periods (lower precision).")
    print("  - A HIGHER threshold does the opposite.")
    print("  - Since ALERT should be your most confident tier, pick its threshold")
    print("    where precision is meaningfully higher, even if recall drops some.")


def run_alert_pipeline(model_path: Path, X: pd.DataFrame,
                        watch_threshold: float, warning_threshold: float,
                        alert_threshold: float) -> pd.DataFrame:
    """Full pipeline: load model, compute flare probability, graduated
    alert level. No severity estimate -- see the note above
    compute_flare_probability for why that was removed."""
    model = joblib.load(model_path)
    flare_prob = compute_flare_probability(model, X)
    alert_level = compute_alert_level(flare_prob, watch_threshold, warning_threshold, alert_threshold)

    result = pd.DataFrame({
        "flare_probability": flare_prob,
        "alert_level": alert_level,
    })
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Apply graduated alert thresholds to a trained forecast model")
    parser.add_argument("--model", required=True)
    parser.add_argument("--master-table", required=True)
    parser.add_argument("--split-fraction", type=float, default=0.8)
    parser.add_argument("--suggest-thresholds", action="store_true")
    parser.add_argument("--watch-threshold", type=float, default=0.3)
    parser.add_argument("--warning-threshold", type=float, default=0.5)
    parser.add_argument("--alert-threshold", type=float, default=0.7)
    args = parser.parse_args()

    df = pd.read_parquet(args.master_table)
    if df.index.name != "timestamp" and "timestamp" in df.columns:
        df = df.set_index("timestamp")
    df = df.sort_index()
    df["label_letter"] = df["label_class"].apply(lambda c: "QUIET" if c == "QUIET" else c[0])
    df["is_flare"] = df["label_letter"] != "QUIET"
    df_clean = df.dropna(subset=FEATURE_COLUMNS)

    span_start, span_end = df_clean.index.min(), df_clean.index.max()
    cutoff = span_start + (span_end - span_start) * args.split_fraction
    test_df = df_clean[df_clean.index >= cutoff]

    model = joblib.load(args.model)

    if args.suggest_thresholds:
        print("=== Precision-recall table on held-out test set ===\n")
        suggest_thresholds(model, test_df[FEATURE_COLUMNS], test_df["is_flare"])
    else:
        result = run_alert_pipeline(
            Path(args.model), test_df[FEATURE_COLUMNS],
            args.watch_threshold, args.warning_threshold, args.alert_threshold,
        )
        print(result.head(20))
        print(f"\nAlert level distribution:\n{result['alert_level'].value_counts()}")
