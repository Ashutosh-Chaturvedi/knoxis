"""
train_model.py

Trains and evaluates the Random Forest forecast model on the master
labeled training table.

Design decisions, and why:
    - NaN feature rows are DROPPED, not imputed -- consistent with this
      project's principle throughout (never fabricate missing data;
      SoLEXS GTI gaps and HEL1OS saturation windows were both handled
      the same way: label as missing, don't invent a value).
    - The train/test split is TIME-BASED (a real cutoff date), not
      random shuffling -- critical for time-series data, since random
      shuffling would let near-duplicate windows from the same flare
      leak across the split and inflate apparent performance.
    - The target label is the CLASS LETTER (QUIET/B/C/M/X), not the
      full "M2.6"-style string -- individual exact-magnitude labels are
      far too sparse to classify meaningfully; the letter is the
      natural classification target.
    - class_weight='balanced' handles the severe class imbalance
      (914 X-class windows vs 152,993 QUIET) without fabricating
      synthetic data -- chosen over oversampling for the same
      never-fabricate-data reasoning as the NaN handling above.
    - Evaluation uses per-class precision/recall/F1 and a confusion
      matrix, NEVER plain accuracy -- a model that always predicts
      QUIET would score >99% accuracy while being useless.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

FEATURE_COLUMNS = [
    "solexs_mean", "solexs_slope", "solexs_std",
    "hel1os_mean", "hel1os_slope",
    "hardness_ratio", "hr_rate_of_change",
    "peak_to_mean_ratio", "hel1os_saturation_fraction",
]


def prepare_dataset(master_table_path: Path, split_fraction: float = 0.8) -> dict:
    """
    Loads the master table, derives the classification target, drops
    NaN feature rows, and performs a TIME-BASED train/test split.

    Returns a dict with X_train, y_train, X_test, y_test, and the
    actual cutoff timestamp used (so it's logged, not silently implicit).
    """
    df = pd.read_parquet(master_table_path)
    if df.index.name != "timestamp" and "timestamp" in df.columns:
        df = df.set_index("timestamp")
    df = df.sort_index()

    # Classification target: the class LETTER only (QUIET/B/C/M/X).
    df["label_letter"] = df["label_class"].apply(
        lambda c: "QUIET" if c == "QUIET" else c[0]
    )

    n_total = len(df)
    n_nan = df[FEATURE_COLUMNS].isna().any(axis=1).sum()
    df_clean = df.dropna(subset=FEATURE_COLUMNS)
    print(f"Dropped {n_nan} rows ({100*n_nan/n_total:.1f}%) with NaN features "
          f"(not imputed -- consistent with this project's data-quality policy).")
    print(f"Remaining rows for training/testing: {len(df_clean)}")

    # Time-based cutoff: split_fraction of the way through the ACTUAL
    # data's time span (not the nominal target range), so the cutoff
    # reflects real data coverage.
    span_start, span_end = df_clean.index.min(), df_clean.index.max()
    cutoff = span_start + (span_end - span_start) * split_fraction

    train_df = df_clean[df_clean.index < cutoff]
    test_df = df_clean[df_clean.index >= cutoff]

    print(f"\nTime-based split cutoff: {cutoff}")
    print(f"Train: {len(train_df)} rows ({train_df.index.min()} -> {train_df.index.max()})")
    print(f"Test:  {len(test_df)} rows ({test_df.index.min()} -> {test_df.index.max()})")

    print("\nTrain label distribution:")
    print(train_df["label_letter"].value_counts())
    print("\nTest label distribution:")
    print(test_df["label_letter"].value_counts())

    return {
        "X_train": train_df[FEATURE_COLUMNS],
        "y_train": train_df["label_letter"],
        "X_test": test_df[FEATURE_COLUMNS],
        "y_test": test_df["label_letter"],
        "cutoff": cutoff,
    }


def train_and_evaluate(data: dict, n_estimators: int = 200, random_state: int = 42) -> dict:
    """Trains the Random Forest and evaluates it on the held-out test set."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(data["X_train"], data["y_train"])

    y_pred = model.predict(data["X_test"])

    print("\n=== Classification report (per-class precision/recall/F1) ===")
    report = classification_report(data["y_test"], y_pred, zero_division=0)
    print(report)

    print("=== Confusion matrix ===")
    labels = sorted(data["y_test"].unique())
    cm = confusion_matrix(data["y_test"], y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])
    print(cm_df)

    print("\n=== Feature importances ===")
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
    print(importances)

    return {"model": model, "y_pred": y_pred, "importances": importances}


def run(master_table_path: Path, model_output_path: Path, split_fraction: float = 0.8) -> None:
    data = prepare_dataset(master_table_path, split_fraction)
    result = train_and_evaluate(data)

    import joblib
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result["model"], model_output_path)
    print(f"\nModel saved to: {model_output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train and evaluate the Knoxis forecast model")
    parser.add_argument("--master-table", required=True)
    parser.add_argument("--model-output", default="knoxis_forecast_model.joblib")
    parser.add_argument("--split-fraction", type=float, default=0.8)
    args = parser.parse_args()

    run(Path(args.master_table), Path(args.model_output), args.split_fraction)
