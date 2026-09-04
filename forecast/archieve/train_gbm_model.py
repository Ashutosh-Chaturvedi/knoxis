"""
train_gbm_model.py

Gradient boosting (LightGBM) forecast model -- a direct swap of the
ALGORITHM only, keeping everything else identical to the original
single-stage Random Forest run (same features, same NaN handling, same
time-based split, same evaluation), for a clean, fair before/after
comparison.

Why try this: today's investigation established that real, statistically
significant signal exists in the features (p-values down to 1e-164 for
several), but with modest effect sizes relative to within-class noise.
Gradient boosting builds trees SEQUENTIALLY, each one correcting the
previous trees' errors -- often more effective than Random Forest's
independent-bagged-trees approach at squeezing out weak, noisy signal,
since RF's trees don't learn from each other's mistakes at all.

Kept as a single-stage 5-class model (not two-stage) -- today's
comparison test showed the two-stage architecture didn't meaningfully
help on real data, so the fairest, most interpretable test here is
swapping the algorithm alone, not stacking multiple changes at once.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix

FEATURE_COLUMNS = [
    "solexs_mean", "solexs_slope", "solexs_std",
    "hel1os_mean", "hel1os_slope",
    "hardness_ratio", "hr_rate_of_change",
    "peak_to_mean_ratio", "hel1os_saturation_fraction",
]


def prepare_dataset(master_table_path: Path, split_fraction: float = 0.8) -> dict:
    """Identical to train_model.py's prepare_dataset -- same NaN
    dropping, same time-based split -- so this is a fair, apples-to-
    apples comparison against the original Random Forest baseline."""
    df = pd.read_parquet(master_table_path)
    if df.index.name != "timestamp" and "timestamp" in df.columns:
        df = df.set_index("timestamp")
    df = df.sort_index()
    df["label_letter"] = df["label_class"].apply(lambda c: "QUIET" if c == "QUIET" else c[0])

    n_total = len(df)
    n_nan = df[FEATURE_COLUMNS].isna().any(axis=1).sum()
    df_clean = df.dropna(subset=FEATURE_COLUMNS)
    print(f"Dropped {n_nan} rows ({100*n_nan/n_total:.1f}%) with NaN features.")

    span_start, span_end = df_clean.index.min(), df_clean.index.max()
    cutoff = span_start + (span_end - span_start) * split_fraction
    train_df = df_clean[df_clean.index < cutoff]
    test_df = df_clean[df_clean.index >= cutoff]

    print(f"Split cutoff: {cutoff}")
    print(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows")

    return {
        "X_train": train_df[FEATURE_COLUMNS],
        "y_train": train_df["label_letter"],
        "X_test": test_df[FEATURE_COLUMNS],
        "y_test": test_df["label_letter"],
    }


def train_and_evaluate(data: dict, n_estimators: int = 300, random_state: int = 42) -> dict:
    model = LGBMClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        num_leaves=31,
        learning_rate=0.05,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(data["X_train"], data["y_train"])
    y_pred = model.predict(data["X_test"])

    print("\n=== Classification report (LightGBM, per-class precision/recall/F1) ===")
    print(classification_report(data["y_test"], y_pred, zero_division=0))

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
    parser = argparse.ArgumentParser(description="Train and evaluate the LightGBM forecast model")
    parser.add_argument("--master-table", required=True)
    parser.add_argument("--model-output", default="knoxis_gbm_model.joblib")
    parser.add_argument("--split-fraction", type=float, default=0.8)
    args = parser.parse_args()

    run(Path(args.master_table), Path(args.model_output), args.split_fraction)
