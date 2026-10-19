"""
train_binary_flare_model.py

Scopes the forecast target down from 5-class (QUIET/B/C/M/X) to binary
(flare / no-flare) -- aligning the model with what today's investigation
actually showed working: real, consistent signal for "will a flare
happen," and near-chance-level signal for "which exact severity."

Same features, same data, same pipeline -- just a more honest target.

Evaluated with ROC-AUC and PR-AUC (average precision), the standard
metrics for imbalanced binary classification.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score, precision_recall_curve,
)

FEATURE_COLUMNS = [
    "solexs_mean", "solexs_slope", "solexs_std",
    "hel1os_mean", "hel1os_slope",
    "hardness_ratio", "hr_rate_of_change",
    "peak_to_mean_ratio", "hel1os_saturation_fraction",
]


def prepare_dataset(master_table_path: Path, split_fraction: float = 0.8) -> dict:
    df = pd.read_parquet(master_table_path)
    if df.index.name != "timestamp" and "timestamp" in df.columns:
        df = df.set_index("timestamp")
    df = df.sort_index()
    df["is_flare"] = df["label_class"] != "QUIET"

    n_total = len(df)
    n_nan = df[FEATURE_COLUMNS].isna().any(axis=1).sum()
    df_clean = df.dropna(subset=FEATURE_COLUMNS)
    print(f"Dropped {n_nan} rows ({100*n_nan/n_total:.1f}%) with NaN features.")

    span_start, span_end = df_clean.index.min(), df_clean.index.max()
    cutoff = span_start + (span_end - span_start) * split_fraction
    train_df = df_clean[df_clean.index < cutoff]
    test_df = df_clean[df_clean.index >= cutoff]

    print(f"Split cutoff: {cutoff}")
    print(f"Train: {len(train_df)} rows ({train_df['is_flare'].mean():.1%} flare)")
    print(f"Test:  {len(test_df)} rows ({test_df['is_flare'].mean():.1%} flare)")

    return {
        "X_train": train_df[FEATURE_COLUMNS],
        "y_train": train_df["is_flare"],
        "X_test": test_df[FEATURE_COLUMNS],
        "y_test": test_df["is_flare"],
    }


def train_and_evaluate(data: dict, n_estimators: int = 300, random_state: int = 42) -> dict:
    model = LGBMClassifier(
        n_estimators=n_estimators, class_weight="balanced",
        num_leaves=31, learning_rate=0.05,
        random_state=random_state, n_jobs=-1, verbose=-1,
    )
    model.fit(data["X_train"], data["y_train"])

    y_proba = model.predict_proba(data["X_test"])[:, 1]
    y_pred = model.predict(data["X_test"])

    roc_auc = roc_auc_score(data["y_test"], y_proba)
    pr_auc = average_precision_score(data["y_test"], y_proba)
    baseline_pr_auc = data["y_test"].mean()

    print(f"\n=== Ranking quality (threshold-independent) ===")
    print(f"ROC-AUC: {roc_auc:.4f}  (0.5 = random, 1.0 = perfect)")
    print(f"PR-AUC (average precision): {pr_auc:.4f}  "
          f"(a no-skill classifier scores {baseline_pr_auc:.4f} here -- compare against THIS)")

    print(f"\n=== Classification report at default 0.5 threshold ===")
    print(classification_report(data["y_test"], y_pred, target_names=["QUIET", "FLARE"], zero_division=0))

    print("=== Confusion matrix ===")
    cm = confusion_matrix(data["y_test"], y_pred)
    print(pd.DataFrame(cm, index=["true_QUIET", "true_FLARE"], columns=["pred_QUIET", "pred_FLARE"]))

    print(f"\n=== Precision-recall table ===")
    precision, recall, thresholds = precision_recall_curve(data["y_test"], y_proba)
    sample_idx = np.linspace(0, len(thresholds) - 1, min(15, len(thresholds))).astype(int)
    print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>8}")
    for i in sample_idx:
        print(f"{thresholds[i]:>10.3f} {precision[i]:>10.3f} {recall[i]:>8.3f}")

    print(f"\n=== Feature importances ===")
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
    print(importances)

    return {"model": model, "roc_auc": roc_auc, "pr_auc": pr_auc, "baseline_pr_auc": baseline_pr_auc}


def run(master_table_path: Path, model_output_path: Path, split_fraction: float = 0.8) -> None:
    data = prepare_dataset(master_table_path, split_fraction)
    result = train_and_evaluate(data)

    import joblib
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result["model"], model_output_path)
    print(f"\nModel saved to: {model_output_path}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    lift = result["pr_auc"] / result["baseline_pr_auc"]
    print(f"PR-AUC is {lift:.2f}x the no-skill baseline "
          f"({'genuinely better than chance' if lift > 1.1 else 'barely better than chance' if lift > 1.02 else 'essentially no better than chance'})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train and evaluate the binary flare/no-flare model")
    parser.add_argument("--master-table", required=True)
    parser.add_argument("--model-output", default="knoxis_binary_flare_model.joblib")
    parser.add_argument("--split-fraction", type=float, default=0.8)
    args = parser.parse_args()

    run(Path(args.master_table), Path(args.model_output), args.split_fraction)
