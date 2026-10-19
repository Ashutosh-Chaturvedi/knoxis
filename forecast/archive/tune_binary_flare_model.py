"""
tune_binary_flare_model.py

Hyperparameter search for the binary flare/no-flare LightGBM model.

Uses sklearn's TimeSeriesSplit for cross-validation, NOT random k-fold
-- consistent with this project's discipline throughout (no random
shuffling of time-series data). Each CV fold trains on an earlier
period and validates on a strictly later one.

The held-out TEST set (the real, final 20%) is NEVER touched during
tuning -- only the training set is used for the search, and the best
parameters are evaluated ONCE on the test set at the end.

Optimizes for PR-AUC (average precision).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
)

FEATURE_COLUMNS = [
    "solexs_mean", "solexs_slope", "solexs_std",
    "hel1os_mean", "hel1os_slope",
    "hardness_ratio", "hr_rate_of_change",
    "peak_to_mean_ratio", "hel1os_saturation_fraction",
]

PARAM_DISTRIBUTIONS = {
    "num_leaves": [15, 31, 63, 127],
    "max_depth": [-1, 5, 8, 12],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "n_estimators": [200, 300, 500, 800],
    "min_child_samples": [10, 20, 50, 100],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_alpha": [0, 0.1, 0.5, 1.0],
    "reg_lambda": [0, 0.1, 0.5, 1.0],
}


def prepare_dataset(master_table_path: Path, split_fraction: float = 0.8) -> dict:
    df = pd.read_parquet(master_table_path)
    if df.index.name != "timestamp" and "timestamp" in df.columns:
        df = df.set_index("timestamp")
    df = df.sort_index()
    df["is_flare"] = df["label_class"] != "QUIET"

    df_clean = df.dropna(subset=FEATURE_COLUMNS)
    span_start, span_end = df_clean.index.min(), df_clean.index.max()
    cutoff = span_start + (span_end - span_start) * split_fraction
    train_df = df_clean[df_clean.index < cutoff]
    test_df = df_clean[df_clean.index >= cutoff]

    return {
        "X_train": train_df[FEATURE_COLUMNS], "y_train": train_df["is_flare"],
        "X_test": test_df[FEATURE_COLUMNS], "y_test": test_df["is_flare"],
    }


def tune(data: dict, n_iter: int = 20, n_splits: int = 3, random_state: int = 42) -> dict:
    base_model = LGBMClassifier(class_weight="balanced", random_state=random_state, n_jobs=-1, verbose=-1)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    search = RandomizedSearchCV(
        base_model, PARAM_DISTRIBUTIONS,
        n_iter=n_iter, cv=tscv, scoring="average_precision",
        random_state=random_state, n_jobs=-1, verbose=1,
    )

    print(f"Running randomized search: {n_iter} parameter combinations x {n_splits} "
          f"time-series folds = {n_iter * n_splits} total fits. This may take a while...")
    search.fit(data["X_train"], data["y_train"])

    print(f"\nBest CV PR-AUC (training folds only, test set untouched): {search.best_score_:.4f}")
    print(f"Best parameters:\n{search.best_params_}")

    return {"best_model": search.best_estimator_, "best_params": search.best_params_, "cv_score": search.best_score_}


def evaluate_on_test(model, data: dict) -> None:
    y_proba = model.predict_proba(data["X_test"])[:, 1]
    y_pred = model.predict(data["X_test"])

    roc_auc = roc_auc_score(data["y_test"], y_proba)
    pr_auc = average_precision_score(data["y_test"], y_proba)
    baseline_pr_auc = data["y_test"].mean()

    print(f"\n=== TUNED MODEL: held-out test set evaluation ===")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}  (no-skill baseline: {baseline_pr_auc:.4f}, "
          f"lift: {pr_auc/baseline_pr_auc:.2f}x)")

    print(f"\n=== Classification report at default 0.5 threshold ===")
    print(classification_report(data["y_test"], y_pred, target_names=["QUIET", "FLARE"], zero_division=0))

    print("=== Confusion matrix ===")
    cm = confusion_matrix(data["y_test"], y_pred)
    print(pd.DataFrame(cm, index=["true_QUIET", "true_FLARE"], columns=["pred_QUIET", "pred_FLARE"]))


def run(master_table_path: Path, model_output_path: Path,
        n_iter: int = 20, n_splits: int = 3, split_fraction: float = 0.8) -> None:
    data = prepare_dataset(master_table_path, split_fraction)
    print(f"Train: {len(data['X_train'])} rows, Test: {len(data['X_test'])} rows (untouched during tuning)\n")

    result = tune(data, n_iter=n_iter, n_splits=n_splits)
    evaluate_on_test(result["best_model"], data)

    import joblib
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result["best_model"], model_output_path)
    print(f"\nTuned model saved to: {model_output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hyperparameter-tune the binary flare model")
    parser.add_argument("--master-table", required=True)
    parser.add_argument("--model-output", default="knoxis_binary_flare_model_tuned.joblib")
    parser.add_argument("--n-iter", type=int, default=20)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--split-fraction", type=float, default=0.8)
    args = parser.parse_args()

    run(Path(args.master_table), Path(args.model_output), args.n_iter, args.n_splits, args.split_fraction)
