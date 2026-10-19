"""
train_two_stage_model.py

Two-stage forecast model, replacing the single 5-class Random Forest
that collapsed to predicting QUIET almost everywhere (real result:
0.00-0.01 recall on B/C/M/X despite class_weight='balanced').

Stage 1 -- binary "flare or not" (QUIET vs FLARE). Roughly a 2:1
imbalance given the real dataset's numbers, vs. the ~150:1 imbalance
the single-stage model faced -- a fundamentally easier problem for a
tree ensemble's Gini-impurity splitting to actually learn from.

Stage 2 -- severity classifier (B/C/M/X), trained ONLY on windows
Stage 1 considers a real flare. Still imbalanced (X is rare relative to
C), but far gentler than fighting QUIET's dominance at the same time.

Evaluation is done on the FULL two-stage PIPELINE end-to-end (Stage 1
then, conditionally, Stage 2), compared against the true label on the
held-out test set -- not each stage evaluated in isolation, which would
look artificially good and hide real pipeline-level mistakes (e.g. a
real flare that Stage 1 incorrectly calls QUIET, so Stage 2 never even
gets to see it).
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
    """Same NaN-dropping + time-based split as the single-stage model,
    plus a derived binary is_flare target for Stage 1."""
    df = pd.read_parquet(master_table_path)
    if df.index.name != "timestamp" and "timestamp" in df.columns:
        df = df.set_index("timestamp")
    df = df.sort_index()

    df["label_letter"] = df["label_class"].apply(lambda c: "QUIET" if c == "QUIET" else c[0])
    df["is_flare"] = df["label_letter"] != "QUIET"

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
    print(f"\nTrain is_flare balance: {train_df['is_flare'].mean():.1%} flare / "
          f"{1-train_df['is_flare'].mean():.1%} quiet  (Stage 1's actual problem, "
          f"vs the ~150:1 imbalance the single-stage model faced)")

    return {
        "X_train": train_df[FEATURE_COLUMNS],
        "y_train_binary": train_df["is_flare"],
        "y_train_letter": train_df["label_letter"],
        "X_test": test_df[FEATURE_COLUMNS],
        "y_test_binary": test_df["is_flare"],
        "y_test_letter": test_df["label_letter"],
    }


def train_two_stage(data: dict, n_estimators: int = 200, random_state: int = 42) -> dict:
    # --- Stage 1: binary flare detector ---
    stage1 = RandomForestClassifier(
        n_estimators=n_estimators, class_weight="balanced_subsample",
        random_state=random_state, n_jobs=-1,
    )
    stage1.fit(data["X_train"], data["y_train_binary"])

    # --- Stage 2: severity classifier, trained ONLY on real flare windows ---
    flare_mask_train = data["y_train_binary"]
    stage2 = RandomForestClassifier(
        n_estimators=n_estimators, class_weight="balanced_subsample",
        random_state=random_state, n_jobs=-1,
    )
    stage2.fit(data["X_train"][flare_mask_train], data["y_train_letter"][flare_mask_train])

    print(f"\nStage 2 trained on {flare_mask_train.sum()} real flare windows "
          f"(vs {len(data['X_train'])} total -- QUIET's dominance is completely removed here).")

    return {"stage1": stage1, "stage2": stage2}


def predict_pipeline(models: dict, X: pd.DataFrame) -> np.ndarray:
    """Runs the full two-stage pipeline: Stage 1 decides flare/no-flare;
    only flare-predicted rows go to Stage 2 for severity classification."""
    stage1_pred = models["stage1"].predict(X)
    final_pred = np.full(len(X), "QUIET", dtype=object)

    flare_rows = X[stage1_pred]
    if len(flare_rows) > 0:
        stage2_pred = models["stage2"].predict(flare_rows)
        final_pred[stage1_pred] = stage2_pred

    return final_pred


def evaluate(models: dict, data: dict) -> None:
    y_pred = predict_pipeline(models, data["X_test"])
    y_true = data["y_test_letter"].values

    print("\n=== STAGE 1 ALONE: flare-detection performance ===")
    stage1_pred = models["stage1"].predict(data["X_test"])
    print(classification_report(data["y_test_binary"], stage1_pred, target_names=["QUIET", "FLARE"], zero_division=0))

    print("=== FULL PIPELINE (end-to-end, this is what actually matters) ===")
    print(classification_report(y_true, y_pred, zero_division=0))

    print("=== Full pipeline confusion matrix ===")
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])
    print(cm_df)


def run(master_table_path: Path, model_output_dir: Path, split_fraction: float = 0.8) -> None:
    data = prepare_dataset(master_table_path, split_fraction)
    models = train_two_stage(data)
    evaluate(models, data)

    import joblib
    model_output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(models["stage1"], model_output_dir / "stage1_flare_detector.joblib")
    joblib.dump(models["stage2"], model_output_dir / "stage2_severity_classifier.joblib")
    print(f"\nModels saved to: {model_output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--master-table", required=True)
    parser.add_argument("--model-output-dir", default="two_stage_models")
    parser.add_argument("--split-fraction", type=float, default=0.8)
    args = parser.parse_args()

    run(Path(args.master_table), Path(args.model_output_dir), args.split_fraction)
