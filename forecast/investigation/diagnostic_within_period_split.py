"""
diagnostic_within_period_split.py

Decisive diagnostic: trains and tests ENTIRELY within the early period
(no crossing of the suspected covariate-shift boundary at all).

If performance here is dramatically better than the real cross-period
run, that CONFIRMS covariate shift is the dominant problem -- the
model CAN learn the relationship when train/test conditions match.

If performance is STILL poor even within a single, internally
consistent period, that points to something more fundamental (weak
features, or an undiscovered bug), not primarily a shift problem.
"""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

FEATURE_COLUMNS = [
    "solexs_mean", "solexs_slope", "solexs_std",
    "hel1os_mean", "hel1os_slope",
    "hardness_ratio", "hr_rate_of_change",
    "peak_to_mean_ratio", "hel1os_saturation_fraction",
]


def run(master_table_path: Path, early_train_end: str, early_test_end: str):
    df = pd.read_parquet(master_table_path)
    if df.index.name != "timestamp" and "timestamp" in df.columns:
        df = df.set_index("timestamp")
    df = df.sort_index()
    df["label_letter"] = df["label_class"].apply(lambda c: "QUIET" if c == "QUIET" else c[0])
    df["is_flare"] = df["label_letter"] != "QUIET"

    df_clean = df.dropna(subset=FEATURE_COLUMNS)

    train_df = df_clean[df_clean.index < early_train_end]
    test_df = df_clean[(df_clean.index >= early_train_end) & (df_clean.index < early_test_end)]

    print(f"WITHIN-EARLY-PERIOD split (no shift boundary crossed):")
    print(f"Train: {len(train_df)} rows ({train_df.index.min()} -> {train_df.index.max()})")
    print(f"Test:  {len(test_df)} rows ({test_df.index.min()} -> {test_df.index.max()})")
    print(f"\nTrain is_flare balance: {train_df['is_flare'].mean():.1%} flare")
    print(f"Test is_flare balance: {test_df['is_flare'].mean():.1%} flare")

    if len(test_df) == 0 or len(train_df) == 0:
        print("ERROR: empty train or test set -- adjust the date ranges.")
        return

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["is_flare"]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df["is_flare"]

    model = RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample", random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n=== Binary flare-detection performance, WITHIN early period ===")
    print(classification_report(y_test, y_pred, target_names=["QUIET", "FLARE"], zero_division=0))
    print("\nCompare this FLARE recall against the real cross-period run's Stage 1 "
          "FLARE recall of 0.05. A large improvement here confirms covariate shift "
          "is the dominant problem, not fundamentally weak features.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--master-table", required=True)
    parser.add_argument("--early-train-end", default="2024-11-01")
    parser.add_argument("--early-test-end", default="2025-02-01")
    args = parser.parse_args()

    run(Path(args.master_table), args.early_train_end, args.early_test_end)
