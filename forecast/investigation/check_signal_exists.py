"""
check_signal_exists.py

The decisive remaining question: within a single, internally-consistent
period (so shift isn't a confound), do the 9 features show ANY
statistically meaningful difference between windows that precede a real
flare within the next 60 minutes, versus windows that stay quiet?
"""

from pathlib import Path

import pandas as pd
from scipy import stats

FEATURE_COLUMNS = [
    "solexs_mean", "solexs_slope", "solexs_std",
    "hel1os_mean", "hel1os_slope",
    "hardness_ratio", "hr_rate_of_change",
    "peak_to_mean_ratio", "hel1os_saturation_fraction",
]


def run(master_table_path: Path, period_end: str):
    df = pd.read_parquet(master_table_path)
    if df.index.name != "timestamp" and "timestamp" in df.columns:
        df = df.set_index("timestamp")
    df = df.sort_index()
    df["label_letter"] = df["label_class"].apply(lambda c: "QUIET" if c == "QUIET" else c[0])
    df["is_flare"] = df["label_letter"] != "QUIET"

    period = df[df.index < period_end].dropna(subset=FEATURE_COLUMNS)
    pre_flare = period[period["is_flare"]]
    stays_quiet = period[~period["is_flare"]]

    print(f"Within-period comparison: {len(pre_flare)} pre-flare windows vs "
          f"{len(stays_quiet)} stays-quiet windows\n")
    print(f"{'Feature':<28} {'pre-flare median':>18} {'quiet median':>15} "
          f"{'p-value':>12}  Different?")
    print("-" * 85)

    any_real_signal = False
    for feat in FEATURE_COLUMNS:
        a = pre_flare[feat].dropna()
        b = stays_quiet[feat].dropna()
        if len(a) < 10 or len(b) < 10:
            continue
        stat, p_value = stats.mannwhitneyu(a, b, alternative="two-sided")
        is_different = p_value < 0.001
        if is_different:
            any_real_signal = True
        print(f"{feat:<28} {a.median():>18.4f} {b.median():>15.4f} {p_value:>12.2e}  "
              f"{'YES' if is_different else 'no'}")

    print(f"\n{'='*70}")
    if any_real_signal:
        print("At least one feature shows a statistically real difference -- SOME")
        print("signal exists. If the model still can't exploit it well, that points")
        print("to a MODELING issue, not absent signal.")
    else:
        print("NO feature shows a meaningful difference, even within one consistent")
        print("period. Supports the hypothesis that X-ray-flux-derived features alone")
        print("may genuinely lack the information needed to predict flare ONSET.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--master-table", required=True)
    parser.add_argument("--period-end", default="2025-02-01")
    args = parser.parse_args()
    run(Path(args.master_table), args.period_end)
