"""
check_monthly_17th_pattern.py

Tests the hypothesis that SoLEXS undergoes a regular, scheduled
reduced-validity period around the 17th of each month (very plausibly
a monthly calibration window, given the instrument's onboard Fe-55
calibration source) -- checking EVERY month in the dataset, not just
the ones that happened to rank in a "worst 20" list, and comparing
against a random control day that should show no such pattern.
"""

from pathlib import Path

import pandas as pd


def run(gti_report_path: Path):
    report = pd.read_parquet(gti_report_path)
    report.index = pd.to_datetime(report.index)

    seventeenths = report[report.index.day == 17].copy()

    print(f"Total 17th-of-month days in range: {len(seventeenths)}")
    print(f"\nValidity fraction on EVERY 17th in the dataset:")
    print(seventeenths[["valid_fraction_pct", "n_windows"]].to_string())

    low_validity_17ths = (seventeenths["valid_fraction_pct"] < 50).sum()
    print(f"\n17ths with validity < 50%: {low_validity_17ths} / {len(seventeenths)} "
          f"({100*low_validity_17ths/len(seventeenths):.1f}%)")

    # Control: a random other day of the month, which should show no
    # special pattern if the 17th effect is real and specific.
    control_day = 5
    control = report[report.index.day == control_day]
    control_low = (control["valid_fraction_pct"] < 50).sum()
    print(f"\nFor comparison, day-of-month={control_day} (control, no reason to be special):")
    print(f"  {control_low} / {len(control)} days below 50% validity "
          f"({100*control_low/len(control):.1f}%)")

    print(f"\nAverage validity on the 17th: {seventeenths['valid_fraction_pct'].mean():.1f}%")
    print(f"Average validity on day {control_day} (control): {control['valid_fraction_pct'].mean():.1f}%")

    if len(seventeenths) > 0 and low_validity_17ths / len(seventeenths) > 0.5 and \
       len(control) > 0 and control_low / len(control) < 0.2:
        print("\n-> CONFIRMED: the 17th shows a systematic, recurring validity drop that")
        print("   a random control day does not. This is very likely a scheduled")
        print("   instrument event (e.g. monthly calibration), not random data loss.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gti-report", default="gti_completeness_full_report.parquet")
    args = parser.parse_args()
    run(Path(args.gti_report))
