"""
Step 5: concatenates every labeled per-day parquet into one master
training table, ready for the time-based train/test split.
"""

from pathlib import Path
import pandas as pd


def build_master_table(labeled_dir: Path, output_path: Path) -> pd.DataFrame:
    labeled_files = sorted(labeled_dir.glob("*_labeled.parquet"))
    if not labeled_files:
        raise FileNotFoundError(f"No *_labeled.parquet files found in {labeled_dir}")

    frames = []
    for f in labeled_files:
        df = pd.read_parquet(f)
        if df.index.name != "timestamp" and "timestamp" in df.columns:
            df = df.set_index("timestamp")
        frames.append(df)

    master = pd.concat(frames)
    master = master.sort_index()

    n_dupes = master.index.duplicated().sum()
    if n_dupes > 0:
        print(f"WARNING: {n_dupes} duplicate timestamps found across day files -- "
              f"investigate before trusting this table (day-boundary bug?).")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    master.to_parquet(output_path)

    print(f"Master table built: {len(master)} rows, {len(labeled_files)} day-files combined.")
    print(f"Date range: {master.index.min()} -> {master.index.max()}")
    print(f"Saved to: {output_path}")

    return master


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build the master labeled training table")
    parser.add_argument("--labeled-dir", required=True)
    parser.add_argument("--output", default="master_training_table.parquet")
    args = parser.parse_args()

    build_master_table(Path(args.labeled_dir), Path(args.output))
