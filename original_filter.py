import pandas as pd
from pathlib import Path

for year in range(2021, 2025):
    infile = Path(f"keogram_segment_stats{year}_24hours.csv")
    outfile = Path(f"keogram_segment_stats{year}._24hours_filtered.csv")

    if not infile.exists():
        print(f"[skip] {infile} not found")
        continue

    df = pd.read_csv(infile)

    # Drop rows where median < 15 (case-insensitive column match just in case)
    median_col = next((c for c in df.columns if c.strip().lower() == "median"), None)
    if median_col is None:
        print(f"[warn] 'median' column not found in {infile.name}; skipping.")
        continue

    filtered_df = df[df[median_col] >= 15]

    filtered_df.to_csv(outfile, index=False)
    print(f"[ok] {infile.name} -> {outfile.name} (kept {len(filtered_df)} rows)")
