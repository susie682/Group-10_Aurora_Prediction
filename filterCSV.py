import pandas as pd
from pathlib import Path
import re

def add_dark_segment_flag(input_csv, output_csv=None, median_threshold=15):
    """
    Add 'dark_segment' column to CSV instead of deleting rows.
    Preserves all existing columns including 'sunlight_contamination'.
    
    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output CSV file (if None, overwrites input)
        median_threshold: Threshold for median value (default: 15)
    
    Returns:
        DataFrame with new column added
    """
    # Load the CSV - this preserves ALL existing columns
    df = pd.read_csv(input_csv)
    
    # Check if 'median' column exists
    if 'median' not in df.columns:
        print(f"  [error] 'median' column not found in {Path(input_csv).name}")
        print(f"  [info] Available columns: {', '.join(df.columns)}")
        return None
    
    # Add 'dark_segment' column: 1 if median < threshold, else 0
    df['dark_segment'] = (df['median'] < median_threshold).astype(int)
    
    # Save to output - this saves ALL columns including existing ones
    output_path = output_csv if output_csv else input_csv
    df.to_csv(output_path, index=False)
    
    # Statistics
    total_rows = len(df)
    dark_rows = df['dark_segment'].sum()
    dark_pct = (dark_rows / total_rows * 100) if total_rows > 0 else 0
    
    print(f"  [ok] {Path(input_csv).name}")
    print(f"       Total rows: {total_rows}, Dark segments: {dark_rows} ({dark_pct:.1f}%)")
    
    return df


def process_all_year_csvs(
        base_dir=".",
        csv_pattern="keogram_segment_stats{}_{:d}hours_weighted.csv",
        hours=[24],  # List of hour values to search for
        output_pattern=None,  # If None, will insert "_filtered" before extension
        year_range=None,
        median_threshold=15,
        overwrite_original=False
    ):
    """
    Process all year CSV files and add 'dark_segment' flag.
    
    Args:
        base_dir: Base directory containing CSV files
        csv_pattern: Pattern for input CSV files (use {} for year, {:d} for hours)
        hours: List of hour values to search for (e.g., [24] or [1, 8, 24])
        output_pattern: Pattern for output CSV files (None = auto-generate with "_filtered")
        year_range: Tuple of (start_year, end_year) inclusive, or None to auto-detect
        median_threshold: Threshold for median value (default: 15)
        overwrite_original: If True, overwrites original files instead of creating new ones
    """
    base_path = Path(base_dir)
    
    # Auto-detect CSV files if year_range not specified
    if year_range is None:
        csv_files = []
        # Look for files matching the exact pattern
        for csv_file in base_path.glob("keogram_segment_stats*hours_weighted.csv"):
            # Extract year and hours from filename
            # Pattern: keogram_segment_stats2012_24hours-weighted.csv
            match = re.search(r'keogram_segment_stats(\d{4})_(\d+)hours_weighted\.csv', csv_file.name)
            if match:
                year = int(match.group(1))
                hour_val = int(match.group(2))
                if hour_val in hours:  # Only include if hours match
                    csv_files.append((year, hour_val, csv_file))
        csv_files.sort()
    else:
        start_year, end_year = year_range
        csv_files = []
        for year in range(start_year, end_year + 1):
            for hour_val in hours:
                csv_file = base_path / csv_pattern.format(year, hour_val)
                if csv_file.exists():
                    csv_files.append((year, hour_val, csv_file))
    
    if not csv_files:
        print(f"[warn] No CSV files found in {base_path}")
        print(f"[info] Looking for pattern: keogram_segment_stats*_*hours_weighted.csv")
        print(f"[info] With hours: {hours}")
        return
    
    print(f"[info] Found {len(csv_files)} CSV files to process")
    print(f"[info] Median threshold: {median_threshold}")
    print(f"{'='*70}")
    
    summary_stats = []
    
    for year, hour_val, csv_file in csv_files:
        print(f"\n[info] Processing year {year} ({hour_val}h segments): {csv_file.name}")
        
        if overwrite_original:
            output_file = csv_file
        else:
            # Auto-generate output filename: replace "-weighted" with "-weighted_final"
            if output_pattern:
                output_file = base_path / output_pattern.format(year, hour_val)
            else:
                # keogram_segment_stats2012_24hours-weighted.csv -> keogram_segment_stats2012_24hours-weighted_final.csv
                new_name = csv_file.name.replace('_weighted.csv', '_weighted_final.csv')
                output_file = base_path / new_name
        
        df = add_dark_segment_flag(
            input_csv=str(csv_file),
            output_csv=str(output_file),
            median_threshold=median_threshold
        )
        
        # Collect summary statistics
        total_rows = len(df)
        dark_rows = df['dark_segment'].sum()
        sunlight_rows = df['sunlight_contamination'].sum() if 'sunlight_contamination' in df.columns else 0
        clean_rows = total_rows - dark_rows - sunlight_rows
        
        summary_stats.append({
            'year': year,
            'hours': hour_val,
            'total_segments': total_rows,
            'dark_segments': dark_rows,
            'sunlight_contaminated': sunlight_rows,
            'clean_segments': clean_rows,
            'output_file': output_file.name
        })
    
    print(f"\n{'='*70}")
    print(f"[done] Processed all {len(csv_files)} CSV files")
    print(f"{'='*70}")
    
    # Print summary table
    print("\n[summary] Statistics by Year:")
    print(f"{'Year':<6} {'Hours':<7} {'Total':<8} {'Dark':<8} {'Sunlight':<10} {'Clean':<8} {'Output File'}")
    print(f"{'-'*80}")
    for stat in summary_stats:
        print(f"{stat['year']:<6} {stat['hours']:<7} {stat['total_segments']:<8} {stat['dark_segments']:<8} "
              f"{stat['sunlight_contaminated']:<10} {stat['clean_segments']:<8} {stat['output_file']}")
    
    # Grand totals
    total_all = sum(s['total_segments'] for s in summary_stats)
    dark_all = sum(s['dark_segments'] for s in summary_stats)
    sunlight_all = sum(s['sunlight_contaminated'] for s in summary_stats)
    clean_all = sum(s['clean_segments'] for s in summary_stats)
    
    print(f"{'-'*80}")
    print(f"{'TOTAL':<6} {'':<7} {total_all:<8} {dark_all:<8} {sunlight_all:<10} {clean_all:<8}")
    print(f"\n[info] Clean segments: {clean_all}/{total_all} ({clean_all/total_all*100:.1f}%)")


def process_single_csv(input_csv, median_threshold=15, output_csv=None):
    """
    Process a single CSV file (convenience function).
    
    Args:
        input_csv: Path to input CSV file
        median_threshold: Threshold for median value (default: 15)
        output_csv: Path to output CSV file (if None, creates filtered version)
    """
    input_path = Path(input_csv)
    
    if not input_path.exists():
        print(f"[error] File not found: {input_csv}")
        return None
    
    if output_csv is None:
        # Create output filename: insert "_filtered" before extension
        output_csv = input_path.with_name(
            input_path.stem + "_filtered" + input_path.suffix
        )
    
    print(f"[info] Processing {input_path.name}")
    print(f"[info] Median threshold: {median_threshold}")
    
    df = add_dark_segment_flag(
        input_csv=str(input_csv),
        output_csv=str(output_csv),
        median_threshold=median_threshold
    )
    
    print(f"[done] Saved to {Path(output_csv).name}")
    return df


if __name__ == "__main__":
    # OPTION 1: Process all 24-hour weighted CSVs (recommended for your data)
    # Finds: keogram_segment_stats2012_24hours-weighted.csv, etc.
    process_all_year_csvs(
        base_dir=".",
        hours=[8,24],  # Only process 24-hour segments
        year_range=None,  # Auto-detect all years
        median_threshold=15,
        overwrite_original=False  # Creates new files with "_filtered" suffix
    )
    
    # OPTION 2: Process multiple hour segments (if you have 1h, 8h, 24h versions)
    # process_all_year_csvs(
    #     base_dir=".",
    #     hours=[1, 8, 24],  # Process all three segment sizes
    #     year_range=None,
    #     median_threshold=15,
    #     overwrite_original=False
    # )
    
    # OPTION 3: Process specific year range with 24-hour segments
    # process_all_year_csvs(
    #     base_dir=".",
    #     csv_pattern="keogram_segment_stats{}_{:d}hours-weighted.csv",
    #     hours=[24],
    #     year_range=(2012, 2024),
    #     median_threshold=15,
    #     overwrite_original=False
    # )
    
    # OPTION 4: Overwrite original files (no new files created)
    # process_all_year_csvs(
    #     base_dir=".",
    #     hours=[24],
    #     median_threshold=15,
    #     overwrite_original=True  # Adds column to existing files
    # )
    
    # OPTION 5: Process single CSV file
    # process_single_csv(
    #     input_csv="keogram_segment_stats2012_24hours-weighted.csv",
    #     median_threshold=15
    # )
    
    # OPTION 6: Custom output pattern
    # process_all_year_csvs(
    #     base_dir=".",
    #     csv_pattern="keogram_segment_stats{}_{:d}hours-weighted.csv",
    #     output_pattern="keogram_segment_stats{}_{:d}hours-final.csv",
    #     hours=[24],
    #     median_threshold=15
    # )