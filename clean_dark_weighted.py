#!/usr/bin/env python3
import sys
import pandas as pd
from pathlib import Path

USAGE = "Usage: python clean_dark_segments.py <input.csv> <output.csv>"

def find_column(cols):
    """Find the 'dark segment' column (case/space tolerant)."""
    normalized = {c: c.strip().lower().replace("_", " ") for c in cols}
    for original, norm in normalized.items():
        if norm == "dark segment":
            return original
    return None

def main():
    if len(sys.argv) != 3:
        sys.exit(USAGE)
    inp = Path(sys.argv[1])
    out = Path(sys.argv[2])

    if not inp.exists():
        sys.exit(f"Input not found: {inp}")

    df = pd.read_csv(inp)

    col = find_column(df.columns)
    if col is None:
        sys.exit("Error: Could not find a column named 'dark segment' (case/space-insensitive).")

    # Drop rows where 'dark segment' == 1 (robust to strings like "1" or " 1 ")
    dark_values = pd.to_numeric(df[col].astype(str).str.strip(), errors="coerce")
    keep_mask = dark_values.ne(1)  # keep rows where value is NOT 1
    cleaned = df.loc[keep_mask].drop(columns=[col])

    cleaned.to_csv(out, index=False)
    print(f"Rows in:  {len(df)}")
    print(f"Rows out: {len(cleaned)}")
    print(f"Wrote:    {out}")

if __name__ == "__main__":
    main()
