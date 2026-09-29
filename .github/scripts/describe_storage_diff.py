"""Print a best-effort summary of differences between two storage files.

Only intended for interactive/log inspection (invoked from CI), not for
building the PR-facing report. Falls back to a generic message if the files
aren't parquet or can't be compared this way.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl


def describe_parquet_diff(path_a: Path, path_b: Path) -> str:
    try:
        df_a = pl.read_parquet(path_a)
        df_b = pl.read_parquet(path_b)
    except Exception as exc:
        return f"Could not read as parquet: {exc}"

    if df_a.shape != df_b.shape:
        return f"Shape differs: {df_a.shape} vs {df_b.shape}"

    if df_a.columns != df_b.columns:
        return f"Columns differ: {df_a.columns} vs {df_b.columns}"

    lines = []
    for column in df_a.columns:
        series_a = df_a[column]
        series_b = df_b[column]

        if series_a.equals(series_b):
            continue

        if series_a.dtype.is_numeric() and series_b.dtype.is_numeric():
            max_abs_diff = (series_a - series_b).abs().max()
            lines.append(f"  {column}: max abs diff = {max_abs_diff}")
        else:
            mismatches = (series_a != series_b).sum()
            lines.append(f"  {column}: {mismatches} mismatching values")

    if not lines:
        return "No column-level differences detected despite files differing."

    return "Columns with differences:\n" + "\n".join(lines)


def main() -> int:
    path_a = Path(sys.argv[1])
    path_b = Path(sys.argv[2])
    print(describe_parquet_diff(path_a, path_b))
    return 0


if __name__ == "__main__":
    sys.exit(main())
