"""Compare two storage files, tolerant of float noise.

Exit code 0 means the files are considered equal, exit code 1 means they
differ. Intended to be called only after a raw byte-for-byte diff has
already failed, as a second-chance check before flagging a file as a real
regression.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import polars as pl
from polars.testing import assert_frame_equal

FLOAT_PATTERN = re.compile(r"-?\d+\.\d+")
ABS_TOL = 1e-8


def round_floats_in_text(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        value = float(match.group())
        return f"{value:.8f}"

    return FLOAT_PATTERN.sub(replace, text)


def compare_as_text(path_a: Path, path_b: Path) -> bool:
    try:
        text_a = path_a.read_text(encoding="utf-8")
        text_b = path_b.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return False

    return round_floats_in_text(text_a) == round_floats_in_text(text_b)


def compare_as_parquet(path_a: Path, path_b: Path) -> bool:
    try:
        df_a = pl.read_parquet(path_a)
        df_b = pl.read_parquet(path_b)
    except Exception:
        return False

    try:
        assert_frame_equal(df_a, df_b, abs_tol=ABS_TOL)
    except AssertionError:
        return False

    return True


def main() -> int:
    path_a = Path(sys.argv[1])
    path_b = Path(sys.argv[2])

    if compare_as_text(path_a, path_b):
        return 0

    if compare_as_parquet(path_a, path_b):
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
