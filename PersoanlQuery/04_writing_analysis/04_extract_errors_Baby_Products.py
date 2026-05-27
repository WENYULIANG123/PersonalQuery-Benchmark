#!/usr/bin/env python3
"""Extract writing errors for Baby_Products."""

from pathlib import Path

_COMMON = Path(__file__).resolve().parent / "common"
import sys
sys.path.insert(0, str(_COMMON))
from extract_errors_common import extract_and_filter_errors

if __name__ == "__main__":
    extract_and_filter_errors("Baby_Products")
