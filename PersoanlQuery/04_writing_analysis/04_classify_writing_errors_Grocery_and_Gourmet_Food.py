#!/usr/bin/env python3
"""Classify writing errors for Grocery_and_Gourmet_Food."""

from pathlib import Path

CATEGORY = "Grocery_and_Gourmet_Food"

_COMMON = Path(__file__).resolve().parent / "common"
import sys
sys.path.insert(0, str(_COMMON))
from classify_writing_errors_common import classify_category

if __name__ == "__main__":
    classify_category(CATEGORY)
