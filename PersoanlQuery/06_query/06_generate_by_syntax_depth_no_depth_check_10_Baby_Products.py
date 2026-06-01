#!/usr/bin/env python3
"""Generate 10 Baby_Products syntax-depth queries per user (no depth check)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.syntax_depth_no_depth_check import main


CATEGORY = "Baby_Products"


if __name__ == "__main__":
    main(CATEGORY)
