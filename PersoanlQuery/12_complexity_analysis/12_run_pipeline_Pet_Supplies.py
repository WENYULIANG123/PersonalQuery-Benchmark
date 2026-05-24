#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_run_stage12():
    module_path = Path(__file__).resolve().parent / "run_stage12.py"
    spec = importlib.util.spec_from_file_location("stage12_run_stage12", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    run_stage12 = _load_run_stage12()
    sys.argv = ["12_run_pipeline_Pet_Supplies.py", "--category", "Pet_Supplies", "--task", "pipeline"]
    run_stage12.main()
