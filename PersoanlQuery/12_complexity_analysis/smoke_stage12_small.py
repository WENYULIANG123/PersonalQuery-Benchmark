#!/usr/bin/env python3
"""Run a small Stage 12 smoke test in an isolated output directory."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path("/fs04/ar57/wenyu")
STAGE12_PATH = REPO_ROOT / "PersoanlQuery" / "12_complexity_analysis" / "12_complexity_analysis.py"
SMOKE_OUTPUT_ROOT = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_smoke"


def load_stage12_module():
    spec = importlib.util.spec_from_file_location("stage12_complexity_analysis", STAGE12_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 stage12 模块: {STAGE12_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    module = load_stage12_module()
    module.STAGE12_ROOT = SMOKE_OUTPUT_ROOT
    summary = module.process_category("Baby_Products", max_rows=5)
    output = {
        "category": summary["category"],
        "num_feature_rows": summary["num_feature_rows"],
        "query_types": {
            query_type: {
                "num_queries": info["num_queries"],
                "methods": sorted(info["methods"].keys()),
            }
            for query_type, info in summary["query_types"].items()
        },
    }
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
