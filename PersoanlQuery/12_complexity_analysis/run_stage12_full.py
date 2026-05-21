#!/usr/bin/env python3
"""Unique entrypoint for full Stage 12 execution."""

from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path("/fs04/ar57/wenyu")
STAGE12_PATH = REPO_ROOT / "PersoanlQuery" / "12_complexity_analysis" / "12_complexity_analysis.py"


def load_stage12_module():
    spec = importlib.util.spec_from_file_location("stage12_complexity_analysis", STAGE12_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 stage12 模块: {STAGE12_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    module = load_stage12_module()
    summaries = {}
    total_rows = 0
    for category in module.DEFAULT_CATEGORIES:
        summaries[category] = module.process_category(category, None)
        total_rows += int(summaries[category]["num_feature_rows"])

    root_summary = {
        "timestamp": module.datetime.now().isoformat(),
        "output_root": str(module.STAGE12_ROOT),
        "categories": list(summaries.keys()),
        "num_categories": len(summaries),
        "num_feature_rows": total_rows,
        "summaries": {
            category: {
                "summary_file": str(module.STAGE12_ROOT / category / "stage12_summary.json"),
                "feature_file": str(module.STAGE12_ROOT / category / "query_features.jsonl"),
                "num_feature_rows": summaries[category]["num_feature_rows"],
            }
            for category in summaries
        },
    }
    module.write_json_file(module.STAGE12_ROOT / "stage12_root_summary.json", root_summary)

    module.log("=" * 80)
    module.log("Stage 12 完成")
    for category in summaries:
        module.log(f"  {category}: {summaries[category]['num_feature_rows']} 条特征记录")
    module.log(f"  总记录数: {total_rows}")
    module.log(f"  根汇总: {module.STAGE12_ROOT / 'stage12_root_summary.json'}")
    module.log("=" * 80)


if __name__ == "__main__":
    main()
