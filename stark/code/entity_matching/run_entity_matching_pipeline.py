#!/usr/bin/env python3
"""
Complete pipeline to build entity matching graph and generate HTML explorer.

This script combines the following steps:
1. Build entity matching graph from product_entities.json
2. Generate static HTML explorer for visualization

Usage:
    python run_entity_matching_pipeline.py --input /path/to/product_entities.json --output_dir ./processed/entity_matching_graph
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_build_graph(input_file: str, output_dir: str, include_main_cat: bool = True,
                   add_category_hierarchy: bool = True, entity_map: str = None) -> None:
    """
    Run the graph building script.

    Args:
        input_file: Path to product_entities.json
        output_dir: Directory to save graph files
        include_main_cat: Whether to include main category edges
        add_category_hierarchy: Whether to add category hierarchy edges
        entity_map: Path to entity resolution map (optional)
    """
    print("🔄 Step 1: Building entity matching graph...")

    cmd = [
        sys.executable,
        str(Path(__file__).parent / "build_entity_matching_graph.py"),
        "--input", input_file,
        "--output_dir", output_dir
    ]

    if entity_map:
        cmd.extend(["--entity_map", entity_map])

    if not include_main_cat:
        cmd.append("--no_include_main_cat")

    if not add_category_hierarchy:
        cmd.append("--no_add_category_hierarchy")

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)

    if result.returncode != 0:
        print("❌ Graph building failed!")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        raise RuntimeError("Graph building step failed")

    print("✅ Graph building completed successfully")


def run_build_html(graph_dir: str, output_html: str = "", default_asin: str = "B000I7OIPI") -> str:
    """
    Run the HTML explorer building script.

    Args:
        graph_dir: Directory containing graph files
        output_html: Output HTML file path (optional)
        default_asin: Default ASIN to display on load

    Returns:
        Path to the generated HTML file
    """
    print("🔄 Step 2: Generating HTML explorer...")

    cmd = [
        sys.executable,
        str(Path(__file__).parent / "build_entity_matching_explorer_html.py"),
        "--graph_dir", graph_dir,
        "--default_asin", default_asin
    ]

    if output_html:
        cmd.extend(["--out", output_html])

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)

    if result.returncode != 0:
        print("❌ HTML generation failed!")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        raise RuntimeError("HTML generation step failed")

    # Determine output path
    if output_html:
        html_path = output_html
    else:
        html_path = os.path.join(graph_dir, "entity_matching_explorer.html")

    if os.path.exists(html_path):
        print(f"✅ HTML explorer generated successfully: {html_path}")
        return html_path
    else:
        raise FileNotFoundError(f"Expected HTML file not found: {html_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Complete pipeline: Build entity matching graph and generate HTML explorer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default paths
  python run_entity_matching_pipeline.py

  # Specify custom input and output
  python run_entity_matching_pipeline.py \\
    --input ./result/product_entities.json \\
    --output_dir ./processed/entity_matching_graph \\
    --html_output ./entity_matching_explorer.html \\
    --default_asin B000I7OIPI

  # Customize graph building options
  python run_entity_matching_pipeline.py \\
    --no_include_main_cat \\
    --no_add_category_hierarchy
        """
    )

    # Input/output options
    parser.add_argument(
        "--input",
        type=str,
        default="/home/wlia0047/ar57/wenyu/result/product_entities.json",
        help="Path to product_entities.json (default: %(default)s)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./processed/entity_matching_graph",
        help="Directory to save graph files (default: %(default)s)"
    )
    parser.add_argument(
        "--html_output",
        type=str,
        default="",
        help="Output HTML file path (default: <output_dir>/entity_matching_explorer.html)"
    )
    parser.add_argument(
        "--entity_map",
        type=str,
        default=None,
        help="Path to entity_resolution_map.json (optional)"
    )

    # Graph building options
    parser.add_argument(
        "--include_main_cat",
        action="store_true",
        default=True,
        help="Include main category edges in graph (default: True)"
    )
    parser.add_argument(
        "--no_include_main_cat",
        action="store_false",
        dest="include_main_cat",
        help="Do not include main category edges"
    )
    parser.add_argument(
        "--add_category_hierarchy",
        action="store_true",
        default=True,
        help="Add category hierarchy edges (default: True)"
    )
    parser.add_argument(
        "--no_add_category_hierarchy",
        action="store_false",
        dest="add_category_hierarchy",
        help="Do not add category hierarchy edges"
    )

    # HTML generation options
    parser.add_argument(
        "--default_asin",
        type=str,
        default="B000I7OIPI",
        help="Default ASIN to display on load (default: %(default)s)"
    )

    args = parser.parse_args()

    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        sys.exit(1)

    print("🚀 Starting entity matching pipeline...")
    print(f"Input: {input_path}")
    print(f"Graph output dir: {args.output_dir}")
    print(f"HTML output: {args.html_output or '<output_dir>/entity_matching_explorer.html'}")
    print()

    try:
        # Step 1: Build graph
        run_build_graph(
            str(input_path),
            args.output_dir,
            args.include_main_cat,
            args.add_category_hierarchy,
            args.entity_map
        )

        # Step 2: Build HTML
        html_path = run_build_html(
            args.output_dir,
            args.html_output,
            args.default_asin
        )

        print()
        print("🎉 Pipeline completed successfully!")
        print(f"📁 Graph files: {args.output_dir}")
        print(f"🌐 HTML explorer: {html_path}")
        print()
        print("To view the explorer, open the HTML file in a web browser.")

    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

