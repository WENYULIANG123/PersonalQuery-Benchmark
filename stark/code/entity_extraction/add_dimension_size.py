#!/usr/bin/env python3
"""
Script to add dimension_size (大/中/小) field to existing standardized_dimensions in JSON file.
"""
import json
import sys
import os

# Make sibling modules importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate_query.normalize_dimensions import map_dimension_to_size  # type: ignore


def update_json_file(input_path: str, output_path: str = None) -> None:
    """
    Update existing JSON file to add dimension_size field to standardized_dimensions.
    
    Args:
        input_path: Path to input JSON file
        output_path: Path to output JSON file (if None, overwrites input)
    """
    if output_path is None:
        output_path = input_path
    
    print(f"📖 Reading from: {input_path}")
    
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Handle two possible JSON shapes
    if isinstance(data, dict):
        products = data.get("products", [])
        total = len(products)
        updated_count = 0
        
        for idx, item in enumerate(products, 1):
            standardized_dims = item.get("standardized_dimensions", [])
            if standardized_dims:
                for dim_entry in standardized_dims:
                    # Only add dimension_size if it doesn't exist
                    if "dimension_size" not in dim_entry:
                        dim_entry["dimension_size"] = map_dimension_to_size(
                            dim_entry.get("dimension_raw", ""),
                            dim_entry.get("spec_name", "")
                        )
                        updated_count += 1
            
            if idx % 100 == 0:
                print(f"  Processed {idx}/{total} products...")
        
        print(f"✅ Updated {updated_count} dimension entries")
        data["products"] = products
    else:
        # List format
        total = len(data)
        updated_count = 0
        
        for idx, item in enumerate(data, 1):
            standardized_dims = item.get("standardized_dimensions", [])
            if standardized_dims:
                for dim_entry in standardized_dims:
                    # Only add dimension_size if it doesn't exist
                    if "dimension_size" not in dim_entry:
                        dim_entry["dimension_size"] = map_dimension_to_size(
                            dim_entry.get("dimension_raw", ""),
                            dim_entry.get("spec_name", "")
                        )
                        updated_count += 1
            
            if idx % 100 == 0:
                print(f"  Processed {idx}/{total} products...")
        
        print(f"✅ Updated {updated_count} dimension entries")
    
    print(f"💾 Saving to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print("✅ Done!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Add dimension_size field to standardized_dimensions")
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to input JSON file"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Path to output JSON file (default: overwrite input file)"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"❌ Error: Input file not found: {args.input_file}")
        sys.exit(1)
    
    update_json_file(args.input_file, args.output)
