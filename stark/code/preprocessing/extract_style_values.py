
import json
import gzip
import os
from collections import defaultdict
from tqdm import tqdm

def extract_style_values(input_file, output_file):
    # Mapping various keys to the 9 target categories
    # Using lowercase for key matching to be robust
    key_mapping = {
        'size:': 'Size',
        'size name:': 'Size',
        'color:': 'Color',
        'color name:': 'Color',
        'style:': 'Style',
        'style name:': 'Style',
        'format:': 'Format',
        'package type:': 'Package Type',
        'design:': 'Design',
        'package quantity:': 'Package Quantity',
        'item package quantity:': 'Package Quantity',
        'length:': 'Length',
        'item display length:': 'Length',
        'product packaging:': 'Product Packaging', 
        'pattern:': 'Pattern'
    }

    # Initialize storage for values: category -> {value: set of ASINs}
    collected_values = defaultdict(lambda: defaultdict(set))

    print(f"Processing file: {input_file}")
    
    try:
        # Determine open function based on file extension
        open_func = gzip.open if input_file.endswith('.gz') else open
        
        with open_func(input_file, 'rt', encoding='utf-8') as f:
            for line in tqdm(f, desc="Reading reviews"):
                try:
                    data = json.loads(line.strip())
                    
                    # Get ASIN from the review data
                    asin = data.get("asin", "")
                    if not asin:
                        continue
                    
                    if "style" in data and isinstance(data["style"], dict):
                        for key, value in data["style"].items():
                            # Normalize key: remove whitespace and lower case for matching
                            norm_key = key.strip().lower()
                            
                            # Get target category: use mapping if exists, else use Title Case of the key
                            target_category = key_mapping.get(norm_key)
                            if not target_category:
                                # Clean up key provided (remove colon if at end, title case)
                                clean_key = norm_key.rstrip(':')
                                target_category = clean_key.title()
                            
                            # Add the ASIN to the set for this value in this category
                            if value:
                                val_str = str(value).strip()
                                if val_str:
                                    collected_values[target_category][val_str].add(asin)
                                    
                except json.JSONDecodeError:
                    continue
                    
    except FileNotFoundError:
        print(f"Error: Input file not found: {input_file}")
        return
    except Exception as e:
        print(f"An error occurred: {e}")
        return

    # Convert sets to sorted lists for JSON serialization
    final_output = {
        category: {
            value: sorted(list(asins))
            for value, asins in sorted(values.items())
        }
        for category, values in collected_values.items()
    }
    
    # Print stats
    print("\nExtraction Statistics:")
    for category, values in final_output.items():
        total_asins = sum(len(asins) for asins in values.values())
        print(f"  {category}: {len(values)} unique values, {total_asins} total ASIN references")

    # Save to file
    print(f"\nSaving results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    
    print("Done!")

if __name__ == "__main__":
    # Configuration
    INPUT_FILE = "/home/wlia0047/ar57/wenyu/data/Amazon-Reviews-2018/raw/Arts_Crafts_and_Sewing.json.gz"
    OUTPUT_FILE = "/home/wlia0047/ar57/wenyu/result/KgData/style_values_collection.json"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    extract_style_values(INPUT_FILE, OUTPUT_FILE)
