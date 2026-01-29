
import json
import gzip
import os
from collections import defaultdict
from tqdm import tqdm

def extract_details_values(input_file, output_file):
    """
    从商品元数据的 details 字段中提取属性值
    
    Args:
        input_file: 元数据文件路径 (gzip 压缩的 JSON Lines)
        output_file: 输出文件路径
    """
    # Mapping various keys to normalized categories
    # Using lowercase for key matching to be robust
    key_mapping = {
        'shipping weight:': 'Shipping Weight',
        'item model number:': 'Item Model Number',
        'asin:': 'ASIN',
        'asin: ': 'ASIN',
        'product dimensions:': 'Product Dimensions',
        '\n    product dimensions: \n    ': 'Product Dimensions',
        'domestic shipping: ': 'Domestic Shipping',
        'international shipping: ': 'International Shipping',
        'upc:': 'UPC',
        'item weight:': 'Item Weight',
        '\n    item weight: \n    ': 'Item Weight',
        'publisher:': 'Publisher',
        'language:': 'Language',
        'label:': 'Label',
        'number of discs:': 'Number Of Discs',
        'audio cd:': 'Audio CD',
        'package dimensions:': 'Package Dimensions',
        '\n    package dimensions: \n    ': 'Package Dimensions',
        'shipping advisory:': 'Shipping Advisory',
        'paperback:': 'Paperback',
        'date first listed on amazon:': 'Date First Listed',
        'isbn-10:': 'ISBN-10',
        'isbn-13:': 'ISBN-13',
        'run time:': 'Run Time',
        'discontinued by manufacturer:': 'Discontinued',
        'subtitles:': 'Subtitles',
        'spars code:': 'SPARS Code',
        'original release date:': 'Original Release Date',
        'pamphlet:': 'Pamphlet',
        'diary:': 'Diary',
        'series:': 'Series',
        'spiral-bound:': 'Spiral-bound',
        'plastic comb:': 'Plastic Comb',
        'hardcover:': 'Hardcover',
        'manufacturer:': 'Manufacturer',
        'item dimensions:': 'Item Dimensions',
        'color:': 'Color',
        'size:': 'Size',
        'material:': 'Material',
        'brand:': 'Brand',
        'model:': 'Model',
        'batteries required:': 'Batteries Required',
        'batteries included:': 'Batteries Included',
        'battery type:': 'Battery Type',
    }

    # Initialize storage for values: category -> {value: set of ASINs}
    collected_values = defaultdict(lambda: defaultdict(set))

    print(f"Processing file: {input_file}")
    
    try:
        # Determine open function based on file extension
        open_func = gzip.open if input_file.endswith('.gz') else open
        
        with open_func(input_file, 'rt', encoding='utf-8') as f:
            for line in tqdm(f, desc="Reading metadata"):
                try:
                    data = json.loads(line.strip())
                    
                    # Get ASIN from the metadata
                    asin = data.get("asin", "")
                    if not asin:
                        continue
                    
                    # Extract from details field
                    if "details" in data and isinstance(data["details"], dict):
                        for key, value in data["details"].items():
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
    for category, values in sorted(final_output.items()):
        total_asins = sum(len(asins) for asins in values.values())
        print(f"  {category}: {len(values)} unique values, {total_asins} total ASIN references")

    # Save to file
    print(f"\nSaving results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    
    print("Done!")

if __name__ == "__main__":
    # Configuration
    INPUT_FILE = "/home/wlia0047/ar57/wenyu/data/Amazon-Reviews-2023/raw/meta_All_Beauty.json.gz"
    OUTPUT_FILE = "/home/wlia0047/ar57/wenyu/result/KgData/details_values_collection.json"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    extract_details_values(INPUT_FILE, OUTPUT_FILE)
