
import json
import gzip
import os
import re
from collections import defaultdict
from tqdm import tqdm

def extract_direct_attributes(input_file, output_file):
    """
    Explore metadata and extract attributes.
    Strict logic: brand value is ONLY taken from the 'brand' field in metadata.

    Args:
        input_file: Raw metadata .json.gz
        output_file: Path to save extracted attributes
    """
    # Pass: Extract and Collect
    collected_attributes = {
        'brand': defaultdict(set),
        'main_category': defaultdict(set),
        'category': defaultdict(set),
        'price_range': defaultdict(set),
    }

    stats = {
        'total_products': 0,
        'with_brand': 0,
        'with_main_cat': 0,
        'with_category': 0,
        'with_price': 0,
        'with_rank': 0,
    }

    print(f"Extracting attributes from {input_file} (Strict Brand logic)...")
    open_func = gzip.open if input_file.endswith('.gz') else open
    with open_func(input_file, 'rt', encoding='utf-8') as f:
        for line in tqdm(f, desc="Processing Items"):
            try:
                data = json.loads(line.strip())
                asin = data.get("asin", "")
                if not asin:
                    continue

                stats['total_products'] += 1
                brand = data.get("brand", "").strip()

                if brand:
                    collected_attributes['brand'][brand].add(asin)
                    stats['with_brand'] += 1

                # (Standard Extraction for other fields)
                main_cat = data.get("main_cat", "").strip()
                if main_cat and '<' not in main_cat:
                    stats['with_main_cat'] += 1
                    collected_attributes['main_category'][main_cat].add(asin)

                category = data.get("category", [])
                if category and isinstance(category, list):
                    stats['with_category'] += 1
                    for cat in category:
                        cat_clean = cat.strip()
                        if cat_clean and cat_clean.lower() != 'category':
                            collected_attributes['category'][cat_clean].add(asin)

                price = data.get("price", "").strip()
                if price and '$' in price:
                    stats['with_price'] += 1
                    try:
                        price_str = price.replace('$', '').replace(',', '')
                        price_val = float(price_str.split()[0])
                        if price_val < 10: price_range = "Under $10"
                        elif price_val < 25: price_range = "$10-$25"
                        elif price_val < 50: price_range = "$25-$50"
                        elif price_val < 100: price_range = "$50-$100"
                        elif price_val < 200: price_range = "$100-$200"
                        else: price_range = "$200+"
                        collected_attributes['price_range'][price_range].add(asin)
                    except:
                        pass

                if data.get("rank"):
                    stats['with_rank'] += 1

            except Exception:
                continue

    # Final output conversion
    final_output = {
        attr_type: {
            value: sorted(list(asins))
            for value, asins in sorted(values.items())
        }
        for attr_type, values in collected_attributes.items()
        if values
    }

    # Print statistics
    print("\n" + "=" * 80)
    print("Extraction Statistics")
    print("=" * 80)
    print(f"Total products processed: {stats['total_products']}")
    print(f"  - With brand: {stats['with_brand']} ({stats['with_brand']/stats['total_products']*100:.1f}%)")

    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract attributes from Amazon metadata.")
    parser.add_argument('--input_file', type=str, default="/home/wlia0047/ar57/wenyu/data/Amazon-Reviews-2018/raw/meta_Arts_Crafts_and_Sewing.json.gz")
    parser.add_argument('--output_file', type=str, default="/home/wlia0047/ar57/wenyu/result/KgData/direct_attributes_collection.json")

    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    extract_direct_attributes(args.input_file, args.output_file)
