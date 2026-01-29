
import json
import os
import re
from collections import defaultdict

def normalize_style_string(style_str):
    if not style_str:
        return ""
    
    # 1. Basic Cleaning
    s = style_str.lower().strip()
    
    # Remove leading noise symbols like #, *, &, .
    s = re.sub(r'^[#*&.\s]+', '', s)
    
    # 2. Redundancy Removal: Dimensions and Units (Redundant with Size/Details)
    # Match patterns like 11" x 14", 12mm, 1.4oz, 1/8", 10x10, etc.
    dimension_patterns = [
        r'\d+(?:\.\d+)?\s?(?:"|inches?|in\.?|in\b)\s?x\s?\d+(?:\.\d+)?\s?(?:"|inches?|in\.?|in\b)', # 11" x 14"
        r'\d+(?:\.\d+)?\s?(?:mm|cm|oz|in|yd|ft|lb|magnification)\b', # 12mm, 1.4oz
        r'\d+\s?x\s?\d+\b', # 10x10
        r'\d+/\d+(?:"|in|inch|inches)\b', # 1/8"
        r'\b\d+\s?grid\b'
    ]
    for pattern in dimension_patterns:
        if re.search(pattern, s):
            # If the entire style is just a dimension, discard it
            if len(re.sub(pattern, '', s).strip()) < 3:
                return ""
            # Otherwise just strip the dimension part
            s = re.sub(pattern, '', s).strip()

    # 3. Packaging/Quantity Cleanup
    noise_patterns = [
        r'\(?pack of \d+\)?', r'\d+[- ]?pack', r'pack of \d+',
        r'\(?set of \d+\)?', r'set of \d+',
        r'\(?qty \d+\)?', r'\(?quantity \d+\)?', r'qty \d+',
        r'box of \d+', r'\d+ pieces?', r'\d+\s?pcs',
        r'\d+[- ]?count', r'\(?\d+ rolls?\)?', r'\(?\d+ sheets?\)?'
    ]
    for pattern in noise_patterns:
        s = re.sub(pattern, '', s)
    
    # 4. Final Polish
    s = s.strip(' -,\t\n\r.!"():/_&')
    
    # Discard if it's too short, too long (likely a title), or just numbers
    if len(s) < 2 or len(s) > 50 or s.isdigit():
        return ""
    
    # Discard nonsense/garbled
    if re.match(r'^[^\w\s]+$', s):
        return ""
        
    return s.title()

def map_styles(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if "Style" not in data:
        print("Error: No 'Style' attribute found.")
        return

    raw_styles = data["Style"]
    
    # Stats before
    unique_before = len(raw_styles)
    asin_refs_before = sum(len(asins) for asins in raw_styles.values())

    # Perform Mapping
    mapped_styles = defaultdict(set)
    unmapped_garbage_count = 0

    for style_name, asins in raw_styles.items():
        norm_name = normalize_style_string(style_name)
        
        if not norm_name:
            unmapped_garbage_count += len(asins)
            continue
            
        mapped_styles[norm_name].update(asins)

    # 5. Frequency Pruning: After normalization, discard items with < 5 ASINs
    final_styles = {}
    pruned_count = 0
    for style, asins in mapped_styles.items():
        if len(asins) >= 5:
            final_styles[style] = sorted(list(asins))
        else:
            pruned_count += 1

    data["Style"] = final_styles

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Stats after
    unique_after = len(final_styles)
    asin_refs_after = sum(len(asins) for asins in final_styles.values())

    print("="*50)
    print("Style Normalization Report")
    print("="*50)
    print(f"Unique Styles: {unique_before} -> {unique_after}")
    print(f"Total ASIN References: {asin_refs_before} -> {asin_refs_after}")
    print(f"Pruned (Low Frequency < 5): {pruned_count} entries")
    print(f"Discarded (Noise/Garbage): {unmapped_garbage_count} ASIN references")
    
    # Sample top 10
    print("\nTop 10 Standardized Styles:")
    sorted_styles = sorted(final_styles.items(), key=lambda x: len(x[1]), reverse=True)
    for name, asins in sorted_styles[:10]:
        print(f"  - {name}: {len(asins)} ASINs")
    print("="*50)

if __name__ == "__main__":
    INPUT = "/home/wlia0047/ar57/wenyu/result/KgData/style_values_collection.json"
    OUTPUT = "/home/wlia0047/ar57/wenyu/result/KgData/style_values_collection.json"
    map_styles(INPUT, OUTPUT)
