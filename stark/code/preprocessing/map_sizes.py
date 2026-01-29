
import json
import os
import re
from collections import defaultdict

def normalize_size_string(size_str):
    if not size_str:
        return ""
    
    # 1. Basic Cleaning
    s = size_str.lower().strip()
    
    # Remove HTML entities like &#x215B;
    s = re.sub(r'&#x[0-9a-fA-F]+;', ' ', s)
    
    # Remove leading noise symbols like !!, #, *
    s = re.sub(r'^[!#*]+', '', s)
    
    # 2. Noise Phrase Removal (e.g., "Pack of 12", "Qty 5")
    noise_patterns = [
        r'\(?pack of \d+\)?', r'\d+[- ]?pack', r'pack of \d+',
        r'\(?set of \d+\)?', r'set of \d+',
        r'\(?qty \d+\)?', r'\(?quantity \d+\)?', r'qty \d+',
        r'box of \d+', r'\d+ pieces?', r'\d+\s?pcs',
        r'\(us \d+-\d+\)', r'text only', r'standard',
        r'\d+[- ]?count', r'\(?\d+ rolls?\)?', r'\(?\d+ sheets?\)?'
    ]
    for pattern in noise_patterns:
        s = re.sub(pattern, '', s)

    # Handle "Size-Us-X-(...)" patterns
    size_us_match = re.search(r'size-us-([\d.]+)', s)
    if size_us_match:
        # Extract the content inside parentheses if it exists as it's often more standard (e.g. 5mm)
        paren_match = re.search(r'\((.*?)\)', s)
        if paren_match:
            s = paren_match.group(1)
        else:
            s = size_us_match.group(1)

    # 3. Unit Standardization
    # Inches: ", inch, inches, in. -> in
    s = re.sub(r'(\d+(?:\.\d+)?)\s?(?:"|inches?|in\.?|in\b)', r'\1in', s)
    # Ounces: oz, ounce, ounces -> oz
    s = re.sub(r'(\d+(?:\.\d+)?)\s?(?:ounces?|oz\.?)', r'\1oz', s)
    # Millimeters: mm, millimeter -> mm
    s = re.sub(r'(\d+(?:\.\d+)?)\s?(?:millimeters?|mm\.?)', r'\1mm', s)
    # Yards: yd, yard, yards -> yd
    s = re.sub(r'(\d+(?:\.\d+)?)\s?(?:yards?|yd\.?|yds\.?)', r'\1yd', s)
    # Centimeters: cm, centimeter -> cm
    s = re.sub(r'(\d+(?:\.\d+)?)\s?(?:centimeters?|cm\.?)', r'\1cm', s)
    
    # 4. Standard Size Mapping (Clothing/etc)
    std_mapping = {
        r'\bxs\b': 'X-Small',
        r'\bs\b': 'Small',
        r'\bsmall\b': 'Small',
        r'\bm\b': 'Medium',
        r'\bmedium\b': 'Medium',
        r'\bl\b': 'Large',
        r'\blarge\b': 'Large',
        r'\bxl\b': 'X-Large',
        r'\bxxl\b': 'XX-Large',
        r'\bxxxl\b': 'XXX-Large',
        r'\bextra small\b': 'X-Small',
        r'\bextra large\b': 'X-Large',
        r'\bone size\b': 'One Size'
    }

    for pattern, replacement in std_mapping.items():
        if re.search(pattern, s):
            s = replacement
            break
            
    # Final cleanup: remove trailing/leading punctuation
    # Added : to the strip list
    s = s.strip(' -,\t\n\r.!"():')
    
    # If the remaining string is just a single digit like '1', '2', it's often a weight or count that leaked
    # But for hooks/needles it could be size. However '1' and '1:' are usually noise.
    # We will let the frequency pruning handle it if it's real, 
    # but we can filter out empty strings after stripping.
    if not s:
        return ""
    
    # Capitalize first letter of each part if it's text, or title case
    if s and not s[0].isdigit():
        return s.title()
    return s

def map_sizes(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if "Size" not in data:
        print("Error: No 'Size' attribute found.")
        return

    raw_sizes = data["Size"]
    
    # Stats before
    unique_before = len(raw_sizes)
    asin_refs_before = sum(len(asins) for asins in raw_sizes.values())

    # Perform Mapping
    mapped_sizes = defaultdict(set)
    unmapped_garbage = 0

    for size_name, asins in raw_sizes.items():
        norm_name = normalize_size_string(size_name)
        
        # Stricter filtering for the final set
        # Discard if empty or just symbols or very long messy strings
        if not norm_name or len(norm_name) > 50 or re.match(r'^[^\w\s]+$', norm_name):
            unmapped_garbage += len(asins)
            continue
            
        mapped_sizes[norm_name].update(asins)

    # 5. Frequency Pruning: After normalization, discard items with < 5 ASINs
    final_sizes = {}
    pruned_count = 0
    for size, asins in mapped_sizes.items():
        if len(asins) >= 5:
            final_sizes[size] = sorted(list(asins))
        else:
            pruned_count += 1

    data["Size"] = final_sizes

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Stats after
    unique_after = len(final_sizes)
    asin_refs_after = sum(len(asins) for asins in final_sizes.values())

    print("="*50)
    print("Size Normalization Report")
    print("="*50)
    print(f"Unique Sizes: {unique_before} -> {unique_after}")
    print(f"Total ASIN References: {asin_refs_before} -> {asin_refs_after}")
    print(f"Pruned (Low Frequency < 5): {pruned_count} entries")
    print(f"Discarded (Noise/Garbage): {unmapped_garbage} ASIN references")
    
    # Sample top 10
    print("\nTop 10 Standardized Sizes:")
    sorted_sizes = sorted(final_sizes.items(), key=lambda x: len(x[1]), reverse=True)
    for name, asins in sorted_sizes[:10]:
        print(f"  - {name}: {len(asins)} ASINs")
    print("="*50)

if __name__ == "__main__":
    INPUT = "/home/wlia0047/ar57/wenyu/result/KgData/style_values_collection.json"
    OUTPUT = "/home/wlia0047/ar57/wenyu/result/KgData/style_values_collection.json"
    map_sizes(INPUT, OUTPUT)
