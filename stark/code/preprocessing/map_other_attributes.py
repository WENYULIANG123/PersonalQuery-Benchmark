
import json
import os
import re
from collections import defaultdict

def normalize_quantity(qty_str):
    if not qty_str:
        return ""
    try:
        # Try to convert to float then to int to remove .0
        val = float(qty_str)
        if val == int(val):
            return str(int(val))
        return str(val)
    except ValueError:
        return qty_str.strip()

def normalize_length(length_str):
    if not length_str:
        return ""
    s = length_str.lower().strip()
    
    # 1. Standardize Units
    # Inches: inches, inch, in. -> in
    s = re.sub(r'(\d+(?:\.\d+)?)\s?(?:inches?|in\.?|in\b)', r'\1in', s)
    # Yards: yards, yard, yd. -> yd
    s = re.sub(r'(\d+(?:\.\d+)?)\s?(?:yards?|yd\.?|yds\.?|yd\b)', r'\1yd', s)
    
    # 2. Numeric cleanup (remove trailing .00)
    s = re.sub(r'(\d+)\.00(?=\D|$)', r'\1', s)
    
    return s.title() if not s[0].isdigit() else s

def normalize_pattern(pattern_str):
    if not pattern_str:
        return ""
    s = pattern_str.lower().strip()
    
    # Remove Model IDs like BIJ-xxx
    if re.search(r'bij-\d+', s):
        return ""
    
    # Remove obvious noise (10 pack, 8 inch)
    if re.search(r'\d+\s?(?:pack|inch|kits|mv|needles?)', s):
        return ""
        
    # Synonyms merging
    synonyms = {
        'floral': ['floral', 'flowers', 'flower'],
        'striped': ['striped', 'stripes', 'stripe'],
        'checkered': ['checkered', 'check', 'checks'],
        'dotted': ['dotted', 'dots', 'dot'],
        'animal': ['leopard', 'zebra', 'tiger', 'wolves', 'wolves'],
        'angel': ['angel', 'angel wings'],
    }
    
    for standard, list_of_syns in synonyms.items():
        if any(syn in s for syn in list_of_syns):
            return standard.title()
            
    return s.title()

def normalize_material(mat_str):
    if not mat_str:
        return ""
    s = mat_str.lower().strip()
    
    # Casing normalization is handled by .title() at the end
    # Basic synonyms
    synonyms = {
        'sterling silver': ['sterling silver', '.925 sterling', '.925 silver'],
        'gold filled': ['gold filled', 'gold-filled', '14/20 gold filled'],
        'metallic': ['metallic', 'foil metallic'],
    }
    
    for standard, list_of_syns in synonyms.items():
        if any(syn in s for syn in list_of_syns):
            s = standard
            break
            
    # Remove obvious non-materials that leaked
    if any(noise in s for noise in ['cinnamon', 'paint set', 'hammer', '60"', 'only fabric', 'apples']):
        return ""
        
    return s.title()

def clean_attribute(attr_data):
    """Generic cleaner that handles frequency pruning after normalization."""
    mapped = defaultdict(set)
    garbage_asin_count = 0
    
    # Logic picker based on some heuristic or we could pass a function
    # For now, let's keep it simple and handle it in the caller loop
    return mapped

def map_other_attributes(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    target_attrs = {
        "Length": normalize_length,
        "Pattern": normalize_pattern,
        "Package Quantity": normalize_quantity,
        "Material": normalize_material,
        "Material Type": normalize_material
    }

    print("="*50)
    print("Other Attributes Normalization Report")
    print("="*50)

    for attr_key, norm_func in target_attrs.items():
        if attr_key not in data:
            continue
            
        raw_values = data[attr_key]
        unique_before = len(raw_values)
        
        mapped = defaultdict(set)
        for val, asins in raw_values.items():
            norm_val = norm_func(val)
            if norm_val:
                mapped[norm_val].update(asins)
        
        # Frequency Pruning
        final_values = {}
        pruned_count = 0
        for val, asins in mapped.items():
            if len(asins) >= 5:
                final_values[val] = sorted(list(asins))
            else:
                pruned_count += 1
                
        unique_after = len(final_values)
        data[attr_key] = final_values
        
        print(f"{attr_key}: {unique_before} unique -> {unique_after} unique (Pruned: {pruned_count})")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print("="*50)

if __name__ == "__main__":
    INPUT = "/home/wlia0047/ar57/wenyu/result/KgData/style_values_collection.json"
    OUTPUT = "/home/wlia0047/ar57/wenyu/result/KgData/style_values_collection.json"
    map_other_attributes(INPUT, OUTPUT)
