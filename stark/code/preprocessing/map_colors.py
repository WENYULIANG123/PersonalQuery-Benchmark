
import json
import os
import re
from collections import defaultdict

def map_colors(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if "Color" not in data:
        print("Error: No 'Color' attribute found in the file.")
        return

    raw_colors = data["Color"]
    
    # Define Base Colors and their keywords
    BASE_COLOR_MAP = {
        'Black': ['black', 'charcoal', 'jet', 'coal', 'onyx', 'obsidian', 'noir'],
        'White': ['white', 'ivory', 'cream', 'snow', 'pearl', 'alabaster', 'eggshell', 'blanc', 'ecru', 'linen', 'alabaster'],
        'Grey': ['grey', 'gray', 'silver', 'slate', 'ash', 'anthracite', 'platinum', 'nickel', 'chrome', 'pewter', 'gunmetal', 'nickle', 'aluminum', 'pewter', 'metallic'],
        'Red': ['red', 'crimson', 'maroon', 'burgundy', 'cherry', 'ruby', 'scarlet', 'wine', 'rust', 'vermilion', 'garnet', 'cranberry', 'raspberry', 'apple'],
        'Blue': ['blue', 'navy', 'sky', 'cyan', 'teal', 'royal', 'azure', 'sapphire', 'indigo', 'aqua', 'turquoise', 'cobalt', 'denim', 'peacock', 'teal blue', 'ocean', 'meridian'],
        'Green': ['green', 'olive', 'lime', 'emerald', 'forest', 'mint', 'sage', 'jade', 'army', 'moss', 'peridot', 'seafoam', 'avocado'],
        'Yellow': ['yellow', 'gold', 'lemon', 'mustard', 'canary', 'blonde', 'champagne', 'maize'],
        'Orange': ['orange', 'coral', 'amber', 'pumpkin', 'tangerine', 'apricot', 'peach'],
        'Purple': ['purple', 'violet', 'lavender', 'magenta', 'plum', 'orchid', 'lilac', 'amethyst', 'grape', 'eggplant', 'mauve', 'aubergine', 'periwinkle'],
        'Pink': ['pink', 'rose', 'fuchsia', 'salmon', 'flamingo', 'blush'],
        'Brown': ['brown', 'beige', 'tan', 'chocolate', 'khaki', 'bronze', 'coffee', 'copper', 'espresso', 'nut', 'sand', 'taupe', 'wood', 'mahogany', 'sienna', 'umber', 'coyote', 'camel', 'brass', 'desert', 'arid', 'canvas', 'almond', 'terracotta'],
        'Clear': ['clear', 'transparent', 'translucent', 'crystal', 'clear ab'],
        'Multicolor': ['multicolor', 'multi', 'rainbow', 'variegated', 'mixed', 'assortment', 'colorful', 'assorted', 'full color', 'acu', 'flower', 'pastel', 'stripes', 'autumn', 'assorted colors']
    }

    NOISE_WORDS = {
        # Descriptive / Version noise
        'opaque', 'natural', 'sew-on', 'original version', 'without frame', 'with frame', 
        'round', 'square', 'large', 'small', 'medium', 'pack', 'set of', 'piece', 
        'handmade', 'quality', 'new', 'official', 'licensed', 'version', 'genuine', 
        'item', 'accessory', 'original', 'basic', 'variety', 'random', 'color', 
        'aged', 'antique', 'matte', 'polished', 'flat', 'framed', 'frameless', 
        'wholesale', 'price', 'counts', 'box', 'mixed', 'assorted', 'variety',
        
        # Shapes
        'circle', 'oval', 'heart', 'star', 'alphabet', 'letter', 'number',
        
        # Events / Themes / Seasons
        'christmas', 'halloween', 'wedding', 'birthday', 'holiday', 'party',
        'autumn', 'spring', 'summer', 'winter', 'country', 'retro', 'vintage',
        'anniversary', 'baby shower', 'baptism', 'graduation',
        
        # Animals / Objects (that are often patterns, not individual colors)
        'butterfly', 'zebra', 'leopard', 'cat', 'dog', 'dolphin', 'elephant', 'owl', 
        'panda', 'unicorn', 'ladybug', 'flower', 'floral', 'leaf', 'tree', 'clover',
        'butterflys', 'butterflies', 'bear', 'giraffe', 'horse', 'lion', 'monkey',
        
        # Technical / Units
        '8mm', '6mm', 'mm', 'inch', 'mil', 'gauge', 'lb', 'qty', 'pcs', 'pkg',
        '10mm', '12mm', '14mm', '16mm', '18mm', '20mm', '3mm', '4mm', '5mm'
    }

    # Tracking stats
    before_asins = set()
    for asins in raw_colors.values():
        before_asins.update(asins)
    
    total_associations_before = sum(len(asins) for asins in raw_colors.values())
    unique_colors_before = len(raw_colors)

    # Perform mapping
    mapped_colors = defaultdict(set)
    unmapped_colors = {} 

    for color_name, asins in raw_colors.items():
        lower_name = color_name.lower()
        
        # 1. Stricter Noise check
        # Remove if contains numbers, units, or is a known noise word
        is_noise = (
            lower_name in NOISE_WORDS or 
            len(lower_name) < 2 or 
            re.search(r'\d', lower_name) or # Any digit: 10mm, #1, 24 colors
            any(nw in lower_name for nw in ['pack', 'pcs', 'colors', 'qty', 'pkg'])
        )
        
        if is_noise:
            continue
            
        found = False
        for base_color, keywords in BASE_COLOR_MAP.items():
            if any(kw in lower_name for kw in keywords):
                mapped_colors[base_color].update(asins)
                found = True
                break
        
        if not found:
            # 2. Pruning check: Discard low-frequency unmapped noise
            # Only keep unmapped if it looks like a clean word and has enough frequency
            if len(asins) >= 5 and lower_name.isalpha(): 
                mapped_colors[color_name.title()].update(asins)
                unmapped_colors[color_name] = len(asins)

    # Convert sets back to sorted lists
    final_mapped = {
        color: sorted(list(asins))
        for color, asins in sorted(mapped_colors.items())
    }
    
    data["Color"] = final_mapped

    # Stats after
    after_asins = set()
    for asins in final_mapped.values():
        after_asins.update(asins)
    
    total_associations_after = sum(len(asins) for asins in final_mapped.values())
    unique_colors_after = len(final_mapped)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print("="*50)
    print("Color Mapping Report")
    print("="*50)
    print(f"Unique Color Names: {unique_colors_before} -> {unique_colors_after}")
    print(f"Total ASIN-Color Associations: {total_associations_before} -> {total_associations_after}")
    print(f"Unique ASINs covered: {len(before_asins)} -> {len(after_asins)}")
    print("="*50)

if __name__ == "__main__":
    INPUT = "/home/wlia0047/ar57/wenyu/result/KgData/style_values_collection.json"
    OUTPUT = "/home/wlia0047/ar57/wenyu/result/KgData/style_values_collection.json"
    map_colors(INPUT, OUTPUT)
