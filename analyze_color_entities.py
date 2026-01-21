#!/usr/bin/env python3
"""Analyze color entities from product_entities.json"""
import json
import sys
from collections import Counter

# Redirect output to file
output_file = open('color_entities_analysis.txt', 'w')
sys.stdout = output_file

# Load data
with open('result/product_entities.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

products = data.get('products', [])

# Collect all color entities
color_entities = []
color_keys = ['Color', 'Color/Finish', 'Colour', 'Colour/Finish']  # Various possible keys

for product in products:
    product_entities = product.get('product_entities', {})
    asin = product.get('asin', 'Unknown')
    
    # Check all possible color keys
    for key in color_keys:
        if key in product_entities:
            values = product_entities[key]
            if isinstance(values, list):
                for val in values:
                    if isinstance(val, str) and val.strip():
                        color_entities.append({
                            'asin': asin,
                            'color': val.strip(),
                            'key': key
                        })

# Count occurrences
color_counter = Counter([item['color'] for item in color_entities])
unique_colors = sorted(set([item['color'] for item in color_entities]))

print(f"=== Color Entities Analysis ===")
print(f"Total products: {len(products)}")
print(f"Products with color entities: {len(set(item['asin'] for item in color_entities))}")
print(f"Total color mentions: {len(color_entities)}")
print(f"Unique color values: {len(unique_colors)}")
print()

print("=== All Unique Color Entities (sorted) ===")
for i, color in enumerate(unique_colors, 1):
    count = color_counter[color]
    print(f"{i:3d}. {color:<40} (appears {count} time{'s' if count > 1 else ''})")

print()
print("=== Top 20 Most Frequent Colors ===")
for color, count in color_counter.most_common(20):
    print(f"{color:<40} : {count}")

print()
print("=== Color Entities by Product (first 20 products) ===")
seen_asins = set()
count = 0
for item in color_entities:
    if item['asin'] not in seen_asins:
        seen_asins.add(item['asin'])
        count += 1
        if count > 20:
            break
        # Get all colors for this product
        product_colors = [i['color'] for i in color_entities if i['asin'] == item['asin']]
        print(f"ASIN: {item['asin']:<15} Colors: {', '.join(set(product_colors))}")

output_file.close()
print("Analysis saved to color_entities_analysis.txt", file=sys.stderr)
