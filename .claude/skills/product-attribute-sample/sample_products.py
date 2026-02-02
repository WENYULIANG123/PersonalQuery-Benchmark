import os
import sys
import json
import random
import csv
import argparse

# Ensure stark code and root are on Python path
CODE_DIR = "/home/wlia0047/ar57/wenyu/stark/code"
STARK_ROOT = "/home/wlia0047/ar57/wenyu/stark"

if STARK_ROOT not in sys.path:
    sys.path.insert(0, STARK_ROOT)
if CODE_DIR not in sys.path:
    sys.path.append(CODE_DIR)

try:
    from stark_qa.skb.amazon import AmazonSKB
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

def select_attributes(skb, node_idx) -> list:
    """Select 3 random attributes for a given product node."""
    node = skb[node_idx]
    candidates = []
    
    # 1. Brand
    if hasattr(node, 'brand') and node.brand:
        candidates.append({"type": "Brand", "value": node.brand})
    
    # 2. Category
    if hasattr(node, 'category') and node.category:
        candidates.append({"type": "Category", "value": node.category[-1]})
        
    # 3. Color
    if hasattr(node, 'color_name') and node.color_name:
         candidates.append({"type": "Color", "value": node.color_name})

    # 4. Features
    if hasattr(node, 'feature') and node.feature:
        for f in node.feature[:5]:
            if f and len(f) < 100:
                candidates.append({"type": "Feature", "value": f})

    if len(candidates) < 3:
        return candidates
    
    return random.sample(candidates, 3)

def main():
    parser = argparse.ArgumentParser(description="Sample products and extract attributes.")
    parser.add_argument("--size", type=int, default=500, help="Number of products to sample.")
    parser.add_argument("--output", type=str, default="/home/wlia0047/ar57/wenyu/result/sample_product_attributes.csv", help="Output CSV path.")
    args = parser.parse_args()

    skb_root = "/home/wlia0047/ar57/wenyu/data/Amazon-Reviews-2018"
    print(f"📦 Loading AmazonSKB from {skb_root}...")
    skb = AmazonSKB(root=skb_root, categories=['Arts_Crafts_and_Sewing'])
    
    product_indices = [i for i, t in enumerate(skb.node_types) if skb.node_type_dict[int(t)] == 'product']
    print(f"Found {len(product_indices)} total products in SKB.")
    
    random.shuffle(product_indices)
    print(f"Filtering for products with at least 3 attributes...")
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    results = []
    count = 0
    for idx in product_indices:
        if count >= args.size:
            break
            
        node = skb[idx]
        asin = getattr(node, 'asin', 'UNKNOWN')
        attrs = select_attributes(skb, idx)
        
        # Only accept products with at least 3 attributes
        if len(attrs) < 3:
            continue
            
        results.append({
            "id": count,
            "query": json.dumps(attrs, ensure_ascii=False),
            "answer_ids_source": json.dumps([asin], ensure_ascii=False)
        })
        count += 1
        
    print(f"Successfully sampled {len(results)} high-quality products.")
        
    print(f"💾 Saving results to {args.output}...")
    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'query', 'answer_ids_source'])
        writer.writeheader()
        writer.writerows(results)
        
    print("✅ Done!")

if __name__ == "__main__":
    main()
