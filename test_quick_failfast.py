#!/usr/bin/env python3
"""快速fail-fast验证 - 处理前 5 个产品"""

import sys
import json
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/.claude/skills")

from PersoanlQuery.src.preference_extraction_module import process_product

with open("/fs04/ar57/wenyu/result/personal_query/00_data_preparation/reviews_A13OFOB1394G31.json") as f:
    data = json.load(f)

products = data['results'][:5]
print(f"处理 {len(products)} 个产品\n", flush=True)

for i, product in enumerate(products, 1):
    asin = product.get('asin', 'UNKNOWN')
    try:
        result = process_product(product)
        print(f"[{i}] ✅ {asin}: {len(result.get('target_user_aspects', []))} 个方面", flush=True)
    except Exception as e:
        print(f"[{i}] ❌ {asin}: {type(e).__name__}: {str(e)[:100]}", flush=True)

print("\n完成", flush=True)
